from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from gurobipy import GRB, quicksum
from gym_multigrid.envs.mdp import ProjectMDP

from src.components.episode_buffer import EpisodeBatch
from src.modules.gurobi_opt import (
    OptimizationProblem,
    SolutionBase,
    VariablesBase,
)


class InfeasiblePolicyError(RuntimeError):
    """Raised when no high-level policy can meet the requested success rate."""


@dataclass
class Variables(VariablesBase):
    task_occupancy: dict[tuple[int, int], object] = field(default_factory=dict)
    message_budget: dict[tuple[int, int, float], object] = field(default_factory=dict)
    selected_occupancy: dict[tuple[int, int, float], object] = field(
        default_factory=dict
    )


@dataclass
class Solution(SolutionBase):
    task_policy: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    comms_budget_policy: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    objective_value: float = 0.0
    goal_reach_probability: float = 0.0


class ILPModel(OptimizationProblem):
    """Optimize a stochastic subtask policy and deterministic capacity per edge.

    The paper's edge-occupancy formulation is a mixed-integer bilinear model.
    Since task graphs are DAGs, each edge occupancy is bounded by one. Exact
    binary-product linearization makes this a MILP while retaining the same
    flow, reachability, and expected communication-cost equations.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._hlmdp: ProjectMDP
        self.success_rate_spec: float
        self.opt_vars: Variables
        self.policy: Solution
        self._edge_budgets: dict[tuple[int, int], list[float]] = {}
        self._success_probs: dict[tuple[int, int, float], float] = {}
        self._graph = nx.DiGraph()
        self._cost_normalizer = 1
        self.fixed_comms_budget: float | None = None

    def optimize_policy(
        self,
        hlmdp: ProjectMDP,
        success_rate_spec: float = 0.95,
        fixed_comms_budget: float | None = None,
    ) -> Solution:
        """Solve from the transition probabilities estimated during LL evaluation."""
        if not 0.0 <= success_rate_spec <= 1.0:
            raise ValueError("success_rate_spec must be in [0, 1].")
        if fixed_comms_budget is not None and not np.isfinite(fixed_comms_budget):
            raise ValueError("fixed_comms_budget must be finite.")
        self._hlmdp = hlmdp
        self.success_rate_spec = success_rate_spec
        self.fixed_comms_budget = fixed_comms_budget
        self._load_model_data()
        self.opt_vars = Variables()
        self.policy = Solution()

        self.build_model()
        self.optimize()
        if not self.check_if_optima_found():
            if self.model.Status == GRB.INFEASIBLE:
                raise InfeasiblePolicyError(
                    "No high-level policy can meet the requested success rate "
                    f"specification {success_rate_spec}."
                )
            raise RuntimeError(
                "High-level policy optimization did not find an optimal solution. "
                "The problem may be infeasible; try a lower success rate spec."
            )
        return self._build_solution()

    def dispose(self) -> None:
        """Release the solver model and its explicitly-created Gurobi environment."""
        model = getattr(self, "model", None)
        model_env = getattr(self, "_model_env", None)
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "_model_env"):
            del self._model_env
        try:
            if model is not None:
                model.dispose()
        finally:
            if model_env is not None:
                model_env.dispose()

    def select_actions(
        self, ep_batch: EpisodeBatch, t_ep: int, t_env: int, test_mode: bool
    ) -> dict:
        """Sample the next task from the solved policy and return its capacity."""
        if self.policy.task_policy.empty:
            raise RuntimeError("Optimize the high-level policy before selecting actions.")

        hl_state = ep_batch["hl_state"][:, t_ep, :].cpu().numpy().flatten()
        state = int(hl_state[0])
        state_actions = self.policy.task_policy.loc[
            self.policy.task_policy.state == state
        ]
        if state_actions.empty:
            raise ValueError(f"The optimized high-level policy has no action for state {state}.")

        probabilities = state_actions.action_probability.to_numpy(
            dtype=float, copy=True
        )
        probabilities /= probabilities.sum()
        selected_index = int(np.random.choice(len(state_actions), p=probabilities))
        selected_action = state_actions.iloc[selected_index]
        next_state = int(selected_action.next_state)

        budget_rows = self.policy.comms_budget_policy.loc[
            (self.policy.comms_budget_policy.state == state)
            & (self.policy.comms_budget_policy.next_state == next_state)
            & (self.policy.comms_budget_policy.action_probability > 0.5)
        ]
        if len(budget_rows) > 1:
            raise RuntimeError(
                f"Multiple capacities selected for task {state}->{next_state}."
            )
        comms_budget = (
            float(budget_rows.iloc[0].msg_budget_per_agent)
            if not budget_rows.empty
            else 0.0
        )
        return {"chosen_next_state": next_state, "comms_budget": comms_budget}

    def cuda(self) -> None:
        pass

    def _load_model_data(self) -> None:
        transitions = self._hlmdp.transition_probs
        normal_rows = transitions.loc[transitions.state_type == "normal"]
        if self.fixed_comms_budget is not None:
            available_edges = {
                (int(row.state), int(row.action[0]))
                for row in normal_rows.itertuples()
            }
            matching_rows = normal_rows.loc[
                normal_rows.action.map(
                    lambda action: np.isclose(
                        float(action[1]), self.fixed_comms_budget
                    )
                )
            ]
            fixed_budget_edges = {
                (int(row.state), int(row.action[0]))
                for row in matching_rows.itertuples()
            }
            missing_edges = sorted(available_edges - fixed_budget_edges)
            if missing_edges:
                raise ValueError(
                    f"Fixed communication budget {self.fixed_comms_budget} is "
                    f"unavailable for task edges {missing_edges}."
                )
            normal_rows = matching_rows

        self._edge_budgets = {}
        self._success_probs = {}
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from(
            int(state)
            for state in transitions.loc[
                transitions.state_type.isin(["normal", "goal"]), "state"
            ].unique()
        )
        outcomes: dict[tuple[int, int, float], dict[int, float]] = {}

        for _, row in normal_rows.iterrows():
            state = int(row.state)
            next_state = int(row.action[0])
            budget = float(row.action[1])
            if next_state == self._hlmdp.fail_state:
                raise ValueError("High-level actions must not target the fail state.")
            if pd.isna(row.prob) or not np.isfinite(float(row.prob)):
                raise ValueError(
                    "Populate high-level transition probabilities from independent "
                    "low-level evaluations before optimization."
                )

            edge = (state, next_state)
            key = (state, next_state, budget)
            self._edge_budgets.setdefault(edge, [])
            if budget not in self._edge_budgets[edge]:
                self._edge_budgets[edge].append(budget)
            if row.next_state not in (next_state, self._hlmdp.fail_state):
                raise ValueError(
                    f"Unexpected outcome {row.next_state} for task {state}->{next_state}."
                )
            outcomes.setdefault(key, {})[int(row.next_state)] = float(row.prob)
            if row.next_state == next_state:
                self._success_probs[key] = float(row.prob)
            if state != next_state:
                self._graph.add_edge(state, next_state)

        invalid_rates = [
            (key, values)
            for key, values in outcomes.items()
            if (
                len(values) != 2
                or not np.isclose(sum(values.values()), 1.0)
                or not 0.0 <= values.get(key[1], -1.0) <= 1.0
            )
        ]
        if invalid_rates:
            raise ValueError(
                "Each task/capacity must have complementary success and failure "
                f"probabilities: {invalid_rates}."
            )
        if not nx.is_directed_acyclic_graph(self._graph):
            raise ValueError(
                "High-level policy optimization requires an acyclic task graph."
            )

        initial_state = int(self._hlmdp.init_state)
        goal_states = {int(state) for state in self._hlmdp.goal_states}
        distances = {state: -1 for state in self._graph.nodes}
        if initial_state not in distances:
            raise ValueError(
                f"Initial state {initial_state} is absent from the high-level graph."
            )
        distances[initial_state] = 0
        for state in nx.topological_sort(self._graph):
            if distances[state] < 0:
                continue
            for successor in self._graph.successors(state):
                distances[successor] = max(
                    distances[successor], distances[state] + 1
                )
        longest_goal_path = max(
            (distances.get(goal, -1) for goal in goal_states),
            default=-1,
        )
        if longest_goal_path < 0:
            raise ValueError(
                "No path exists from the high-level initial state to a goal state."
            )
        self._cost_normalizer = max(1, longest_goal_path)

    def _build_variables(self) -> None:
        for state, next_state in self._edge_budgets:
            self.opt_vars.task_occupancy[state, next_state] = self.model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=f"occupancy_{state}_{next_state}",
            )
            for budget in self._edge_budgets[(state, next_state)]:
                key = (state, next_state, budget)
                self.opt_vars.message_budget[key] = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=f"capacity_{state}_{next_state}_{budget}",
                )
                self.opt_vars.selected_occupancy[key] = self.model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    ub=1.0,
                    name=f"selected_occupancy_{state}_{next_state}_{budget}",
                )

        for state in self._hlmdp.goal_states:
            self.opt_vars.task_occupancy[int(state), int(state)] = self.model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=f"goal_occupancy_{state}",
            )

    def _build_constraints(self) -> None:
        goal_states = {int(state) for state in self._hlmdp.goal_states}
        for edge, budgets in self._edge_budgets.items():
            self.model.addConstr(
                quicksum(
                    self.opt_vars.message_budget[(*edge, budget)]
                    for budget in budgets
                )
                == 1,
                name=f"choose_capacity_{edge[0]}_{edge[1]}",
            )
            occupancy = self.opt_vars.task_occupancy[edge]
            for budget in budgets:
                key = (*edge, budget)
                selected_occupancy = self.opt_vars.selected_occupancy[key]
                indicator = self.opt_vars.message_budget[key]
                self.model.addConstr(selected_occupancy <= occupancy)
                self.model.addConstr(selected_occupancy <= indicator)
                self.model.addConstr(
                    selected_occupancy >= occupancy + indicator - 1
                )

        incoming_edges: dict[int, list[tuple[int, int]]] = {}
        for edge in self._edge_budgets:
            incoming_edges.setdefault(edge[1], []).append(edge)

        for state in self._graph.nodes:
            if state in goal_states:
                outgoing = self.opt_vars.task_occupancy[state, state]
            else:
                outgoing = quicksum(
                    self.opt_vars.task_occupancy[edge]
                    for edge in self._edge_budgets
                    if edge[0] == state
                )
            incoming = 1.0 if state == self._hlmdp.init_state else 0.0
            for edge in incoming_edges.get(state, []):
                incoming += quicksum(
                    self.opt_vars.selected_occupancy[(*edge, budget)]
                    * self._success_probs[(*edge, budget)]
                    for budget in self._edge_budgets[edge]
                )
            self.model.addConstr(
                outgoing == incoming,
                name=f"flow_conservation_{state}",
            )

        goal_reach_probability = quicksum(
            self.opt_vars.task_occupancy[state, state] for state in goal_states
        )
        self.model.addConstr(
            goal_reach_probability >= self.success_rate_spec,
            name="minimum_goal_reach_probability",
        )

    def _build_objective(self) -> None:
        expected_communication = quicksum(
            self.opt_vars.selected_occupancy[key] * key[2]
            for key in self.opt_vars.selected_occupancy
        )
        self.model.setObjective(
            expected_communication / self._cost_normalizer,
            GRB.MINIMIZE,
        )

    def _build_solution(self) -> Solution:
        sol = Solution()
        occupancy_by_state: dict[int, float] = {}
        for (state, _), variable in self.opt_vars.task_occupancy.items():
            occupancy_by_state[state] = occupancy_by_state.get(state, 0.0) + variable.X

        task_rows = []
        for (state, next_state), variable in self.opt_vars.task_occupancy.items():
            outgoing_occupancy = occupancy_by_state[state]
            if outgoing_occupancy > 0.0:
                action_probability = variable.X / outgoing_occupancy
            else:
                action_count = sum(
                    1
                    for edge in self.opt_vars.task_occupancy
                    if edge[0] == state
                )
                action_probability = 1.0 / action_count if action_count else 0.0
            task_rows.append(
                {
                    "state": state,
                    "next_state": next_state,
                    "occupancy": variable.X,
                    "action_probability": action_probability,
                }
            )

        capacity_rows = [
            {
                "state": state,
                "next_state": next_state,
                "msg_budget_per_agent": budget,
                "action_probability": variable.X,
            }
            for (state, next_state, budget), variable in (
                self.opt_vars.message_budget.items()
            )
        ]
        sol.task_policy = pd.DataFrame.from_records(task_rows)
        sol.comms_budget_policy = pd.DataFrame.from_records(capacity_rows)
        sol.objective_value = float(self.get_objective_value())
        sol.goal_reach_probability = float(
            sum(
                self.opt_vars.task_occupancy[int(state), int(state)].X
                for state in self._hlmdp.goal_states
            )
        )
        sol.feasible = True
        self.policy = sol
        return sol
