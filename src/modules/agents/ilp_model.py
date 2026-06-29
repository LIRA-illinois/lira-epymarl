import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple

from gym_multigrid.envs.mdp import ProjectMDP

from src.components.episode_buffer import EpisodeBatch
from src.modules.gurobi_opt import (
    OptimizationProblem,
    VariablesBase,
    SolutionBase,
)

from gurobipy import GRB


@dataclass
class Variables(VariablesBase):
    task_occupancy: Dict[Tuple[int, int], object] = field(default_factory=dict)
    comms_action: Dict[Tuple[int, int, float], object] = field(default_factory=dict)


@dataclass
class Solution(SolutionBase):
    # TODO these can be combined into a single HL policy that outputs the next state and comms value together
    task_policy: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    comms_policy: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())


class ILPModel(OptimizationProblem):
    """integer linear program, takes an MDP model with transition probabilities and solves for the optimal policy
    works great for small problems, but this model + solution method does not scale super well with the number of states
    """

    """Gurobi-based ILP that mirrors the high-level comms optimization in `hl_cm_agent.py`.

    Usage: instantiate with a `ProjectMDP` object and call `build_model()` then `optimize()`.
    After optimization call `_build_solution()` to extract results.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self._hlmdp: ProjectMDP
        self.success_rate_spec: float
        self.opt_vars: Variables
        self.policy: Solution

    def optimize_policy(
        self, hlmdp: ProjectMDP, success_rate_spec: float = 0.95
    ) -> Solution:
        """optimize a tabular policy"""
        # stat table is a df (or similar) that has data for each subtask + success rates
        # returns a policy to navigate in the HLMDP + choose comms values
        self._hlmdp = hlmdp
        self.success_rate_spec = success_rate_spec

        ###############
        # # fake data for development
        # df = self.hlmdp.transition_probs
        # success_rate_task_0_comms_0 = 0.85
        # success_rate_task_0_comms_1 = 0.85
        # success_rate_task_1_comms_0 = 0.91
        # success_ratDe_task_1_comms_1 = 0.98

        # probs = [
        #     success_rate_task_0_comms_0, 1 - success_rate_task_0_comms_0,
        #     success_rate_task_0_comms_1, 1 - success_rate_task_0_comms_1,
        #     success_rate_task_1_comms_0, 1 - success_rate_task_1_comms_0,
        #     success_rate_task_1_comms_1, 1 - success_rate_task_1_comms_1,
        #     1.0, 1.0]
        # df.prob = pd.array(probs)
        ###############

        self.opt_vars = Variables()
        self.policy = Solution()

        self.build_model()
        self.optimize()
        if self.check_if_optima_found():
            policy: Solution = self._build_solution()
        else:
            print(
                "High-level policy optimization did not find an optimal solution. Problem may be infeasible, try a lower success rate spec."
            )
            import sys

            sys.exit("Exiting")

        # print(self.hlmdp.transition_probs)
        # print(self.policy.task_policy)
        # print(self.policy.comms_policy)

        return policy

    def select_actions(
        self, ep_batch: EpisodeBatch, t_ep: int, t_env: int, test_mode: bool
    ) -> dict:
        # return action with shape (n_hierarchical_actions)

        # state is a scalar in the HLMDP
        hl_state = ep_batch["hl_state"][:, t_ep, :].cpu().numpy().flatten()
        mdp_state = int(hl_state[0])

        chosen_next_mdp_state: int
        comms_allocation: float

        # can have a null action while the low-level policy is executing its task
        if test_mode:
            # actually choose the comms value to be used during evaluation of MAIC
            comms_allocation = 0.0
        else:
            comms_allocation = 0.0

        if (t_env == 0) and (t_ep == 0):
            print("solving model-based MDP problem (ILP) to get full tabular policy")

        # assume the high-level model never changes for a given env during training
        # you can pre-solve this one time for each env and then save the tabular policy to a dict or something
        # walk thru each state in the HLMDP
        # construct avail actions given the current state using the avail_actions method from the env itself
        # solve the optimization problem to populate a tabular policy
        # grab an action for the current HLMDP state
        ## idk how I want to handle comms values yet, but its only currently relevant for evaluation so don't worry about it for now
        # if you just completd a task, low-level agents need a new task assigned to them
        # then once we're running this in the loop, just grab the correct action from the tabular policy based on the current state
        if mdp_state == 2:
            chosen_next_mdp_state = mdp_state

        else:
            chosen_next_mdp_state = mdp_state + 1

        actions: dict = {
            "chosen_next_state": chosen_next_mdp_state,
            "comms_allocation": comms_allocation,
        }

        return actions

    def cuda(self):
        pass

    def _build_variables(self) -> None:
        # state-action occupancy variables (continuous)
        # action = "next state" action
        df_trans = self._hlmdp.transition_probs
        for _, row in df_trans.iterrows():
            if row.next_state == self._hlmdp.fail_state:
                continue

            state, next_state, comms_val = row.state, row.action[0], row.action[1]

            self.opt_vars.task_occupancy[state, next_state] = self.model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                name=f"task_occupancy_ss'={state, next_state}",
            )

            self.opt_vars.comms_action[state, next_state, comms_val] = (
                self.model.addVar(
                    vtype=GRB.BINARY,
                    name=f"comms_action_ss'c={state,next_state,comms_val}",
                )
            )

        self.model.update()

    def _build_constraints(self) -> None:
        df_trans = self._hlmdp.transition_probs

        # deterministic comms allocation: chose one comms level for each task
        # for each edge wtih the same start and end states, sum over all the comms vals and set to 1
        for state, next_state in self.opt_vars.task_occupancy:
            # get the relevant comms values for this transition
            actions = df_trans.loc[
                (df_trans.state == state) & (df_trans.next_state == next_state)
            ].action
            comms_vals = [action[1] for action in actions]

            constraint = 0
            for comms_val in comms_vals:
                constraint += self.opt_vars.comms_action[state, next_state, comms_val]

            self.model.addConstr(
                constraint == 1.0,
                name=f"deterministic_comms_policy_constr_{(state, next_state)}",
            )

        # Bellman flow constraints: sum occupancy of outgoing actions == sum of incoming occupancy for each state
        for state in pd.unique(df_trans.state).tolist():
            if state == self._hlmdp.fail_state:
                continue

            # get set of successor states
            df_successor = df_trans.loc[
                (df_trans.state == state)
                & (df_trans.next_state != self._hlmdp.fail_state)
            ]

            # define outgoing occupancy from state
            outgoing = 0

            # add occupancy for available next tasks
            for next_state in pd.unique(df_successor.next_state):
                outgoing += self.opt_vars.task_occupancy[(state, next_state)]

            # define incoming occupancy to state
            if state == self._hlmdp.init_state:
                # initial state occupancy is defined to be 1 since
                # we assume there is only one initial state
                incoming = 1

            else:
                # add incoming occupancy for predecessor state-actions
                incoming = 0
                df_pred = df_trans.loc[df_trans.next_state == state]

                for pred_state in pd.unique(df_pred.state).tolist():
                    # ignore goal state's self transition
                    if pred_state in [self._hlmdp.goal_state, self._hlmdp.fail_state]:
                        continue

                    pred_actions = df_pred.loc[
                        (df_pred.state == pred_state) & (df_pred.next_state == state)
                    ].action

                    comms_vals = [action[1] for action in pred_actions]

                    for comms_val in comms_vals:

                        comms_action = self.opt_vars.comms_action[
                            pred_state, state, comms_val
                        ]

                        success_rate = df_pred.loc[
                            (df_pred.state == pred_state)
                            & (df_pred.action == (state, comms_val))
                        ].prob.item()

                        chosen_comms_success_rate = comms_action * success_rate

                        incoming_task_occupancy = self.opt_vars.task_occupancy[
                            (pred_state, state)
                        ]

                        incoming += incoming_task_occupancy * chosen_comms_success_rate

            self.model.addConstr(
                outgoing == incoming, name=f"bellman_flow_conservation_s={state}"
            )

        # define constraint on successful global task completion probability (i.e., reaching the goal state)
        df_goal = df_trans.loc[df_trans.state_type == "goal"]
        self.model.addConstr(
            self.opt_vars.task_occupancy[df_goal.state.item(), df_goal.action.item()[0]]
            >= self.success_rate_spec,
            name=f"success_rate_spec_s={state}",
        )
        self.model.update()

    def _build_objective(self) -> None:
        objective = 0
        df_trans = self._hlmdp._transition_probs

        for state, next_state in self.opt_vars.task_occupancy:
            # get the relevant comms values for this transition
            actions = df_trans.loc[
                (df_trans.state == state) & (df_trans.next_state == next_state)
            ].action
            comms_vals = [action[1] for action in actions]

            for comms_val in comms_vals:
                objective += (
                    self.opt_vars.task_occupancy[state, next_state]
                    * self.opt_vars.comms_action[state, next_state, comms_val]
                    * comms_val
                )

        self.model.setObjective(objective, GRB.MINIMIZE)
        self.model.update()

    def _build_solution(self) -> Solution:
        # unpack Gurobi variables into dataframe rows to form a tabular policy
        sol = Solution()

        env_data, comms_data = [], []
        for (state, next_state), var in self.opt_vars.task_occupancy.items():
            env_data.append(
                {"state": state, "next_state": next_state, "occupancy": var.X}
            )

        for (state, next_state, comms_val), var in self.opt_vars.comms_action.items():
            comms_data.append(
                {
                    "state": state,
                    "next_state": next_state,
                    "comms_val": comms_val,
                    "action_probability": var.X,
                }
            )

        sol.task_policy = pd.DataFrame.from_records(env_data)
        sol.comms_policy = pd.DataFrame.from_records(comms_data)

        sol.objective_value = self.get_objective_value()
        sol.feasible = True if self.check_if_optima_found() else False
        self.policy = sol
        return sol
