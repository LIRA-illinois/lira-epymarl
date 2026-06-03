import sys
from typing import Literal
from dataclasses import dataclass, field
from inspect import get_annotations
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from cm_extension.core.optimization_problem import (
    ProblemConfigBase,
    VariablesBase,
    ConstraintsBase,
    SolutionBase,
    OptimizationProblem,
)

from cm_extension.environments.finite_mdp import FiniteMDP
from cm_extension.utils.utils import LowLevelAgentData


@dataclass
class ProblemConfig(ProblemConfigBase):
    """problem config"""

    comms_vals: list[float]
    seeds: list[int]
    success_prob_spec: float

    success_probs: dict[tuple[int, float], float] = field(default_factory=dict)
    best_seeds: dict[tuple[int, float], int | None] = field(default_factory=dict)
    n_comms_vals: int = field(init=False)

    def __post_init__(self):
        self.n_comms_vals = len(self.comms_vals)


@dataclass
class Variables(VariablesBase):
    """problem decision variables"""

    # occupancy measure for state and env-action pairs, x[s, a]
    state_action_occupancy: dict[tuple[int, int], gp.Var] = field(default_factory=dict)

    # choice of what level of communication to allocate to a given subtask
    comms_action: dict[tuple[int, float], gp.Var] = field(default_factory=dict)


@dataclass
class Constraints(ConstraintsBase):
    """problem constraints"""

    pass


@dataclass
class EnvPolicyData:
    state_idx: int
    action_idx: int
    probability: float | None
    occupancy: float | None


@dataclass
class CommsPolicyData:
    state_idx: int
    action_idx: int
    comms_val: float
    seed: int | None
    probability: float


class Solution(SolutionBase):
    """problem solution"""

    def __init__(self) -> None:
        self.env_policy: pd.DataFrame = pd.DataFrame(
            columns=list(get_annotations(EnvPolicyData).keys())
        )
        self.comms_policy: pd.DataFrame = pd.DataFrame(
            columns=list(get_annotations(CommsPolicyData).keys())
        )

        self.goal_reach_prob: float | None = None
        self.objective_value: float | None = None
        self.feasible: bool | None = None
        self.opt_config: ProblemConfig


class HighLevelCMAgent(OptimizationProblem):
    def __init__(
        self,
        success_prob_spec: float,
        plot_dir: str,
        low_level_agent_data: LowLevelAgentData,
        low_level_env_name: str,
        num_agents: int,
        subtask_type: Literal["dependent", "independent"],
    ) -> None:
        """
        Parameters
        ----------
        success_prob_spec : float
            probabilistic task completion specification for the high-level problem
        plot_dir : str
            directory for saving plots
        low_level_env_name :str
            name of the low-level envirionment, used to config the high-level env
        """
        super().__init__()
        self.env = FiniteMDP(
            low_level_env_name=low_level_env_name,
            subtask_type=subtask_type,
            num_agents=num_agents,
        )
        self.plot_dir = plot_dir
        self.low_level_agent_data = low_level_agent_data

        self.opt_config = ProblemConfig(
            comms_vals=self.low_level_agent_data.comms_vals,
            seeds=self.low_level_agent_data.seeds,
            success_prob_spec=success_prob_spec,
        )
        self.opt_vars = Variables()
        self.opt_constrs = Constraints()

    # external interface
    def get_policy(
        self, df_agent: pd.DataFrame, uniform_policy: bool = False
    ) -> Solution:
        """
        Parameters
        ----------
        df_agent : pd.DataFrame
            df_agent from the hierarchical CM agent
        uniform_policy : bool
            whether to compute uniform policies or not
        """
        opt_sol = Solution()

        if uniform_policy:
            opt_sol: Solution = self._get_uniform_policies()

        else:
            self._build_model(df_agent=df_agent)
            self.optimize()
            optima_found = self.check_if_optima_found()

            if optima_found:
                opt_sol: Solution = self._build_solution()

            else:
                print(
                    "High-level policy optimization did not find an optimal solution. Check you debugging the problem."
                )
                sys.exit()

            # Delete the model object so it isn't part of the parent process
            # when we start up parallel processes for training low-level agents.
            # Gurobi models cause issues for multiprocessing in the hierarchical agent
            # b/c they cannot be pickled or dilled.
            del self.model

        return opt_sol

    def eval_policy(self, df_agent: pd.DataFrame, hl_sol: Solution) -> Solution:
        """evaluates the goal-reaching probability of a hierarchical policy
        with low-level policy data in df_agent and a high-level policy in hl_sol

        Parameters
        ----------
        df_agent : pd.DataFrame
        hl_sol : Solution

        Returns
        -------
        float
            _description_
        """
        self._build_model(df_agent=df_agent, eval_hl_sol=hl_sol)
        self.optimize()
        optima_found = self.check_if_optima_found()

        if optima_found:
            eval_sol = self._build_solution()
        else:
            eval_sol = Solution()

        # Delete the model object so it isn't part of the parent process
        del self.model

        return eval_sol

    def _get_uniform_policies(self) -> Solution:
        # build policy dataframes
        all_env_policy_data: dict[int, EnvPolicyData] = {}
        all_comms_policy_data: dict[int, CommsPolicyData] = {}

        row_idx: int = 0
        for _, state_data in self.env.df_state.iterrows():
            if not (state_data.goal or state_data.fail):
                state_idx = state_data.state_idx

                avail_actions = state_data.avail_actions
                for action_idx in avail_actions:
                    all_env_policy_data[row_idx] = EnvPolicyData(
                        state_idx=state_idx,
                        action_idx=action_idx,
                        probability=1.0 / len(avail_actions),
                        occupancy=0.0,
                    )

                    for comms_val in self.opt_config.comms_vals:
                        # use a filler value for the seed here since it
                        # isn't used in computing the policy
                        all_comms_policy_data[row_idx] = CommsPolicyData(
                            state_idx=state_idx,
                            action_idx=action_idx,
                            comms_val=comms_val,
                            seed=None,
                            probability=1.0 / self.opt_config.n_comms_vals,
                        )
                        row_idx += 1

        # put output data into the Solution class
        opt_sol = Solution()
        opt_sol.env_policy = pd.DataFrame.from_dict(all_env_policy_data, orient="index")
        # reset the row indices to match the lengths of the dfs
        opt_sol.env_policy.reset_index(drop=True, inplace=True)
        opt_sol.comms_policy = pd.DataFrame.from_dict(
            all_comms_policy_data, orient="index"
        )

        return opt_sol

    # model setup
    def _build_model(
        self,
        df_agent: pd.DataFrame,
        eval_hl_sol: Solution | None = None,
    ) -> None:
        """build computational model of the optimization problem"""
        gurobi_env = gp.Env()
        # suppress gurobi logging
        # gurobi_env.setParam("OutputFlag", 0)

        self.model = gp.Model(self.__class__.__name__, env=gurobi_env)

        self._build_config(df_agent=df_agent)

        self._build_variables()
        self.model.update()

        self._build_constraints()
        self.model.update()

        if eval_hl_sol is not None:
            self._build_eval_constraints(eval_hl_sol=eval_hl_sol)
            self.model.update()

        self._build_objective()
        self.model.update()

        assert isinstance(self.opt_config, ProblemConfigBase)
        assert isinstance(self.opt_vars, VariablesBase)
        assert isinstance(self.opt_constrs, ConstraintsBase)

    def _build_config(self, df_agent: pd.DataFrame) -> None:
        """build the success probabilities to be used in the high-level model by
        aggregating over low-level policy performance data

        Parameters
        ----------
        df_agent : pd.DataFrame
            df_agent from the hierarchical CM agent
        """
        success_probs, best_seeds = self.get_success_probs(df_agent)
        self.opt_config.success_probs = success_probs
        self.opt_config.best_seeds = best_seeds

    def get_success_probs(
        self, df_agent: pd.DataFrame
    ) -> tuple[dict[tuple[int, float], float], dict[tuple[int, float], int | None]]:
        """aggregate the success probabilities over multiple seeds to get a single
        one per subtask and comms value to be used in the high-level agent's optimization problem
        """
        success_probs: dict[tuple[int, float], float] = {}
        best_seeds: dict[tuple[int, float], int | None] = {}

        for _, row in self.env.df_action.iterrows():
            if not row.dummy:
                env_action = row.action_idx

                for comms_val in self.opt_config.comms_vals:

                    df_data = df_agent.loc[
                        (df_agent.subtask_idx == env_action)
                        & (df_agent.comms_val == comms_val)
                    ]

                    # filter out policies that didn't perform above the
                    # success threshold for sampling the success prob
                    df_data = df_data.loc[df_data.n_eval_episodes.notna()]

                    if len(df_data) == 0:
                        high_level_success_prob: float = 0.0
                        best_seed = None
                    else:
                        # doing max here, but you could also imagine doing mean or some other
                        # aggregation over low-level policies. However, that would mean a totally different
                        # concept of policy deployment, where you then have some mixed strategy over the
                        # low-level policies, which gets to be implausible. Also not clear why you would want
                        # to do that when you can just take the best-performing low-level policy and deploy it
                        max_success_prob: float = np.max(df_data.eval_success_prob)

                        # save the best seed to output for later too
                        # there may be multiple rows that have the max success
                        # if that's the case, just pick the 0th row in the resulting dataframe
                        best_seed = (
                            df_data.loc[df_data.eval_success_prob == max_success_prob]
                            .iloc[0]
                            .seed.item()
                        )

                        high_level_success_prob = max_success_prob

                    success_probs[env_action, comms_val] = high_level_success_prob
                    best_seeds[env_action, comms_val] = best_seed

        return success_probs, best_seeds

    def _build_variables(self) -> None:
        """builds all relevant variables used in the problem"""

        # state-action occupancy
        for _, row in self.env.df_state.iterrows():
            if not row.fail:
                state = row.state_idx
                for env_action in row.avail_actions:
                    self.opt_vars.state_action_occupancy[state, env_action] = (
                        self.model.addVar(
                            vtype=GRB.CONTINUOUS,
                            lb=0,
                            name=f"state_action_occupancy_s={state}_a={env_action}",
                        )
                    )

        # subtask communication allocation decision var
        for _, row in self.env.df_action.iterrows():
            # we should not have a comms allocation action for the dummy action for the goal state
            if not row.dummy:
                env_action = row.action_idx

                for comms_val in self.opt_config.comms_vals:
                    self.opt_vars.comms_action[env_action, comms_val] = (
                        self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f"comms_action_a={env_action}_w={comms_val}",
                        )
                    )

        self.model.update()

    def _build_constraints(self) -> None:
        """builds all relevant constraints used in the problem"""

        # define a deterministic comms policy that allocates only one level
        # of comms to any given subtask (as opposed to a mixed strategy
        # over comms levels)
        for _, row in self.env.df_action.iterrows():
            if not row.dummy:
                env_action = row.action_idx
                constraint = 0
                for comms_val in self.opt_config.comms_vals:
                    constraint += self.opt_vars.comms_action[env_action, comms_val]

                self.model.addConstr(
                    constraint == 1,
                    name=f"deterministic_comms_policy_constr_{env_action}",
                )

        # Bellman flow constraint to define the state-action occupancy measure
        for _, row in self.env.df_state.iterrows():
            state = row.state_idx

            # avoid fail state since we can never take an action to
            # intentionally go there
            if not row.fail:
                # left-hand side of the constraint
                lhs = 0

                # add occupancy for available actions
                for env_action in row.avail_actions:
                    lhs += self.opt_vars.state_action_occupancy[state, env_action]

                # right-hand side of the constraint
                if state == self.env.init_state:
                    # initial state occupancy is defined to be 1 since
                    # we assume there is only one initial state
                    rhs = 1

                else:
                    rhs = 0
                    # add incoming occupancy for predecessor state-actions
                    for state_pred, env_action_pred in self.env.predecessors[state]:
                        # ignore the goal state's self-transition since it is technically an accepting state
                        if state_pred != self.env.goal_state:
                            for comms_val in self.opt_config.comms_vals:
                                comms_action = self.opt_vars.comms_action[
                                    env_action_pred, comms_val
                                ]
                                success_prob = self.opt_config.success_probs[
                                    env_action_pred, comms_val
                                ]

                                incoming_state_action_occupancy = (
                                    self.opt_vars.state_action_occupancy[
                                        state_pred, env_action_pred
                                    ]
                                )

                                subtask_completion_prob = comms_action * success_prob

                                rhs += (
                                    incoming_state_action_occupancy
                                    * subtask_completion_prob
                                )
                self.model.addConstr(
                    lhs == rhs, name=f"in_and_out_bellman_flow_equality_{state}"
                )

        # define constraint on successful global task completion probability (i.e., reaching the goal state)
        for _, row in self.env.df_state.iterrows():
            if row.goal:
                state = row.state_idx
                dummy_env_action = row.avail_actions[0]

                self.model.addConstr(
                    self.opt_vars.state_action_occupancy[state, dummy_env_action]
                    >= self.opt_config.success_prob_spec,
                    name=f"success_prob_spec_u={state}",
                )

        self.model.update()

    def _build_eval_constraints(self, eval_hl_sol: Solution) -> None:
        # constrain the values of all decision vars to be from eval_hl_sol
        # set state-action occupancy
        for _, row in eval_hl_sol.env_policy.iterrows():
            state = row.state_idx
            env_action = row.action_idx

            self.model.addConstr(
                self.opt_vars.state_action_occupancy[state, env_action]
                == row.occupancy,
                name=f"eval_state_action_u={state}, a={env_action}",
            )

        # set comms value
        for _, row in eval_hl_sol.comms_policy.iterrows():
            env_action = row.action_idx
            comms_val = row.comms_val
            self.model.addConstr(
                self.opt_vars.comms_action[env_action, comms_val] == row.probability,
                name=f"eval_comms_action_a={env_action}, w={comms_val}",
            )

    def _build_objective(self) -> None:
        """builds all the objective function used in the problem"""
        objective = 0
        for _, row in self.env.df_state.iterrows():
            if not (row.fail or row.goal):
                state = row.state_idx
                for env_action in row.avail_actions:
                    for comms_val in self.opt_config.comms_vals:

                        objective += (
                            self.opt_vars.state_action_occupancy[state, env_action]
                            * self.opt_vars.comms_action[env_action, comms_val]
                            * comms_val
                        )

        self.model.setObjective(objective, GRB.MINIMIZE)
        self.model.update()

    # build solution
    def _build_solution(self) -> Solution:
        """build solution data for the optimization problem"""
        # model feasibility
        # if we're building the solution, it is feasible by default
        opt_sol = Solution()
        opt_sol.opt_config = self.opt_config
        opt_sol.feasible = True

        # translate model output into chosen communication values and success probabilites
        all_env_policy_data: dict[int, EnvPolicyData] = {}
        all_comms_policy_data: dict[int, CommsPolicyData] = {}

        # build the env policy from state-action occupancy by normalizing state-action by
        # total occupancy for each state
        env_policy: dict[tuple[int, int], float | None] = {}
        for _, row in self.env.df_state.iterrows():
            if not (row.goal or row.fail):
                state = row.state_idx
                total_state_occupancy = 0
                avail_actions = row.avail_actions

                for env_action in avail_actions:
                    total_state_occupancy += self.opt_vars.state_action_occupancy[
                        state, env_action
                    ].X

                for env_action in avail_actions:
                    if total_state_occupancy == 0.0:
                        # set env policy to a filler value
                        env_policy[state, env_action] = None
                    else:
                        env_policy[state, env_action] = (
                            self.opt_vars.state_action_occupancy[state, env_action].X
                            / total_state_occupancy
                        )

        row_idx: int = 0
        for _, row in self.env.df_state.iterrows():
            if not (row.goal or row.fail):
                state = row.state_idx
                avail_actions = row.avail_actions

                for env_action in avail_actions:
                    all_env_policy_data[row_idx] = EnvPolicyData(
                        state_idx=state,
                        action_idx=env_action,
                        probability=env_policy[state, env_action],
                        occupancy=self.opt_vars.state_action_occupancy[
                            state, env_action
                        ].X,
                    )

                    for comms_val in self.opt_config.comms_vals:
                        all_comms_policy_data[row_idx] = CommsPolicyData(
                            state_idx=state,
                            action_idx=env_action,
                            comms_val=comms_val,
                            seed=self.opt_config.best_seeds[env_action, comms_val],
                            probability=self.opt_vars.comms_action[
                                env_action, comms_val
                            ].X,
                        )

                        row_idx += 1

        # put output data into the Solution class
        opt_sol.env_policy = pd.DataFrame.from_dict(all_env_policy_data, orient="index")
        # reset the row indices to match the lengths of the dfs
        opt_sol.env_policy.reset_index(drop=True, inplace=True)
        opt_sol.comms_policy = pd.DataFrame.from_dict(
            all_comms_policy_data, orient="index"
        )

        # get objective function value
        opt_sol.objective_value = self.get_objective_value()

        # get goal-reaching probability
        # state_action_occupancy[self.env.goal_state, goal_dummy_action] is the probability
        # of reaching the final state starting from the initial state in the high-level MDP b/c
        # it has info about success probs of all predecessor subtasks
        df_state = self.env.df_state
        goal_dummy_action = df_state.loc[
            df_state.state_idx == self.env.goal_state
        ].avail_actions.item()[0]

        opt_sol.goal_reach_prob = self.opt_vars.state_action_occupancy[
            self.env.goal_state, goal_dummy_action
        ].X

        return opt_sol
