"""Dependent-subtask coordination-machine training orchestration."""

from collections import defaultdict
from copy import deepcopy
from os import makedirs
from os.path import join
from shutil import rmtree
from types import SimpleNamespace as SN

import numpy as np
import pandas as pd
import torch as th

from src.simulation.build import build_sim
from src.simulation.evaluate import collect_successful_final_state_dist


class DependentSubtaskTrainingMixin:
    """CM candidate training and composed evaluation for dependent subtasks."""

    def train_dependent_subtasks(self, hl_task_sequence: list) -> None:
        """Alternate candidate low-level CM passes with high-level optimization.

        A distinct low-level model is trained for each edge, communication
        budget, and policy seed. In each pass, successor spawn distributions
        are assembled from successful predecessor rollouts, weighted by the
        previous CM solution's occupancy and selected communication budget.
        """
        edges: list[tuple[int, int]] = [
            (int(edge[0]), int(edge[1])) for edge in hl_task_sequence
        ]
        if not edges or any(from_state == to_state for from_state, to_state in edges):
            raise ValueError("hl_task_sequence must contain non-self-loop edges.")
        if len(set(edges)) != len(edges):
            raise ValueError("hl_task_sequence must not contain duplicate edges.")

        source_states = {from_state for from_state, _ in edges}
        destination_states = {to_state for _, to_state in edges}
        root_states = source_states - destination_states
        if len(root_states) != 1:
            raise ValueError(
                "hl_task_sequence must describe a DAG with exactly one root state."
            )
        root_state = root_states.pop()

        predecessors: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for from_state, to_state in edges:
            predecessors[to_state].append((from_state, to_state))

        outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for edge in edges:
            outgoing[edge[0]].append(edge)

        if not hasattr(self.args, "factored_hierarchical_policy"):
            raise ValueError(
                "Dependent CM training requires the factored high-level controller "
                "configuration (for example, config=maic_comm_opt)."
            )
        configured_budgets = getattr(self.args, "msg_budget_per_agent", None)
        if configured_budgets is None:
            configured_budgets = [self.args.n_agents - 1]
        budgets = [float(budget) for budget in configured_budgets]
        if not budgets or len(set(budgets)) != len(budgets):
            raise ValueError(
                "msg_budget_per_agent must contain unique configured budgets."
            )
        policy_seeds = [
            int(seed)
            for seed in getattr(self.args, "dependent_policy_seeds", [self.args.seed])
        ]
        if not policy_seeds or len(set(policy_seeds)) != len(policy_seeds):
            raise ValueError(
                "dependent_policy_seeds must contain unique integer seeds."
            )

        n_eval_eps = int(
            getattr(self.args, "n_eval_eps_dependent_subtask", self.n_eval_eps)
        )
        if n_eval_eps <= 0:
            raise ValueError("n_eval_eps_dependent_subtask must be greater than zero.")
        max_cm_iterations = max(1, int(getattr(self.args, "cm_max_iterations", 20)))
        cm_epsilon = float(getattr(self.args, "cm_convergence_epsilon", 0.2))
        terminal_states = destination_states - source_states
        if not terminal_states:
            raise ValueError(
                "hl_task_sequence must contain at least one terminal state."
            )

        original_args = self.args
        original_runner = self.runner
        if not hasattr(original_runner, "env") or not hasattr(
            original_runner.env, "hlmdp"
        ):
            raise ValueError(
                "Dependent CM optimization requires a runner exposing env.hlmdp."
            )
        hlmdp = original_runner.env.hlmdp
        root_init_state_dist = self._dependent_root_init_state_dist(root_state, edges)

        previous_solution = None
        previous_init_state_dists: dict[int, dict] | None = None
        final_policy_bank: dict[tuple[int, int, float, int], dict] = {}
        self._dependent_cm_total_t_env = 0
        original_runner.close_env()

        for cm_iter in range(max_cm_iterations):
            self.logger.info(
                f"Starting dependent-subtask CM iteration {cm_iter + 1}/"
                f"{max_cm_iterations}",
                log_header=True,
            )

            edge_success_rates: dict[tuple[int, int, float], float] = {}
            edge_final_state_dists: dict[tuple[int, int, float], dict] = {}
            learned_init_state_dists: dict[int, dict] = {}
            state_arrival_flow = {root_state: 1.0}
            pass_policy_bank: dict[tuple[int, int, float, int], dict] = {}

            for from_state, to_state in edges:
                self.logger.info(
                    f"Training dependent subtask {from_state} -> {to_state} "
                    f"(CM iteration {cm_iter + 1})",
                    log_header=True,
                )

                if from_state == root_state:
                    init_state_dist = root_init_state_dist
                else:
                    if from_state not in learned_init_state_dists:
                        raise ValueError(
                            "hl_task_sequence must be topologically ordered; "
                            f"no initial distribution is available for state {from_state}."
                        )
                    init_state_dist = learned_init_state_dists[from_state]

                reset_options = {
                    "hl_start_state": from_state,
                    "hl_task": [from_state, to_state],
                    "navigation_task_transition": [from_state, to_state],
                    "navigation_init_state_dist": init_state_dist,
                }

                self.logger.info(
                    f"CM candidate pass: {from_state} -> {to_state}; "
                    f"{len(budgets)} budgets x {len(policy_seeds)} policy seeds"
                )
                edge = (from_state, to_state)
                for budget in budgets:
                    seed_dists = []
                    seed_success_rates = []
                    for policy_seed in policy_seeds:
                        final_state_dist, success_rate, policy_weights = (
                            self._train_dependent_candidate(
                                reset_options=reset_options,
                                budget=budget,
                                policy_seed=policy_seed,
                                n_eval_eps=n_eval_eps,
                            )
                        )
                        candidate_key = (
                            from_state,
                            to_state,
                            budget,
                            policy_seed,
                        )
                        pass_policy_bank[candidate_key] = policy_weights
                        seed_dists.append(final_state_dist)
                        seed_success_rates.append(success_rate)
                        self.logger.info(
                            f"Candidate {from_state}->{to_state}, budget={budget:g}, "
                            f"seed={policy_seed}: success={success_rate:.3f}, "
                            f"successful terminal states="
                            f"{len(final_state_dist['states'])}"
                        )

                    edge_budget = (from_state, to_state, budget)
                    edge_success_rates[edge_budget] = float(np.mean(seed_success_rates))
                    if any(dist["states"] for dist in seed_dists):
                        contributing_dists = [
                            (dist, rate)
                            for dist, rate in zip(seed_dists, seed_success_rates)
                            if dist["states"] and rate > 0.0
                        ]
                        edge_final_state_dists[edge_budget] = self._merge_state_dists(
                            [dist for dist, _ in contributing_dists],
                            [rate for _, rate in contributing_dists],
                        )
                    else:
                        edge_final_state_dists[edge_budget] = {
                            "states": [],
                            "probs": [],
                        }

                incoming_edges = predecessors[to_state]
                if to_state not in terminal_states and all(
                    all(
                        (edge[0], edge[1], budget) in edge_success_rates
                        for budget in budgets
                    )
                    for edge in incoming_edges
                ):
                    incoming_dists = []
                    incoming_weights = []
                    for incoming_edge in incoming_edges:
                        if previous_solution is None:
                            edge_flow = state_arrival_flow.get(
                                incoming_edge[0], 0.0
                            ) / len(outgoing[incoming_edge[0]])
                            budget_probabilities = {
                                budget: 1.0 / len(budgets) for budget in budgets
                            }
                        else:
                            edge_flow = self._cm_edge_occupancy(
                                previous_solution, incoming_edge
                            )
                            budget_probabilities = self._cm_budget_probabilities(
                                previous_solution, incoming_edge, budgets
                            )

                        for budget in budgets:
                            weight = (
                                edge_flow
                                * budget_probabilities[budget]
                                * edge_success_rates[
                                    (incoming_edge[0], incoming_edge[1], budget)
                                ]
                            )
                            dist = edge_final_state_dists[
                                (incoming_edge[0], incoming_edge[1], budget)
                            ]
                            if weight > 0.0 and dist["states"]:
                                incoming_dists.append(dist)
                                incoming_weights.append(weight)

                    if not incoming_dists:
                        for incoming_edge in incoming_edges:
                            for budget in budgets:
                                dist = edge_final_state_dists[
                                    (incoming_edge[0], incoming_edge[1], budget)
                                ]
                                if dist["states"]:
                                    incoming_dists.append(dist)
                                    incoming_weights.append(1.0)
                        self.logger.info(
                            f"No incoming flow reached state {to_state} under the "
                            "current CM policy; using an unweighted successful-data "
                            "mixture so its outgoing candidates can still be trained."
                        )

                    learned_init_state_dists[to_state] = self._merge_state_dists(
                        incoming_dists,
                        incoming_weights,
                    )
                    state_arrival_flow[to_state] = sum(incoming_weights)

            current_solution = self._optimize_dependent_cm_policy(
                hlmdp, edges, edge_success_rates
            )
            if current_solution is not None:
                previous_solution = current_solution
            elif previous_solution is None and cm_iter == max_cm_iterations - 1:
                raise RuntimeError(
                    "No feasible dependent high-level policy was found after "
                    f"{max_cm_iterations} CM passes."
                )

            delta_isd = self._state_dist_delta(
                previous_init_state_dists, learned_init_state_dists
            )
            success_probability = (
                previous_solution.goal_reach_probability
                if previous_solution is not None
                else float("nan")
            )
            self.logger.info(
                f"CM iteration {cm_iter + 1}: estimated hierarchical success "
                f"{success_probability:.3f}, initial-state distance "
                f"{delta_isd:.3f}, ILP "
                f"{'updated' if current_solution is not None else 'infeasible; retained prior policy'}"
            )
            if previous_solution is not None:
                self.logger.log_stat(
                    "dependent_cm_success_rate", success_probability, cm_iter
                )
            self.logger.log_stat("dependent_cm_isd_delta", delta_isd, cm_iter)

            previous_init_state_dists = learned_init_state_dists
            final_policy_bank = pass_policy_bank
            if cm_iter > 0 and delta_isd <= cm_epsilon and current_solution is not None:
                self.logger.info(
                    f"Dependent-subtask CM converged after {cm_iter + 1} iterations."
                )
                break

        if previous_solution is None:
            raise RuntimeError("Dependent CM training ended without an ILP solution.")

        self.args, self.runner, self.buffer, self.learner = build_sim(
            original_args, self.logger
        )
        try:
            if not hasattr(self.runner.mac, "set_dependent_policy_bank"):
                raise RuntimeError(
                    "The configured high-level controller cannot load dependent "
                    "subtask candidate policies."
                )
            self.runner.mac.comms_agent.policy = previous_solution
            self.runner.mac.set_dependent_policy_bank(final_policy_bank, policy_seeds)
            self.dependent_cm_policy = previous_solution
            if self.args.save_model:
                self._save_dependent_cm_artifacts(final_policy_bank, previous_solution)
            self._evaluate_dependent_cm_policy(
                root_state,
                {root_state: root_init_state_dist, **learned_init_state_dists},
                policy_seeds,
            )
        finally:
            self.runner.close_env()
        self.logger.info("Finished Dependent Subtask Training")

    def _dependent_root_init_state_dist(
        self, root_state: int, edges: list[tuple[int, int]]
    ) -> dict:
        navigation_env = self.runner.env.unwrapped
        catalog = getattr(navigation_env, "navigation_task_catalog", None)
        if catalog is None:
            raise ValueError(
                "Dependent CM training requires a state-keyed navigation task catalog."
            )

        root_dists = []
        for edge in edges:
            if edge[0] != root_state:
                continue
            task = catalog.task_for(*edge)
            if task.init_state_dist is None:
                raise ValueError(
                    f"Root task {edge[0]}->{edge[1]} has no configured initial "
                    "state distribution."
                )
            root_dists.append({
                "states": [
                    [list(position) for position in joint_state]
                    for joint_state in task.init_state_dist.states
                ],
                "probs": list(task.init_state_dist.probs),
            })

        first_dist = root_dists[0]
        if any(
            dist["states"] != first_dist["states"]
            or not np.allclose(dist["probs"], first_dist["probs"])
            for dist in root_dists[1:]
        ):
            raise ValueError(
                "All root outgoing tasks must share one initial-state distribution "
                "so the policy-selected first task can use the same spawn state."
            )
        return first_dist

    def _train_dependent_candidate(
        self,
        reset_options: dict,
        budget: float,
        policy_seed: int,
        n_eval_eps: int,
    ) -> tuple[dict, float, dict]:
        original_context = (
            self.args,
            self.runner,
            self.buffer,
            self.learner,
            self.n_eval_eps,
        )
        original_args = self.args
        candidate_args = SN(**vars(original_args))
        if hasattr(candidate_args, "factored_hierarchical_policy"):
            del candidate_args.factored_hierarchical_policy
        candidate_args.env_args = deepcopy(original_args.env_args)
        candidate_args.env_args["seed"] = policy_seed
        candidate_args.msg_budget_per_agent = [int(budget)]
        candidate_args.env_args["msg_budget_per_agent"] = [int(budget)]
        candidate_args.seed = policy_seed
        candidate_args.unique_policy_per_msg_budget = True
        candidate_args.save_model = False

        numpy_state = np.random.get_state()
        torch_state = th.random.get_rng_state()
        cuda_states = th.cuda.get_rng_state_all() if th.cuda.is_available() else None
        candidate_runner = None
        try:
            np.random.seed(policy_seed)
            th.manual_seed(policy_seed)
            self.args = candidate_args
            self.args, self.runner, self.buffer, self.learner = build_sim(
                candidate_args, self.logger
            )
            candidate_runner = self.runner
            self.n_eval_eps = max(1, self.args.test_nepisode // self.runner.batch_size)
            self.runner.mac.msg_budget_per_agent = int(budget)
            self.train_single_task(
                reset_options=reset_options,
                close_env=False,
            )
            final_state_dist, success_rate = collect_successful_final_state_dist(
                self.runner,
                n_eval_eps=n_eval_eps,
                reset_options=reset_options,
            )
            policy_weights = {
                key: value.detach().cpu().clone()
                for key, value in self.runner.mac.agent.state_dict().items()
            }
            self._dependent_cm_total_t_env += self.runner.t_env
            return final_state_dist, success_rate, policy_weights
        finally:
            if candidate_runner is not None:
                candidate_runner.close_env()
            (
                self.args,
                self.runner,
                self.buffer,
                self.learner,
                self.n_eval_eps,
            ) = original_context
            np.random.set_state(numpy_state)
            th.random.set_rng_state(torch_state)
            if cuda_states is not None:
                th.cuda.set_rng_state_all(cuda_states)

    @staticmethod
    def _cm_edge_occupancy(solution, edge: tuple[int, int]) -> float:
        rows = solution.task_policy.loc[
            (solution.task_policy.state == edge[0])
            & (solution.task_policy.next_state == edge[1]),
            "occupancy",
        ]
        return float(rows.sum())

    @staticmethod
    def _cm_budget_probabilities(
        solution, edge: tuple[int, int], budgets: list[float]
    ) -> dict[float, float]:
        rows = solution.comms_budget_policy.loc[
            (solution.comms_budget_policy.state == edge[0])
            & (solution.comms_budget_policy.next_state == edge[1])
        ]
        return {
            budget: float(
                rows.loc[
                    np.isclose(rows.msg_budget_per_agent, budget),
                    "action_probability",
                ].sum()
            )
            for budget in budgets
        }

    def _evaluate_dependent_cm_policy(
        self,
        root_state: int,
        state_init_state_dists: dict[int, dict],
        policy_seeds: list[int],
    ) -> None:
        runner = self.runner
        env = runner.env
        set_init_state_dists = getattr(
            env, "set_dependent_navigation_init_state_dists", None
        )
        if set_init_state_dists is None:
            raise RuntimeError(
                "The composed evaluation environment cannot install learned "
                "navigation initial-state distributions."
            )
        set_init_state_dists(state_init_state_dists)
        previous_termination_setting = env.terminate_on_task_completed
        test_returns = list(runner.test_returns)
        test_stats = dict(runner.test_stats)
        env.terminate_on_task_completed = False
        reset_options = {
            "hl_start_state": root_state,
            "navigation_init_state_dist": state_init_state_dists[root_state],
            "select_navigation_task_from_policy": True,
        }
        n_eval_eps = max(1, int(self.args.test_nepisode))

        try:
            for policy_seed in policy_seeds:
                runner.mac.set_dependent_policy_seed(policy_seed)
                successes = 0
                for episode_idx in range(n_eval_eps):
                    result = runner.run(
                        test_mode=True,
                        return_log_stats=episode_idx == n_eval_eps - 1,
                        reset_options=reset_options,
                    )
                    final_infos = result.get("final_infos", [result.get("final_info")])
                    if not final_infos or final_infos[0] is None:
                        raise RuntimeError(
                            "End-to-end evaluation runner did not return terminal info."
                        )
                    successes += int(final_infos[0].get("task_completed", False))

                success_rate = successes / n_eval_eps
                self.logger.info(
                    f"End-to-end dependent policy seed {policy_seed}: "
                    f"{success_rate:.3f} ({successes}/{n_eval_eps})"
                )
                self.logger.log_stat(
                    "dependent_end_to_end_success_rate",
                    success_rate,
                    policy_seed,
                )
        finally:
            env.terminate_on_task_completed = previous_termination_setting
            runner.test_returns = test_returns
            runner.test_stats = test_stats
            runner.mac.set_dependent_policy_seed(None)

    def _save_dependent_cm_artifacts(self, policy_bank: dict, solution) -> None:
        save_path = join(
            self.args.local_results_path,
            "models",
            self.args.unique_token,
            "dependent_cm",
        )
        makedirs(save_path, exist_ok=True)

        candidate_rows = []
        for (from_state, to_state, budget, policy_seed), policy_weights in sorted(
            policy_bank.items()
        ):
            budget_token = f"{budget:g}".replace("-", "m").replace(".", "p")
            filename = (
                f"edge_{from_state}_{to_state}_budget_{budget_token}_"
                f"seed_{policy_seed}.th"
            )
            th.save(policy_weights, join(save_path, filename))
            candidate_rows.append({
                "from_state": from_state,
                "to_state": to_state,
                "msg_budget_per_agent": budget,
                "seed": policy_seed,
                "filename": filename,
            })

        pd.DataFrame.from_records(candidate_rows).to_csv(
            join(save_path, "candidate_manifest.csv"), index=False
        )
        solution.task_policy.to_csv(join(save_path, "task_policy.csv"), index=False)
        solution.comms_budget_policy.to_csv(
            join(save_path, "comms_budget_policy.csv"), index=False
        )
        self.logger.info(f"Saving dependent CM artifacts to {save_path}")
        if self.args.use_wandb:
            self.logger.log_agent(
                save_path=save_path,
                t=self._dependent_cm_total_t_env,
            )
        if self.args.delete_local_models:
            rmtree(save_path, ignore_errors=True)

    def _optimize_dependent_cm_policy(
        self,
        hlmdp,
        edges: list[tuple[int, int]],
        edge_success_rates: dict[tuple[int, int, float], float],
    ):
        """Solve the measured edge-and-budget transition model."""
        from gurobipy import GRB

        from src.modules.agents.ilp_model import ILPModel

        transition_probs = hlmdp.transition_probs.copy(deep=True)
        edge_set = set(edges)
        for (from_state, to_state, budget), success_rate in edge_success_rates.items():
            if (from_state, to_state) not in edge_set:
                raise ValueError(
                    f"Measured success rate references an unknown task edge "
                    f"{from_state}->{to_state}."
                )
            success_rate = float(np.clip(success_rate, 0.0, 1.0))
            edge_rows = transition_probs.loc[
                (transition_probs.state == from_state)
                & transition_probs.action.apply(
                    lambda action, target_state=to_state, selected_budget=budget: (
                        int(action[0]) == target_state
                        and np.isclose(float(action[1]), selected_budget)
                    )
                )
            ].index
            if len(edge_rows) != 2:
                raise ValueError(
                    f"Expected success and failure outcomes for task "
                    f"{from_state}->{to_state} at budget {budget:g}; "
                    f"found {len(edge_rows)}."
                )
            for row_index in edge_rows:
                next_state = int(transition_probs.at[row_index, "next_state"])
                if next_state == to_state:
                    probability = success_rate
                elif next_state == hlmdp.fail_state:
                    probability = 1.0 - success_rate
                else:
                    raise ValueError(
                        f"Unexpected transition outcome {next_state} for task "
                        f"{from_state}->{to_state}."
                    )
                transition_probs.at[row_index, "prob"] = probability

        hlmdp._transition_probs = transition_probs
        optimizer = ILPModel(self.args)
        try:
            solution = optimizer.optimize_policy(
                hlmdp,
                float(getattr(self.args, "success_rate_spec", 0.85)),
            )
            if solution.task_policy.empty:
                raise ValueError("The high-level optimizer returned an empty policy.")
            return solution
        except RuntimeError:
            if optimizer.model.Status == GRB.INFEASIBLE:
                self.logger.info(
                    "High-level ILP model is infeasible for the measured "
                    "edge-and-budget success rates."
                )
                return None
            raise
        finally:
            optimizer.model.dispose()

    @staticmethod
    def _uniform_cm_policy(
        edges: list[tuple[int, int]],
    ) -> dict[int, dict[tuple[int, int], float]]:
        outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for edge in edges:
            outgoing[edge[0]].append(edge)
        return {
            state: {edge: 1.0 / len(state_edges) for edge in state_edges}
            for state, state_edges in outgoing.items()
        }

    @classmethod
    def _build_cm_policy(
        cls,
        edges: list[tuple[int, int]],
        edge_success_rates: dict[tuple[int, int], float],
        terminal_states: set[int],
    ) -> dict[int, dict[tuple[int, int], float]]:
        outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for edge in edges:
            outgoing[edge[0]].append(edge)

        state_values = {state: 1.0 for state in terminal_states}
        for from_state, to_state in reversed(edges):
            state_values[from_state] = max(
                edge_success_rates[(from_state, next_state)]
                * state_values.get(next_state, 0.0)
                for _, next_state in outgoing[from_state]
            )

        policy = {}
        for state, state_edges in outgoing.items():
            scores = np.array(
                [
                    edge_success_rates[edge] * state_values.get(edge[1], 0.0)
                    for edge in state_edges
                ],
                dtype=float,
            )
            best = np.flatnonzero(np.isclose(scores, scores.max()))
            probability = 1.0 / len(best)
            policy[state] = {edge: 0.0 for edge in state_edges}
            for index in best:
                policy[state][state_edges[index]] = probability
        return policy

    @staticmethod
    def _evaluate_cm_policy(
        edges: list[tuple[int, int]],
        edge_success_rates: dict[tuple[int, int], float],
        policy: dict[int, dict[tuple[int, int], float]],
        root_state: int,
        terminal_states: set[int],
    ) -> float:
        occupancy = {root_state: 1.0}
        for from_state, to_state in edges:
            flow = occupancy.get(from_state, 0.0)
            occupancy[to_state] = (
                occupancy.get(to_state, 0.0)
                + flow
                * policy.get(from_state, {}).get((from_state, to_state), 0.0)
                * edge_success_rates[(from_state, to_state)]
            )
        return sum(occupancy.get(state, 0.0) for state in terminal_states)

    @staticmethod
    def _state_dist_delta(
        previous: dict[int, dict] | None, current: dict[int, dict]
    ) -> float:
        if previous is None:
            return float("inf")
        deltas = []
        for state in set(previous) | set(current):
            previous_dist = previous.get(state, {"states": [], "probs": []})
            current_dist = current.get(state, {"states": [], "probs": []})
            probabilities = defaultdict(lambda: [0.0, 0.0])
            for index, (joint_state, probability) in enumerate(
                zip(previous_dist["states"], previous_dist["probs"])
            ):
                probabilities[tuple(tuple(position) for position in joint_state)][0] = (
                    probability
                )
            for joint_state, probability in zip(
                current_dist["states"], current_dist["probs"]
            ):
                probabilities[tuple(tuple(position) for position in joint_state)][1] = (
                    probability
                )
            deltas.append(
                sum(
                    abs(previous_probability - current_probability)
                    for previous_probability, current_probability in probabilities.values()
                )
                / 2.0
            )
        return max(deltas, default=0.0)

    @staticmethod
    def _merge_state_dists(dists: list[dict], weights: list[float]) -> dict:
        """Combine several `{"states": [...], "probs": [...]}` distributions into
        one, weighted by how often trajectories reach each distribution's edge."""
        total_weight = sum(weights)
        if total_weight <= 0:
            weights = [1.0] * len(weights)
            total_weight = float(len(weights))

        merged: dict[tuple, float] = defaultdict(float)
        for dist, weight in zip(dists, weights):
            if not dist["states"]:
                continue
            scale = weight / total_weight
            for state, prob in zip(dist["states"], dist["probs"]):
                merged[tuple(tuple(position) for position in state)] += scale * prob

        if not merged:
            raise ValueError(
                "No successful terminal states available to build the next "
                "subtask's spawn distribution; predecessor subtask(s) never succeeded."
            )

        states = list(merged.keys())
        probs = np.array(list(merged.values()))
        probs = probs / probs.sum()
        return {"states": [list(state) for state in states], "probs": probs.tolist()}
