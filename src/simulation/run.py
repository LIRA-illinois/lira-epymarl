import datetime
import multiprocessing as mp
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from os import listdir, makedirs, walk
from os.path import abspath, isdir, join, splitext
from shutil import rmtree
from statistics import NormalDist
from types import SimpleNamespace as SN
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from matplotlib.patches import FancyArrowPatch

import wandb
from src.simulation.build import build_sim
from src.simulation.evaluate import collect_successful_final_state_dist, run_eval_episodes
from src.utils.general_reward_support import test_alg_config_supports_reward
from src.utils.logging import MainLogger, log_setup
from src.utils.navigation_rendering import (
    load_navigation_mdp,
    render_navigation_task_frames,
)
from src.utils.timehelper import time_left, time_str

# use agg backend to support multiprocessing
plt.switch_backend("agg")


class Simulation:
    def __init__(self, _config, _log) -> None:
        mp.set_start_method("spawn", force=True)
        args: SN = self._parse_config(_config, _log)

        self.logger: MainLogger = self._build_logger(args, _config, _log)
        self.args, self.runner, self.buffer, self.learner = build_sim(args, self.logger)
        self.n_eval_eps = max(1, self.args.test_nepisode // self.runner.batch_size)

    def run(self) -> None:
        if self.args.evaluate:
            self._load_checkpoint()

            # run evaluation on loaded checkpoint
            if self.args.evaluate or self.args.save_replay:
                self.evaluate_loaded()
                return

        elif hasattr(self.args, "post_processing"):
            assert self.args.post_processing in [
                "optimize_high_level_policy",
                "aggregate_runs",
            ]

            if self.args.post_processing == "optimize_high_level_policy":
                df_data = self._load_experiment_table_artifact(
                    art_name="eval_stats",
                    art_version="latest",
                    art_type="run_table",
                )
                self.logger.info("Optimizing High-Level Policy", log_header=True)
                self._train_high_level_policy(df_data)
                print(self.runner.mac.comms_agent.policy.task_policy)
                print(self.runner.mac.comms_agent.policy.comms_budget_policy)
                self.runner.close_env()
                return

            elif self.args.post_processing == "aggregate_runs":
                df_data, runs = self._load_experiment_table_artifact(
                    art_name="eval_stats",
                    art_version="latest",
                    art_type="run_table",
                    return_runs=True,
                )

                # Group scenarios by experimental conditions while allowing
                # different model implementations to contribute to one plot.
                ignored_config_keys = {
                    "_wandb",
                    "msg_budget_per_agent",
                    "scenario",
                    "seed",
                    "config",
                    "agent",
                    "learner",
                    "mac",
                    "mixer",
                }

                def without_seed(value):
                    if isinstance(value, dict):
                        return {
                            key: without_seed(item)
                            for key, item in value.items()
                            if key != "seed"
                        }
                    if isinstance(value, (list, tuple)):
                        return type(value)(without_seed(item) for item in value)
                    return value

                scenario_groups = {}
                group_runs = {}
                scenario_runs = {}
                for run in runs:
                    scenario_value = run.config.get("scenario")
                    try:
                        scenario = int(scenario_value)
                    except (TypeError, ValueError):
                        continue

                    group_key = tuple(
                        sorted(
                            (key, repr(without_seed(value)))
                            for key, value in run.config.items()
                            if key not in ignored_config_keys
                        )
                    )
                    scenario_groups[scenario] = group_key
                    scenario_runs[scenario] = run
                    group_runs.setdefault(group_key, run)

                df_data["scenario_group"] = df_data["scenario"].map(scenario_groups)
                df_data = df_data.loc[df_data["scenario_group"].notna()]

                source_run = next(iter(group_runs.values()), runs[0])

                postprocess_name = f"{self.args.time_id}_postprocess"
                api = wandb.Api()
                postprocess_runs = [
                    run
                    for run in api.runs(
                        self.args.wandb_project,
                        filters={"config.time_id": self.args.time_id},
                    )
                    if (getattr(run, "name", "") or "") in {postprocess_name}
                    or (getattr(run, "name", "") or "").startswith(
                        f"{postprocess_name}_"
                    )
                ]
                active_postprocess_run = next(
                    (
                        run
                        for run in postprocess_runs
                        if (getattr(run, "state", "") or "").lower() == "running"
                    ),
                    None,
                )

                if active_postprocess_run is not None:
                    self.logger.info(
                        f"Resuming active post-processing run {active_postprocess_run.id}"
                    )
                    wandb_run = wandb.init(
                        entity=getattr(active_postprocess_run, "entity", None),
                        project=getattr(active_postprocess_run, "project", None),
                        id=active_postprocess_run.id,
                        resume="allow",
                    )
                else:
                    finished_postprocess_run = any(
                        (getattr(run, "state", "") or "").lower() == "finished"
                        for run in postprocess_runs
                    )
                    if finished_postprocess_run:
                        revision = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        postprocess_name = f"{postprocess_name}_{revision}"

                    # Use a dedicated online run for aggregate images.
                    wandb_run = wandb.init(
                        entity=getattr(source_run, "entity", None),
                        project=getattr(source_run, "project", None),
                        name=postprocess_name,
                        config={
                            "experiment": self.args.experiment,
                            "scenario": "postprocess",
                            "time_id": self.args.time_id,
                            "post_processing": self.args.post_processing,
                        },
                        mode="online",
                    )

                combined_groups_by_map = {}
                combined_source_runs_by_map = {}
                source_runs_by_map = {}
                for group_idx, (group_key, df_group) in enumerate(
                    df_data.groupby("scenario_group", sort=True), start=1
                ):
                    group_columns = [
                        "msg_budget_per_agent",
                        "t_env_rounded",
                    ]
                    df_avg = (
                        df_group.groupby(group_columns, dropna=False)
                        .mean(numeric_only=True)
                        .reset_index()
                    )
                    df_avg["t_env"] = df_avg["t_env_rounded"]

                    n_seeds_per_run = df_group[
                        ["t_env_rounded", "scenario", "msg_budget_per_agent"]
                    ].value_counts()
                    min_n_seeds, max_n_seeds = (
                        min(n_seeds_per_run),
                        max(n_seeds_per_run),
                    )

                    parameter_info = ", ".join(
                        f"{key}={value}" for key, value in group_key
                    )
                    scenario_indices = sorted(
                        {
                            int(scenario)
                            for scenario in df_group["scenario"].dropna().unique()
                        }
                    )
                    group_source_run = group_runs.get(group_key)
                    if group_source_run is None:
                        continue
                    map_name = self._map_name_from_config(group_source_run.config)
                    map_name = map_name or "unknown_map"
                    hl_task = group_source_run.config.get("hl_task")
                    scenario_label = "_".join(
                        str(scenario) for scenario in scenario_indices
                    )
                    if isinstance(hl_task, (list, tuple)):
                        hl_task_label = "-".join(str(state) for state in hl_task)
                    elif hl_task is not None:
                        hl_task_label = str(hl_task)
                    else:
                        hl_task_label = None

                    plot_group = f"s{scenario_label}"
                    if hl_task_label is not None:
                        plot_group += f"_t{hl_task_label}"
                    plot_group = f"map_{map_name}_{plot_group}"
                    combined_groups_by_map.setdefault(map_name, []).append(
                        (
                            f"task {hl_task_label or group_idx}",
                            df_avg.copy(),
                            hl_task,
                        )
                    )
                    source_runs_by_budget = {}
                    for scenario in scenario_indices:
                        scenario_run = scenario_runs.get(scenario)
                        if scenario_run is None:
                            continue
                        for budget in self._configured_message_budgets(
                            scenario_run.config.get("msg_budget_per_agent")
                        ):
                            source_runs_by_budget[budget] = scenario_run
                    combined_source_runs_by_map.setdefault(map_name, []).append(
                        source_runs_by_budget
                    )
                    source_runs_by_map.setdefault(map_name, group_source_run)
                    self.logger.info(
                        f"Plotting aggregated eval stats for group {group_idx}: "
                        f"{parameter_info}"
                    )
                    if hl_task is not None:
                        aggregate_table = wandb.Table(dataframe=df_avg)
                        self._make_comms_eval_plots(
                            aggregate_table,
                            t=np.max(df_avg.t_env_rounded),
                            wandb_run=wandb_run,
                            info_str=(
                                f"Group {group_idx}; Min Seeds: {min_n_seeds}, "
                                f"Max Seeds: {max_n_seeds}"
                            ),
                            plot_group=plot_group,
                        )

                for map_name in sorted(combined_groups_by_map):
                    map_groups = combined_groups_by_map[map_name]
                    map_t = max(
                        int(df["t_env_rounded"].max()) for _, df, _ in map_groups
                    )
                    if not any(hl_task is not None for _, _, hl_task in map_groups):
                        map_df = pd.concat(
                            [df for _, df, _ in map_groups], ignore_index=True
                        )
                        map_df = (
                            map_df.groupby(
                                ["msg_budget_per_agent", "t_env"], dropna=False
                            )
                            .mean(numeric_only=True)
                            .reset_index()
                        )
                        self._make_comms_eval_plots(
                            wandb.Table(dataframe=map_df),
                            t=map_t,
                            wandb_run=wandb_run,
                            plot_group=f"map_{map_name}",
                        )
                    if any(hl_task is not None for _, _, hl_task in map_groups):
                        self._make_combined_comms_eval_plots(
                            map_groups,
                            t=map_t,
                            map_name=map_name,
                            wandb_run=wandb_run,
                        )
                        self._make_task_success_frame_plot(
                            map_groups,
                            source_runs=combined_source_runs_by_map[map_name],
                            source_run=source_runs_by_map[map_name],
                            t=map_t,
                            map_name=map_name,
                            wandb_run=wandb_run,
                        )
                wandb_run.finish()

                return

        # # run training
        # if hierarchical:
        #     self.train_hierarchical()

        # else:

        hl_task_sequence = getattr(self.args, "hl_task_sequence", None)
        if hl_task_sequence is not None:
            self.train_dependent_subtasks(hl_task_sequence)
            return

        hl_task = getattr(self.args, "hl_task", None)
        reset_options = None
        if hl_task is not None:
            reset_options = {"hl_start_state": hl_task[0], "hl_task": hl_task}

        self.train_single_task(reset_options=reset_options)

    def train_single_task(
        self, reset_options: Optional[dict] = None, close_env: bool = True
    ) -> None:
        """
        original EPYMARL training for non-hierarchical policies for a project with a single task
        """
        self.logger.info("Training Policy", log_header=True)

        # timing setup
        episode = 0
        last_test_t = 0
        last_log_t = 0
        model_save_time = 0

        start_time = time.time()
        last_time = start_time
        episode_reward_history: list[tuple[int, float]] = []
        evaluation_success_lcb: Optional[float] = None

        # training loop
        self.logger.info("Beginning training for {} timesteps".format(self.args.t_max))

        if getattr(self.args, "unique_policy_per_msg_budget", False):
            self.runner.mac.msg_budget_per_agent = self.args.msg_budget_per_agent[0]

        while self.runner.t_env <= self.args.t_max:
            # Run for a whole episode at a time
            result = self.runner.run(test_mode=False, reset_options=reset_options)

            episode_batch = result["batch"] if isinstance(result, dict) else result
            self.buffer.insert_episode_batch(episode_batch)

            if getattr(self.args, "early_stopping", False):
                episode_reward_history.extend(
                    (self.runner.t_env, episode_reward_total)
                    for episode_reward_total in self._episode_batch_reward_totals(
                        episode_batch
                    )
                )
                window = getattr(self.args, "early_stopping_window", 500000)
                episode_reward_history = [
                    (timestamp, episode_reward_total)
                    for timestamp, episode_reward_total in episode_reward_history
                    if timestamp >= self.runner.t_env - window
                ]

            # Keep the learner update count proportional to collected episodes.
            n_updates = episode_batch.batch_size
            for update_idx in range(n_updates):
                if not self.buffer.can_sample(self.args.batch_size):
                    break

                episode_sample = self.buffer.sample(self.args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != self.args.device:
                    episode_sample.to(self.args.device)

                self.learner.train(
                    episode_sample,
                    self.runner.t_env,
                    episode + update_idx,
                )

            # run evaluation episodes
            if self.runner.t_env - last_test_t >= self.args.test_interval:
                self.logger.info(f"t_env: {self.runner.t_env} / {self.args.t_max}")
                self.logger.info(
                    (
                        "Estimated time left: "
                        f"{time_left(last_time, last_test_t, self.runner.t_env, self.args.t_max)}. "
                        "Time passed: "
                        f"{time_str(time.time() - start_time)}"
                    )
                )

                last_time = time.time()
                last_test_t = self.runner.t_env
                evaluation_success_lcb = self.evaluate(
                    n_eval_eps=self.n_eval_eps, reset_options=reset_options
                )
                if evaluation_success_lcb is not None:
                    self.logger.log_stat(
                        "early_stopping_success_lcb",
                        evaluation_success_lcb,
                        self.runner.t_env,
                    )

            reward_stabilized = self._training_reward_stabilized(
                episode_reward_history,
                self.runner.t_env,
                getattr(self.args, "early_stopping_window", 500000),
                getattr(self.args, "early_stopping_min_delta", 0.01),
            )
            min_success_lcb = getattr(self.args, "early_stopping_min_success_lcb", None)
            success_lcb_ready = min_success_lcb is None or (
                evaluation_success_lcb is not None
                and evaluation_success_lcb >= min_success_lcb
            )
            if reward_stabilized and success_lcb_ready:
                stopping_reason = "episode reward stabilized"
                if min_success_lcb is not None:
                    stopping_reason += (
                        f"; success-rate LCB reached {evaluation_success_lcb:.3f}"
                    )
                self.logger.info(
                    "Stopping training early: "
                    f"{stopping_reason} over "
                    f"{self.args.early_stopping_window} timesteps."
                )
                break

            # save model to disk
            if (
                self.args.save_model
                and self.runner.t_env - model_save_time >= self.args.save_model_interval
            ):
                model_save_time = self.runner.t_env
                self.save()

            episode += self.args.batch_size_run

            # log key training / eval metrics
            if (self.runner.t_env - last_log_t) >= self.args.log_interval:
                self.logger.log_stat("episode", episode, self.runner.t_env)
                self.logger.print_recent_stats()
                last_log_t = self.runner.t_env

        if close_env:
            self.runner.close_env()
        self.logger.info("Finished Training")

    def train_dependent_subtasks(self, hl_task_sequence: list) -> None:
        """
        Train a DAG of dependent subtasks in topological order.

        Unlike independent subtasks (`hl_task`), which assume a fixed,
        hand-specified spawn distribution per edge, each dependent subtask's
        initial state distribution is learned from its predecessor edge(s)'
        trained policies: after training an edge, its successful terminal
        joint-agent positions are evaluated empirically and used as the spawn
        distribution for the subtask(s) leaving its destination state. When a
        state has multiple incoming edges, their final-state distributions are
        combined, weighted by the fraction of trajectories (from the root
        state) that survive to traverse each edge.

        Each subtask is trained as its own independent low-level policy (own
        network weights, own `t_env` budget), since edges represent distinct
        tasks -- only the learned spawn distribution is threaded between them.

        Parameters
        ----------
        hl_task_sequence : list
            Subtask edges [from_state, to_state] listed in topological order,
            e.g. [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]].
        """
        edges: list[tuple[int, int]] = [
            (int(edge[0]), int(edge[1])) for edge in hl_task_sequence
        ]
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

        n_eval_eps = getattr(self.args, "n_eval_eps_dependent_subtask", self.n_eval_eps)
        max_cm_iterations = max(1, int(getattr(self.args, "cm_max_iterations", 20)))
        cm_epsilon = float(getattr(self.args, "cm_convergence_epsilon", 0.2))
        terminal_states = destination_states - source_states
        if not terminal_states:
            raise ValueError("hl_task_sequence must contain at least one terminal state.")

        previous_policy: dict[int, dict[tuple[int, int], float]] | None = None
        previous_init_state_dists: dict[int, dict] | None = None
        self.runner.close_env()

        for cm_iter in range(max_cm_iterations):
            self.logger.info(
                f"Starting dependent-subtask CM iteration {cm_iter + 1}/"
                f"{max_cm_iterations}",
                log_header=True,
            )

            # State occupancy is the fraction of root trajectories reaching each
            # state under the previous CM policy. Iteration zero is uniform.
            occupancy = {root_state: 1.0}
            edge_success_rates: dict[tuple[int, int], float] = {}
            edge_final_state_dists: dict[tuple[int, int], dict] = {}
            learned_init_state_dists: dict[int, dict] = {}
            merge_policy = previous_policy or self._uniform_cm_policy(edges)

            for from_state, to_state in edges:
                self.logger.info(
                    f"Training dependent subtask {from_state} -> {to_state} "
                    f"(CM iteration {cm_iter + 1})",
                    log_header=True,
                )

                if from_state == root_state:
                    reset_options = {"hl_task": [from_state, to_state]}
                else:
                    if from_state not in learned_init_state_dists:
                        raise ValueError(
                            "hl_task_sequence must be topologically ordered; "
                            f"no initial distribution is available for state {from_state}."
                        )
                    reset_options = {
                        "navigation_task_transition": [from_state, to_state],
                        "navigation_init_state_dist": learned_init_state_dists[from_state],
                    }

                self.args, self.runner, self.buffer, self.learner = build_sim(
                    self.args, self.logger
                )
                self.train_single_task(reset_options=reset_options, close_env=False)

                final_state_dist, success_rate = collect_successful_final_state_dist(
                    self.runner, n_eval_eps=n_eval_eps, reset_options=reset_options
                )
                self.runner.close_env()

                self.logger.info(
                    f"Subtask {from_state} -> {to_state} success rate: "
                    f"{success_rate:.3f} ({len(final_state_dist['states'])} unique "
                    "successful terminal states)"
                )
                edge = (from_state, to_state)
                edge_success_rates[edge] = success_rate
                edge_final_state_dists[edge] = final_state_dist

                incoming_edges = predecessors[to_state]
                if all(edge in edge_success_rates for edge in incoming_edges):
                    weights = [
                        occupancy.get(edge[0], 0.0)
                        * merge_policy.get(edge[0], {}).get(
                            edge, 1.0 / len(merge_policy.get(edge[0], {}))
                        )
                        * edge_success_rates[edge]
                        for edge in incoming_edges
                    ]
                    learned_init_state_dists[to_state] = self._merge_state_dists(
                        [edge_final_state_dists[edge] for edge in incoming_edges],
                        weights,
                    )
                    occupancy[to_state] = sum(weights)

            cm_policy, policy_source = self._optimize_dependent_cm_policy(
                edges, edge_success_rates, terminal_states
            )
            hierarchical_success = self._evaluate_cm_policy(
                edges, edge_success_rates, cm_policy, root_state, terminal_states
            )
            delta_isd = self._state_dist_delta(
                previous_init_state_dists, learned_init_state_dists
            )
            self.logger.info(
                f"CM iteration {cm_iter + 1}: hierarchical success probability "
                f"{hierarchical_success:.3f}, initial-state distance {delta_isd:.3f}, "
                f"policy source: {policy_source}"
            )
            self.logger.log_stat(
                "dependent_cm_success_rate", hierarchical_success, cm_iter
            )
            self.logger.log_stat("dependent_cm_isd_delta", delta_isd, cm_iter)

            previous_policy = cm_policy
            previous_init_state_dists = learned_init_state_dists
            self.dependent_cm_policy = cm_policy
            if cm_iter > 0 and delta_isd <= cm_epsilon:
                self.logger.info(
                    f"Dependent-subtask CM converged after {cm_iter + 1} iterations."
                )
                break

        self.logger.info("Finished Dependent Subtask Training")

    def _optimize_dependent_cm_policy(
        self,
        edges: list[tuple[int, int]],
        edge_success_rates: dict[tuple[int, int], float],
        terminal_states: set[int],
    ) -> tuple[dict[int, dict[tuple[int, int], float]], str]:
        """Solve the measured dependent-task model and return a state policy.

        Dependent-task training currently measures one success rate per edge, so
        that rate is assigned to every configured communication budget. The ILP
        can therefore optimize routing and select the least costly budget while
        respecting the global success specification.
        """
        fallback = self._build_cm_policy(edges, edge_success_rates, terminal_states)
        try:
            from src.modules.agents.ilp_model import ILPModel

            hlmdp = self.runner.env.hlmdp
            transition_probs = hlmdp.transition_probs.copy(deep=True)
            edge_set = set(edges)
            for edge, success_rate in edge_success_rates.items():
                from_state, to_state = edge
                if edge not in edge_set:
                    continue
                success_rate = float(np.clip(success_rate, 0.0, 1.0))
                edge_rows = transition_probs.loc[
                    (transition_probs.state == from_state)
                    & (transition_probs.action.apply(lambda action: action[0] == to_state))
                ].index
                if len(edge_rows) == 0:
                    raise ValueError(f"No high-level actions found for edge {edge}.")
                for row_index in edge_rows:
                    next_state = transition_probs.at[row_index, "next_state"]
                    transition_probs.at[row_index, "prob"] = (
                        success_rate
                        if next_state == to_state
                        else 1.0 - success_rate
                    )

            hlmdp._transition_probs = transition_probs
            optimizer = ILPModel(self.args)
            solution = optimizer.optimize_policy(
                hlmdp,
                float(getattr(self.args, "success_rate_spec", 0.85)),
            )
            occupancy = solution.task_policy
            if occupancy.empty:
                raise ValueError("The high-level optimizer returned an empty policy.")

            outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for edge in edges:
                outgoing[edge[0]].append(edge)
            policy = {state: {edge: 0.0 for edge in state_edges} for state, state_edges in outgoing.items()}
            for state, state_edges in outgoing.items():
                state_occupancy = occupancy.loc[occupancy.state == state, "occupancy"].sum()
                if state_occupancy <= 0.0:
                    for edge in state_edges:
                        policy[state][edge] = 1.0 / len(state_edges)
                    continue
                for edge in state_edges:
                    edge_occupancy = occupancy.loc[
                        (occupancy.state == edge[0])
                        & (occupancy.next_state == edge[1]),
                        "occupancy",
                    ].sum()
                    policy[state][edge] = float(edge_occupancy / state_occupancy)
                total_probability = sum(policy[state].values())
                if total_probability <= 0.0:
                    for edge in state_edges:
                        policy[state][edge] = 1.0 / len(state_edges)
                else:
                    for edge in state_edges:
                        policy[state][edge] /= total_probability
            optimizer.model.dispose()
            return policy, "ILP"
        except Exception as error:
            self.logger.info(
                f"High-level ILP policy unavailable ({error}); using analytical policy."
            )
            return fallback, "analytical fallback"

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
            occupancy[to_state] = occupancy.get(to_state, 0.0) + flow * policy.get(
                from_state, {}
            ).get((from_state, to_state), 0.0) * edge_success_rates[
                (from_state, to_state)
            ]
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
                sum(abs(previous_probability - current_probability) for previous_probability, current_probability in probabilities.values())
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

    @staticmethod
    def _episode_batch_reward_totals(episode_batch) -> np.ndarray:
        rewards = episode_batch["reward"].detach().cpu().numpy()
        filled = episode_batch["filled"].detach().cpu().numpy().squeeze(-1)
        episode_reward_totals = (rewards * filled[..., None]).sum(axis=1)
        if episode_reward_totals.ndim > 1:
            episode_reward_totals = episode_reward_totals.sum(axis=-1)
        return np.asarray(episode_reward_totals, dtype=float)

    @staticmethod
    def _training_reward_stabilized(
        episode_reward_history: list[tuple[int, float]],
        t_env: int,
        window: int,
        min_delta: float,
    ) -> bool:
        if window <= 0 or t_env < window:
            return False

        window_start = t_env - window
        midpoint = window_start + window / 2
        first_half = [
            episode_reward_total
            for timestamp, episode_reward_total in episode_reward_history
            if window_start <= timestamp < midpoint
        ]
        second_half = [
            episode_reward_total
            for timestamp, episode_reward_total in episode_reward_history
            if timestamp >= midpoint
        ]

        if len(first_half) < 5 or len(second_half) < 5:
            return False

        return (
            abs(float(np.mean(second_half)) - float(np.mean(first_half))) <= min_delta
        )

    @staticmethod
    def _success_rate_lcb(
        success_rate: float, n_episodes: int, confidence: float
    ) -> Optional[float]:
        """Return the Wilson lower confidence bound for a success rate."""
        if n_episodes <= 0 or not 0.0 <= success_rate <= 1.0:
            return None
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1")

        z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
        denominator = 1.0 + z**2 / n_episodes
        center = success_rate + z**2 / (2.0 * n_episodes)
        margin = z * np.sqrt(
            success_rate * (1.0 - success_rate) / n_episodes
            + z**2 / (4.0 * n_episodes**2)
        )
        return float((center - margin) / denominator)

    def _evaluation_success_lcb(self, log_stats: dict) -> Optional[float]:
        success_rate = log_stats.get("test_task_completed_mean")
        n_episodes = log_stats.get("test_n_episodes")
        if success_rate is None or n_episodes is None:
            return None
        return self._success_rate_lcb(
            float(success_rate),
            int(n_episodes),
            getattr(self.args, "early_stopping_confidence", 0.95),
        )

    def _train_high_level_policy(self, df_data: pd.DataFrame) -> None:
        self.runner.env.hlmdp.transition_probs = df_data
        self.learner.optimize_hl_agent(
            self.runner.env.hlmdp, self.args.success_rate_spec
        )

    def evaluate(
        self, n_eval_eps: int, reset_options: Optional[dict] = None
    ) -> Optional[float]:
        """Evaluation entry point."""

        # always comms sweep if hierarchical or not
        if hasattr(self.args, "msg_budget_per_agent"):
            msg_budget_per_agent_list = self.args.msg_budget_per_agent
            self.logger.info(
                f"Evaluating Policy Across Message Budgets: {msg_budget_per_agent_list}",
                log_header=True,
            )

            eval_data: list[dict] = []

            # Keep evaluation layouts matched across communication budgets.
            # Process-backed runners keep their RNGs in the worker processes.
            if hasattr(self.runner, "get_env_rng_states"):
                init_rng_state = self.runner.get_env_rng_states()
            else:
                env_rng = self.runner.env.get_wrapper_attr("np_random")
                init_rng_state = env_rng.bit_generator.state

            for budget in msg_budget_per_agent_list:
                if hasattr(self.runner, "set_env_rng_states"):
                    self.runner.set_env_rng_states(init_rng_state)
                else:
                    env_rng = self.runner.env.get_wrapper_attr("np_random")
                    env_rng.bit_generator.state = init_rng_state

                self.logger.info(f"Evaluating with msg_budget_per_agent = {budget}")

                if reset_options is None:
                    reset_options = {"msg_budget_per_agent": budget}
                else:
                    reset_options["msg_budget_per_agent"] = budget

                result = run_eval_episodes(
                    args=self.args,
                    runner=self.runner,
                    n_eval_eps=n_eval_eps,
                    t_env=self.runner.t_env,
                    reset_options=reset_options,
                )
                eval_data.append(result["log_stats"])

            df_eval = pd.DataFrame.from_records(eval_data)
            success_lcbs = [
                lcb
                for log_stats in eval_data
                if (lcb := self._evaluation_success_lcb(log_stats)) is not None
            ]

            self.logger.log_table(key="eval_stats", value=df_eval, t=self.runner.t_env)
            self._make_comms_eval_plots(
                self.logger.data_tables["eval_stats"], t=self.runner.t_env
            )
            return max(success_lcbs, default=None)

            """
            if self.args.parallel_comms_eval:
                agent_state_dict = {k: v.cpu() for k, v in self.runner.mac.agent.state_dict().items()}
                wandb_attrs = ["entity", "project", "id", "name"]
                wandb_config = {attr: getattr(self.logger.wandb, attr) for attr in wandb_attrs}

                n_procs = getattr(self.args, "max_parallel_eval_processes", min(len(msg_budget_per_agents), max(1, (cpu_count() or 1) - 1)))

                inputs = []
                for msg_budget_per_agent in msg_budget_per_agents:
                    input_args = {
                        "function": eval_worker,
                        "args": self.args,
                        "n_eval_e
                        ps": n_eval_eps,
                        "t_env": self.runner.t_env,
                        "agent_state_dict": agent_state_dict,
                        "logger_dir": self.logger.dir,
                        "wandb_config": wandb_config,
                        "reset_options": {"msg_budget_per_agent": msg_budget_per_agent},
                    }
                    inputs.append(input_args)

                with mp.Pool(processes=n_procs, maxtasksperchild=2) as pool:
                    results: list[dict] = list(pool.map(mp_kwargs_wrapper, inputs))

                eval_data = [res["log_stats"] for res in results]

            else:
            """

        # non-hierarchical evaluation w/ no comms sweep
        else:
            self.logger.info("Evaluating Policy", log_header=True)
            result = run_eval_episodes(
                args=self.args, runner=self.runner, n_eval_eps=n_eval_eps
            )
            if result is None:
                return None
            return self._evaluation_success_lcb(result["log_stats"])

        """
        # Hierarchical env handling
        only needed if you want to eval multiple tasks in one process
        if hasattr(self.runner.env, "hlmdp"):
            hlmdp = self.runner.env.hlmdp

            # goal-conditioned approach, 1 policy for multiple tasks, not quite there yet
            # # Full sweep across all HL actions
            # if reset_options is None:
            #     df_actions = hlmdp.transition_probs.copy()
            #     df_actions = df_actions.loc[df_actions.state_type == "normal"]
            #     hl_actions = df_actions.action.drop_duplicates().tolist()

            #     self.logger.info(f"Evaluating Policy Across Tasks", log_header=True)

            #     eval_data: list[dict] = []
            #     for action in hl_actions:
            #         state = df_actions.loc[df_actions.action == action, "state"].unique().item()
            #         chosen_next_state, comms_val = action
            #         ro = {"hl_start_state": int(state), "msg_budget_per_agent": comms_val}

            #         result = run_eval_episodes(
            #             args=self.args,
            #             runner=self.runner,
            #             n_eval_eps=n_eval_eps,
            #             t_env=self.runner.t_env,
            #             reset_options=ro,
            #         )

            #         eval_data.append(result["log_stats"])

            #         # update transition probs
            #         success_rate = result["log_stats"].get("test_task_completed_mean")
            #         df = hlmdp.transition_probs
            #         df.loc[(df.action == action) & (df.next_state == chosen_next_state), "prob"] = success_rate
            #         df.loc[(df.action == action) & (df.next_state != chosen_next_state), "prob"] = (1.0 - success_rate)

            #     df_eval = pd.DataFrame.from_records(eval_data)
            #     self.logger.log_table(df_eval, t=self.runner.t_env)

            # Single-task evaluation
            result = run_eval_episodes(
                args=self.args,
                runner=self.runner,
                n_eval_eps=n_eval_eps,
                t_env=self.runner.t_env,
                reset_options=reset_options,
            )

            df_data = pd.DataFrame.from_records([result["log_stats"]])
            self.logger.log_table(df_data, t=self.runner.t_env)

            # Optionally update HLMDP if caller provided the exact action tuple
            action = reset_options.get("action") if reset_options is not None else None
            if action is not None:
                success_rate = result["log_stats"].get("test_task_completed_mean")
                df = hlmdp.transition_probs
                chosen_next_state, _ = action
                df.loc[(df.action == action) & (df.next_state == chosen_next_state), "prob"] = success_rate
                df.loc[(df.action == action) & (df.next_state != chosen_next_state), "prob"] = (1.0 - success_rate)
        """

    def _evaluate_all_tasks(
        self,
        hlmdp,
        n_eval_eps: int,
    ) -> None:
        """
        Evaluate the policy starting from every non-terminating HLMDP state.

        Parameters
        ----------
        hlmdp : ProjectMDP
            High-level MDP instance used by the hierarchical environment.
        """
        # gather all normal-state outgoing actions (tuples)
        df_actions = hlmdp.transition_probs.copy()
        df_actions = df_actions.loc[df_actions.state_type == "normal"]

        # unique action tuples: (chosen_next_state, comms_val)
        hl_actions = df_actions.action.drop_duplicates().tolist()

        self.logger.info("Evaluating Policy Across Tasks", log_header=True)

        eval_data: list[dict] = []

        # For each unique HL action, run evals from the state it goes out of
        for action in hl_actions:
            # action is expected to be a tuple (chosen_next_state, comms_val)
            state = df_actions.loc[df_actions.action == action, "state"].unique().item()
            chosen_next_state, message_budget = action
            reset_options = {
                "hl_start_state": int(state),
                "msg_budget_per_agent": message_budget,
            }

            # set comms value if provided
            result = run_eval_episodes(
                args=self.args,
                runner=self.runner,
                n_eval_eps=n_eval_eps,
                t_env=self.runner.t_env,
                reset_options=reset_options,
            )

            eval_data.append(result["log_stats"])

            # update transition probs in hlmdp, 2 possible outcomes of task success or failure
            df = hlmdp.transition_probs
            success_rate = result["log_stats"].get("test_task_completed_mean")
            df.loc[
                (df.action == action) & (df.next_state == chosen_next_state), "prob"
            ] = success_rate
            df.loc[
                (df.action == action) & (df.next_state != chosen_next_state), "prob"
            ] = 1.0 - success_rate

        df_eval = pd.DataFrame.from_records(eval_data)
        self.logger.log_table(key="eval_stats", value=df_eval, t=self.runner.t_env)

        # TODO it may make sense to log each tasks's success rate to wandb too

        # if msg_budget_per_agents is not None:
        #     self._make_comms_eval_plots(self.logger.data_table, t=self.runner.t_env)

    def _evaluate_multi_comms(
        self,
        msg_budget_per_agents: list[float],
        n_eval_eps: int,
        parallel_eval: bool = True,
    ) -> None:
        """
        Evaluate a trained policy across multiple comms allocation values.

        Parameters
        ----------
        msg_budget_per_agents : list[float]
            List of comms values to evaluate (e.g., [0.0, 0.5, 1.0])
        """
        self.logger.info(
            f"Evaluating Policy Across Comms Values: {msg_budget_per_agents}",
            log_header=True,
        )

        eval_data: list[dict] = []

        # Serial evaluation
        if not parallel_eval:
            for mb in msg_budget_per_agents:
                self.logger.info(f"Evaluating with msg_budget_per_agent = {mb}")

                result = run_eval_episodes(
                    args=self.args,
                    runner=self.runner,
                    n_eval_eps=n_eval_eps,
                    t_env=self.runner.t_env,
                    reset_options={"msg_budget_per_agent": mb},
                )

                eval_data.append(result["log_stats"])

        # Convert to DataFrame and log
        df_eval = pd.DataFrame.from_records(eval_data)
        self.logger.log_table(key="eval_stats", value=df_eval, t=self.runner.t_env)
        self._make_comms_eval_plots(
            self.logger.data_tables["eval_stats"], t=self.runner.t_env
        )

    def _make_comms_eval_plots(
        self,
        data_table: wandb.Table,
        t: int,
        wandb_run=None,
        info_str: str = "",
        plot_group: str = "",
    ) -> None:
        """Make plots for comms evaluation.

        Plots each metric in `cols` vs `t_env` for every comms value present
        (or provided in `msg_budget_per_agents`) and logs images to wandb if enabled.
        """
        df = data_table.get_dataframe()
        df["msg_budget_per_agent"] = pd.to_numeric(
            df["msg_budget_per_agent"], errors="coerce"
        )
        save_dir = abspath(join(self.logger.dir, "images", f"t_{t}"))
        if plot_group:
            save_dir = join(save_dir, plot_group)
        makedirs(save_dir, exist_ok=True)

        # Columns to plot (exclude t_env as it's the x axis)
        cols = [
            "test_return_mean",
            "test_return_std",
            "test_task_completed_mean",
            "test_ep_length_mean",
        ]
        msg_budget_per_agents = sorted(df["msg_budget_per_agent"].dropna().unique())

        for col in cols:
            plt.figure()

            for idx, msg_budget_per_agent in enumerate(msg_budget_per_agents):
                df_plot = df[
                    df.get("msg_budget_per_agent") == msg_budget_per_agent
                ].copy()
                label = f"Comms: {msg_budget_per_agent}"
                n_samples = df_plot["test_n_episodes"].astype(int)

                # show N in legend title using first row's n
                n_samples = int(n_samples.iloc[0]) if len(n_samples) > 0 else 0

                plt.plot(
                    df_plot["t_env"],
                    df_plot[col],
                    marker="o",
                    alpha=1.0,
                    label=label,
                )

                if col == "test_task_completed_mean":
                    plt.ylim(-0.05, 1.05)
                legend_title = f"samples={n_samples} (per seed)"
                if info_str != "":
                    legend_title += f"\n{info_str}"

            plt.xlabel("t_env")
            plt.ylabel(col)
            plt.title(f"{col}")

            plt.legend(title=legend_title)
            plt.grid(True)

            save_path = join(save_dir, f"comms_eval_{col}.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        # log all images in the image dir
        if wandb_run is not None:
            log_prefix = "comms_eval_aggregated"
            if plot_group:
                log_prefix = f"{log_prefix}/{plot_group}"
            for _, _, files in walk(save_dir):
                for file in files:
                    data = log_setup(self.logger.step_metric, t)
                    path = join(save_dir, file)
                    fn = splitext(file)[0]
                    data[f"{log_prefix}/{fn}{self.logger.log_suffix}"] = wandb.Image(
                        path
                    )
                    wandb_run.log(data=data)
            return

        self.logger.log_images(save_dir, t=self.runner.t_env, group="comms_eval/")

    def _make_combined_comms_eval_plots(
        self,
        groups: list[tuple[str, pd.DataFrame, object]],
        t: int,
        map_name: str = "",
        wandb_run=None,
    ) -> None:
        """Make one vertically stacked figure for each metric across tasks."""
        if not groups:
            return

        cols = [
            "test_return_mean",
            "test_return_std",
            "test_task_completed_mean",
            "test_ep_length_mean",
        ]
        save_dir = abspath(join(self.logger.dir, "images", f"t_{t}", "combined"))
        if map_name:
            save_dir = join(save_dir, map_name)
        makedirs(save_dir, exist_ok=True)

        for col in cols:
            figure, axes = plt.subplots(
                nrows=len(groups),
                ncols=1,
                figsize=(8, max(3.5 * len(groups), 4.0)),
                squeeze=False,
                sharex=True,
            )
            axes = axes[:, 0]

            for axis, (task_label, df, _) in zip(axes, groups):
                msg_budget_per_agents = sorted(
                    df["msg_budget_per_agent"].dropna().unique()
                )
                for msg_budget_per_agent in msg_budget_per_agents:
                    df_plot = df[df["msg_budget_per_agent"] == msg_budget_per_agent]
                    axis.plot(
                        df_plot["t_env"],
                        df_plot[col],
                        marker="o",
                        label=f"Comms: {msg_budget_per_agent}",
                    )

                axis.set_title(task_label, loc="left")
                axis.set_ylabel(col)
                axis.grid(True)
                if col == "test_task_completed_mean":
                    axis.set_ylim(-0.05, 1.05)
                axis.legend()

            axes[-1].set_xlabel("t_env")
            figure.suptitle(col)
            figure.tight_layout()
            save_path = join(save_dir, f"comms_eval_combined_{col}.png")
            figure.savefig(save_path)
            plt.close(figure)

            if wandb_run is not None:
                data = log_setup(self.logger.step_metric, t)
                data[
                    f"comms_eval_aggregated/combined/"
                    f"{map_name + '/' if map_name else ''}{col}"
                    f"{self.logger.log_suffix}"
                ] = wandb.Image(save_path)
                wandb_run.log(data=data)

    def _make_task_success_frame_plot(
        self,
        groups: list[tuple[str, pd.DataFrame, object]],
        source_runs: list,
        source_run,
        t: int,
        map_name: str = "",
        wandb_run=None,
    ) -> None:
        """Pair each task's initial frame with its success-rate curve."""
        if not groups:
            return

        frames = self._render_task_frames(groups, source_run)
        figure = plt.figure(
            figsize=(max(3.0 * len(groups) + 2.5, 13.0), 11),
        )
        grid = figure.add_gridspec(
            nrows=4,
            ncols=len(groups) + 2,
            height_ratios=[1.4, 1.0, 1.0, 1.0],
            width_ratios=[1] * len(groups) + [1.4, 0.8],
            hspace=0.45,
        )
        frame_axes = [
            figure.add_subplot(grid[0, column]) for column in range(len(groups))
        ]
        high_level_axis = figure.add_subplot(grid[0, -2])
        self._plot_high_level_env(high_level_axis, source_run, map_name)
        curve_axes = [figure.add_subplot(grid[1, 0])]
        curve_axes.extend(
            figure.add_subplot(grid[1, column], sharey=curve_axes[0])
            for column in range(1, len(groups))
        )
        for curve_axis in curve_axes[1:]:
            curve_axis.tick_params(axis="y", labelleft=False)
        training_axes = [figure.add_subplot(grid[2, 0])]
        training_axes.extend(
            figure.add_subplot(grid[2, column], sharey=training_axes[0])
            for column in range(1, len(groups))
        )
        for training_axis in training_axes[1:]:
            training_axis.tick_params(axis="y", labelleft=False)
        training_axes[0].set_ylabel("training return")
        legend_axis = figure.add_subplot(grid[1, -1])
        legend_axis.axis("off")
        bottom_width = min(2, len(groups))
        bottom_start = (len(groups) + 1 - bottom_width) // 2
        final_axis = figure.add_subplot(
            grid[3, bottom_start : bottom_start + bottom_width]
        )
        final_legend_axis = figure.add_subplot(grid[3, -1])
        final_legend_axis.axis("off")
        curve_handles = []
        curve_labels = []
        final_handles = []
        final_labels = []

        for column, ((task_label, df, hl_task), source_runs_by_budget) in enumerate(
            zip(groups, source_runs)
        ):
            frame_axis = frame_axes[column]
            curve_axis = curve_axes[column]
            frame = frames.get(
                tuple(hl_task) if isinstance(hl_task, (list, tuple)) else hl_task
            )
            if frame is not None:
                frame_axis.imshow(frame)
            else:
                frame_axis.text(
                    0.5,
                    0.5,
                    "frame unavailable",
                    ha="center",
                    va="center",
                )
            frame_axis.set_title(task_label, loc="left")
            frame_axis.axis("off")

            success_lines = {}
            msg_budgets = sorted(df["msg_budget_per_agent"].dropna().unique())
            for msg_budget_per_agent in msg_budgets:
                df_plot = df[df["msg_budget_per_agent"] == msg_budget_per_agent]
                (line,) = curve_axis.plot(
                    df_plot["t_env"],
                    df_plot["test_task_completed_mean"],
                    marker="o",
                    label=f"Comms: {msg_budget_per_agent}",
                )
                success_lines[float(msg_budget_per_agent)] = line
                if column == 0:
                    curve_handles.append(line)
                    curve_labels.append(f"Comms: {msg_budget_per_agent}")

            if column == 0:
                curve_axis.set_ylabel("success rate")
            curve_axis.set_ylim(-0.05, 1.05)
            curve_axis.set_xlabel("t_env")
            curve_axis.grid(True)

            training_axis = training_axes[column]
            for msg_budget_per_agent in msg_budgets:
                source_run = source_runs_by_budget.get(float(msg_budget_per_agent))
                if source_run is None:
                    continue
                training_df = self._load_training_returns(source_run)
                if training_df.empty:
                    continue
                training_axis.plot(
                    training_df["t_env"],
                    training_df["training_return"],
                    color=success_lines[float(msg_budget_per_agent)].get_color(),
                    linewidth=1.5,
                    label=f"Comms: {msg_budget_per_agent:g}",
                )
            training_axis.set_xlabel("t_env")
            training_axis.grid(True)

        final_step = max(df["t_env"].max() for _, df, _ in groups)
        for task_label, df, _ in groups:
            final_t = df["t_env"].max()
            final_df = df[df["t_env"] == final_t].sort_values("msg_budget_per_agent")
            (line,) = final_axis.plot(
                final_df["msg_budget_per_agent"],
                final_df["test_task_completed_mean"],
                marker="o",
                alpha=0.65,
                label=task_label,
            )
            final_handles.append(line)
            final_labels.append(task_label)
        final_axis.set_title(f"{final_step / 1_000_000:g}M steps")
        final_axis.set_xlabel("comms budget")
        final_axis.set_ylabel("success rate")
        final_axis.set_ylim(-0.05, 1.05)
        final_axis.grid(True)
        if final_handles:
            final_legend_axis.legend(
                final_handles,
                final_labels,
                loc="center",
                title="task",
                fontsize="small",
            )

        if curve_handles:
            legend_axis.legend(
                curve_handles,
                curve_labels,
                loc="center",
                title="communication budget",
            )
        figure.tight_layout()

        save_dir = abspath(join(self.logger.dir, "images", f"t_{t}", "combined"))
        if map_name:
            save_dir = join(save_dir, map_name)
        makedirs(save_dir, exist_ok=True)
        save_path = join(save_dir, "comms_eval_combined_task_frames.png")
        figure.savefig(save_path)
        plt.close(figure)

        if wandb_run is not None:
            data = log_setup(self.logger.step_metric, t)
            data[
                f"comms_eval_aggregated/combined/"
                f"{map_name + '/' if map_name else ''}task_frames"
                f"{self.logger.log_suffix}"
            ] = wandb.Image(save_path)
            wandb_run.log(data=data)

    @staticmethod
    def _map_name_from_config(config: dict) -> str:
        env_args = config.get("env_args", {}) or {}
        return str(config.get("env_args.map_name", env_args.get("map_name", "")))

    def _plot_high_level_env(self, axis, source_run, map_name: str) -> None:
        mdp = load_navigation_mdp(source_run.config, logger=self.logger)
        states = mdp.get("states", [])
        transitions = mdp.get("transitions", [])
        if not states or not transitions:
            axis.text(
                0.5, 0.5, "high-level graph unavailable", ha="center", va="center"
            )
            axis.axis("off")
            return

        levels = {states[0]: 0}
        for _ in states:
            for transition in transitions:
                start = transition["from_state"]
                end = transition["to_state"]
                if start in levels:
                    levels[end] = max(levels.get(end, 0), levels[start] + 1)
        positions = {}
        for level in sorted(set(levels.values())):
            level_states = [state for state in states if levels.get(state) == level]
            offset = (len(level_states) - 1) / 2
            positions.update(
                {
                    state: (level, offset - index)
                    for index, state in enumerate(level_states)
                }
            )

        for transition in transitions:
            start = positions.get(transition["from_state"])
            end = positions.get(transition["to_state"])
            if start is not None and end is not None:
                axis.add_patch(
                    FancyArrowPatch(
                        start,
                        end,
                        arrowstyle="->",
                        mutation_scale=12,
                        linewidth=1.2,
                        color="0.35",
                        connectionstyle="arc3,rad=0.08",
                    )
                )

        goal_states = set(mdp.get("goal_states", []))
        for state, position in positions.items():
            axis.scatter(
                *position,
                s=500,
                color="tab:green" if state in goal_states else "white",
                edgecolor="0.2",
                zorder=2,
            )
            axis.text(*position, state, ha="center", va="center", zorder=3)

        axis.set_title(f"high-level env: {map_name}", fontsize="small")
        axis.set_xlim(-0.5, max(levels.values()) + 0.5)
        axis.set_ylim(min(position[1] for position in positions.values()) - 0.7, 1.0)
        axis.axis("off")

    @staticmethod
    def _configured_message_budgets(value) -> list[float]:
        if isinstance(value, (list, tuple, set)):
            values = value
        elif isinstance(value, str):
            values = value.strip("[]").replace(",", " ").split()
        else:
            values = [value]
        try:
            return [float(item) for item in values]
        except (TypeError, ValueError):
            return []

    @staticmethod
    def _load_training_returns(source_run) -> pd.DataFrame:
        """Load the source run's training return history for plotting."""
        for metric_name in ("return_mean", "total_return_mean"):
            try:
                history = source_run.history(
                    keys=["t_env", metric_name], pandas=True, samples=10000
                )
            except (AttributeError, TypeError, ValueError):
                continue
            if history is None or history.empty or metric_name not in history:
                continue
            training_df = history[["t_env", metric_name]].dropna()
            return training_df.rename(columns={metric_name: "training_return"})
        return pd.DataFrame(columns=["t_env", "training_return"])

    def _render_task_frames(self, groups, source_run) -> dict:
        return render_navigation_task_frames(
            groups,
            getattr(source_run, "config", {}),
            logger=self.logger,
        )

    def evaluate_loaded(self) -> None:
        """probably doesn't work given new eval functions"""
        self.runner.log_train_stats_t = self.runner.t_env

        self.evaluate(n_eval_eps=self.n_eval_eps)

        self.runner.close_env()
        self.logger.log_stat("episode", self.runner.t_env, self.runner.t_env)
        self.logger.print_recent_stats()
        self.logger.info("Finished Evaluation")

    def save(self) -> None:
        model_dir = join(
            self.args.local_results_path,
            "models",
        )
        save_path = join(
            model_dir,
            self.args.unique_token,
            str(self.runner.t_env),
        )

        # "results/models/{}".format(unique_token)
        makedirs(save_path, exist_ok=True)
        self.logger.info("Saving models to {}".format(save_path))

        # learner should handle saving/loading -- delegate actor save/load to mac,
        # use appropriate filenames to do critics, optimizer states
        self.learner.save_models(save_path)

        if self.args.use_wandb:
            self.logger.log_agent(
                save_path=save_path,
                t=self.runner.t_env,
            )

        # models are saved locally and on the wandb server
        # as wandb artifacts and can be accessed with the wandb API
        if self.args.delete_local_models:
            rmtree(model_dir, ignore_errors=True)

    def _load_experiment_table_artifact(
        self,
        art_name: str = "eval_stats",
        art_version: str = "latest",
        art_type: str = "run_table",
        return_runs: bool = False,
    ):
        api = wandb.Api()

        # load all runs w/ the given time_id and get eval stats tables
        runs = api.runs(
            self.args.wandb_project,
            filters={"config.time_id": self.args.time_id},
        )
        runs = [
            run
            for run in runs
            if "_postprocess" not in (getattr(run, "name", "") or "")
        ]

        if len(runs) == 0:
            self.logger.info(f"No wandb runs found for time_id={self.args.time_id}")
            return

        run_ids = [run.id for run in runs]
        self.logger.info(f"Time ID: {self.args.time_id}")
        self.logger.info(f"Loading {len(run_ids)} runs with ids: {run_ids}")

        dfs = []

        # parallel download
        def get_wandb_data(wandb_run):
            try:
                art_info = f"run-{wandb_run.id}-{art_name}:{art_version}"
                artifact = api.artifact(
                    name=join(wandb_run.entity, wandb_run.project, art_info),
                    type=art_type,
                )
                data = artifact.get(art_name)
                df = data.get_dataframe()
                df["scenario"] = int(wandb_run.config["scenario"])
                configured_budget = wandb_run.config.get("msg_budget_per_agent")
                if (
                    isinstance(configured_budget, (list, tuple))
                    and len(configured_budget) == 1
                ):
                    configured_budget = configured_budget[0]
                if configured_budget is not None:
                    if wandb_run.config.get("unique_policy_per_msg_budget", False):
                        # For one-policy-per-budget runs, the run config is the
                        # authoritative budget even if the table has stale values.
                        df["msg_budget_per_agent"] = configured_budget
                    elif "msg_budget_per_agent" not in df:
                        df["msg_budget_per_agent"] = configured_budget
                    else:
                        df["msg_budget_per_agent"] = df["msg_budget_per_agent"].fillna(
                            configured_budget
                        )
                df["test_interval"] = wandb_run.config.get(
                    "test_interval", self.args.test_interval
                )
                return df

            except wandb.CommError:
                # artifact is missing or deleted, run may have died early
                return

        with ThreadPoolExecutor() as ex:
            dfs = ex.map(
                get_wandb_data,
                [run for run in runs],
            )

        df_data = pd.concat(dfs, ignore_index=True)
        # round to the
        # nearest eval time since different seeds eval at slightly different times
        df_data["t_env_rounded"] = (
            df_data["t_env"] / df_data["test_interval"]
        ).round() * df_data["test_interval"]
        df_data.drop(columns=["test_interval"], inplace=True)

        df_data.sort_values("scenario").reset_index(drop=True, inplace=True)

        if return_runs:
            return df_data, runs
        else:
            return df_data

    def _load_checkpoint(self) -> None:
        # get load time step for both cases
        timesteps = []
        timestep_to_load = 0

        if self.args.eval_run_id is not None:
            artifacts = self.logger.wandb_inactive.logged_artifacts()

            # go thru metadata and get all time steps saved
            agent_artifacts = []
            for artifact in artifacts:
                # Check if this artifact is a model and has the correct step in its metadata
                if artifact.type == "agent":
                    agent_artifacts.append(artifact)
                    timesteps.append(artifact.metadata["t_env"])

        else:
            if not isdir(self.args.checkpoint_path):
                self.logger.info(
                    "Checkpoint directiory {} doesn't exist".format(
                        self.args.checkpoint_path
                    )
                )
                return

            # Go through all files in args.checkpoint_path
            for name in listdir(self.args.checkpoint_path):
                full_name = join(self.args.checkpoint_path, name)
                # Check if they are dirs the names of which are numbers
                if isdir(full_name) and name.isdigit():
                    timesteps.append(int(name))

        if self.args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(
                timesteps, key=lambda x: abs(x - self.args.load_step)
            )

        if self.args.eval_run_id is not None:
            for artifact in agent_artifacts:
                if artifact.metadata["t_env"] == timestep_to_load:
                    model_path = artifact.download()
        else:
            model_path = join(self.args.checkpoint_path, str(timestep_to_load))

        self.logger.info(f"Loading model from t={timestep_to_load} ({model_path})")
        self.learner.load_models(model_path)
        self.runner.t_env = timestep_to_load

        if self.args.eval_run_id is not None:
            # clean up local files that have been loaded into memory
            rmtree("artifacts", ignore_errors=True)

    def _parse_config(self, _config, _log) -> SN:
        _config = self._args_sanity_check(_config, _log)

        args = SN(**_config)
        args.device = "cuda" if args.use_cuda else "cpu"
        assert test_alg_config_supports_reward(args), (
            "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."
        )

        # update for parallel comms eval, can't be done offline
        # due to parallel to a single wandb run on a remote server
        if args.parallel_comms_eval:
            args.wandb_mode = "shared"

        return args

    def _args_sanity_check(self, config, _log):
        # set CUDA flags
        # config["use_cuda"] = True # Use cuda whenever possible!
        if config["use_cuda"] and not th.cuda.is_available():
            config["use_cuda"] = False
            _log.warning(
                "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
            )

        if config["test_nepisode"] < config["batch_size_run"]:
            config["test_nepisode"] = config["batch_size_run"]
        else:
            config["test_nepisode"] = (
                config["test_nepisode"] // config["batch_size_run"]
            ) * config["batch_size_run"]

        return config

    def _build_logger(self, args: SN, _config, _log) -> MainLogger:
        # get unique token for this run
        if hasattr(_config["env_args"], "map_name"):
            map_name = _config["env_args"]["map_name"]
        else:
            map_name = _config["env_args"]["key"]

        # run_name has a unique datetime in it, so only include curr_time if that is not available
        curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[2:][:-3]
        unique_token = (
            f"{args.run_name if args.run_name != '' else curr_time}_"
            f"{args.env}_{map_name + '_' if map_name != args.env else ''}"
            f"{args.name}_seed_{args.seed}"
        )

        args.unique_token = unique_token

        # logger setup
        return MainLogger(_log, config=_config, args=args)

    def finish(self) -> None:
        # Finish logging
        self.logger.finish()

        # Clean up after finishing
        print("Exiting Main")

        print("Stopping all threads")
        for t in threading.enumerate():
            if t.name != "MainThread":
                print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
                t.join(timeout=1)
                print("Thread joined")

        print("Exiting script")

    # def train_hierarchical(self) -> None:
    #     """
    #     Two-stage training: first train low-level policy, then use its success rates
    #     in the high-level agent that interfaces with the HLMDP.
    #     """

    #     # Train low-level policy for a single task
    #     self.logger.info("Training Low-Level Policy", log_header=True)

    #     self.train_single_task()

    #     # may need to eval here for more samples than during training to get good statistical estimates of init state dists
    #     # only really needed for the dependent tasks

    #     # # Train high-level policy with learned success rates
    #     self.logger.info("Optimizing High-Level Policy", log_header=True)
    #     self._train_high_level_policy(self.runner.env.hlmdp)

    #     # evaluate Hl policy (only relevant for dependent tasks)

    #     self.runner.close_env()
    #     self.logger.info("Finished Hierarchical Training")
