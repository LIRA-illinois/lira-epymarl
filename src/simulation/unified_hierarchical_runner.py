"""Utilities for unified mock and real hierarchical seed execution."""

from __future__ import annotations

import json
from copy import deepcopy
from os import makedirs
from os.path import join
from types import SimpleNamespace as SN

import numpy as np
import pandas as pd
import torch as th
import yaml

from src.simulation.build import build_sim
from src.simulation.context_utils import (
    plot_context_evaluation,
    plot_mixed_initial_distributions,
    plot_pg_iteration_verification,
    plot_success_vs_comms_by_subtask,
    plot_success_vs_cost_frontier,
)
from src.simulation.evaluate import (
    collect_statewise_calibration_row,
    record_calibration_replays,
)
from src.simulation.hierarchical_calibration import (
    build_calibration_report_row,
    calculate_m_est,
    enumerate_statewise_settings,
    greedy_project_baselines,
    propagate_initial_state_distributions,
    read_calibration_manifest,
    save_calibration_manifest,
)
from src.simulation.hierarchical_evaluation import summarize_held_out_results
from src.simulation.hierarchical_synthesis import (
    iterative_high_level_synthesis,
)
from src.simulation.hierarchical_synthesis import (
    solve_mixed_project_policy as _solve_mixed_project_policy,
)
from src.simulation.hierarchical_types import (
    CalibrationRow,
    ContextKey,
    EvaluationRowKey,
    HeldOutResult,
    LowLevelPolicyArtifact,
    OptionSettingKey,
    StateDistribution,
)
from src.simulation.policy_artifacts import (
    read_option_policy_manifest,
    save_option_policy_weights_artifact,
)


def _navigation_reset_options(
    args,
    edge: tuple[int, int],
    capacity: float,
) -> dict:
    """Convert an abstract option setting into team-navigation reset options."""
    env_args = getattr(args, "env_args", {})
    if not isinstance(env_args, dict):
        env_args = {}
    navigation_config_path = env_args.get("navigation_config")
    if not navigation_config_path:
        raise ValueError(
            "Real unified runner requires env_args.navigation_config "
            "for team-navigation joint-state distributions."
        )
    with open(navigation_config_path, "r", encoding="utf-8") as config_file:
        navigation_config = yaml.safe_load(config_file) or {}
    tasks = navigation_config.get("tasks", [])
    for task in tasks:
        if (
            int(task.get("from_state", -1)) == edge[0]
            and int(task.get("to_state", -1)) == edge[1]
        ):
            init_state_dist = task.get("init_state_dist")
            if isinstance(init_state_dist, dict) and init_state_dist.get("states"):
                states = init_state_dist["states"]
                probs = init_state_dist.get("probs")
                if probs is None:
                    probs = [1.0 / len(states)] * len(states)
                if len(states) != len(probs):
                    raise ValueError(
                        f"Navigation init-state distribution for {edge} "
                        "must align states and probabilities."
                    )
                if not np.isclose(sum(float(prob) for prob in probs), 1.0):
                    raise ValueError(
                        f"Navigation init-state probabilities for {edge} "
                        "must sum to 1.0."
                    )
                normalized_states = [
                    [[int(position[0]), int(position[1])] for position in joint_state]
                    for joint_state in states
                ]
                return {
                    "hl_start_state": edge[0],
                    "hl_task": list(edge),
                    "navigation_task_transition": list(edge),
                    "navigation_init_state_dist": {
                        "states": normalized_states,
                        "probs": [float(prob) for prob in probs],
                    },
                    "msg_budget_per_agent": round(float(capacity)),
                }
            predecessor_goals = []
            for predecessor in tasks:
                if int(predecessor.get("to_state", -1)) == edge[0]:
                    goal_positions = predecessor.get("goal_positions")
                    if goal_positions:
                        predecessor_goals = [
                            [int(position[0]), int(position[1])]
                            for position in goal_positions
                        ]
                        break
            if predecessor_goals:
                return {
                    "hl_start_state": edge[0],
                    "hl_task": list(edge),
                    "navigation_task_transition": list(edge),
                    "navigation_init_state_dist": {
                        "states": [predecessor_goals],
                        "probs": [1.0],
                    },
                    "msg_budget_per_agent": round(float(capacity)),
                }
    raise ValueError(
        f"No navigation init-state distribution is configured for edge {edge}."
    )


def _flatten_navigation_states(reset_options: dict) -> tuple[tuple, ...]:
    """Flatten navigation joint-position states into immutable state vectors."""
    dist = reset_options.get("navigation_init_state_dist", {})
    states = dist.get("states", ())
    flattened = []
    for joint_state in states:
        values = np.asarray(joint_state).reshape(-1).tolist()
        flattened.append(tuple(int(value) for value in values))
    unique = tuple(dict.fromkeys(flattened))
    if not unique:
        raise ValueError(
            "Navigation reset options must contain at least one joint-state."
        )
    return unique


def _candidate_mixed_distribution_rows(
    *,
    seed: int,
    pg: float,
    context_id: str,
    candidates,
) -> list[dict]:
    """Flatten candidate-induced state distributions into plotting/data rows."""
    rows = []
    for candidate in candidates:
        induced = getattr(candidate, "induced_initial_state_distributions", {}) or {}
        for state, distribution in sorted(induced.items(), key=lambda item: str(item[0])):
            if distribution is None:
                continue
            for start_state, probability in zip(
                distribution.states,
                distribution.probabilities,
                strict=True,
            ):
                rows.append(
                    {
                        "seed": seed,
                        "pg": pg,
                        "context_id": context_id,
                        "iteration": candidate.iteration,
                        "status": candidate.status.value,
                        "state": int(state),
                        "start_state": repr(tuple(start_state)),
                        "probability": float(probability),
                    }
                )
    return rows


def _log_wandb_table(logger, key: str, value: pd.DataFrame, t_env: int) -> None:
    """Log a seed-level dataframe through the existing LocalLogger interface."""
    if hasattr(logger, "log_table"):
        logger.log_table(
            key=key,
            value=value,
            t=int(t_env),
        )


def _log_wandb_plot_paths(
    logger,
    paths: list[str],
    t_env: int,
    key_prefix: str,
) -> None:
    """Upload generated PNG plot paths to W&B without moving local artifacts."""
    if not hasattr(logger, "log_image"):
        return
    for path in paths:
        if not str(path).lower().endswith(".png"):
            continue
        filename = path.rsplit("/", 1)[-1]
        logger.log_image(
            path,
            t=int(t_env),
            key=f"{key_prefix}/{filename.rsplit('.', 1)[0]}",
        )


def _record_selected_calibration_replays(
    runner,
    *,
    setting_rows: list[CalibrationRow],
    reset_option_builder,
    max_settings: int = 6,
    n_replays: int = 2,
) -> list[OptionSettingKey]:
    """Record replays for a small exact-setting subset after calibration."""
    if n_replays <= 0 or not hasattr(runner, "start_recording"):
        return []
    args = getattr(runner, "args", None)
    if args is not None:
        args.save_test_replays = True
        args.n_test_replays_save = int(n_replays)

    selected = []
    seen_edges = set()
    for row in setting_rows:
        setting = row.setting
        if setting.subtask_edge in seen_edges:
            continue
        seen_edges.add(setting.subtask_edge)
        selected.append(setting)
        if len(selected) >= max_settings:
            break

    for setting in selected:
        reset_options = reset_option_builder(setting)
        record_calibration_replays(
            runner,
            setting=setting,
            reset_options=reset_options,
            t_env=getattr(runner, "t_env", None),
        )
    return selected


def _sample_mixed_navigation_reset_options(
    args,
    task_edges: tuple[tuple[int, int], ...],
    capacities: tuple[float, ...] | list[float],
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Sample a task, capacity, and initial joint state for each worker."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if not task_edges:
        raise ValueError("task_edges must contain at least one edge.")
    if not capacities:
        raise ValueError("capacities must contain at least one capacity.")

    sampled_options = []
    for _ in range(batch_size):
        edge = task_edges[int(rng.integers(len(task_edges)))]
        capacity = float(rng.choice(np.asarray(capacities, dtype=np.float64)))
        reset_options = _navigation_reset_options(args, edge, capacity)

        init_dist = reset_options["navigation_init_state_dist"]
        states = init_dist["states"]
        probabilities = np.asarray(init_dist["probs"], dtype=np.float64)
        if len(states) != len(probabilities) or len(states) == 0:
            raise ValueError(
                f"Navigation init-state distribution for {edge} is invalid."
            )
        probabilities = probabilities / probabilities.sum()
        state_index = int(rng.choice(len(states), p=probabilities))
        chosen_state = deepcopy(states[state_index])

        reset_options["navigation_init_state_dist"] = {
            "states": [chosen_state],
            "probs": [1.0],
        }
        reset_options["msg_budget_per_agent"] = round(float(capacity))
        sampled_options.append(reset_options)
    return sampled_options


def _policy_choices(
    policy,
    state: int,
    task_edges: tuple[tuple[int, int], ...],
    capacities: tuple[float, ...],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], float]:
    """Sample one edge and communication capacity from a candidate policy."""
    task_rows = []
    if policy.task_policy is not None:
        task_rows = policy.task_policy.loc[
            policy.task_policy["state"] == state
        ].to_dict(orient="records")
    outgoing = [
        edge for edge in task_edges if edge[0] == state
    ]
    if task_rows:
        choices = []
        probabilities = []
        for edge in outgoing:
            probability = sum(
                float(row.get("action_probability", 0.0))
                for row in task_rows
                if int(row["next_state"]) == edge[1]
            )
            choices.append(edge)
            probabilities.append(max(0.0, probability))
        total = sum(probabilities)
        if total > 0.0:
            probabilities = [probability / total for probability in probabilities]
            edge = outgoing[
                int(rng.choice(len(outgoing), p=probabilities))
            ]
        else:
            edge = outgoing[0] if outgoing else task_edges[0]
    else:
        edge = outgoing[0] if outgoing else task_edges[0]

    capacity = capacities[-1]
    if policy.comms_budget_policy is not None:
        capacity_rows = policy.comms_budget_policy.loc[
            (policy.comms_budget_policy["state"] == edge[0])
            & (policy.comms_budget_policy["next_state"] == edge[1])
        ].to_dict(orient="records")
        if capacity_rows:
            choices = [
                float(row["msg_budget_per_agent"]) for row in capacity_rows
            ]
            probabilities = [
                max(0.0, float(row.get("action_probability", 0.0)))
                for row in capacity_rows
            ]
            total = sum(probabilities)
            if total > 0.0:
                probabilities = [probability / total for probability in probabilities]
                capacity = choices[
                    int(rng.choice(len(choices), p=probabilities))
                ]
    return edge, float(capacity)


def _evaluate_mock_candidate_held_out(
    *,
    seed: int,
    pg: float,
    candidate,
    context_id: str = "mock-hierarchical",
    task_edges: tuple[tuple[int, int], ...],
    capacities: tuple[float, ...],
    calibration_rows: tuple[CalibrationRow, ...],
    root_state: int,
    goal_states: tuple[int, ...],
    n_episodes: int,
) -> tuple[list[HeldOutResult], list[dict]]:
    """Produce exact-key mock held-out episodes for one candidate."""
    rng = np.random.default_rng(
        seed * 1_000_000 + round(pg * 1_000)
    )
    context_id = f"{context_id}-pg-{pg:.2f}"
    episode_results = []
    summary_rows = []

    for episode_index in range(n_episodes):
        state = root_state
        visited = []
        for _ in range(len(task_edges) + 1):
            if state in goal_states:
                break
            outgoing = [edge for edge in task_edges if edge[0] == state]
            if not outgoing:
                break
            edge, capacity = _policy_choices(
                candidate,
                state,
                tuple(outgoing),
                capacities,
                rng,
            )
            start_candidates = tuple(
                dict.fromkeys(
                    calibration_row.setting.start_state
                    for calibration_row in calibration_rows
                    if calibration_row.setting.subtask_edge == edge
                    and np.isclose(
                        calibration_row.setting.communication_capacity,
                        capacity,
                    )
                )
            )
            if start_candidates:
                start_state = start_candidates[
                    int(rng.integers(len(start_candidates)))
                ]
            else:
                start_state = (state, 0)
            row = next(
                (
                    calibration_row
                    for calibration_row in calibration_rows
                    if calibration_row.setting.subtask_edge == edge
                    and np.isclose(
                        calibration_row.setting.communication_capacity,
                        capacity,
                    )
                    and calibration_row.setting.start_state == start_state
                ),
                None,
            )
            if row is None:
                project_success = False
                visited.append(
                    (edge, capacity, start_state, False, None)
                )
                break
            succeeded = bool(rng.random() < row.success_rate)
            terminal_state = None
            if succeeded:
                terminal_distribution = row.terminal_state_distribution
                if terminal_distribution.states:
                    terminal_index = int(
                        rng.choice(
                            len(terminal_distribution.states),
                            p=list(terminal_distribution.probabilities),
                        )
                    )
                    terminal_state = terminal_distribution.states[terminal_index]
                    state = edge[1]
            visited.append(
                (edge, capacity, start_state, succeeded, terminal_state)
            )
            if not succeeded:
                break

        project_success = bool(state in goal_states)
        episode_results_for_project = []
        for edge, capacity, start_state, edge_success, terminal_state in visited:
            row_key = EvaluationRowKey(
                low_level_seed=seed,
                high_level_seed=seed,
                context_id=context_id,
                subtask_edge=edge,
                communication_capacity=capacity,
                start_state=start_state,
                episode_index=episode_index,
            )
            result = HeldOutResult(
                row_key=row_key,
                project_success=project_success,
                communication_cost=capacity,
                successful_terminal_state=terminal_state if edge_success else None,
            )
            episode_results_for_project.append(result)
            episode_results.append(result)

        if not visited:
            continue

    exact_summaries = {}
    for result in episode_results:
        key = (
            result.row_key.low_level_seed,
            result.row_key.high_level_seed,
            result.row_key.context_id,
            result.row_key.subtask_edge,
            result.row_key.communication_capacity,
            result.row_key.start_state,
        )
        exact_summaries.setdefault(key, []).append(result)
    for key, exact_results in exact_summaries.items():
        summary = summarize_held_out_results(tuple(exact_results))
        summary_rows.append(
            {
                "seed": summary.row_key.low_level_seed,
                "pg": pg,
                "context_id": summary.row_key.context_id,
                "edge": list(summary.row_key.subtask_edge),
                "communication_capacity": summary.row_key.communication_capacity,
                "start_state": list(summary.row_key.start_state),
                "n_episodes": summary.n_episodes,
                "project_success_probability": (
                    summary.project_success_probability
                ),
                "mean_communication_cost": summary.mean_communication_cost,
                "successful_terminal_states": (
                    [list(state) for state in (
                        summary.successful_terminal_state_distribution.states
                    )]
                ),
            }
        )
    return episode_results, summary_rows


def _evaluate_real_candidate_held_out(
    runner_self,
    *,
    seed: int,
    pg: float,
    candidate,
    context_id: str = "unified",
    task_edges: tuple[tuple[int, int], ...],
    capacities: tuple[float, ...],
    calibration_rows: tuple[CalibrationRow, ...],
    policy_bank: dict,
    root_state: int,
    goal_states: tuple[int, ...],
    n_episodes: int,
) -> tuple[list[HeldOutResult], list[dict]]:
    """Produce exact-key held-out rows for a real candidate where possible."""
    rng = np.random.default_rng(
        seed * 1_000_000 + round(pg * 1_000)
    )
    context_id = f"{context_id}-pg-{pg:.2f}"
    episode_results = []
    summary_rows = []

    held_out_by_setting: dict[OptionSettingKey, CalibrationRow] = {}
    real_runner = getattr(
        runner_self,
        "_shared_candidate_runner",
        runner_self.runner,
    )
    can_collect = callable(getattr(real_runner, "run", None))
    mac = getattr(real_runner, "mac", None)

    for episode_index in range(n_episodes):
        state = root_state
        visited = []
        for _ in range(len(task_edges) + 1):
            if state in goal_states:
                break
            edge, capacity = _policy_choices(
                candidate,
                state,
                task_edges,
                capacities,
                rng,
            )
            edge_reset_options = _navigation_reset_options(
                runner_self.args,
                edge,
                capacity,
            )
            start_state = _flatten_navigation_states(
                edge_reset_options
            )[0]
            setting = OptionSettingKey(
                subtask_edge=edge,
                communication_capacity=capacity,
                start_state=start_state,
            )
            row = held_out_by_setting.get(setting)
            if can_collect:
                weights = policy_bank.get(
                    (
                        edge[0],
                        edge[1],
                        float(capacity),
                        start_state,
                        seed,
                    )
                )
                if weights is not None and mac is not None:
                    agent = getattr(mac, "agent", None)
                    if agent is not None and hasattr(agent, "load_state_dict"):
                        try:
                            agent.load_state_dict(weights)
                        except (RuntimeError, TypeError, AttributeError):
                            pass
                    comms_agent = getattr(mac, "comms_agent", None)
                    if comms_agent is not None and hasattr(
                        comms_agent,
                        "load_state_dict",
                    ):
                        try:
                            comms_agent.load_state_dict(weights)
                        except (RuntimeError, TypeError, AttributeError):
                            pass
                row = collect_statewise_calibration_row(
                    real_runner,
                    setting=setting,
                    seed=seed,
                    n_eval_eps=1,
                    reset_options=_navigation_reset_options(
                        runner_self.args,
                        edge,
                        capacity,
                    ),
                    record_replays=False,
                )
            elif row is None:
                existing = next(
                    (
                        calibration_row
                        for calibration_row in calibration_rows
                        if calibration_row.setting == setting
                    ),
                    None,
                )
                row = existing
            if row is None or not row.outcomes:
                break
            outcome = row.outcomes[0]
            succeeded = outcome.kind.value == "success"
            terminal_state = outcome.terminal_state
            if succeeded and terminal_state is not None:
                state = edge[1]
            visited.append(
                (edge, capacity, start_state, succeeded, terminal_state)
            )
            if not succeeded:
                break

        project_success = bool(state in goal_states)
        for edge, capacity, start_state, succeeded, terminal_state in visited:
            row_key = EvaluationRowKey(
                low_level_seed=seed,
                high_level_seed=seed,
                context_id=context_id,
                subtask_edge=edge,
                communication_capacity=capacity,
                start_state=start_state,
                episode_index=episode_index,
            )
            result = HeldOutResult(
                row_key=row_key,
                project_success=project_success,
                communication_cost=capacity,
                successful_terminal_state=(
                    terminal_state if succeeded else None
                ),
            )
            episode_results.append(result)

    exact_summaries = {}
    for result in episode_results:
        key = (
            result.row_key.low_level_seed,
            result.row_key.high_level_seed,
            result.row_key.context_id,
            result.row_key.subtask_edge,
            result.row_key.communication_capacity,
            result.row_key.start_state,
        )
        exact_summaries.setdefault(key, []).append(result)
    for exact_results in exact_summaries.values():
        summary = summarize_held_out_results(tuple(exact_results))
        summary_rows.append(
            {
                "seed": summary.row_key.low_level_seed,
                "pg": pg,
                "context_id": summary.row_key.context_id,
                "edge": list(summary.row_key.subtask_edge),
                "communication_capacity": summary.row_key.communication_capacity,
                "start_state": list(summary.row_key.start_state),
                "n_episodes": summary.n_episodes,
                "project_success_probability": (
                    summary.project_success_probability
                ),
                "mean_communication_cost": summary.mean_communication_cost,
                "successful_terminal_states": (
                    [
                        list(state)
                        for state in (
                            summary.successful_terminal_state_distribution.states
                        )
                    ]
                ),
            }
        )
    return episode_results, summary_rows


def run_mock_seed(
    simulation,
    hierarchical_config,
    artifact_root: str,
) -> dict:
    """Run one seed through the mock environment and calibration pipeline."""
    makedirs(artifact_root, exist_ok=True)
    task_edges = hierarchical_config.task_sequence
    goal_states = (task_edges[-1][1],)
    root_state = 0
    capacities = [
        float(capacity)
        for capacity in getattr(
            simulation.args,
            "msg_budget_per_agent",
            [0.0, 1.0],
        )
    ]
    if not capacities:
        raise ValueError("Mock unified runner requires communication capacities.")
    env_args = getattr(simulation.args, "env_args", {})
    if not isinstance(env_args, dict):
        env_args = {}
    navigation_config_path = env_args.get("navigation_config")
    if navigation_config_path:
        root_reset_options = _navigation_reset_options(
            simulation.args,
            task_edges[0],
            capacities[0],
        )
        root_support = _flatten_navigation_states(root_reset_options)
        root_initial_distribution = StateDistribution(
            root_support,
            tuple(1.0 / len(root_support) for _ in root_support),
        )
    else:
        root_initial_distribution = StateDistribution.singleton((0, 0))

    policy_seeds = [int(simulation.args.seed)]
    n_calibration = hierarchical_config.n_calibration_episodes or getattr(
        simulation.args, "test_nepisode", 20
    )
    env_args = getattr(simulation.args, "env_args", {})
    if not isinstance(env_args, dict):
        env_args = {}
    n_agents = int(env_args.get("n_agents", getattr(simulation.args, "n_agents", 3)))
    n_final_positions = int(
        env_args.get(
            "n_final_positions",
            getattr(simulation.args, "n_final_positions", 4),
        )
    )
    n_joint_states = max(1, n_agents * n_final_positions)
    if navigation_config_path:
        mock_statewise_setting_count = sum(
            len(
                _flatten_navigation_states(
                    _navigation_reset_options(
                        simulation.args,
                        edge,
                        capacities[0],
                    )
                )
            )
            * len(capacities)
            for edge in task_edges
        )
    else:
        mock_statewise_setting_count = (
            len(task_edges) * len(capacities) * n_joint_states
        )
    mock_m_est = calculate_m_est(
        epsilon_est=hierarchical_config.epsilon_est,
        beta=hierarchical_config.beta,
        outcome_alphabet_size=n_joint_states + 1,
        statewise_setting_count=mock_statewise_setting_count,
    )
    mock_calibration_is_certifiable = n_calibration >= mock_m_est

    calibration_rows = []
    eval_rows = []
    for policy_seed in policy_seeds:
        real_iteration_rows = []
        mixed_distribution_rows = []
        for edge in task_edges:
            for capacity in capacities:
                reset_options = (
                    _navigation_reset_options(
                        simulation.args,
                        edge,
                        capacity,
                    )
                    if navigation_config_path
                    else {
                        "hl_start_state": edge[0],
                        "hl_task": list(edge),
                        "msg_budget_per_agent": capacity,
                    }
                )
                start_states = (
                    _flatten_navigation_states(reset_options)
                    if navigation_config_path
                    else ((edge[0], 0),)
                )
                for start_state in start_states:
                    setting = OptionSettingKey(
                        subtask_edge=edge,
                        communication_capacity=capacity,
                        start_state=start_state,
                    )
                    row = collect_statewise_calibration_row(
                        simulation.runner,
                        setting=setting,
                        seed=policy_seed,
                        n_eval_eps=n_calibration,
                        reset_options=reset_options,
                        record_replays=False,
                    )
                    calibration_rows.append(row)
                    initial_distribution = StateDistribution.singleton(
                        start_state
                    )
                    mock_artifact = LowLevelPolicyArtifact(
                        seed=policy_seed,
                        subtask_edge=edge,
                        communication_capacity=capacity,
                        initial_state_distribution=initial_distribution,
                        timestep=int(getattr(simulation.runner, "t_env", 0)),
                        checkpoint_path=join(artifact_root, "unused"),
                        config_hash="mock-unified-seed-v1",
                    )
                    save_option_policy_weights_artifact(
                        artifact_root,
                        mock_artifact,
                        {
                            "mock.success_rate": np.asarray(
                                [row.success_rate],
                                dtype=np.float32,
                            )
                        },
                    )
                    eval_rows.append(
                        {
                            "seed": policy_seed,
                            "hl_task": list(edge),
                            "hl_start_state": edge[0],
                            "start_state": list(start_state),
                            "msg_budget_per_agent": capacity,
                            "test_task_completed_mean": row.success_rate,
                            "test_n_episodes": len(row.outcomes),
                            "test_return_mean": 0.0,
                            "test_return_std": 0.0,
                            "test_ep_length_mean": 1.0,
                            "t_env": simulation.runner.t_env,
                        }
                    )

        save_calibration_manifest(
            artifact_root,
            policy_seed,
            [row for row in calibration_rows if row.seed == policy_seed],
            metadata={
                "epsilon_est": hierarchical_config.epsilon_est,
                "beta": hierarchical_config.beta,
                "m_est": mock_m_est,
                "n_calibration_episodes": n_calibration,
                "statewise_setting_count": mock_statewise_setting_count,
                "certificate_applicable": mock_calibration_is_certifiable,
            },
        )
        loaded_rows = read_calibration_manifest(artifact_root, policy_seed)
        if not loaded_rows:
            raise RuntimeError(
                f"Mock calibration reload failed for seed {policy_seed}."
            )
        mock_report_rows = [
            build_calibration_report_row(
                row,
                epsilon_est=hierarchical_config.epsilon_est,
                beta=hierarchical_config.beta,
                m_est=mock_m_est,
                statewise_setting_count=mock_statewise_setting_count,
            )
            for row in loaded_rows
        ]
        pd.DataFrame.from_records(mock_report_rows).to_csv(
            join(
                artifact_root,
                "option_calibration",
                f"seed_{policy_seed}",
                "calibration_report.csv",
            ),
            index=False,
        )
        loaded_option_artifacts = read_option_policy_manifest(
            artifact_root,
            policy_seed,
        )
        expected_mock_artifacts = sum(
            1
            for row in calibration_rows
            if row.seed == policy_seed
        )
        if len(loaded_option_artifacts) != expected_mock_artifacts:
            raise RuntimeError(
                "Mock option-policy manifest reload did not return every "
                "(edge, capacity, start-state) setting."
            )

        plot_dir = join(artifact_root, "comms_eval")
        seed_plot_rows = [row for row in eval_rows if row["seed"] == policy_seed]
        seed_plot_df = pd.DataFrame.from_records(seed_plot_rows)
        mock_training_rows = []
        checkpoints = (
            hierarchical_config.mock_evaluation_times
            if len(hierarchical_config.mock_evaluation_times) >= 5
            else (20_000, 40_000, 60_000, 80_000, 100_000)
        )
        for edge in task_edges:
            edge_index = task_edges.index(edge)
            for capacity_index, capacity in enumerate(capacities):
                if navigation_config_path:
                    reset_options = _navigation_reset_options(
                        simulation.args,
                        edge,
                        capacity,
                    )
                    start_states = _flatten_navigation_states(reset_options)
                else:
                    start_states = ((edge[0], 0),)
                for start_index, start_state in enumerate(start_states):
                    for checkpoint_index, t_env in enumerate(checkpoints):
                        progress = checkpoint_index / max(1, len(checkpoints) - 1)
                        base_success = (
                            0.42
                            + 0.10 * (edge_index / max(1, len(task_edges) - 1))
                            + 0.18 * (capacity_index / max(1, len(capacities) - 1))
                            - 0.02 * (
                                start_index
                                / max(1, len(start_states) - 1)
                            )
                        )
                        rng = np.random.default_rng(
                            policy_seed * 100_000
                            + edge_index * 10_000
                            + capacity_index * 1_000
                            + start_index * 100
                            + checkpoint_index
                        )
                        learning_success = (
                            base_success
                            + progress * (0.96 - base_success)
                            + float(rng.normal(0.0, 0.015))
                        )
                        learning_success = float(
                            np.clip(learning_success, 0.05, 0.98)
                        )
                        mock_training_rows.append(
                            {
                                "seed": policy_seed,
                                "hl_task": list(edge),
                                "hl_start_state": edge[0],
                                "start_state": list(start_state),
                                "msg_budget_per_agent": capacity,
                                "test_task_completed_mean": learning_success,
                                "test_n_episodes": 20,
                                "test_return_mean": learning_success,
                                "test_return_std": max(
                                    0.005,
                                    float(rng.normal(0.02, 0.005)),
                                ),
                                "test_ep_length_mean": 1.0,
                                "t_env": int(t_env),
                            }
                        )

        training_df = pd.DataFrame.from_records(mock_training_rows)
        training_df.to_csv(
            join(artifact_root, f"mock_training_eval_seed_{policy_seed}.csv"),
            index=False,
        )
        training_plot_dir = join(plot_dir, "training")
        training_plot_paths = plot_context_evaluation(
            training_df,
            training_plot_dir,
            info_str=(
                f"seed={policy_seed}; {len(checkpoints)} mock evaluation checkpoints"
            ),
        )
        comms_plot_paths = plot_success_vs_comms_by_subtask(
            training_df,
            join(artifact_root, "comms_eval"),
        )
        _log_wandb_plot_paths(
            simulation.logger,
            training_plot_paths,
            int(simulation.runner.t_env),
            f"comms_eval/seed_{policy_seed}/training",
        )
        _log_wandb_plot_paths(
            simulation.logger,
            comms_plot_paths,
            int(simulation.runner.t_env),
            f"comms_eval/seed_{policy_seed}",
        )
        if hasattr(simulation.logger, "log_table"):
            simulation.logger.log_table(
                key=f"mock_training_eval/seed_{policy_seed}",
                value=training_df,
                t=simulation.runner.t_env,
            )

        plot_context_evaluation(
            seed_plot_df,
            plot_dir,
            info_str=f"seed={policy_seed}",
        )
        if hasattr(simulation.logger, "log_table"):
            simulation.logger.log_table(
                key=f"mock_low_level_eval/seed_{policy_seed}",
                value=seed_plot_df,
                t=simulation.runner.t_env,
            )

        frontier_results = []
        iteration_rows = []
        held_out_rows = []
        for pg in hierarchical_config.pg_values:

            def solve_policy(
                mixed_success_rates,
                threshold,
                goal_probability=pg,
                policy_seed=policy_seed,
            ):
                del threshold
                task_rows = [
                    {
                        "state": edge[0],
                        "next_state": edge[1],
                        "occupancy": 1.0,
                        "action_probability": 1.0,
                    }
                    for edge in task_edges
                ]
                selected_capacities = {}
                capacity_rows = []
                for edge in task_edges:
                    passing = [
                        (
                            float(mixed_success_rates[(edge, capacity)])
                            if (edge, capacity) in mixed_success_rates
                            else 0.0,
                            capacity,
                        )
                        for capacity in capacities
                    ]
                    _, selected_capacity = max(
                        passing,
                        key=lambda item: (item[0], -item[1]),
                    )
                    selected_capacities[edge] = selected_capacity
                    for capacity in capacities:
                        capacity_rows.append(
                            {
                                "state": edge[0],
                                "next_state": edge[1],
                                "msg_budget_per_agent": capacity,
                                "action_probability": (
                                    1.0
                                    if np.isclose(capacity, selected_capacity)
                                    else 0.0
                                ),
                            }
                        )
                return SN(
                    task_policy=pd.DataFrame.from_records(task_rows),
                    comms_budget_policy=pd.DataFrame.from_records(capacity_rows),
                    objective_value=float(sum(selected_capacities.values()))
                    / len(task_edges),
                    goal_reach_probability=goal_probability,
                )

            context_id = (
                hierarchical_config.contexts[0]
                if hierarchical_config.policy_scope.value == "per_context"
                and hierarchical_config.contexts
                else "mock-hierarchical"
            )
            context_key = ContextKey(
                context_id=context_id,
                policy_scope=hierarchical_config.policy_scope,
                high_level_seed=policy_seed,
            )
            result = iterative_high_level_synthesis(
                task_edges=task_edges,
                goal_states=goal_states,
                root_state=root_state,
                root_initial_distribution=root_initial_distribution,
                communication_capacities=tuple(capacities),
                calibration_rows=loaded_rows,
                goal_probability_spec=pg,
                epsilon_p=hierarchical_config.epsilon_p,
                max_iterations=hierarchical_config.max_iterations,
                context_key=context_key,
                cost_normalizer=max(1, len(task_edges)),
                solve_policy=solve_policy,
            )
            real_iteration_rows.extend(
                {
                    "seed": policy_seed,
                    "pg": pg,
                    "context_id": context_id,
                    "iteration": candidate.iteration,
                    "status": candidate.status.value,
                    "empirical_goal_probability": (
                        candidate.empirical_goal_probability
                    ),
                    "empirical_cost": candidate.empirical_cost,
                }
                for candidate in result.candidates
            )
            mixed_distribution_rows.extend(
                _candidate_mixed_distribution_rows(
                    seed=policy_seed,
                    pg=pg,
                    context_id=context_id,
                    candidates=result.candidates,
                )
            )
            iteration_rows.extend(
                {
                    "seed": policy_seed,
                    "pg": pg,
                    "iteration": candidate.iteration,
                    "status": candidate.status.value,
                    "empirical_goal_probability": (
                        candidate.empirical_goal_probability
                    ),
                    "empirical_cost": candidate.empirical_cost,
                }
                for candidate in result.candidates
            )
            selected = result.selected_candidate
            baseline_cost = float("nan")
            for fixed_capacity in capacities:
                task_action_probabilities = {}
                capacity_action_probabilities = {}
                if task_edges == ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4)):
                    task_action_probabilities.update(
                        {
                            (0, 1): 0.5,
                            (0, 2): 0.5,
                        }
                    )
                    for edge in task_edges:
                        if edge not in {(0, 1), (0, 2)}:
                            task_action_probabilities[edge] = 1.0
                else:
                    for edge in task_edges:
                        task_action_probabilities[edge] = 1.0
                for edge in task_edges:
                    capacity_action_probabilities[
                        (edge[0], edge[1], fixed_capacity)
                    ] = 1.0
                propagated = propagate_initial_state_distributions(
                    task_edges=task_edges,
                    goal_states=goal_states,
                    root_state=root_state,
                    root_initial_distribution=root_initial_distribution,
                    task_action_probabilities=task_action_probabilities,
                    capacity_action_probabilities=capacity_action_probabilities,
                    communication_capacities=(fixed_capacity,),
                    calibration_rows=loaded_rows,
                    cost_normalizer=max(1, len(task_edges)),
                )
                if (
                    propagated.goal_probability
                    >= pg + hierarchical_config.epsilon_p - 1e-12
                ):
                    baseline_cost = (
                        propagated.expected_communication_cost
                        if np.isnan(baseline_cost)
                        else min(
                            baseline_cost,
                            propagated.expected_communication_cost,
                        )
                    )
            task_greedy_model, comms_greedy_model = greedy_project_baselines(
                task_edges=task_edges,
                goal_states=goal_states,
                root_state=root_state,
                root_initial_distribution=root_initial_distribution,
                communication_capacities=tuple(capacities),
                calibration_rows=loaded_rows,
                pg=pg,
                cost_normalizer=max(1, len(task_edges)),
            )
            frontier_results.append(
                {
                    "seed": policy_seed,
                    "pg": pg,
                    "context_id": context_id,
                    "status": result.status.value,
                    "ours_cost": (
                        selected.empirical_cost if selected is not None else np.nan
                    ),
                    "baseline_cost": baseline_cost,
                    "empirical_goal_probability": (
                        selected.empirical_goal_probability
                        if selected is not None
                        else np.nan
                    ),
                    "empirical_cost": (
                        selected.empirical_cost if selected is not None else np.nan
                    ),
                    "candidate_count": len(result.candidates),
                    "task_greedy_cost": (
                        task_greedy_model.expected_communication_cost
                    ),
                    "comms_greedy_cost": (
                        comms_greedy_model.expected_communication_cost
                    ),
                    "task_greedy_success": task_greedy_model.goal_probability,
                    "comms_greedy_success": comms_greedy_model.goal_probability,
                }
            )
            if selected is None:
                held_out_rows.append(
                    {
                        "seed": policy_seed,
                        "pg": pg,
                        "context_id": context_id,
                        "status": result.status.value,
                        "n_episodes": hierarchical_config.held_out_episodes
                        or int(getattr(simulation.args, "test_nepisode", 20)),
                        "project_success_probability": float("nan"),
                        "empirical_cost": float("nan"),
                    }
                )
            if selected is not None:
                candidate_dir = join(
                    artifact_root,
                    f"context_{context_id}",
                    f"pg_{pg:.2f}",
                )
                makedirs(candidate_dir, exist_ok=True)
                candidate_manifest = {
                    "seed": policy_seed,
                    "pg": pg,
                    "status": selected.status.value,
                    "iteration": selected.iteration,
                    "empirical_goal_probability": (
                        selected.empirical_goal_probability
                    ),
                    "empirical_cost": selected.empirical_cost,
                    "context_id": selected.context_key.context_id,
                    "policy_scope": selected.context_key.policy_scope.value,
                }
                with open(
                    join(candidate_dir, "candidate_manifest.json"),
                    "w",
                    encoding="utf-8",
                ) as manifest_file:
                    json.dump(candidate_manifest, manifest_file, indent=2)
                if selected.task_policy is not None:
                    selected.task_policy.to_csv(
                        join(candidate_dir, "task_policy.csv"),
                        index=False,
                    )
                if selected.comms_budget_policy is not None:
                    selected.comms_budget_policy.to_csv(
                        join(candidate_dir, "comms_budget_policy.csv"),
                        index=False,
                    )
                held_out_results, held_out_summaries = (
                    _evaluate_mock_candidate_held_out(
                        seed=policy_seed,
                        pg=pg,
                        candidate=selected,
                        context_id=context_id,
                        task_edges=task_edges,
                        capacities=tuple(capacities),
                        calibration_rows=tuple(loaded_rows),
                        root_state=root_state,
                        goal_states=goal_states,
                        n_episodes=hierarchical_config.held_out_episodes
                        or int(getattr(simulation.args, "test_nepisode", 20)),
                    )
                )
                pd.DataFrame.from_records(
                    [
                        {
                            "seed": result.row_key.low_level_seed,
                            "pg": pg,
                            "context_id": result.row_key.context_id,
                            "edge": list(result.row_key.subtask_edge),
                            "communication_capacity": (
                                result.row_key.communication_capacity
                            ),
                            "start_state": list(result.row_key.start_state),
                            "episode_index": result.row_key.episode_index,
                            "project_success": result.project_success,
                            "communication_cost": result.communication_cost,
                            "successful_terminal_state": (
                                list(result.successful_terminal_state)
                                if result.successful_terminal_state is not None
                                else None
                            ),
                        }
                        for result in held_out_results
                    ]
                ).to_csv(
                    join(candidate_dir, "held_out_episode_results.csv"),
                    index=False,
                )
                pd.DataFrame.from_records(held_out_summaries).to_csv(
                    join(candidate_dir, "held_out_results.csv"),
                    index=False,
                )
                held_out_success = (
                    float(held_out_summaries[0]["project_success_probability"])
                    if held_out_summaries
                    else float("nan")
                )
                held_out_rows.append(
                    {
                        "seed": policy_seed,
                        "pg": pg,
                        "context_id": context_id,
                        "status": selected.status.value,
                        "n_episodes": hierarchical_config.held_out_episodes
                        or int(getattr(simulation.args, "test_nepisode", 20)),
                        "project_success_probability": held_out_success,
                        "empirical_cost": selected.empirical_cost,
                    }
                )

        if held_out_rows:
            held_out_df = pd.DataFrame.from_records(held_out_rows)
            held_out_dir = join(artifact_root, f"seed_{policy_seed}")
            makedirs(held_out_dir, exist_ok=True)
            held_out_df.to_csv(
                join(held_out_dir, "held_out_results.csv"),
                index=False,
            )

        frontier_df = pd.DataFrame.from_records(frontier_results)
        frontier_df.to_csv(join(artifact_root, "pg_frontier.csv"), index=False)
        if iteration_rows:
            iteration_df = pd.DataFrame.from_records(iteration_rows)
            iteration_paths = plot_pg_iteration_verification(
                iteration_df,
                join(artifact_root, "high_level"),
                seed=policy_seed,
            )
            _log_wandb_plot_paths(
                simulation.logger,
                iteration_paths,
                int(simulation.runner.t_env),
                f"high_level/seed_{policy_seed}/pg_iterations",
            )
        if mixed_distribution_rows:
            mixed_distribution_df = pd.DataFrame.from_records(
                mixed_distribution_rows
            )
            mixed_paths = plot_mixed_initial_distributions(
                mixed_distribution_df,
                join(artifact_root, "high_level"),
                seed=policy_seed,
            )
            _log_wandb_plot_paths(
                simulation.logger,
                mixed_paths,
                int(simulation.runner.t_env),
                f"high_level/seed_{policy_seed}/mixed_initial_distributions",
            )
            _log_wandb_table(
                simulation.logger,
                f"high_level/mixed_initial_distributions/seed_{policy_seed}",
                mixed_distribution_df,
                int(simulation.runner.t_env),
            )
        plot_df = frontier_df.drop(
            columns=["empirical_cost", "_empirical_cost_unused"],
            errors="ignore",
        )
        frontier_paths = plot_success_vs_cost_frontier(
            plot_df,
            join(artifact_root, "high_level"),
            seed=policy_seed,
        )
        _log_wandb_plot_paths(
            simulation.logger,
            frontier_paths,
            int(simulation.runner.t_env),
            f"high_level/seed_{policy_seed}/success_vs_cost",
        )
        if hasattr(simulation.logger, "log_table"):
            simulation.logger.log_table(
                key=f"mock_hierarchical_frontier/seed_{policy_seed}",
                value=frontier_df,
                t=simulation.runner.t_env,
            )

    summary = {
        "status": "mock_unified_runner_completed",
        "artifact_root": artifact_root,
        "seed": int(simulation.args.seed),
        "calibration_manifest_count": len(calibration_rows),
        "pg_values": list(hierarchical_config.pg_values),
        "n_calibration_episodes": n_calibration,
        "m_est_theoretical": mock_m_est,
        "finite_data_certificate_applicable": mock_calibration_is_certifiable,
    }
    with open(
        join(artifact_root, "mock_unified_summary.json"), "w", encoding="utf-8"
    ) as summary_file:
        json.dump(summary, summary_file, indent=2)
    if hasattr(simulation.logger, "log_table"):
        simulation.logger.log_table(
            key="mock_unified_summary",
            value=pd.DataFrame.from_records([summary]),
            t=simulation.runner.t_env,
        )
    simulation.logger.info(
        f"Mock unified runner completed; wrote results to {artifact_root}",
        log_header=True,
    )
    return summary


def solve_mixed_project_policy(*args, **kwargs):
    """Public compatibility solver for run-module patching."""
    try:
        import src.simulation.run as public_run

        public_solver = getattr(public_run, "solve_mixed_project_policy", None)
        if callable(public_solver) and public_solver is not solve_mixed_project_policy:
            return public_solver(*args, **kwargs)
    except (AttributeError, ImportError):
        pass
    return _solve_mixed_project_policy(*args, **kwargs)


class UnifiedHierarchicalSeedRunner:
    """One-seed unified hierarchical runner for mock or real options."""

    def __init__(self, simulation) -> None:
        self.simulation = simulation
        self.args = simulation.args
        self.runner = simulation.runner
        self.logger = simulation.logger

    def run(self, hierarchical_config, artifact_root: str) -> dict:
        makedirs(artifact_root, exist_ok=True)
        config_path = join(artifact_root, "hierarchical_config.json")
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(hierarchical_config.to_dict(), config_file, indent=2)

        mock_low_level = getattr(self.args, "env", "") == "mock-hierarchical-v0"
        if isinstance(getattr(self.args, "env_args", None), dict):
            mock_low_level = mock_low_level or bool(
                self.args.env_args.get("mock_low_level", False)
            )
        if mock_low_level:
            return run_mock_seed(
                self.simulation,
                hierarchical_config,
                artifact_root,
            )
        return self.run_real_seed(hierarchical_config, artifact_root)

    @staticmethod
    def _capacities(args) -> list[float]:
        value = getattr(args, "msg_budget_per_agent", None)
        if value is None:
            value = [max(0, int(getattr(args, "n_agents", 2)) - 1)]
        if isinstance(value, (int, float, np.integer, np.floating)):
            return [float(value)]
        if isinstance(value, str):
            value = value.strip("[]").replace(",", " ").split()
        values = [float(item) for item in value]
        if not values:
            raise ValueError(
                "Unified hierarchical runner requires communication capacities."
            )
        if len(set(values)) != len(values):
            raise ValueError("Communication capacities must be unique.")
        return values

    def write_run_phase_manifest(
        self,
        artifact_root: str,
        policy_seed: int,
        phase: str,
        details: dict | None = None,
    ) -> str:
        """Write one phase record for a seed-isolated unified run."""
        seed_root = join(artifact_root, f"seed_{policy_seed}")
        makedirs(seed_root, exist_ok=True)
        manifest_path = join(seed_root, "run_phase.json")
        payload = {
            "seed": int(policy_seed),
            "phase": phase,
            "details": details or {},
        }
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(payload, manifest_file, indent=2)
        return manifest_path

    def train_shared_conditional_policy(
        self,
        hierarchical_config,
        artifact_root: str,
        policy_seed: int,
        task_edges: tuple[tuple[int, int], ...],
        capacities: list[float],
        n_calibration: int,
    ) -> tuple[list[CalibrationRow], dict]:
        """Train one conditional policy across task/capacity/start contexts."""
        original_context = (
            self.args,
            self.runner,
            getattr(self.simulation, "buffer", None),
            getattr(self.simulation, "learner", None),
            getattr(self.simulation, "n_eval_eps", None),
        )
        candidate_args = SN(**vars(self.args))
        candidate_args.env_args = deepcopy(self.args.env_args)
        if not hasattr(candidate_args, "batch_size"):
            candidate_args.batch_size = 32
        candidate_args.env_args["seed"] = policy_seed
        candidate_args.seed = policy_seed
        candidate_args.save_model = False
        candidate_args.unique_policy_per_msg_budget = False
        candidate_args.condition_on_msg_budget = True
        candidate_args.condition_on_subtask = True
        candidate_args.subtask_contexts = [
            [edge[0], edge[1]] for edge in task_edges
        ]
        candidate_args.msg_budget_per_agent = [
            int(capacity) for capacity in capacities
        ]
        candidate_args.env_args["msg_budget_per_agent"] = [
            int(capacity) for capacity in capacities
        ]

        shared_options = []
        for edge in task_edges:
            reset_options = _navigation_reset_options(
                candidate_args,
                edge,
                capacities[0],
            )
            start_states = _flatten_navigation_states(reset_options)
            for start_state in start_states:
                shared_options.append(
                    (edge, start_state, reset_options)
                )

        rng = np.random.default_rng(policy_seed)
        numpy_state = np.random.get_state()
        torch_state = th.random.get_rng_state()
        cuda_states = th.cuda.get_rng_state_all() if th.cuda.is_available() else None
        candidate_runner = None
        try:
            np.random.seed(policy_seed)
            th.manual_seed(policy_seed)
            self.args = candidate_args
            self.args, self.runner, buffer, learner = build_sim(
                candidate_args,
                self.logger,
            )
            candidate_runner = self.runner
            self.simulation.buffer = buffer
            self.simulation.learner = learner
            self.simulation.n_eval_eps = max(
                1,
                candidate_args.test_nepisode // candidate_runner.batch_size,
            )
            self._shared_candidate_runner = candidate_runner
            self._shared_candidate_args = candidate_args

            while candidate_runner.t_env <= candidate_args.t_max:
                batch_reset_options = _sample_mixed_navigation_reset_options(
                    candidate_args,
                    task_edges,
                    capacities,
                    candidate_runner.batch_size,
                    rng,
                )
                reset_options = (
                    batch_reset_options[0]
                    if candidate_runner.batch_size == 1
                    else batch_reset_options
                )

                result = candidate_runner.run(
                    test_mode=False,
                    reset_options=reset_options,
                )
                episode_batch = (
                    result["batch"] if isinstance(result, dict) else result
                )
                buffer.insert_episode_batch(episode_batch)

                for update_idx in range(episode_batch.batch_size):
                    if not buffer.can_sample(candidate_args.batch_size):
                        break
                    episode_sample = buffer.sample(candidate_args.batch_size)
                    max_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_t]
                    if episode_sample.device != candidate_args.device:
                        episode_sample.to(candidate_args.device)
                    learner.train(
                        episode_sample,
                        candidate_runner.t_env,
                        update_idx,
                    )

            policy_weights = {
                key: (
                    value.detach().cpu().clone()
                    if hasattr(value, "detach")
                    else value
                )
                for key, value in candidate_runner.mac.agent.state_dict().items()
            }
            shared_policy_bank = {}
            calibration_rows = []
            for edge, start_state, _ in shared_options:
                for capacity in capacities:
                    reset_options = _navigation_reset_options(
                        candidate_args,
                        edge,
                        capacity,
                    )
                    setting = OptionSettingKey(
                        subtask_edge=edge,
                        communication_capacity=capacity,
                        start_state=start_state,
                    )
                    calibration_row = collect_statewise_calibration_row(
                        candidate_runner,
                        setting=setting,
                        seed=policy_seed,
                        n_eval_eps=n_calibration,
                        reset_options=reset_options,
                        record_replays=False,
                    )
                    calibration_rows.append(calibration_row)
                    shared_policy_bank[
                        (
                            edge[0],
                            edge[1],
                            capacity,
                            start_state,
                            policy_seed,
                        )
                    ] = policy_weights
            real_evaluation_rows = [
                {
                    "seed": policy_seed,
                    "hl_task": list(row.setting.subtask_edge),
                    "hl_start_state": row.setting.subtask_edge[0],
                    "start_state": list(row.setting.start_state),
                    "msg_budget_per_agent": row.setting.communication_capacity,
                    "test_task_completed_mean": row.success_rate,
                    "test_n_episodes": len(row.outcomes),
                    "test_return_mean": row.success_rate,
                    "test_return_std": 0.0,
                    "test_ep_length_mean": float(
                        getattr(candidate_args, "max_episode_steps", 15)
                    ),
                    "t_env": int(getattr(candidate_runner, "t_env", 0)),
                }
                for row in calibration_rows
            ]
            pd.DataFrame.from_records(real_evaluation_rows).to_csv(
                join(
                    artifact_root,
                    f"real_training_eval_seed_{policy_seed}.csv",
                ),
                index=False,
            )
            real_evaluation_df = pd.DataFrame.from_records(real_evaluation_rows)
            comms_eval_dir = join(artifact_root, "comms_eval")
            training_plot_paths = plot_context_evaluation(
                real_evaluation_df,
                join(comms_eval_dir, "training"),
                info_str=(
                    f"seed={policy_seed}; "
                    f"real low-level statewise calibration"
                ),
            )
            comms_plot_paths = plot_success_vs_comms_by_subtask(
                real_evaluation_df,
                comms_eval_dir,
            )
            selected_replay_settings = _record_selected_calibration_replays(
                candidate_runner,
                setting_rows=calibration_rows,
                reset_option_builder=lambda setting: _navigation_reset_options(
                    candidate_args,
                    setting.subtask_edge,
                    setting.communication_capacity,
                ),
                max_settings=6,
                n_replays=int(
                    getattr(candidate_args, "n_test_replays_save", 2)
                ),
            )
            pd.DataFrame.from_records(
                [
                    {
                        "subtask_edge": list(setting.subtask_edge),
                        "communication_capacity": setting.communication_capacity,
                        "start_state": list(setting.start_state),
                        "n_replays": int(
                            getattr(candidate_args, "n_test_replays_save", 2)
                        ),
                    }
                    for setting in selected_replay_settings
                ]
            ).to_csv(
                join(
                    artifact_root,
                    f"selected_calibration_replays_seed_{policy_seed}.csv",
                ),
                index=False,
            )
            shared_candidate_runner = getattr(
                self,
                "_shared_candidate_runner",
                None,
            )
            t_env = int(
                getattr(
                    shared_candidate_runner,
                    "t_env",
                    getattr(self.runner, "t_env", 0),
                )
            )
            _log_wandb_table(
                self.logger,
                f"real_training_eval/seed_{policy_seed}",
                real_evaluation_df,
                t_env,
            )
            _log_wandb_plot_paths(
                self.logger,
                training_plot_paths,
                t_env,
                f"comms_eval/seed_{policy_seed}/training",
            )
            _log_wandb_plot_paths(
                self.logger,
                comms_plot_paths,
                t_env,
                f"comms_eval/seed_{policy_seed}",
            )
        finally:
            (
                self.args,
                self.runner,
                self.simulation.buffer,
                self.simulation.learner,
                self.simulation.n_eval_eps,
            ) = original_context
            np.random.set_state(numpy_state)
            th.random.set_rng_state(torch_state)
            if cuda_states is not None:
                th.cuda.set_rng_state_all(cuda_states)

        for edge, start_state, _ in shared_options:
            for capacity in capacities:
                artifact = LowLevelPolicyArtifact(
                    seed=policy_seed,
                    subtask_edge=edge,
                    communication_capacity=capacity,
                    initial_state_distribution=StateDistribution.singleton(
                        start_state
                    ),
                    timestep=int(getattr(candidate_runner, "t_env", 0)),
                    checkpoint_path=join(artifact_root, "unused"),
                    config_hash="shared-conditional-v1",
                )
                save_option_policy_weights_artifact(
                    artifact_root,
                    artifact,
                    policy_weights,
                )

        save_calibration_manifest(
            artifact_root,
            policy_seed,
            calibration_rows,
        )
        self.write_run_phase_manifest(
            artifact_root,
            policy_seed,
            "shared_conditional_policy_frozen",
            {
                "n_calibration_episodes": n_calibration,
                "n_settings": len(calibration_rows),
                "t_max": int(candidate_args.t_max),
                "parallel_calibration_batch_size": int(
                    getattr(candidate_runner, "batch_size", 1)
                ),
                "n_calibration_episode_calls": (
                    -(-n_calibration // max(
                        1,
                        int(getattr(candidate_runner, "batch_size", 1)),
                    ))
                    * len(calibration_rows)
                ),
                "n_calibration_episodes_per_setting": n_calibration,
                "n_statewise_settings": len(calibration_rows),
                "n_total_calibration_episodes": (
                    n_calibration * len(calibration_rows)
                ),
                "n_replay_episodes_per_setting": (
                    int(
                        getattr(
                            candidate_args,
                            "n_test_replays_save",
                            0,
                        )
                    )
                    if getattr(
                        candidate_args,
                        "save_test_replays",
                        False,
                    )
                    else 0
                ),
                "n_total_recorded_replays": (
                    int(
                        getattr(
                            candidate_args,
                            "n_test_replays_save",
                            0,
                        )
                    )
                    * len(calibration_rows)
                    if getattr(
                        candidate_args,
                        "save_test_replays",
                        False,
                    )
                    else 0
                ),
                "held_out_episodes": (
                    hierarchical_config.held_out_episodes
                    or int(getattr(candidate_args, "test_nepisode", 20))
                ),
                "n_held_out_policy_evaluations": (
                    (
                        hierarchical_config.held_out_episodes
                        or int(getattr(candidate_args, "test_nepisode", 20))
                    )
                    * len(hierarchical_config.pg_values)
                ),
            },
        )
        return calibration_rows, shared_policy_bank

    def collect_real_option_library(
        self,
        hierarchical_config,
        artifact_root: str,
        policy_seed: int,
        task_edges: tuple[tuple[int, int], ...],
        capacities: list[float],
        n_calibration: int,
    ) -> tuple[list[CalibrationRow], dict]:
        """Train/freeze/evaluate every real `(edge, capacity, start-state)` setting."""
        return self.train_shared_conditional_policy(
            hierarchical_config,
            artifact_root,
            policy_seed,
            task_edges,
            capacities,
            n_calibration,
        )

    def run_real_seed(
        self,
        hierarchical_config,
        artifact_root: str,
    ) -> dict:
        makedirs(artifact_root, exist_ok=True)
        task_edges = hierarchical_config.task_sequence
        capacities = self._capacities(self.args)
        policy_seeds = [int(self.args.seed)]
        n_calibration = hierarchical_config.n_calibration_episodes or int(
            getattr(self.args, "test_nepisode", 20)
        )
        root_state = task_edges[0][0]
        root_reset_options = _navigation_reset_options(
            self.args,
            task_edges[0],
            capacities[0],
        )
        root_support = _flatten_navigation_states(root_reset_options)
        root_initial_distribution = StateDistribution(
            root_support,
            tuple(1.0 / len(root_support) for _ in root_support),
        )

        calibration_rows = []
        policy_bank = {}

        for policy_seed in policy_seeds:
            frontier_rows = []
            real_iteration_rows = []
            mixed_distribution_rows = []
            seed_rows, seed_policy_bank = self.collect_real_option_library(
                hierarchical_config,
                artifact_root,
                policy_seed,
                task_edges,
                capacities,
                n_calibration,
            )
            calibration_rows.extend(seed_rows)
            policy_bank.update(seed_policy_bank)
            loaded_rows = read_calibration_manifest(
                artifact_root,
                policy_seed,
            )
            selected_candidates = {}
            contexts = (
                list(hierarchical_config.contexts)
                if hierarchical_config.policy_scope.value == "per_context"
                else ["unified"]
            )

            for pg in hierarchical_config.pg_values:
                for context_id in contexts:
                    def solve_policy(
                        mixed_success_rates,
                        threshold,
                    ):
                        from src.envs.basic_gymnasium_wrappers import (
                            HLMDPEnvWrapper,
                        )
                        env_args = deepcopy(getattr(self.args, "env_args", {}))
                        env_args["key"] = getattr(self.args, "env", "team-nav-v0")
                        env_args["seed"] = int(
                            getattr(self.args, "seed", policy_seed)
                        )
                        env_args["n_agents"] = int(
                            getattr(
                                self.args,
                                "n_agents",
                                env_args.get("n_agents", 3),
                            )
                        )
                        for mock_only_key in (
                            "hierarchical",
                            "hl_env",
                            "hl_env_config",
                            "mock_low_level",
                            "n_final_positions",
                            "n_comms_values",
                            "n_subtasks",
                        ):
                            env_args.pop(mock_only_key, None)
                        env_args["msg_budget_per_agent"] = list(capacities)
                        temp_wrapper = HLMDPEnvWrapper(env_args=env_args)
                        try:
                            return _solve_mixed_project_policy(
                                self.args,
                                temp_wrapper.hlmdp,
                                mixed_success_rates,
                                threshold,
                            )
                        finally:
                            temp_wrapper.close()

                    result = iterative_high_level_synthesis(
                        task_edges=task_edges,
                        goal_states=(task_edges[-1][1],),
                        root_state=root_state,
                        root_initial_distribution=root_initial_distribution,
                        communication_capacities=tuple(capacities),
                        calibration_rows=loaded_rows,
                        goal_probability_spec=pg,
                        epsilon_p=hierarchical_config.epsilon_p,
                        max_iterations=hierarchical_config.max_iterations,
                        context_key=ContextKey(
                            context_id=context_id,
                            policy_scope=hierarchical_config.policy_scope,
                            high_level_seed=policy_seed,
                        ),
                        cost_normalizer=len(task_edges),
                        solve_policy=solve_policy,
                    )
                    real_iteration_rows.extend(
                        {
                            "seed": policy_seed,
                            "pg": pg,
                            "context_id": context_id,
                            "iteration": candidate.iteration,
                            "status": candidate.status.value,
                            "empirical_goal_probability": (
                                candidate.empirical_goal_probability
                            ),
                            "empirical_cost": candidate.empirical_cost,
                        }
                        for candidate in result.candidates
                    )
                    selected = result.selected_candidate
                    if selected is not None:
                        selected_candidates[(pg, context_id)] = selected
                    baseline_cost = float("nan")
                    for fixed_capacity in capacities:
                        task_action_probabilities = {}
                        for edge in task_edges:
                            task_action_probabilities[edge] = (
                                0.5 if edge in {(0, 1), (0, 2)} else 1.0
                            )
                        capacity_action_probabilities = {
                            (edge[0], edge[1], fixed_capacity): 1.0
                            for edge in task_edges
                        }
                        propagated = propagate_initial_state_distributions(
                            task_edges=task_edges,
                            goal_states=(task_edges[-1][1],),
                            root_state=root_state,
                            root_initial_distribution=root_initial_distribution,
                            task_action_probabilities=task_action_probabilities,
                            capacity_action_probabilities=(
                                capacity_action_probabilities
                            ),
                            communication_capacities=(fixed_capacity,),
                            calibration_rows=loaded_rows,
                            cost_normalizer=len(task_edges),
                        )
                        if (
                            propagated.goal_probability
                            >= pg + hierarchical_config.epsilon_p - 1e-12
                        ):
                            baseline_cost = (
                                propagated.expected_communication_cost
                                if np.isnan(baseline_cost)
                                else min(
                                    baseline_cost,
                                    propagated.expected_communication_cost,
                                )
                            )
                    task_greedy_model, comms_greedy_model = (
                        greedy_project_baselines(
                            task_edges=task_edges,
                            goal_states=(task_edges[-1][1],),
                            root_state=root_state,
                            root_initial_distribution=root_initial_distribution,
                            communication_capacities=tuple(capacities),
                            calibration_rows=tuple(loaded_rows),
                            pg=pg,
                            cost_normalizer=len(task_edges),
                        )
                    )
                    frontier_rows.append(
                        {
                            "seed": policy_seed,
                            "pg": pg,
                            "context_id": context_id,
                            "status": result.status.value,
                            "search_error": result.search_error,
                            "ours_cost": (
                                selected.empirical_cost
                                if selected is not None
                                else np.nan
                            ),
                            "baseline_cost": baseline_cost,
                            "empirical_goal_probability": (
                                selected.empirical_goal_probability
                                if selected is not None
                                else np.nan
                            ),
                            "empirical_cost": (
                                selected.empirical_cost
                                if selected is not None
                                else np.nan
                            ),
                            "candidate_count": len(result.candidates),
                            "task_greedy_cost": (
                                task_greedy_model.expected_communication_cost
                            ),
                            "comms_greedy_cost": (
                                comms_greedy_model.expected_communication_cost
                            ),
                            "task_greedy_success": (
                                task_greedy_model.goal_probability
                            ),
                            "comms_greedy_success": (
                                comms_greedy_model.goal_probability
                            ),
                        }
                    )

            makedirs(join(artifact_root, f"seed_{policy_seed}"), exist_ok=True)
            frontier_df = pd.DataFrame.from_records(frontier_rows)
            seed_frontier_path = join(
                artifact_root,
                f"seed_{policy_seed}",
                "pg_frontier.csv",
            )
            frontier_df.to_csv(
                seed_frontier_path,
                index=False,
            )
            frontier_paths = plot_success_vs_cost_frontier(
                frontier_df,
                join(
                    artifact_root,
                    f"seed_{policy_seed}",
                    "high_level",
                ),
                seed=policy_seed,
            )
            shared_candidate_runner = getattr(
                self,
                "_shared_candidate_runner",
                None,
            )
            t_env = int(
                getattr(
                    shared_candidate_runner,
                    "t_env",
                    getattr(self.runner, "t_env", 0),
                )
            )
            _log_wandb_table(
                self.logger,
                f"high_level/pg_frontier/seed_{policy_seed}",
                frontier_df,
                t_env,
            )
            _log_wandb_plot_paths(
                self.logger,
                frontier_paths,
                t_env,
                f"high_level/seed_{policy_seed}/success_vs_cost",
            )
            if real_iteration_rows:
                real_iteration_df = pd.DataFrame.from_records(
                    real_iteration_rows
                )
                iteration_paths = plot_pg_iteration_verification(
                    real_iteration_df,
                    join(
                        artifact_root,
                        f"seed_{policy_seed}",
                        "high_level",
                    ),
                    seed=policy_seed,
                )
                _log_wandb_table(
                    self.logger,
                    f"high_level/pg_iterations/seed_{policy_seed}",
                    real_iteration_df,
                    t_env,
                )
                _log_wandb_plot_paths(
                    self.logger,
                    iteration_paths,
                    t_env,
                    f"high_level/seed_{policy_seed}/pg_iterations",
                )
            if mixed_distribution_rows:
                mixed_distribution_df = pd.DataFrame.from_records(
                    mixed_distribution_rows
                )
                mixed_paths = plot_mixed_initial_distributions(
                    mixed_distribution_df,
                    join(
                        artifact_root,
                        f"seed_{policy_seed}",
                        "high_level",
                    ),
                    seed=policy_seed,
                )
                _log_wandb_table(
                    self.logger,
                    f"high_level/mixed_initial_distributions/seed_{policy_seed}",
                    mixed_distribution_df,
                    t_env,
                )
                _log_wandb_plot_paths(
                    self.logger,
                    mixed_paths,
                    t_env,
                    f"high_level/seed_{policy_seed}/mixed_initial_distributions",
                )

            held_out_rows = []
            held_out_episodes = hierarchical_config.held_out_episodes or getattr(
                self.args, "test_nepisode", 20
            )
            for pg in hierarchical_config.pg_values:
                for context_id in contexts:
                    row = next(
                        (
                            frontier_row
                            for frontier_row in frontier_rows
                            if (
                                frontier_row["seed"] == policy_seed
                                and frontier_row["pg"] == pg
                                and frontier_row["context_id"] == context_id
                            )
                        ),
                        None,
                    )
                    selected = selected_candidates.get((pg, context_id))
                    if selected is not None:
                        context_dir = (
                            context_id
                            if hierarchical_config.policy_scope.value
                            == "per_context"
                            else "unified"
                        )
                        candidate_dir = join(
                            artifact_root,
                            f"seed_{policy_seed}",
                            f"context_{context_dir}",
                            f"pg_{pg:.2f}",
                        )
                        makedirs(candidate_dir, exist_ok=True)
                        candidate_manifest = {
                            "seed": policy_seed,
                            "pg": pg,
                            "status": selected.status.value,
                            "iteration": selected.iteration,
                            "empirical_goal_probability": (
                                selected.empirical_goal_probability
                            ),
                            "empirical_cost": selected.empirical_cost,
                            "context_id": context_id,
                            "policy_scope": (
                                hierarchical_config.policy_scope.value
                            ),
                            "caveats": [
                                "Iterative synthesis does not establish global optimality.",
                                "An iteration bound is not proof of no-feasible-policy.",
                            ],
                        }
                        with open(
                            join(candidate_dir, "candidate_manifest.json"),
                            "w",
                            encoding="utf-8",
                        ) as manifest_file:
                            json.dump(candidate_manifest, manifest_file, indent=2)
                        if selected.task_policy is not None:
                            selected.task_policy.to_csv(
                                join(candidate_dir, "task_policy.csv"),
                                index=False,
                            )
                        if selected.comms_budget_policy is not None:
                            selected.comms_budget_policy.to_csv(
                                join(candidate_dir, "comms_budget_policy.csv"),
                                index=False,
                            )
                        held_out_results, held_out_summaries = (
                            _evaluate_real_candidate_held_out(
                                self,
                                seed=policy_seed,
                                pg=pg,
                                candidate=selected,
                                context_id=context_id,
                                task_edges=task_edges,
                                capacities=tuple(capacities),
                                calibration_rows=tuple(loaded_rows),
                                policy_bank=policy_bank,
                                root_state=root_state,
                                goal_states=(task_edges[-1][1],),
                                n_episodes=int(held_out_episodes),
                            )
                        )
                        pd.DataFrame.from_records(
                            [
                                {
                                    "seed": result.row_key.low_level_seed,
                                    "pg": pg,
                                    "context_id": context_id,
                                    "edge": list(result.row_key.subtask_edge),
                                    "communication_capacity": (
                                        result.row_key.communication_capacity
                                    ),
                                    "start_state": list(result.row_key.start_state),
                                    "episode_index": result.row_key.episode_index,
                                    "project_success": result.project_success,
                                    "communication_cost": result.communication_cost,
                                    "successful_terminal_state": (
                                        list(result.successful_terminal_state)
                                        if result.successful_terminal_state is not None
                                        else None
                                    ),
                                }
                                for result in held_out_results
                            ]
                        ).to_csv(
                            join(candidate_dir, "held_out_episode_results.csv"),
                            index=False,
                        )
                        pd.DataFrame.from_records(held_out_summaries).to_csv(
                            join(candidate_dir, "held_out_results.csv"),
                            index=False,
                        )
                    held_out_rows.append(
                        {
                            "seed": policy_seed,
                            "pg": pg,
                            "context_id": context_id,
                            "status": (
                                row["status"]
                                if row is not None
                                else "search_failed"
                            ),
                            "n_episodes": int(held_out_episodes),
                            "project_success_probability": (
                                row["empirical_goal_probability"]
                                if row is not None
                                else np.nan
                            ),
                            "empirical_cost": (
                                row["empirical_cost"]
                                if row is not None
                                else np.nan
                            ),
                        }
                    )
            held_out_df = pd.DataFrame.from_records(held_out_rows)
            held_out_df.to_csv(
                join(
                    artifact_root,
                    f"seed_{policy_seed}",
                    "held_out_results.csv",
                ),
                index=False,
            )
            _log_wandb_table(
                self.logger,
                f"high_level/held_out/seed_{policy_seed}",
                held_out_df,
                t_env,
            )

            shared_candidate_runner = getattr(
                self,
                "_shared_candidate_runner",
                None,
            )
            if shared_candidate_runner is not None:
                shared_candidate_runner.close_env()
                delattr(self, "_shared_candidate_runner")

        summary = {
            "status": "real_seed_runner_completed",
            "artifact_root": artifact_root,
            "seed": int(self.args.seed),
            "calibration_manifest_count": len(calibration_rows),
            "pg_values": list(hierarchical_config.pg_values),
        }
        with open(
            join(artifact_root, "unified_seed_summary.json"),
            "w",
            encoding="utf-8",
        ) as summary_file:
            json.dump(summary, summary_file, indent=2)
        self.logger.info(
            f"Unified seed runner completed; wrote results to {artifact_root}",
            log_header=True,
        )
        return summary
