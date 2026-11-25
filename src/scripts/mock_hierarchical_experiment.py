"""Mock fixed-option-library experiment for high-level synthesis development.

This script intentionally has no environment or GPU dependencies. It:

1. generates mock categorical statewise calibration episodes,
2. writes and reloads a calibration manifest,
3. writes and reloads mock option-policy artifacts,
4. runs the iterative high-level synthesis algorithm with a small fixed-capacity
   chain-policy solver.
"""

from __future__ import annotations

import argparse
import itertools
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.context_utils import (
    plot_context_evaluation,
    plot_success_vs_comms_by_subtask,
    plot_success_vs_cost_frontier,
)
from src.simulation.hierarchical_calibration import (
    calculate_m_est,
    calculate_project_reliability_error,
    greedy_project_baselines,
    propagate_initial_state_distributions,
    read_calibration_manifest,
    save_calibration_manifest,
    validate_project_screening_margin,
)
from src.simulation.hierarchical_synthesis import (
    SynthesisInfeasibleError,
    iterative_high_level_synthesis,
)
from src.simulation.hierarchical_types import (
    CalibrationOutcome,
    CalibrationRow,
    ContextKey,
    LowLevelPolicyArtifact,
    OptionSettingKey,
    PolicyScope,
    StateDistribution,
)
from src.simulation.policy_artifacts import (
    find_latest_option_policy_artifact,
    load_option_policy_artifact,
    option_manifest_path,
    read_option_policy_manifest,
    save_option_policy_artifact,
)


@dataclass(frozen=True)
class MockExperimentParameters:
    n_agents: int = 3
    n_final_positions: int = 4
    n_comms_values: int = 2
    n_subtasks: int = 5
    mock_evaluation_times: tuple[int, ...] = (20_000, 40_000, 60_000, 80_000, 100_000)

    def __post_init__(self) -> None:
        for name in (
            "n_agents",
            "n_final_positions",
            "n_comms_values",
            "n_subtasks",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if len(self.mock_evaluation_times) < 5:
            raise ValueError(
                "mock_evaluation_times must contain at least five checkpoints."
            )
        evaluation_times = tuple(
            sorted(int(t_env) for t_env in self.mock_evaluation_times)
        )
        if len(set(evaluation_times)) != len(evaluation_times):
            raise ValueError("mock_evaluation_times must contain unique times.")
        if any(t_env < 0 for t_env in evaluation_times):
            raise ValueError("mock_evaluation_times cannot contain negative times.")
        object.__setattr__(self, "mock_evaluation_times", evaluation_times)

    @property
    def communication_capacities(self) -> tuple[float, ...]:
        if self.n_comms_values == 1:
            return (0.0,)
        return tuple(
            index / (self.n_comms_values - 1) for index in range(self.n_comms_values)
        )

    @property
    def final_joint_state_count(self) -> int:
        # Mock representative joint states: one moved agent at a time.
        return self.n_agents * self.n_final_positions

    @property
    def l_setting_count(self) -> int:
        return self.n_comms_values * self.n_subtasks * self.final_joint_state_count


def generate_joint_states(params: MockExperimentParameters) -> tuple:
    """Generate a representative joint-state set of size n_agents*n_positions."""
    states = []
    for state_index in range(params.final_joint_state_count):
        position_offset = state_index % params.n_final_positions
        agent_positions = tuple(
            (position_offset + agent_index) % params.n_final_positions
            for agent_index in range(params.n_agents)
        )
        states.append((state_index, *agent_positions))
    return tuple(states)


def build_mock_task_edges(n_subtasks: int) -> tuple[tuple[int, int], ...]:
    """Use the merge DAG for five subtasks; otherwise use a chain."""
    if n_subtasks == 5:
        return ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4))
    return tuple((index, index + 1) for index in range(n_subtasks))


class MockPolicyLearner:
    """Fake learner interface sufficient for the manifest checkpoint helpers."""

    def __init__(self):
        self.saved_paths: list[str] = []
        self.loaded_paths: list[str] = []

    def save_models(self, path: str) -> None:
        checkpoint = Path(path)
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "mock_policy_weights.json").write_text(
            '{"fixed": "mock-low-level-policy"}\n',
            encoding="utf-8",
        )
        self.saved_paths.append(path)

    def load_models(self, path: str) -> None:
        checkpoint = Path(path)
        if not (checkpoint / "mock_policy_weights.json").is_file():
            raise FileNotFoundError(f"Missing mock policy weights at {checkpoint}")
        self.loaded_paths.append(path)


def _generate_terminal_outcomes(
    rng: random.Random,
    n_episodes: int,
    success_rate: float,
    terminal_distribution: StateDistribution,
) -> tuple[CalibrationOutcome, ...]:
    n_success = round(n_episodes * success_rate)
    n_success = max(0, min(n_episodes, n_success))
    n_failure = n_episodes - n_success

    terminal_states: list[tuple] = []
    if n_success > 0:
        if not terminal_distribution.states:
            raise ValueError("A successful mock setting needs terminal states.")
        remaining = n_success
        allocations: list[int] = []
        for probability in terminal_distribution.probabilities:
            allocation = round(n_success * probability)
            allocation = min(allocation, remaining)
            allocations.append(allocation)
            remaining -= allocation
        while remaining > 0:
            for index in range(len(allocations)):
                if remaining == 0:
                    break
                allocations[index] += 1
                remaining -= 1
        while sum(allocations) > n_success:
            for index in range(len(allocations) - 1, -1, -1):
                if allocations[index] > 0 and sum(allocations) > n_success:
                    allocations[index] -= 1

        for state, count in zip(
            terminal_distribution.states,
            allocations,
            strict=True,
        ):
            terminal_states.extend([state] * count)

    outcomes = [
        (
            CalibrationOutcome.success(rng.choice(terminal_states))
            if terminal_states
            else CalibrationOutcome.failure()
        )
        for _ in range(n_success)
    ]
    outcomes.extend(CalibrationOutcome.failure() for _ in range(n_failure))
    rng.shuffle(outcomes)
    return tuple(outcomes)


def generate_mock_calibration_rows(
    seed: int,
    n_estimation_episodes: int,
    params: MockExperimentParameters | None = None,
) -> tuple[CalibrationRow, ...]:
    """Generate fixed mock categorical rows for every statewise setting."""
    if n_estimation_episodes < 1:
        raise ValueError("n_estimation_episodes must be positive.")
    params = params or MockExperimentParameters()
    rng = random.Random(seed)
    joint_states = generate_joint_states(params)
    terminal_distribution = StateDistribution(
        joint_states,
        tuple(1.0 / len(joint_states) for _ in joint_states),
    )
    task_edges = build_mock_task_edges(params.n_subtasks)
    rows = []
    for edge_index, edge in enumerate(task_edges):
        for capacity_index, capacity in enumerate(params.communication_capacities):
            for start_index, start_state in enumerate(joint_states):
                edge_position = edge_index / max(1, len(task_edges) - 1)
                capacity_position = capacity_index / max(
                    1, len(params.communication_capacities) - 1
                )
                start_position = start_index / max(1, len(joint_states) - 1)
                success_rate = min(
                    0.98,
                    0.58
                    + 0.20 * edge_position
                    + 0.20 * capacity_position
                    + 0.05 * (1.0 - start_position),
                )
                setting = OptionSettingKey(
                    edge,
                    float(capacity),
                    start_state,
                )
                outcomes = _generate_terminal_outcomes(
                    rng,
                    n_estimation_episodes,
                    success_rate,
                    terminal_distribution,
                )
                rows.append(
                    CalibrationRow(
                        setting=setting,
                        outcomes=outcomes,
                        seed=seed,
                    )
                )
    return tuple(rows)


def generate_mock_training_evaluations(
    seed: int,
    params: MockExperimentParameters,
) -> pd.DataFrame:
    """Generate realistic mock success-vs-time and success-vs-comms rows."""
    task_edges = build_mock_task_edges(params.n_subtasks)
    rows = []
    n_checkpoints = len(params.mock_evaluation_times)
    for edge_index, edge in enumerate(task_edges):
        for capacity_index, capacity in enumerate(params.communication_capacities):
            for checkpoint_index, t_env in enumerate(params.mock_evaluation_times):
                progress = checkpoint_index / max(1, n_checkpoints - 1)
                base_success = (
                    0.42
                    + 0.10 * (edge_index / max(1, len(task_edges) - 1))
                    + 0.18
                    * (
                        capacity_index
                        / max(1, len(params.communication_capacities) - 1)
                    )
                )
                rng = np.random.default_rng(
                    seed * 100_000
                    + edge_index * 10_000
                    + capacity_index * 1_000
                    + checkpoint_index
                )
                success = (
                    base_success
                    + progress * (0.96 - base_success)
                    + float(rng.normal(0.0, 0.015))
                )
                success = float(np.clip(success, 0.05, 0.98))
                rows.append(
                    {
                        "seed": seed,
                        "hl_task": list(edge),
                        "hl_start_state": edge[0],
                        "msg_budget_per_agent": capacity,
                        "test_task_completed_mean": success,
                        "test_n_episodes": 20,
                        "test_return_mean": success,
                        "test_return_std": max(
                            0.005,
                            float(rng.normal(0.02, 0.005)),
                        ),
                        "test_ep_length_mean": 1.0,
                        "t_env": int(t_env),
                    }
                )
    return pd.DataFrame.from_records(rows)


def save_mock_option_library(
    output_dir: Path,
    seed: int,
    params: MockExperimentParameters | None = None,
) -> tuple:
    """Save and reload mock low-level artifacts through the real manifest helpers."""
    params = params or MockExperimentParameters()
    task_edges = build_mock_task_edges(params.n_subtasks)
    joint_states = generate_joint_states(params)
    initial_distribution = StateDistribution(
        joint_states,
        tuple(1.0 / len(joint_states) for _ in joint_states),
    )
    manifest_path = option_manifest_path(output_dir, seed)
    if manifest_path.is_file():
        manifest_path.unlink()
    learner = MockPolicyLearner()
    artifacts = []
    for edge in task_edges:
        for capacity in params.communication_capacities:
            artifact = save_option_policy_artifact(
                learner,
                output_dir,
                LowLevelPolicyArtifact(
                    seed=seed,
                    subtask_edge=edge,
                    communication_capacity=capacity,
                    initial_state_distribution=initial_distribution,
                    timestep=1000,
                    checkpoint_path=str(output_dir / "unused"),
                    config_hash="mock-fixed-policy-v1",
                ),
            )
            artifacts.append(artifact)

    return tuple(read_option_policy_manifest(output_dir, seed)), learner


def _mock_fixed_capacity_solution(
    mixed_success_rates,
    goal_probability_threshold: float,
    params: MockExperimentParameters | None = None,
):
    """Enumerate fixed edge-capacity and root-branch policies."""
    params = params or MockExperimentParameters()
    task_edges = build_mock_task_edges(params.n_subtasks)
    communication_capacities = params.communication_capacities
    branch_pairs = ((1.0, 0.0), (0.5, 0.5), (0.0, 1.0))
    capacity_combinations = itertools.product(
        communication_capacities,
        repeat=len(task_edges),
    )
    passing = []
    for selected_capacities in capacity_combinations:
        branch_options = (
            branch_pairs
            if task_edges == ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4))
            else ((1.0,),)
        )
        for branch_probability in branch_options:
            if len(branch_probability) == 1:
                success_probability = 1.0
                expected_cost = 0.0
                for edge, capacity in zip(
                    task_edges,
                    selected_capacities,
                    strict=True,
                ):
                    success_probability *= float(
                        mixed_success_rates[(edge, float(capacity))]
                    )
                    expected_cost += float(capacity)
                expected_cost /= len(task_edges)
                if success_probability + 1e-12 >= goal_probability_threshold:
                    passing.append(
                        (
                            expected_cost,
                            selected_capacities,
                            branch_probability,
                            success_probability,
                        )
                    )
                continue
            root_prob_one, root_prob_two = branch_probability
            capacity_by_edge = dict(
                zip(
                    task_edges,
                    selected_capacities,
                    strict=True,
                )
            )
            root_one_success = float(
                mixed_success_rates[((0, 1), capacity_by_edge[(0, 1)])]
            )
            root_two_success = float(
                mixed_success_rates[((0, 2), capacity_by_edge[(0, 2)])]
            )
            branch_one_flow = root_prob_one * root_one_success
            branch_two_flow = root_prob_two * root_two_success

            branch_one_success = float(
                mixed_success_rates[((1, 3), capacity_by_edge[(1, 3)])]
            )
            branch_two_success = float(
                mixed_success_rates[((2, 3), capacity_by_edge[(2, 3)])]
            )
            merge_flow = (
                branch_one_flow * branch_one_success
                + branch_two_flow * branch_two_success
            )
            final_success = float(
                mixed_success_rates[((3, 4), capacity_by_edge[(3, 4)])]
            )
            success_probability = merge_flow * final_success
            expected_cost = (
                root_prob_one * capacity_by_edge[(0, 1)]
                + root_prob_two * capacity_by_edge[(0, 2)]
                + branch_one_flow * capacity_by_edge[(1, 3)]
                + branch_two_flow * capacity_by_edge[(2, 3)]
                + merge_flow * capacity_by_edge[(3, 4)]
            ) / len(task_edges)
            if success_probability + 1e-12 >= goal_probability_threshold:
                passing.append(
                    (
                        expected_cost,
                        selected_capacities,
                        branch_probability,
                        success_probability,
                    )
                )

    if not passing:
        raise SynthesisInfeasibleError(
            f"No mock fixed-capacity policy reached {goal_probability_threshold:g}."
        )

    (
        expected_cost,
        selected_capacities,
        branch_probability,
        success_probability,
    ) = min(passing)
    root_probability_one = (
        branch_probability[0] if len(branch_probability) == 2 else 1.0
    )
    root_probability_two = (
        branch_probability[1] if len(branch_probability) == 2 else 1.0
    )
    task_rows = [
        {
            "state": edge[0],
            "next_state": edge[1],
            "occupancy": 1.0,
            "action_probability": (
                (
                    root_probability_one
                    if edge == (0, 1)
                    else root_probability_two
                    if edge == (0, 2)
                    else 1.0
                )
                if len(task_edges) == 5
                else 1.0
            ),
        }
        for edge in task_edges
    ]
    capacity_rows = []
    for edge, selected_capacity in zip(
        task_edges,
        selected_capacities,
        strict=True,
    ):
        for capacity in communication_capacities:
            capacity_rows.append(
                {
                    "state": edge[0],
                    "next_state": edge[1],
                    "msg_budget_per_agent": capacity,
                    "action_probability": (
                        1.0 if capacity == selected_capacity else 0.0
                    ),
                }
            )
    return type(
        "MockHighLevelSolution",
        (),
        {
            "task_policy": pd.DataFrame.from_records(task_rows),
            "comms_budget_policy": pd.DataFrame.from_records(capacity_rows),
            "objective_value": expected_cost,
            "goal_reach_probability": success_probability,
        },
    )()


def fixed_capacity_baseline_cost(
    *,
    task_edges: tuple[tuple[int, int], ...],
    goal_states: tuple[int, ...],
    communication_capacities: tuple[float, ...],
    root_initial_distribution: StateDistribution,
    calibration_rows: tuple[CalibrationRow, ...],
    pg: float,
    epsilon_p: float,
) -> float:
    """Return minimum normalized fixed-capacity cost that clears pg+epsilon_p."""
    passing = []
    merge_dag = task_edges == ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4))
    for capacity in communication_capacities:
        task_action_probabilities = {}
        capacity_action_probabilities = {}
        if merge_dag:
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
            capacity_action_probabilities[(edge[0], edge[1], float(capacity))] = 1.0

        propagated = propagate_initial_state_distributions(
            task_edges=task_edges,
            goal_states=goal_states,
            root_state=0,
            root_initial_distribution=root_initial_distribution,
            task_action_probabilities=task_action_probabilities,
            capacity_action_probabilities=capacity_action_probabilities,
            communication_capacities=(float(capacity),),
            calibration_rows=calibration_rows,
            cost_normalizer=len(task_edges),
        )
        if propagated.goal_probability + 1e-12 >= pg + epsilon_p:
            passing.append(propagated.expected_communication_cost)

    if not passing:
        return float("nan")
    return min(passing)


def run_mock_experiment(
    output_dir: str | Path,
    seed: int = 17,
    pg: float = 0.65,
    pg_values: Sequence[float] | None = None,
    epsilon_est: float = 0.05,
    beta: float = 0.95,
    epsilon_p: float = 0.10,
    n_estimation_episodes: int = 100,
    max_iterations: int = 3,
    params: MockExperimentParameters | None = None,
) -> dict:
    """Generate, save, reload, synthesize, and return a mock experiment summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = params or MockExperimentParameters()
    task_edges = build_mock_task_edges(params.n_subtasks)
    goal_state = task_edges[-1][1]
    joint_states = generate_joint_states(params)
    training_df = generate_mock_training_evaluations(seed, params)
    training_df.to_csv(
        output_dir / "mock_training_eval.csv",
        index=False,
    )
    plot_context_evaluation(
        training_df,
        str(output_dir / "mock_training_plots"),
        info_str=(
            f"seed={seed}; "
            f"{len(params.mock_evaluation_times)} mock evaluation checkpoints"
        ),
    )
    plot_success_vs_comms_by_subtask(
        training_df,
        str(output_dir / "mock_training_plots"),
    )
    l_setting_count = params.l_setting_count
    outcome_alphabet_size = 1 + len(joint_states)
    m_est = calculate_m_est(
        epsilon_est,
        beta,
        outcome_alphabet_size,
        l_setting_count,
    )
    delta_p = calculate_project_reliability_error(epsilon_est, n_max=2)
    epsilon_p = validate_project_screening_margin(
        epsilon_est,
        n_max=2,
        epsilon_p=epsilon_p,
    )

    calibration_rows = generate_mock_calibration_rows(
        seed,
        n_estimation_episodes,
        params,
    )
    calibration_path = save_calibration_manifest(
        output_dir,
        seed,
        calibration_rows,
    )
    loaded_calibration = read_calibration_manifest(output_dir, seed)
    if not loaded_calibration:
        raise RuntimeError("Generated mock calibration data did not reload.")

    artifact_manifest, learner = save_mock_option_library(
        output_dir,
        seed,
        params,
    )
    if not artifact_manifest:
        raise RuntimeError("Generated mock option library did not reload.")
    loaded_artifacts = []
    for artifact in artifact_manifest:
        loaded_artifacts.append(load_option_policy_artifact(learner, artifact))
    uniform_initial_distribution = StateDistribution(
        joint_states,
        tuple(1.0 / len(joint_states) for _ in joint_states),
    )
    latest_edge = task_edges[-1]
    latest = find_latest_option_policy_artifact(
        output_dir,
        seed=seed,
        subtask_edge=latest_edge,
        communication_capacity=params.communication_capacities[-1],
        initial_state_distribution=uniform_initial_distribution,
        config_hash="mock-fixed-policy-v1",
    )
    if latest is None:
        raise RuntimeError("Latest mock option-policy artifact was not found.")

    context_key = ContextKey(
        context_id="mock-chain",
        policy_scope=PolicyScope.SHARED_ACROSS_CONTEXTS,
        high_level_seed=seed,
    )

    synthesis = iterative_high_level_synthesis(
        task_edges=task_edges,
        goal_states=(goal_state,),
        root_state=0,
        root_initial_distribution=uniform_initial_distribution,
        communication_capacities=params.communication_capacities,
        calibration_rows=loaded_calibration,
        goal_probability_spec=pg,
        epsilon_p=epsilon_p,
        max_iterations=max_iterations,
        context_key=context_key,
        cost_normalizer=len(task_edges),
        solve_policy=lambda mixed_success_rates, threshold: (
            _mock_fixed_capacity_solution(
                mixed_success_rates,
                threshold,
                params,
            )
        ),
    )

    pg_values_used = tuple(
        sorted(
            {
                float(pg),
                *(float(value) for value in (pg_values or ())),
            }
        )
    )
    frontier_rows = []
    for frontier_pg in pg_values_used:
        frontier_synthesis = iterative_high_level_synthesis(
            task_edges=task_edges,
            goal_states=(goal_state,),
            root_state=0,
            root_initial_distribution=uniform_initial_distribution,
            communication_capacities=params.communication_capacities,
            calibration_rows=loaded_calibration,
            goal_probability_spec=frontier_pg,
            epsilon_p=epsilon_p,
            max_iterations=max_iterations,
            context_key=ContextKey(
                context_id="mock-chain",
                policy_scope=PolicyScope.SHARED_ACROSS_CONTEXTS,
                high_level_seed=seed,
            ),
            cost_normalizer=len(task_edges),
            solve_policy=lambda mixed_success_rates, threshold: (
                _mock_fixed_capacity_solution(
                    mixed_success_rates,
                    threshold,
                    params,
                )
            ),
        )
        ours_cost = (
            frontier_synthesis.selected_candidate.empirical_cost
            if frontier_synthesis.selected_candidate is not None
            else float("nan")
        )
        baseline_cost = fixed_capacity_baseline_cost(
            task_edges=task_edges,
            goal_states=(goal_state,),
            communication_capacities=params.communication_capacities,
            root_initial_distribution=uniform_initial_distribution,
            calibration_rows=loaded_calibration,
            pg=frontier_pg,
            epsilon_p=epsilon_p,
        )
        task_greedy_model, comms_greedy_model = greedy_project_baselines(
            task_edges=task_edges,
            goal_states=(goal_state,),
            root_state=0,
            root_initial_distribution=uniform_initial_distribution,
            communication_capacities=params.communication_capacities,
            calibration_rows=loaded_calibration,
            pg=frontier_pg,
            cost_normalizer=len(task_edges),
        )
        frontier_rows.append(
            {
                "pg": frontier_pg,
                "ours_cost": ours_cost,
                "baseline_cost": baseline_cost,
                "task_greedy_cost": (
                    task_greedy_model.expected_communication_cost
                ),
                "comms_greedy_cost": (
                    comms_greedy_model.expected_communication_cost
                ),
                "task_greedy_success": task_greedy_model.goal_probability,
                "comms_greedy_success": comms_greedy_model.goal_probability,
                "status": frontier_synthesis.status.value,
            }
        )

    frontier_df = pd.DataFrame.from_records(frontier_rows)
    frontier_csv = output_dir / f"success_vs_cost_seed_{seed}.csv"
    frontier_df.to_csv(frontier_csv, index=False)
    plot_success_vs_cost_frontier(
        frontier_df,
        str(output_dir / "mock_training_plots"),
        seed=seed,
    )

    candidate_rows = [
        {
            "iteration": candidate.iteration,
            "status": candidate.status.value,
            "empirical_goal_probability": candidate.empirical_goal_probability,
            "empirical_cost": candidate.empirical_cost,
        }
        for candidate in synthesis.candidates
    ]
    selected = (
        {
            "iteration": synthesis.selected_candidate.iteration,
            "empirical_goal_probability": (
                synthesis.selected_candidate.empirical_goal_probability
            ),
            "empirical_cost": synthesis.selected_candidate.empirical_cost,
            "status": synthesis.selected_candidate.status.value,
        }
        if synthesis.selected_candidate is not None
        else None
    )
    return {
        "output_dir": str(output_dir),
        "calibration_manifest": str(calibration_path),
        "option_artifact_count": len(loaded_artifacts),
        "latest_loaded_artifact": latest.checkpoint_path,
        "n_agents": params.n_agents,
        "n_final_positions": params.n_final_positions,
        "n_final_joint_states": len(joint_states),
        "n_comms_values": params.n_comms_values,
        "communication_capacities": list(params.communication_capacities),
        "n_subtasks": params.n_subtasks,
        "l_setting_count": l_setting_count,
        "outcome_alphabet_size": outcome_alphabet_size,
        "mock_evaluation_times": list(params.mock_evaluation_times),
        "mock_training_eval_csv": str(output_dir / "mock_training_eval.csv"),
        "m_est_theoretical": m_est,
        "delta_p": delta_p,
        "epsilon_p": epsilon_p,
        "pg": pg,
        "screen_threshold": pg + epsilon_p,
        "status": synthesis.status.value,
        "candidates": candidate_rows,
        "selected": selected,
        "pg_values": list(pg_values_used),
        "success_vs_cost_csv": str(frontier_csv),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a mock fixed-option-library iterative synthesis experiment."
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/mock_hierarchical_synthesis",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pg", type=float, default=0.65)
    parser.add_argument(
        "--pg-values",
        nargs="+",
        type=float,
        default=None,
        help="Mission success specifications for the reliability-cost frontier.",
    )
    parser.add_argument("--epsilon-est", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--epsilon-p", type=float, default=0.10)
    parser.add_argument("--n-estimation-episodes", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--n-final-positions", type=int, default=4)
    parser.add_argument("--n-comms-values", type=int, default=2)
    parser.add_argument("--n-subtasks", type=int, default=5)
    parser.add_argument(
        "--mock-evaluation-times",
        nargs="+",
        type=int,
        default=[20_000, 40_000, 60_000, 80_000, 100_000],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = MockExperimentParameters(
        n_agents=args.n_agents,
        n_final_positions=args.n_final_positions,
        n_comms_values=args.n_comms_values,
        n_subtasks=args.n_subtasks,
        mock_evaluation_times=tuple(args.mock_evaluation_times),
    )
    summary = run_mock_experiment(
        output_dir=args.output_dir,
        seed=args.seed,
        pg=args.pg,
        pg_values=args.pg_values,
        epsilon_est=args.epsilon_est,
        beta=args.beta,
        epsilon_p=args.epsilon_p,
        n_estimation_episodes=args.n_estimation_episodes,
        max_iterations=args.iterations,
        params=params,
    )
    print(summary)


if __name__ == "__main__":
    main()
