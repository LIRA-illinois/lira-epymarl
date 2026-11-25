"""Finite-data calibration and mixing utilities for hierarchical options."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.simulation.hierarchical_types import (
    CalibrationOutcome,
    CalibrationOutcomeKind,
    CalibrationRow,
    OptionSettingKey,
    StateDistribution,
    normalize_state,
)


def calculate_m_est(
    epsilon_est: float,
    beta: float,
    outcome_alphabet_size: int,
    statewise_setting_count: int,
) -> int:
    """Compute the paper's independent-episode count per statewise setting."""
    epsilon_est = float(epsilon_est)
    beta = float(beta)
    if not 0.0 < epsilon_est < 1.0:
        raise ValueError("epsilon_est must be strictly between 0.0 and 1.0.")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be strictly between 0.0 and 1.0.")
    if not isinstance(outcome_alphabet_size, int) or outcome_alphabet_size < 1:
        raise TypeError("outcome_alphabet_size must be a positive integer.")
    if not isinstance(statewise_setting_count, int) or statewise_setting_count < 1:
        raise TypeError("statewise_setting_count must be a positive integer.")

    logarithm = math.log(
        (2.0 ** (outcome_alphabet_size + 1)) * statewise_setting_count / (1.0 - beta)
    )
    return math.ceil(logarithm / (2.0 * epsilon_est**2))


def calculate_project_reliability_error(
    epsilon_est: float,
    n_max: int,
) -> float:
    """Compute Delta_p = 1 - (1 - epsilon_est)^N_max."""
    epsilon_est = float(epsilon_est)
    if not 0.0 < epsilon_est < 1.0:
        raise ValueError("epsilon_est must be strictly between 0.0 and 1.0.")
    if not isinstance(n_max, int) or n_max < 1:
        raise TypeError("n_max must be a positive integer.")
    return 1.0 - (1.0 - epsilon_est) ** n_max


def calculate_project_cost_error(
    epsilon_est: float,
    n_max: int,
    maximum_capacity: float,
    cost_normalizer: int = 1,
) -> float:
    """Compute Delta_J = C_max * Delta_p."""
    maximum_capacity = float(maximum_capacity)
    if not math.isfinite(maximum_capacity) or maximum_capacity < 0.0:
        raise ValueError("maximum_capacity must be finite and non-negative.")
    if not isinstance(cost_normalizer, int) or cost_normalizer < 1:
        raise TypeError("cost_normalizer must be a positive integer.")
    delta_p = calculate_project_reliability_error(epsilon_est, n_max)
    c_max = (n_max / cost_normalizer) * maximum_capacity
    return c_max * delta_p


def validate_project_screening_margin(
    epsilon_est: float,
    n_max: int,
    epsilon_p: float,
) -> float:
    """Validate and return epsilon_p when epsilon_p >= Delta_p."""
    delta_p = calculate_project_reliability_error(epsilon_est, n_max)
    epsilon_p = float(epsilon_p)
    if not math.isfinite(epsilon_p) or epsilon_p < 0.0:
        raise ValueError("epsilon_p must be finite and non-negative.")
    if epsilon_p + 1e-12 < delta_p:
        raise ValueError(f"epsilon_p={epsilon_p} must be at least Delta_p={delta_p}.")
    return epsilon_p


def enumerate_statewise_settings(
    task_edges: Sequence[tuple[int, int]],
    communication_capacities: Sequence[float],
    initial_support_by_source: Mapping[int, Sequence[object]],
) -> tuple[OptionSettingKey, ...]:
    """Enumerate every (subtask edge, capacity, source physical start) setting."""
    settings: list[OptionSettingKey] = []
    seen: set[OptionSettingKey] = set()
    capacities = tuple(float(capacity) for capacity in communication_capacities)
    if not capacities:
        raise ValueError("At least one communication capacity is required.")

    for edge_value in task_edges:
        edge = (int(edge_value[0]), int(edge_value[1]))
        if edge[0] not in initial_support_by_source:
            raise ValueError(
                f"No initial-state support is configured for source state {edge[0]}."
            )
        supports = initial_support_by_source[edge[0]]
        if not supports:
            raise ValueError(
                f"Initial-state support for source state {edge[0]} cannot be empty."
            )
        for capacity in capacities:
            for start_state in supports:
                setting = OptionSettingKey(
                    subtask_edge=edge,
                    communication_capacity=capacity,
                    start_state=normalize_state(start_state),
                )
                if setting not in seen:
                    seen.add(setting)
                    settings.append(setting)
    return tuple(settings)


def total_variation_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    """Return half the L1 distance for two categorical distributions."""
    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys
    )


def mix_statewise_option_estimates(
    calibration_rows: Sequence[CalibrationRow],
    subtask_edge: tuple[int, int],
    communication_capacity: float,
    initial_state_distribution: StateDistribution,
) -> tuple[float, StateDistribution]:
    """Mix statewise estimates under an arbitrary option initial distribution."""
    edge = (int(subtask_edge[0]), int(subtask_edge[1]))
    capacity = float(communication_capacity)
    rows_by_setting: dict[tuple, CalibrationRow] = {}

    for row in calibration_rows:
        setting = row.setting
        if setting.subtask_edge != edge:
            continue
        if not np.isclose(setting.communication_capacity, capacity):
            continue
        key = setting.start_state
        if key in rows_by_setting:
            raise ValueError(
                f"Duplicate calibration row for option setting "
                f"{setting.subtask_edge}, capacity={capacity:g}, "
                f"start_state={setting.start_state}."
            )
        rows_by_setting[key] = row

    mixed_success = 0.0
    terminal_masses: dict[tuple, float] = {}
    for start_probability, start_state in zip(
        initial_state_distribution.probabilities,
        initial_state_distribution.states,
        strict=True,
    ):
        if start_probability == 0.0:
            continue
        key = start_state
        row = rows_by_setting.get(key)
        if row is None:
            raise ValueError(
                f"Missing calibration row for option setting "
                f"{edge}, capacity={capacity:g}, start_state={start_state}."
            )
        state_success = start_probability * row.success_rate
        mixed_success += state_success

        terminal_distribution = row.terminal_state_distribution
        for terminal_probability, terminal_state in zip(
            terminal_distribution.probabilities,
            terminal_distribution.states,
            strict=True,
        ):
            terminal_masses[terminal_state] = (
                terminal_masses.get(terminal_state, 0.0)
                + state_success * terminal_probability
            )

    if mixed_success == 0.0:
        return 0.0, StateDistribution.empty()

    if not math.isclose(sum(terminal_masses.values()), mixed_success):
        raise RuntimeError(
            "Mixed terminal-state mass does not match the mixed success probability."
        )
    terminal_states = tuple(terminal_masses)
    terminal_probabilities = tuple(
        terminal_masses[state] / mixed_success for state in terminal_states
    )
    return mixed_success, StateDistribution(
        terminal_states,
        terminal_probabilities,
    )


def _calibration_row_to_record(row: CalibrationRow) -> dict[str, Any]:
    outcomes = []
    for outcome in row.outcomes:
        outcomes.append(
            {
                "kind": outcome.kind.value,
                "terminal_state": (
                    list(outcome.terminal_state)
                    if outcome.terminal_state is not None
                    else None
                ),
            }
        )
    return {
        "subtask_edge": list(row.setting.subtask_edge),
        "communication_capacity": row.setting.communication_capacity,
        "start_state": list(row.setting.start_state),
        "seed": row.seed,
        "outcomes": outcomes,
    }


def _calibration_row_from_record(record: Mapping[str, Any]) -> CalibrationRow:
    edge = record["subtask_edge"]
    if not isinstance(edge, list) or len(edge) != 2:
        raise ValueError("Calibration record subtask_edge must contain two states.")
    outcomes = []
    for outcome_record in record["outcomes"]:
        kind = CalibrationOutcomeKind(outcome_record["kind"])
        if kind == CalibrationOutcomeKind.FAILURE:
            outcomes.append(CalibrationOutcome.failure())
        else:
            outcomes.append(
                CalibrationOutcome.success(tuple(outcome_record["terminal_state"]))
            )
    return CalibrationRow(
        setting=OptionSettingKey(
            subtask_edge=(int(edge[0]), int(edge[1])),
            communication_capacity=float(record["communication_capacity"]),
            start_state=tuple(record["start_state"]),
        ),
        outcomes=tuple(outcomes),
        seed=int(record["seed"]),
    )


def calibration_manifest_path(root: str | Path, seed: int) -> Path:
    """Return the JSON calibration manifest path for one seed."""
    return Path(root) / "option_calibration" / f"seed_{int(seed)}" / "calibration.json"


def build_calibration_report_row(
    row: CalibrationRow,
    *,
    epsilon_est: float,
    beta: float,
    m_est: int,
    statewise_setting_count: int,
) -> dict[str, object]:
    """Report one statewise calibration row and finite-data context."""
    n_episodes = len(row.outcomes)
    return {
        "seed": row.seed,
        "subtask_edge": list(row.setting.subtask_edge),
        "communication_capacity": row.setting.communication_capacity,
        "start_state": list(row.setting.start_state),
        "n_episodes": n_episodes,
        "success_count": row.success_count,
        "failure_count": row.failure_count,
        "empirical_success_rate": row.success_rate,
        "epsilon_est": float(epsilon_est),
        "beta": float(beta),
        "m_est": int(m_est),
        "statewise_setting_count": int(statewise_setting_count),
        "certificate_applicable": bool(n_episodes >= m_est),
    }


def save_calibration_manifest(
    root: str | Path,
    seed: int,
    calibration_rows: Sequence[CalibrationRow],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically write a seed-isolated categorical calibration manifest."""
    if not calibration_rows:
        raise ValueError("A calibration manifest cannot be empty.")
    for row in calibration_rows:
        if row.seed != int(seed):
            raise ValueError(
                f"Calibration row seed {row.seed} does not match manifest seed {seed}."
            )
    manifest_path = calibration_manifest_path(root, seed)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": int(seed),
        "metadata": dict(metadata or {}),
        "rows": [_calibration_row_to_record(row) for row in calibration_rows],
    }
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return manifest_path


def read_calibration_manifest(
    root: str | Path,
    seed: int,
) -> tuple[CalibrationRow, ...]:
    """Read a seed-isolated categorical calibration manifest."""
    manifest_path = calibration_manifest_path(root, seed)
    if not manifest_path.is_file():
        return ()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload["seed"]) != int(seed):
        raise ValueError(
            f"Calibration manifest seed {payload['seed']} "
            f"does not match requested seed {seed}."
        )
    return tuple(
        _calibration_row_from_record(record) for record in payload.get("rows", [])
    )


def read_calibration_metadata(
    root: str | Path,
    seed: int,
) -> dict[str, object]:
    """Read seed-isolated finite-data calibration metadata."""
    manifest_path = calibration_manifest_path(root, seed)
    if not manifest_path.is_file():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload["seed"]) != int(seed):
        raise ValueError(
            f"Calibration manifest seed {payload['seed']} "
            f"does not match requested seed {seed}."
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("Calibration manifest metadata must be a mapping.")
    return dict(metadata)


def merge_weighted_state_distributions(
    distributions: Sequence[StateDistribution],
    weights: Sequence[float],
) -> StateDistribution:
    """Merge state distributions weighted by successful project flow."""
    if len(distributions) != len(weights):
        raise ValueError("State distributions and weights must have the same length.")
    weighted_masses: dict[tuple, float] = {}
    total_weight = 0.0
    for distribution, weight in zip(distributions, weights, strict=True):
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                "State-distribution weights must be finite and non-negative."
            )
        if weight == 0.0:
            continue
        total_weight += weight
        for state, probability in zip(
            distribution.states,
            distribution.probabilities,
            strict=True,
        ):
            weighted_masses[state] = (
                weighted_masses.get(state, 0.0) + weight * probability
            )
    if total_weight == 0.0 or not weighted_masses:
        return StateDistribution.empty()

    normalized_states = tuple(weighted_masses)
    normalized_probabilities = tuple(
        weighted_masses[state] / total_weight for state in normalized_states
    )
    return StateDistribution(normalized_states, normalized_probabilities)


@dataclass(frozen=True)
class PropagatedProjectModel:
    """Candidate-induced empirical project flow and successor starts."""

    initial_state_distributions: dict[int, StateDistribution]
    state_flows: dict[int, float]
    edge_success_rates: dict[tuple[tuple[int, int], float], float]
    edge_terminal_distributions: dict[
        tuple[tuple[int, int], float],
        StateDistribution,
    ]
    goal_probability: float
    expected_communication_cost: float


def propagate_initial_state_distributions(
    *,
    task_edges: Sequence[tuple[int, int]],
    goal_states: Sequence[int],
    root_state: int,
    root_initial_distribution: StateDistribution,
    task_action_probabilities: Mapping[tuple[int, int], float],
    capacity_action_probabilities: Mapping[
        tuple[int, int, float],
        float,
    ],
    communication_capacities: Sequence[float],
    calibration_rows: Sequence[CalibrationRow],
    cost_normalizer: int = 1,
) -> PropagatedProjectModel:
    """Propagate one candidate's empirical flow through a project DAG."""
    edges = tuple((int(edge[0]), int(edge[1])) for edge in task_edges)
    if not edges:
        raise ValueError("At least one task edge is required.")
    if len(set(edges)) != len(edges):
        raise ValueError("Task edges must be unique.")
    if any(edge[0] == edge[1] for edge in edges):
        raise ValueError("Task edges cannot contain self-loops.")

    goals = {int(state) for state in goal_states}
    if not goals:
        raise ValueError("At least one goal state is required.")
    if not isinstance(cost_normalizer, int) or cost_normalizer < 1:
        raise TypeError("cost_normalizer must be a positive integer.")

    outgoing: dict[int, list[tuple[int, int]]] = {}
    incoming: dict[int, list[tuple[int, int]]] = {}
    indegree: dict[int, int] = {}
    states = {root_state}
    for edge in edges:
        outgoing.setdefault(edge[0], []).append(edge)
        incoming.setdefault(edge[1], []).append(edge)
        indegree[edge[1]] = indegree.get(edge[1], 0) + 1
        indegree.setdefault(edge[0], 0)
        states.update(edge)

    ready = [state for state in states if indegree.get(state, 0) == 0]
    topological: list[int] = []
    indegree_remaining = dict(indegree)
    while ready:
        state = ready.pop()
        topological.append(state)
        for edge in outgoing.get(state, []):
            successor = edge[1]
            indegree_remaining[successor] -= 1
            if indegree_remaining[successor] == 0:
                ready.append(successor)
    if len(topological) != len(states):
        raise ValueError("Task edges must describe a directed acyclic graph.")

    state_flows = {state: 0.0 for state in states}
    state_flows[root_state] = 1.0
    initial_state_distributions = {
        root_state: root_initial_distribution,
    }
    incoming_terminal_distributions: dict[
        int,
        list[tuple[StateDistribution, float]],
    ] = {}
    edge_success_rates: dict[
        tuple[tuple[int, int], float],
        float,
    ] = {}
    edge_terminal_distributions: dict[
        tuple[tuple[int, int], float],
        StateDistribution,
    ] = {}

    goal_probability = 0.0
    expected_communication_cost = 0.0
    for state in topological:
        distributions_with_weights = incoming_terminal_distributions.get(state)
        if distributions_with_weights:
            distributions = [
                distribution for distribution, _ in distributions_with_weights
            ]
            weights = [weight for _, weight in distributions_with_weights]
            initial_state_distributions[state] = merge_weighted_state_distributions(
                distributions, weights
            )
        source_flow = state_flows[state]
        if state not in initial_state_distributions:
            initial_state_distributions[state] = StateDistribution.empty()
        source_initial_distribution = initial_state_distributions[state]

        if source_flow == 0.0:
            continue
        for edge in outgoing.get(state, []):
            task_probability = float(task_action_probabilities.get(edge, 0.0))
            if not 0.0 <= task_probability <= 1.0:
                raise ValueError(
                    f"Task action probability for {edge} must be in [0, 1]."
                )
            if task_probability == 0.0:
                continue
            for capacity in communication_capacities:
                capacity = float(capacity)
                capacity_probability = float(
                    capacity_action_probabilities.get(
                        (edge[0], edge[1], capacity),
                        0.0,
                    )
                )
                if not 0.0 <= capacity_probability <= 1.0:
                    raise ValueError(
                        f"Capacity probability for {edge} at {capacity:g} "
                        "must be in [0, 1]."
                    )
                if capacity_probability == 0.0:
                    continue

                mixed_success, terminal_distribution = mix_statewise_option_estimates(
                    calibration_rows,
                    edge,
                    capacity,
                    source_initial_distribution,
                )
                edge_key = (edge, capacity)
                edge_success_rates[edge_key] = mixed_success
                edge_terminal_distributions[edge_key] = terminal_distribution

                edge_success_flow = (
                    source_flow
                    * task_probability
                    * capacity_probability
                    * mixed_success
                )
                destination = edge[1]
                if destination in goals:
                    goal_probability += edge_success_flow
                else:
                    state_flows[destination] += edge_success_flow

                if edge_success_flow > 0.0 and terminal_distribution.states:
                    incoming_terminal_distributions.setdefault(destination, []).append(
                        (terminal_distribution, edge_success_flow)
                    )

                expected_communication_cost += (
                    source_flow * task_probability * capacity_probability * capacity
                )

    for state, distributions_with_weights in incoming_terminal_distributions.items():
        distributions = [distribution for distribution, _ in distributions_with_weights]
        weights = [weight for _, weight in distributions_with_weights]
        initial_state_distributions[state] = merge_weighted_state_distributions(
            distributions,
            weights,
        )

    for state in states:
        if state not in initial_state_distributions:
            if state == root_state:
                initial_state_distributions[state] = root_initial_distribution
            else:
                initial_state_distributions[state] = StateDistribution.empty()

    if not math.isfinite(goal_probability):
        raise RuntimeError("Propagated goal probability must be finite.")
    if not math.isfinite(expected_communication_cost):
        raise RuntimeError("Propagated communication cost must be finite.")
    goal_probability = float(np.clip(goal_probability, 0.0, 1.0))
    normalized_cost = expected_communication_cost / cost_normalizer

    return PropagatedProjectModel(
        initial_state_distributions=initial_state_distributions,
        state_flows={state: float(flow) for state, flow in state_flows.items()},
        edge_success_rates=edge_success_rates,
        edge_terminal_distributions=edge_terminal_distributions,
        goal_probability=goal_probability,
        expected_communication_cost=normalized_cost,
    )


def greedy_project_baselines(
    *,
    task_edges: Sequence[tuple[int, int]],
    goal_states: Sequence[int],
    root_state: int,
    root_initial_distribution: StateDistribution,
    communication_capacities: Sequence[float],
    calibration_rows: Sequence[CalibrationRow],
    pg: float,
    cost_normalizer: int = 1,
) -> tuple[PropagatedProjectModel, PropagatedProjectModel]:
    """Evaluate task-greedy and communication-greedy high-level policies.

    Task greedy chooses the outgoing edge with the largest mixed success at
    maximum communication and uses maximum communication on that edge.

    Communication greedy uses the same edge choice, but selects the lowest
    communication capacity whose mixed success reaches a per-edge threshold
    derived from the mission success specification.
    """
    pg = float(pg)
    if not math.isfinite(pg) or not 0.0 <= pg <= 1.0:
        raise ValueError("pg must be between 0.0 and 1.0.")
    capacities = tuple(sorted({float(capacity) for capacity in communication_capacities}))
    if not capacities:
        raise ValueError("Communication capacities are required.")
    if any(not math.isfinite(capacity) or capacity < 0.0 for capacity in capacities):
        raise ValueError("Communication capacities must be finite and non-negative.")

    edges = tuple((int(edge[0]), int(edge[1])) for edge in task_edges)
    if not edges:
        raise ValueError("At least one task edge is required.")
    goals = {int(state) for state in goal_states}
    if not goals:
        raise ValueError("At least one goal state is required.")

    edge_threshold = pg ** (1.0 / max(1, len(edges)))
    edge_threshold = float(np.clip(edge_threshold, 0.0, 1.0))

    def evaluate(policy_kind: str) -> PropagatedProjectModel:
        outgoing: dict[int, list[tuple[int, int]]] = {}
        indegree: dict[int, int] = {}
        states = {root_state}
        for edge in edges:
            outgoing.setdefault(edge[0], []).append(edge)
            indegree[edge[1]] = indegree.get(edge[1], 0) + 1
            indegree.setdefault(edge[0], 0)
            states.update(edge)

        ready = [state for state in states if indegree.get(state, 0) == 0]
        topological = []
        indegree_remaining = dict(indegree)
        while ready:
            state = ready.pop()
            topological.append(state)
            for edge in outgoing.get(state, []):
                successor = edge[1]
                indegree_remaining[successor] -= 1
                if indegree_remaining[successor] == 0:
                    ready.append(successor)
        if len(topological) != len(states):
            raise ValueError("Task edges must describe a directed acyclic graph.")

        state_flows = {state: 0.0 for state in states}
        state_flows[root_state] = 1.0
        initial_distributions = {root_state: root_initial_distribution}
        incoming_terminal_distributions: dict[
            int,
            list[tuple[StateDistribution, float]],
        ] = {}
        edge_success_rates: dict[
            tuple[tuple[int, int], float],
            float,
        ] = {}
        edge_terminal_distributions: dict[
            tuple[tuple[int, int], float],
            StateDistribution,
        ] = {}
        goal_probability = 0.0
        expected_communication_cost = 0.0

        for state in topological:
            distributions_with_weights = incoming_terminal_distributions.get(state)
            if distributions_with_weights:
                distributions = [
                    distribution
                    for distribution, _ in distributions_with_weights
                ]
                weights = [
                    weight
                    for _, weight in distributions_with_weights
                ]
                initial_distributions[state] = merge_weighted_state_distributions(
                    distributions,
                    weights,
                )
            source_flow = state_flows[state]
            if source_flow == 0.0:
                continue
            source_distribution = initial_distributions.get(
                state,
                StateDistribution.empty(),
            )

            candidates = []
            for edge in outgoing.get(state, []):
                edge_success_by_capacity = []
                for capacity in capacities:
                    mixed_success, _ = mix_statewise_option_estimates(
                        calibration_rows,
                        edge,
                        capacity,
                        source_distribution,
                    )
                    edge_success_by_capacity.append(
                        (mixed_success, capacity)
                    )
                best_success = max(
                    success
                    for success, _ in edge_success_by_capacity
                )
                best_capacity = max(
                    capacity
                    for success, capacity in edge_success_by_capacity
                    if math.isclose(success, best_success)
                )
                candidates.append(
                    (best_success, edge, best_capacity, edge_success_by_capacity)
                )

            if not candidates:
                continue
            _, chosen_edge, _, edge_success_by_capacity = min(
                candidates,
                key=lambda item: (
                    -item[0],
                    item[1],
                ),
            )

            if policy_kind == "task_greedy":
                chosen_capacity = max(capacities)
            else:
                passing = [
                    (success, capacity)
                    for success, capacity in edge_success_by_capacity
                    if success >= edge_threshold
                ]
                chosen_capacity = (
                    min(capacity for _, capacity in passing)
                    if passing
                    else max(capacities)
                )

            mixed_success, terminal_distribution = (
                mix_statewise_option_estimates(
                    calibration_rows,
                    chosen_edge,
                    chosen_capacity,
                    source_distribution,
                )
            )
            edge_key = (chosen_edge, chosen_capacity)
            edge_success_rates[edge_key] = mixed_success
            edge_terminal_distributions[edge_key] = terminal_distribution

            edge_success_flow = source_flow * mixed_success
            destination = chosen_edge[1]
            if destination in goals:
                goal_probability += edge_success_flow
            else:
                state_flows[destination] += edge_success_flow
            if edge_success_flow > 0.0 and terminal_distribution.states:
                incoming_terminal_distributions.setdefault(destination, []).append(
                    (terminal_distribution, edge_success_flow)
                )
            expected_communication_cost += source_flow * chosen_capacity

        goal_probability = float(np.clip(goal_probability, 0.0, 1.0))
        normalized_cost = expected_communication_cost / max(1, cost_normalizer)
        for state in states:
            if state not in initial_distributions:
                if state == root_state:
                    initial_distributions[state] = root_initial_distribution
                else:
                    initial_distributions[state] = StateDistribution.empty()
        return PropagatedProjectModel(
            initial_state_distributions={
                state: distribution
                for state, distribution in initial_distributions.items()
            },
            state_flows={state: float(flow) for state, flow in state_flows.items()},
            edge_success_rates=edge_success_rates,
            edge_terminal_distributions=edge_terminal_distributions,
            goal_probability=goal_probability,
            expected_communication_cost=normalized_cost,
        )

    return (
        evaluate("task_greedy"),
        evaluate("comms_greedy"),
    )
