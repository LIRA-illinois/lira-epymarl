"""Iterative high-level synthesis over a frozen statewise calibration model."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src.simulation.hierarchical_calibration import (
    mix_statewise_option_estimates,
    propagate_initial_state_distributions,
)
from src.simulation.hierarchical_types import (
    CalibrationRow,
    CertificateStatus,
    ContextKey,
    HighLevelCandidate,
    StateDistribution,
)


class SynthesisInfeasibleError(RuntimeError):
    """Raised by a high-level solver when a tightened threshold has no solution."""


@dataclass(frozen=True)
class IterativeSynthesisResult:
    """All candidates and the final screening result for one seed/context."""

    candidates: tuple[HighLevelCandidate, ...]
    status: CertificateStatus
    selected_candidate: HighLevelCandidate | None
    search_error: str | None = None
    caveats: tuple[str, ...] = (
        "Iterative synthesis does not establish global optimality.",
        (
            "Terminal convergence/non-progression is evaluated by iteration "
            "bound and is not a proof of no-feasible-policy."
        ),
    )


def extract_solution_action_probabilities(
    solution,
) -> tuple[
    dict[tuple[int, int], float],
    dict[tuple[int, int, float], float],
]:
    """Extract task and capacity probabilities from an ILP-style solution."""
    task_policy = getattr(solution, "task_policy", None)
    comms_policy = getattr(solution, "comms_budget_policy", None)
    if task_policy is None or comms_policy is None:
        raise ValueError(
            "High-level solution must expose task_policy and comms_budget_policy."
        )

    task_probabilities: dict[tuple[int, int], float] = {}
    for row in task_policy.to_dict(orient="records"):
        edge = (int(row["state"]), int(row["next_state"]))
        probability = float(row.get("action_probability", 0.0))
        task_probabilities[edge] = task_probabilities.get(edge, 0.0) + probability

    capacity_probabilities: dict[tuple[int, int, float], float] = {}
    for row in comms_policy.to_dict(orient="records"):
        edge = (int(row["state"]), int(row["next_state"]))
        capacity = float(row["msg_budget_per_agent"])
        probability = float(row.get("action_probability", 0.0))
        key = (edge[0], edge[1], capacity)
        capacity_probabilities[key] = capacity_probabilities.get(key, 0.0) + probability

    return task_probabilities, capacity_probabilities


def apply_mixed_success_rates(
    hlmdp,
    mixed_success_rates: Mapping[tuple[tuple[int, int], float], float],
) -> None:
    """Update the HLMDP transition table for one fixed mixed-outcome model."""
    transition_probs = hlmdp.transition_probs.copy(deep=True)
    known_edges = set()
    for (edge, capacity), success_rate in mixed_success_rates.items():
        edge = (int(edge[0]), int(edge[1]))
        capacity = float(capacity)
        success_rate = float(success_rate)
        if not 0.0 <= success_rate <= 1.0:
            raise ValueError(
                f"Mixed success rate for {edge} at capacity {capacity:g} "
                "must be in [0, 1]."
            )
        known_edges.add((edge, capacity))
        matching_rows = transition_probs.loc[
            (transition_probs.state == edge[0])
            & transition_probs.action.apply(
                lambda action, target_state=edge[1], selected_capacity=capacity: (
                    int(action[0]) == target_state
                    and np.isclose(float(action[1]), selected_capacity)
                )
            )
        ]
        if len(matching_rows) != 2:
            raise ValueError(
                f"Expected success and failure transitions for task "
                f"{edge[0]}->{edge[1]} at capacity {capacity:g}; "
                f"found {len(matching_rows)}."
            )
        for row_index in matching_rows.index:
            next_state = int(transition_probs.at[row_index, "next_state"])
            if next_state == edge[1]:
                transition_probs.at[row_index, "prob"] = success_rate
            elif next_state == hlmdp.fail_state:
                transition_probs.at[row_index, "prob"] = 1.0 - success_rate
            else:
                raise ValueError(
                    f"Unexpected transition outcome {next_state} for "
                    f"task {edge[0]}->{edge[1]}."
                )
    if not known_edges:
        raise ValueError("Mixed success rates cannot be empty.")
    if hasattr(hlmdp, "_transition_probs"):
        hlmdp._transition_probs = transition_probs
    else:
        hlmdp.transition_probs = transition_probs


def solve_mixed_project_policy(
    args,
    hlmdp,
    mixed_success_rates: Mapping[
        tuple[tuple[int, int], float],
        float,
    ],
    success_threshold: float,
):
    """Apply mixed rates and solve once with the existing ILPModel."""
    from src.modules.agents.ilp_model import ILPModel, InfeasiblePolicyError

    apply_mixed_success_rates(hlmdp, mixed_success_rates)
    optimizer = ILPModel(args)
    try:
        return optimizer.optimize_policy(
            hlmdp,
            success_rate_spec=float(success_threshold),
        )
    except InfeasiblePolicyError as error:
        raise SynthesisInfeasibleError(
            f"No high-level policy can meet success threshold {success_threshold:g}."
        ) from error
    finally:
        optimizer.dispose()


def iterative_high_level_synthesis(
    *,
    task_edges: Sequence[tuple[int, int]],
    goal_states: Sequence[int],
    root_state: int,
    root_initial_distribution: StateDistribution,
    communication_capacities: Sequence[float],
    calibration_rows: Sequence[CalibrationRow],
    goal_probability_spec: float,
    epsilon_p: float,
    max_iterations: int,
    context_key: ContextKey,
    cost_normalizer: int = 1,
    solve_policy: Callable[
        [
            Mapping[tuple[tuple[int, int], float], float],
            float,
        ],
        object | None,
    ],
) -> IterativeSynthesisResult:
    """Run the paper's fixed-library alternating candidate/screening procedure."""
    goal_probability_spec = float(goal_probability_spec)
    epsilon_p = float(epsilon_p)
    if not 0.0 <= goal_probability_spec <= 1.0:
        raise ValueError("goal_probability_spec must be between 0.0 and 1.0.")
    if not np.isfinite(epsilon_p) or epsilon_p < 0.0:
        raise ValueError("epsilon_p must be finite and non-negative.")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if not isinstance(cost_normalizer, int) or cost_normalizer < 1:
        raise TypeError("cost_normalizer must be a positive integer.")

    edges = tuple((int(edge[0]), int(edge[1])) for edge in task_edges)
    capacities = tuple(float(capacity) for capacity in communication_capacities)
    if not edges or not capacities:
        raise ValueError("Task edges and communication capacities are required.")

    current_initial_distributions = {
        root_state: root_initial_distribution,
    }
    outgoing_edges: dict[int, list[tuple[int, int]]] = {}
    for edge in edges:
        outgoing_edges.setdefault(edge[0], []).append(edge)
    initial_task_action_probabilities: dict[tuple[int, int], float] = {}
    initial_capacity_action_probabilities: dict[
        tuple[int, int, float],
        float,
    ] = {}
    for source_state, source_edges in outgoing_edges.items():
        task_probability = 1.0 / len(source_edges)
        for edge in source_edges:
            initial_task_action_probabilities[edge] = task_probability
            for capacity in capacities:
                initial_capacity_action_probabilities[(edge[0], edge[1], capacity)] = (
                    1.0 / len(capacities)
                )
    try:
        initial_propagation = propagate_initial_state_distributions(
            task_edges=edges,
            goal_states=tuple(int(state) for state in goal_states),
            root_state=root_state,
            root_initial_distribution=root_initial_distribution,
            task_action_probabilities=initial_task_action_probabilities,
            capacity_action_probabilities=initial_capacity_action_probabilities,
            communication_capacities=capacities,
            calibration_rows=calibration_rows,
            cost_normalizer=cost_normalizer,
        )
    except (RuntimeError, ValueError) as error:
        return IterativeSynthesisResult(
            candidates=(),
            status=CertificateStatus.SEARCH_FAILED,
            selected_candidate=None,
            search_error=f"{type(error).__name__}: {error}",
        )
    current_initial_distributions = initial_propagation.initial_state_distributions
    candidates: list[HighLevelCandidate] = []

    for iteration in range(max_iterations):
        mixed_success_rates: dict[
            tuple[tuple[int, int], float],
            float,
        ] = {}

        for edge in edges:
            source_state = edge[0]
            source_distribution = current_initial_distributions.get(
                source_state,
                StateDistribution.empty(),
            )
            for capacity in capacities:
                mixed_success, _ = mix_statewise_option_estimates(
                    calibration_rows,
                    edge,
                    capacity,
                    source_distribution,
                )
                mixed_success_rates[(edge, capacity)] = mixed_success

        try:
            certified_threshold = min(
                1.0,
                goal_probability_spec + epsilon_p,
            )
            solution = solve_policy(
                mixed_success_rates,
                certified_threshold,
            )
        except SynthesisInfeasibleError:
            return IterativeSynthesisResult(
                candidates=tuple(candidates),
                status=CertificateStatus.INFEASIBLE,
                selected_candidate=None,
            )
        except RuntimeError as error:
            return IterativeSynthesisResult(
                candidates=tuple(candidates),
                status=CertificateStatus.SEARCH_FAILED,
                selected_candidate=None,
                search_error=f"{type(error).__name__}: {error}",
            )

        if solution is None:
            break

        (
            task_action_probabilities,
            capacity_action_probabilities,
        ) = extract_solution_action_probabilities(solution)
        try:
            propagated = propagate_initial_state_distributions(
                task_edges=edges,
                goal_states=tuple(int(state) for state in goal_states),
                root_state=root_state,
                root_initial_distribution=root_initial_distribution,
                task_action_probabilities=task_action_probabilities,
                capacity_action_probabilities=capacity_action_probabilities,
                communication_capacities=capacities,
                calibration_rows=calibration_rows,
                cost_normalizer=cost_normalizer,
            )
        except (RuntimeError, ValueError) as error:
            return IterativeSynthesisResult(
                candidates=tuple(candidates),
                status=CertificateStatus.SEARCH_FAILED,
                selected_candidate=None,
                search_error=f"{type(error).__name__}: {error}",
            )

        passes_screen = (
            propagated.goal_probability
            >= goal_probability_spec + epsilon_p - 1e-12
        )
        candidate = HighLevelCandidate(
            context_key=context_key,
            goal_probability_spec=goal_probability_spec,
            empirical_goal_probability=propagated.goal_probability,
            empirical_cost=propagated.expected_communication_cost,
            status=(
                CertificateStatus.CERTIFIED
                if passes_screen
                else CertificateStatus.NO_CERTIFIED_CANDIDATE
            ),
            iteration=iteration,
            task_policy=getattr(solution, "task_policy", None),
            comms_budget_policy=getattr(
                solution,
                "comms_budget_policy",
                None,
            ),
            induced_initial_state_distributions=(
                propagated.initial_state_distributions
            ),
        )
        candidates.append(candidate)
        current_initial_distributions = propagated.initial_state_distributions

    passing_candidates = [
        candidate
        for candidate in candidates
        if candidate.status == CertificateStatus.CERTIFIED
    ]
    if not passing_candidates:
        return IterativeSynthesisResult(
            candidates=tuple(candidates),
            status=CertificateStatus.NO_CERTIFIED_CANDIDATE,
            selected_candidate=None,
        )

    selected_candidate = min(
        passing_candidates,
        key=lambda candidate: candidate.empirical_cost,
    )
    return IterativeSynthesisResult(
        candidates=tuple(candidates),
        status=CertificateStatus.CERTIFIED,
        selected_candidate=selected_candidate,
    )
