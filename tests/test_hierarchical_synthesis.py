from types import SimpleNamespace

import pandas as pd
import pytest

from src.simulation.hierarchical_synthesis import (
    IterativeSynthesisResult,
    SynthesisInfeasibleError,
    apply_mixed_success_rates,
    extract_solution_action_probabilities,
    iterative_high_level_synthesis,
)
from src.simulation.hierarchical_types import (
    CalibrationOutcome,
    CalibrationRow,
    CertificateStatus,
    ContextKey,
    OptionSettingKey,
    PolicyScope,
    StateDistribution,
)


def _solution(edge=(0, 1), capacity=1.0):
    return SimpleNamespace(
        task_policy=pd.DataFrame.from_records(
            [
                {
                    "state": edge[0],
                    "next_state": edge[1],
                    "occupancy": 1.0,
                    "action_probability": 1.0,
                }
            ]
        ),
        comms_budget_policy=pd.DataFrame.from_records(
            [
                {
                    "state": edge[0],
                    "next_state": edge[1],
                    "msg_budget_per_agent": capacity,
                    "action_probability": 1.0,
                }
            ]
        ),
    )


def _context(seed=11):
    return ContextKey(
        "test-map",
        PolicyScope.SHARED_ACROSS_CONTEXTS,
        seed,
    )


def test_extract_solution_action_probabilities():
    task_probabilities, capacity_probabilities = extract_solution_action_probabilities(
        _solution()
    )

    assert task_probabilities == {(0, 1): 1.0}
    assert capacity_probabilities == {(0, 1, 1.0): 1.0}


def test_apply_mixed_success_rates_updates_success_and_failure_rows():
    transitions = pd.DataFrame.from_records(
        [
            {
                "state": 0,
                "action": (1, 1.0),
                "next_state": 1,
                "prob": 0.5,
            },
            {
                "state": 0,
                "action": (1, 1.0),
                "next_state": -1,
                "prob": 0.5,
            },
        ]
    )
    hlmdp = SimpleNamespace(
        fail_state=-1,
        transition_probs=transitions,
    )

    apply_mixed_success_rates(
        hlmdp,
        {((0, 1), 1.0): 0.25},
    )

    assert (
        hlmdp.transition_probs.loc[
            hlmdp.transition_probs.next_state == 1,
            "prob",
        ].item()
        == 0.25
    )
    assert (
        hlmdp.transition_probs.loc[
            hlmdp.transition_probs.next_state == -1,
            "prob",
        ].item()
        == 0.75
    )


@pytest.mark.parametrize(
    ("pg", "epsilon_p", "expected_status"),
    [
        (0.40, 0.05, CertificateStatus.CERTIFIED),
        (0.45, 0.05, CertificateStatus.CERTIFIED),
        (0.95, 0.05, CertificateStatus.NO_CERTIFIED_CANDIDATE),
    ],
)
def test_pg_frontier_pass_boundary_and_unattainable_statuses(
    pg,
    epsilon_p,
    expected_status,
):
    edge = (0, 1)
    calibration_rows = (
        CalibrationRow(
            setting=OptionSettingKey(edge, 1.0, (0, 0)),
            outcomes=(
                CalibrationOutcome.success((1, 0)),
                CalibrationOutcome.success((1, 0)),
                CalibrationOutcome.failure(),
                CalibrationOutcome.failure(),
            ),
            seed=11,
        ),
    )

    result = iterative_high_level_synthesis(
        task_edges=(edge,),
        goal_states=(1,),
        root_state=0,
        root_initial_distribution=StateDistribution.singleton((0, 0)),
        communication_capacities=(1.0,),
        calibration_rows=calibration_rows,
        goal_probability_spec=pg,
        epsilon_p=epsilon_p,
        max_iterations=2,
        context_key=_context(),
        solve_policy=lambda mixed_success_rates, threshold: _solution(),
    )

    assert result.status == expected_status


def test_iterative_synthesis_retains_lowest_cost_certified_candidate():
    edge = (0, 1)
    calibration_rows = (
        CalibrationRow(
            setting=OptionSettingKey(edge, 1.0, (0, 0)),
            outcomes=(
                CalibrationOutcome.success((1, 0)),
                CalibrationOutcome.success((1, 0)),
                CalibrationOutcome.failure(),
                CalibrationOutcome.failure(),
            ),
            seed=11,
        ),
    )

    solve_calls = []

    def solve_policy(mixed_success_rates, threshold):
        solve_calls.append((dict(mixed_success_rates), threshold))
        return _solution()

    result = iterative_high_level_synthesis(
        task_edges=(edge,),
        goal_states=(1,),
        root_state=0,
        root_initial_distribution=StateDistribution.singleton((0, 0)),
        communication_capacities=(1.0,),
        calibration_rows=calibration_rows,
        goal_probability_spec=0.4,
        epsilon_p=0.05,
        max_iterations=2,
        context_key=_context(),
        solve_policy=solve_policy,
    )

    assert isinstance(result, IterativeSynthesisResult)
    assert result.status == CertificateStatus.CERTIFIED
    assert result.selected_candidate is not None
    assert result.selected_candidate.empirical_goal_probability == 0.5
    assert len(result.candidates) == 2
    assert len(solve_calls) == 2
    assert solve_calls[0][1] == pytest.approx(0.45)


def test_iterative_synthesis_clamps_certified_threshold_to_one():
    edge = (0, 1)
    calibration_rows = (
        CalibrationRow(
            setting=OptionSettingKey(
                subtask_edge=edge,
                communication_capacity=1.0,
                start_state=(0, 0),
            ),
            outcomes=(
                CalibrationOutcome.success((1, 0)),
            ),
            seed=11,
        ),
    )
    thresholds = []

    def solve_policy(mixed_success_rates, threshold):
        del mixed_success_rates
        thresholds.append(threshold)
        return _solution()

    iterative_high_level_synthesis(
        task_edges=(edge,),
        goal_states=(1,),
        root_state=0,
        root_initial_distribution=StateDistribution.singleton((0, 0)),
        communication_capacities=(1.0,),
        calibration_rows=calibration_rows,
        goal_probability_spec=1.0,
        epsilon_p=0.20,
        max_iterations=1,
        context_key=_context(),
        solve_policy=solve_policy,
    )

    assert thresholds == [1.0]


def test_iterative_synthesis_reports_infeasible_threshold():
    edge = (0, 1)
    calibration_rows = (
        CalibrationRow(
            setting=OptionSettingKey(edge, 1.0, (0, 0)),
            outcomes=(CalibrationOutcome.failure(),),
            seed=11,
        ),
    )

    def infeasible_solver(mixed_success_rates, threshold):
        del mixed_success_rates, threshold
        raise SynthesisInfeasibleError("No policy meets the threshold.")

    result = iterative_high_level_synthesis(
        task_edges=(edge,),
        goal_states=(1,),
        root_state=0,
        root_initial_distribution=StateDistribution.singleton((0, 0)),
        communication_capacities=(1.0,),
        calibration_rows=calibration_rows,
        goal_probability_spec=0.95,
        epsilon_p=0.05,
        max_iterations=2,
        context_key=_context(),
        solve_policy=infeasible_solver,
    )

    assert result.status == CertificateStatus.INFEASIBLE
    assert result.selected_candidate is None
    assert result.candidates == ()


def test_iterative_synthesis_reports_no_certified_candidate():
    edge = (0, 1)
    calibration_rows = (
        CalibrationRow(
            setting=OptionSettingKey(edge, 1.0, (0, 0)),
            outcomes=(
                CalibrationOutcome.success((1, 0)),
                CalibrationOutcome.failure(),
            ),
            seed=11,
        ),
    )

    result = iterative_high_level_synthesis(
        task_edges=(edge,),
        goal_states=(1,),
        root_state=0,
        root_initial_distribution=StateDistribution.singleton((0, 0)),
        communication_capacities=(1.0,),
        calibration_rows=calibration_rows,
        goal_probability_spec=0.7,
        epsilon_p=0.0,
        max_iterations=2,
        context_key=_context(),
        solve_policy=lambda mixed_success_rates, threshold: _solution(),
    )

    assert result.status == CertificateStatus.NO_CERTIFIED_CANDIDATE
    assert result.selected_candidate is None
    assert len(result.candidates) == 2


@pytest.mark.parametrize(
    ("solve_behavior", "expected_status"),
    [
        (lambda: None, CertificateStatus.NO_CERTIFIED_CANDIDATE),
        (
            lambda: (_ for _ in ()).throw(SynthesisInfeasibleError()),
            CertificateStatus.INFEASIBLE,
        ),
        (
            lambda: (_ for _ in ()).throw(RuntimeError("solver failure")),
            CertificateStatus.SEARCH_FAILED,
        ),
    ],
)
def test_iterative_synthesis_distinguishes_search_status(
    solve_behavior,
    expected_status,
):
    edge = (0, 1)
    calibration_rows = (
        CalibrationRow(
            setting=OptionSettingKey(edge, 1.0, (0, 0)),
            outcomes=(CalibrationOutcome.success((1, 0)),),
            seed=11,
        ),
    )

    def solve_policy(mixed_success_rates, threshold):
        result = solve_behavior()
        if result is not None:
            return result
        return None

    result = iterative_high_level_synthesis(
        task_edges=(edge,),
        goal_states=(1,),
        root_state=0,
        root_initial_distribution=StateDistribution.singleton((0, 0)),
        communication_capacities=(1.0,),
        calibration_rows=calibration_rows,
        goal_probability_spec=0.5,
        epsilon_p=0.0,
        max_iterations=2,
        context_key=_context(),
        solve_policy=solve_policy,
    )

    assert result.status == expected_status
