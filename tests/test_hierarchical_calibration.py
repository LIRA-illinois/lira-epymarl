import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.simulation.evaluate import (
    collect_statewise_calibration_row,
    record_calibration_replays,
)
from src.simulation.hierarchical_calibration import (
    build_calibration_report_row,
    calculate_m_est,
    calculate_project_cost_error,
    calculate_project_reliability_error,
    enumerate_statewise_settings,
    greedy_project_baselines,
    merge_weighted_state_distributions,
    mix_statewise_option_estimates,
    propagate_initial_state_distributions,
    total_variation_distance,
    validate_project_screening_margin,
)
from src.simulation.hierarchical_types import (
    CalibrationOutcome,
    CalibrationRow,
    OptionSettingKey,
    StateDistribution,
)


def test_greedy_project_baselines_choose_edge_and_minimum_budget():
    edges = ((0, 1), (0, 2))
    rows = []
    for edge in edges:
        for capacity, success_rate in (
            (0.0, 0.90 if edge == (0, 1) else 0.60),
            (1.0, 0.95 if edge == (0, 1) else 0.65),
            (2.0, 0.98 if edge == (0, 1) else 0.70),
        ):
            outcomes = tuple(
                CalibrationOutcome.success((edge[1], 0))
                if index / 100 < success_rate
                else CalibrationOutcome.failure()
                for index in range(100)
            )
            rows.append(
                CalibrationRow(
                    setting=OptionSettingKey(
                        subtask_edge=edge,
                        communication_capacity=capacity,
                        start_state=(0, 0),
                    ),
                    outcomes=outcomes,
                    seed=101,
                )
            )

    task_greedy, comms_greedy = greedy_project_baselines(
        task_edges=edges,
        goal_states=(1, 2),
        root_state=0,
        root_initial_distribution=StateDistribution.singleton((0, 0)),
        communication_capacities=(0.0, 1.0, 2.0),
        calibration_rows=rows,
        pg=0.55,
        cost_normalizer=2,
    )

    assert task_greedy.goal_probability == pytest.approx(0.98)
    assert task_greedy.expected_communication_cost == pytest.approx(1.0)
    assert comms_greedy.goal_probability == pytest.approx(0.90)
    assert comms_greedy.expected_communication_cost == pytest.approx(0.0)
    assert set(task_greedy.edge_success_rates) == {((0, 1), 2.0)}
    assert set(comms_greedy.edge_success_rates) == {((0, 1), 0.0)}


def test_m_est_matches_finite_sample_formula():
    epsilon_est = 0.1
    beta = 0.95
    outcome_alphabet_size = 2
    setting_count = 10
    expected = math.ceil(
        math.log((2 ** (outcome_alphabet_size + 1) * setting_count) / (1 - beta))
        / (2 * epsilon_est**2)
    )

    assert (
        calculate_m_est(
            epsilon_est,
            beta,
            outcome_alphabet_size,
            setting_count,
        )
        == expected
    )


def test_calibration_report_row_and_manifest_metadata(tmp_path):
    row = CalibrationRow(
        setting=OptionSettingKey((0, 1), 1.0, (0, 0)),
        outcomes=(
            CalibrationOutcome.success((1, 0)),
            CalibrationOutcome.success((1, 0)),
            CalibrationOutcome.failure(),
        ),
        seed=17,
    )
    from src.simulation.hierarchical_calibration import (
        read_calibration_metadata,
        save_calibration_manifest,
    )
    save_calibration_manifest(
        tmp_path,
        17,
        (row,),
        metadata={
            "epsilon_est": 0.05,
            "beta": 0.95,
            "m_est": 100,
            "n_calibration_episodes": 3,
        },
    )

    metadata = read_calibration_metadata(tmp_path, 17)
    report = build_calibration_report_row(
        row,
        epsilon_est=0.05,
        beta=0.95,
        m_est=100,
        statewise_setting_count=10,
    )

    assert metadata["epsilon_est"] == 0.05
    assert metadata["n_calibration_episodes"] == 3
    assert report["n_episodes"] == 3
    assert report["success_count"] == 2
    assert report["failure_count"] == 1
    assert report["certificate_applicable"] is False


def test_project_error_allowances_and_screening_margin():
    delta_p = calculate_project_reliability_error(0.01, 10)

    assert np.isclose(delta_p, 1 - 0.99**10)
    assert np.isclose(
        calculate_project_cost_error(
            epsilon_est=0.01,
            n_max=10,
            maximum_capacity=2.0,
            cost_normalizer=2,
        ),
        10 / 2 * 2.0 * delta_p,
    )
    assert validate_project_screening_margin(0.01, 10, delta_p) == delta_p

    with pytest.raises(ValueError, match="epsilon_p"):
        validate_project_screening_margin(0.01, 10, delta_p / 2)


def test_enumerate_statewise_settings_uses_source_support():
    settings = enumerate_statewise_settings(
        task_edges=((0, 1), (1, 2)),
        communication_capacities=(0.0, 1.0),
        initial_support_by_source={
            0: ((0, 0),),
            1: ((1, 0), (1, 1)),
        },
    )

    assert len(settings) == 6
    assert {
        (setting.subtask_edge, setting.communication_capacity, setting.start_state)
        for setting in settings
    } == {
        ((0, 1), 0.0, (0, 0)),
        ((0, 1), 1.0, (0, 0)),
        ((1, 2), 0.0, (1, 0)),
        ((1, 2), 0.0, (1, 1)),
        ((1, 2), 1.0, (1, 0)),
        ((1, 2), 1.0, (1, 1)),
    }


def test_statewise_mixing_preserves_success_and_terminal_mixture():
    edge = (0, 1)
    setting_a = OptionSettingKey(edge, 1.0, (0, 0))
    setting_b = OptionSettingKey(edge, 1.0, (0, 1))
    rows = (
        CalibrationRow(
            setting=setting_a,
            outcomes=(
                CalibrationOutcome.success((1, 0)),
                CalibrationOutcome.success((1, 0)),
                CalibrationOutcome.failure(),
            ),
            seed=11,
        ),
        CalibrationRow(
            setting=setting_b,
            outcomes=(
                CalibrationOutcome.success((1, 1)),
                CalibrationOutcome.failure(),
                CalibrationOutcome.failure(),
            ),
            seed=11,
        ),
    )
    initial_distribution = StateDistribution(
        ((0, 0), (0, 1)),
        (0.75, 0.25),
    )

    mixed_success, terminal_distribution = mix_statewise_option_estimates(
        rows,
        edge,
        1.0,
        initial_distribution,
    )

    assert np.isclose(mixed_success, 0.75 * 2 / 3 + 0.25 * 1 / 3)
    assert terminal_distribution.states == ((1, 0), (1, 1))
    assert np.allclose(terminal_distribution.probabilities, (6 / 7, 1 / 7))


def test_statewise_mixing_requires_rows_for_positive_probability():
    edge = (0, 1)
    row = CalibrationRow(
        setting=OptionSettingKey(edge, 1.0, (0, 0)),
        outcomes=(CalibrationOutcome.failure(),),
        seed=11,
    )

    with pytest.raises(ValueError, match="Missing calibration row"):
        mix_statewise_option_estimates(
            (row,),
            edge,
            1.0,
            StateDistribution.singleton((0, 1)),
        )


def test_total_variation_distance():
    assert np.isclose(
        total_variation_distance(
            {"failure": 0.25, "success": 0.75},
            {"failure": 0.5, "success": 0.5},
        ),
        0.25,
    )


def test_merge_weighted_state_distributions():
    merged = merge_weighted_state_distributions(
        (
            StateDistribution.singleton((1, 0)),
            StateDistribution(
                ((2, 0), (2, 1)),
                (0.25, 0.75),
            ),
        ),
        (3.0, 1.0),
    )

    assert merged.states == ((1, 0), (2, 0), (2, 1))
    assert np.allclose(merged.probabilities, (0.75, 0.0625, 0.1875))


def test_propagate_initial_state_distributions_chain():
    edge_one = (0, 1)
    edge_two = (1, 2)
    rows = (
        CalibrationRow(
            setting=OptionSettingKey(edge_one, 1.0, (0, 0)),
            outcomes=(
                CalibrationOutcome.success((1, 0)),
                CalibrationOutcome.success((1, 0)),
                CalibrationOutcome.failure(),
                CalibrationOutcome.failure(),
            ),
            seed=11,
        ),
        CalibrationRow(
            setting=OptionSettingKey(edge_two, 1.0, (1, 0)),
            outcomes=(
                CalibrationOutcome.success((2, 0)),
                CalibrationOutcome.success((2, 0)),
                CalibrationOutcome.success((2, 0)),
                CalibrationOutcome.success((2, 0)),
                CalibrationOutcome.failure(),
            ),
            seed=11,
        ),
    )

    propagated = propagate_initial_state_distributions(
        task_edges=(edge_one, edge_two),
        goal_states=(2,),
        root_state=0,
        root_initial_distribution=StateDistribution.singleton((0, 0)),
        task_action_probabilities={
            edge_one: 1.0,
            edge_two: 1.0,
        },
        capacity_action_probabilities={
            (*edge_one, 1.0): 1.0,
            (*edge_two, 1.0): 1.0,
        },
        communication_capacities=(1.0,),
        calibration_rows=rows,
    )

    assert np.isclose(propagated.goal_probability, 0.4)
    assert np.isclose(propagated.expected_communication_cost, 1.5)
    assert np.isclose(propagated.state_flows[1], 0.5)
    assert propagated.initial_state_distributions[1].states == ((1, 0),)
    assert np.isclose(
        propagated.edge_success_rates[(edge_one, 1.0)],
        0.5,
    )
    assert np.isclose(
        propagated.edge_success_rates[(edge_two, 1.0)],
        0.8,
    )


def test_propagate_initial_state_distributions_merges_incoming_branches():
    edge_01 = (0, 1)
    edge_02 = (0, 2)
    edge_13 = (1, 3)
    edge_23 = (2, 3)
    rows = (
        CalibrationRow(
            setting=OptionSettingKey(edge_01, 1.0, (0, 0)),
            outcomes=(CalibrationOutcome.success((1, 0)),),
            seed=11,
        ),
        CalibrationRow(
            setting=OptionSettingKey(edge_02, 1.0, (0, 0)),
            outcomes=(CalibrationOutcome.success((2, 0)),),
            seed=11,
        ),
        CalibrationRow(
            setting=OptionSettingKey(edge_13, 1.0, (1, 0)),
            outcomes=(CalibrationOutcome.success((3, 1)),),
            seed=11,
        ),
        CalibrationRow(
            setting=OptionSettingKey(edge_23, 1.0, (2, 0)),
            outcomes=(CalibrationOutcome.success((3, 2)),),
            seed=11,
        ),
    )

    propagated = propagate_initial_state_distributions(
        task_edges=(edge_01, edge_02, edge_13, edge_23),
        goal_states=(3,),
        root_state=0,
        root_initial_distribution=StateDistribution.singleton((0, 0)),
        task_action_probabilities={
            edge_01: 0.5,
            edge_02: 0.5,
            edge_13: 1.0,
            edge_23: 1.0,
        },
        capacity_action_probabilities={
            (*edge_01, 1.0): 1.0,
            (*edge_02, 1.0): 1.0,
            (*edge_13, 1.0): 1.0,
            (*edge_23, 1.0): 1.0,
        },
        communication_capacities=(1.0,),
        calibration_rows=rows,
    )

    assert np.isclose(propagated.goal_probability, 1.0)
    assert np.isclose(propagated.expected_communication_cost, 2.0)
    assert propagated.initial_state_distributions[3].states == ((3, 1), (3, 2))
    assert np.allclose(
        propagated.initial_state_distributions[3].probabilities,
        (0.5, 0.5),
    )


class _CalibrationRunner:
    batch_size = 1

    def __init__(self, episode_results):
        self.episode_results = iter(episode_results)
        self.mac = SimpleNamespace(msg_budget_per_agent=3)
        self.test_returns = [99]
        self.test_stats = {"n_episodes": 99}
        self.env = SimpleNamespace(terminate_on_task_completed=False)
        self.reset_options = []

    def run(self, test_mode, return_log_stats, reset_options):
        assert test_mode
        self.reset_options.append(reset_options)
        final_info = next(self.episode_results)
        return {"final_infos": [final_info], "final_info": final_info}


def test_collect_statewise_calibration_row_preserves_outcomes():
    setting = OptionSettingKey((0, 1), 1.0, ((0, 0),))
    runner = _CalibrationRunner(
        (
            {"task_completed": True, "final_state": ((1, 0),)},
            {"task_completed": False},
            {"task_completed": True, "final_state": ((1, 1),)},
        )
    )

    row = collect_statewise_calibration_row(
        runner,
        setting=setting,
        seed=17,
        n_eval_eps=3,
    )

    assert row.seed == 17
    assert row.success_count == 2
    assert row.failure_count == 1
    assert row.success_rate == 2 / 3
    assert row.terminal_state_distribution.states == ((1, 0), (1, 1))
    assert runner.mac.msg_budget_per_agent == 3
    assert runner.test_returns == [99]
    assert runner.test_stats == {"n_episodes": 99}
    assert runner.env.terminate_on_task_completed is False
    assert runner.reset_options[0]["hl_task"] == [0, 1]
    assert runner.reset_options[0]["navigation_init_state_dist"] == {
        "states": [[[0, 0]]],
        "probs": [1.0],
    }


def test_record_calibration_replays_uses_setting_prefix():
    recording_calls = []

    class _ReplayRunner:
        args = SimpleNamespace(
            save_test_replays=True,
            n_test_replays_save=2,
        )
        t_env = 800

        def start_recording(self, **kwargs):
            recording_calls.append(("start", kwargs))

        def run(self, **kwargs):
            recording_calls.append(("run", kwargs))

        def stop_recording(self, **kwargs):
            recording_calls.append(("stop", kwargs))

    runner = _ReplayRunner()
    setting = OptionSettingKey((1, 2), 1.0, (1, 1, 1, 3, 1, 5))

    record_calibration_replays(
        runner,
        setting=setting,
        reset_options={
            "hl_start_state": 1,
            "hl_task": [1, 2],
        },
        t_env=800,
    )

    assert recording_calls[0][0] == "start"
    assert recording_calls[0][1]["video_prefix"].startswith(
        "replay-task-1-2-msg-1.00"
    )
    assert sum(call[0] == "run" for call in recording_calls) == 2
    assert recording_calls[-1][0] == "stop"


def test_collect_statewise_calibration_overrides_real_joint_position_dist():
    setting = OptionSettingKey((0, 1), 1.0, ((0, 0),))
    runner = _CalibrationRunner(
        (
            {"task_completed": True, "final_state": ((1, 1), (1, 3), (1, 5))},
        )
    )
    real_dist = {
        "states": [[[1, 1], [1, 3], [1, 5]]],
        "probs": [1.0],
    }

    collect_statewise_calibration_row(
        runner,
        setting=setting,
        seed=17,
        n_eval_eps=1,
        reset_options={
            "hl_start_state": 0,
            "hl_task": [0, 1],
            "navigation_init_state_dist": real_dist,
        },
    )

    assert runner.reset_options[0]["navigation_init_state_dist"] == {
        "states": [[[0, 0]]],
        "probs": [1.0],
    }
