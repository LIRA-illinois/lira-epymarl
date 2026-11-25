import numpy as np
import pytest

from src.simulation.hierarchical_types import (
    CalibrationOutcome,
    CalibrationRow,
    CertificateStatus,
    ContextKey,
    EvaluationRowKey,
    HeldOutResult,
    HighLevelCandidate,
    LowLevelPolicyArtifact,
    OptionSettingKey,
    PolicyScope,
    StateDistribution,
)


def test_state_distribution_singleton_and_dict_round_trip():
    singleton = StateDistribution.singleton([[1, 2], [3, 4]])

    assert singleton.states == (((1, 2), (3, 4)),)
    assert singleton.probabilities == (1.0,)
    assert StateDistribution.from_dict(singleton.to_dict()) == singleton


def test_state_distribution_validates_and_combines_duplicate_states():
    distribution = StateDistribution(
        ([1, 2], [1, 2], [3, 4]),
        (0.2, 0.3, 0.5),
    )

    assert distribution.states == ((1, 2), (3, 4))
    assert distribution.probabilities == (0.5, 0.5)

    with pytest.raises(ValueError, match="sum to 1.0"):
        StateDistribution(([1, 2],), (0.5,))
    with pytest.raises(ValueError, match="non-negative"):
        StateDistribution(([1, 2],), (-0.1,))


def test_option_setting_key_is_hashable_and_normalized():
    key = OptionSettingKey(
        subtask_edge=(0, 1),
        communication_capacity=np.int64(2),
        start_state=np.array([1, 2]),
    )

    assert key.communication_capacity == 2.0
    assert key.start_state == (1, 2)
    assert len({key}) == 1


def test_option_setting_key_rejects_self_loop_and_invalid_capacity():
    with pytest.raises(ValueError, match="self-loop"):
        OptionSettingKey((1, 1), 1.0, (0, 0))
    with pytest.raises(ValueError, match="non-negative"):
        OptionSettingKey((0, 1), -1.0, (0, 0))


def test_calibration_row_has_success_and_terminal_statistics():
    setting = OptionSettingKey((0, 1), 1.0, (0, 0))
    row = CalibrationRow(
        setting=setting,
        outcomes=[
            CalibrationOutcome.success((2, 2)),
            CalibrationOutcome.success((2, 2)),
            CalibrationOutcome.success((3, 3)),
            CalibrationOutcome.failure(),
        ],
        seed=17,
    )

    assert row.success_count == 3
    assert row.failure_count == 1
    assert row.success_rate == 0.75
    assert row.terminal_state_distribution.states == ((2, 2), (3, 3))
    assert row.terminal_state_distribution.probabilities == (2 / 3, 1 / 3)
    assert row.outcome_distribution == {"failure": 0.25, "success": 0.75}


def test_calibration_outcome_rejects_malformed_terminal_fields():
    with pytest.raises(ValueError, match="terminal state"):
        CalibrationOutcome.success(None)
    with pytest.raises(ValueError, match="cannot have"):
        CalibrationOutcome(kind="failure", terminal_state=(1, 1))


def test_policy_artifact_context_and_evaluation_keys_are_typed():
    artifact = LowLevelPolicyArtifact(
        seed=11,
        subtask_edge=(0, 1),
        communication_capacity=1.0,
        initial_state_distribution=StateDistribution.singleton((0, 0)),
        timestep=100,
        checkpoint_path="/tmp/option.pt",
        config_hash="abc123",
    )
    context = ContextKey("map-1", PolicyScope.SHARED_ACROSS_CONTEXTS, 11)
    evaluation_key = EvaluationRowKey(
        low_level_seed=11,
        high_level_seed=11,
        context_id="map-1",
        subtask_edge=(0, 1),
        communication_capacity=1.0,
        start_state=np.array([0, 0]),
        episode_index=3,
    )

    assert artifact.timestep == 100
    assert context.policy_scope == PolicyScope.SHARED_ACROSS_CONTEXTS
    assert evaluation_key.start_state == (0, 0)
    assert len({evaluation_key}) == 1


def test_high_level_candidate_and_held_out_result():
    context = ContextKey("map-1", PolicyScope.SHARED_ACROSS_CONTEXTS, 23)
    candidate = HighLevelCandidate(
        context_key=context,
        goal_probability_spec=0.95,
        empirical_goal_probability=0.96,
        empirical_cost=1.25,
        status=CertificateStatus.CERTIFIED,
        iteration=2,
    )
    row_key = EvaluationRowKey(
        low_level_seed=23,
        high_level_seed=23,
        context_id="map-1",
        subtask_edge=(0, 1),
        communication_capacity=1.0,
        start_state=(0, 0),
        episode_index=0,
    )
    held_out = HeldOutResult(
        row_key=row_key,
        project_success=True,
        communication_cost=1.25,
        successful_terminal_state=(1, 1),
    )

    assert candidate.status == CertificateStatus.CERTIFIED
    assert held_out.project_success
    assert held_out.successful_terminal_state == (1, 1)
