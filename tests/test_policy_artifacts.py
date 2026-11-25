from pathlib import Path

import pytest

from src.simulation.hierarchical_types import (
    LowLevelPolicyArtifact,
    StateDistribution,
)
from src.simulation.policy_artifacts import (
    find_latest_option_policy_artifact,
    find_option_policy_artifact,
    load_option_policy_artifact,
    read_option_policy_manifest,
    save_option_policy_artifact,
)


class _FakeLearner:
    def __init__(self):
        self.saved_paths = []
        self.loaded_paths = []

    def save_models(self, path):
        checkpoint = Path(path)
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "agent.th").write_text("weights", encoding="utf-8")
        self.saved_paths.append(path)

    def load_models(self, path):
        self.loaded_paths.append(path)


def _artifact(tmp_path, timestep):
    return LowLevelPolicyArtifact(
        seed=17,
        subtask_edge=(0, 1),
        communication_capacity=1.0,
        initial_state_distribution=StateDistribution.singleton((0, 0)),
        timestep=timestep,
        checkpoint_path=str(tmp_path / "unused"),
        config_hash="abc123",
    )


def test_save_load_and_find_latest_option_policy_artifact(tmp_path):
    learner = _FakeLearner()
    first = save_option_policy_artifact(
        learner,
        tmp_path,
        _artifact(tmp_path, 100),
    )
    second = save_option_policy_artifact(
        learner,
        tmp_path,
        _artifact(tmp_path, 200),
    )

    assert first.checkpoint_path != second.checkpoint_path
    assert (Path(second.checkpoint_path) / "agent.th").read_text(
        encoding="utf-8"
    ) == "weights"
    manifest = read_option_policy_manifest(tmp_path, seed=17)
    assert len(manifest) == 2

    latest = find_latest_option_policy_artifact(
        tmp_path,
        seed=17,
        subtask_edge=(0, 1),
        communication_capacity=1.0,
        initial_state_distribution=StateDistribution.singleton((0, 0)),
        config_hash="abc123",
    )
    assert latest is not None
    assert latest.timestep == 200

    loaded = load_option_policy_artifact(learner, latest)
    assert loaded == latest
    assert learner.loaded_paths == [latest.checkpoint_path]


def test_specific_timestep_artifact_lookup(tmp_path):
    learner = _FakeLearner()
    save_option_policy_artifact(
        learner,
        tmp_path,
        _artifact(tmp_path, 50),
    )
    save_option_policy_artifact(
        learner,
        tmp_path,
        _artifact(tmp_path, 75),
    )
    matching = read_option_policy_manifest(tmp_path, seed=17)
    selected = next(artifact for artifact in matching if artifact.timestep == 50)
    exact = find_option_policy_artifact(
        tmp_path,
        seed=17,
        subtask_edge=(0, 1),
        communication_capacity=1.0,
        initial_state_distribution=StateDistribution.singleton((0, 0)),
        timestep=50,
        config_hash="abc123",
    )

    assert selected.checkpoint_path.endswith("/50")
    assert exact is not None
    assert exact.timestep == 50


def test_find_latest_returns_none_for_missing_setting(tmp_path):
    assert (
        find_latest_option_policy_artifact(
            tmp_path,
            seed=17,
            subtask_edge=(1, 2),
            communication_capacity=1.0,
            initial_state_distribution=StateDistribution.singleton((1, 1)),
        )
        is None
    )


def test_save_rejects_empty_manifest(tmp_path):
    from src.simulation.policy_artifacts import write_option_policy_manifest

    with pytest.raises(ValueError, match="cannot be empty"):
        write_option_policy_manifest(tmp_path, [])
