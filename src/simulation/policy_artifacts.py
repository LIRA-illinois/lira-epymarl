"""Manifest helpers for resumable low-level option-policy artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch as th

from src.simulation.hierarchical_types import (
    LowLevelPolicyArtifact,
    StateDistribution,
)


def _capacity_token(capacity: float) -> str:
    return f"{float(capacity):g}".replace("-", "m").replace(".", "p")


def _state_hash(distribution: StateDistribution) -> str:
    payload = json.dumps(
        distribution.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def option_setting_directory(
    root: str | Path,
    artifact: LowLevelPolicyArtifact,
) -> Path:
    """Return the immutable setting-specific directory for an artifact."""
    root_path = Path(root)
    from_state, to_state = artifact.subtask_edge
    return (
        root_path
        / "option_library"
        / f"seed_{artifact.seed}"
        / f"edge_{from_state}_{to_state}"
        / f"budget_{_capacity_token(artifact.communication_capacity)}"
        / f"start_{_state_hash(artifact.initial_state_distribution)}"
        / f"config_{artifact.config_hash}"
    )


def option_manifest_path(root: str | Path, seed: int) -> Path:
    """Return the manifest path for one low-level seed."""
    return Path(root) / "option_library" / f"seed_{int(seed)}" / "manifest.json"


def _artifact_to_record(artifact: LowLevelPolicyArtifact) -> dict[str, Any]:
    return {
        "seed": artifact.seed,
        "subtask_edge": list(artifact.subtask_edge),
        "communication_capacity": artifact.communication_capacity,
        "initial_state_distribution": artifact.initial_state_distribution.to_dict(),
        "timestep": artifact.timestep,
        "checkpoint_path": artifact.checkpoint_path,
        "config_hash": artifact.config_hash,
    }


def _artifact_from_record(record: Mapping[str, Any]) -> LowLevelPolicyArtifact:
    edge = record["subtask_edge"]
    if not isinstance(edge, list) or len(edge) != 2:
        raise ValueError("Manifest subtask_edge must contain two states.")
    return LowLevelPolicyArtifact(
        seed=int(record["seed"]),
        subtask_edge=(int(edge[0]), int(edge[1])),
        communication_capacity=float(record["communication_capacity"]),
        initial_state_distribution=StateDistribution.from_dict(
            record["initial_state_distribution"]
        ),
        timestep=int(record["timestep"]),
        checkpoint_path=str(record["checkpoint_path"]),
        config_hash=str(record["config_hash"]),
    )


def write_option_policy_manifest(
    root: str | Path,
    artifacts: list[LowLevelPolicyArtifact],
) -> Path:
    """Atomically write a seed-isolated option-policy manifest."""
    if not artifacts:
        raise ValueError("An option-policy manifest cannot be empty.")
    seeds = {artifact.seed for artifact in artifacts}
    if len(seeds) != 1:
        raise ValueError("One manifest may contain only one low-level seed.")

    seed = next(iter(seeds))
    manifest_path = option_manifest_path(root, seed)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "artifacts": [_artifact_to_record(artifact) for artifact in artifacts],
    }
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return manifest_path


def read_option_policy_manifest(
    root: str | Path,
    seed: int,
) -> tuple[LowLevelPolicyArtifact, ...]:
    """Read a seed-isolated option-policy manifest."""
    manifest_path = option_manifest_path(root, seed)
    if not manifest_path.is_file():
        return ()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_seed = int(payload["seed"])
    if manifest_seed != int(seed):
        raise ValueError(
            f"Manifest seed {manifest_seed} does not match requested seed {seed}."
        )
    return tuple(
        _artifact_from_record(record) for record in payload.get("artifacts", [])
    )


def save_option_policy_artifact(
    learner,
    root: str | Path,
    artifact: LowLevelPolicyArtifact,
) -> LowLevelPolicyArtifact:
    """Save an artifact through the existing learner callback and update the manifest."""
    setting_directory = option_setting_directory(root, artifact)
    checkpoint_directory = setting_directory / str(artifact.timestep)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    learner.save_models(str(checkpoint_directory))

    saved_artifact = replace(
        artifact,
        checkpoint_path=str(checkpoint_directory),
    )
    existing = list(read_option_policy_manifest(root, artifact.seed))
    replacement_key = (
        saved_artifact.subtask_edge,
        saved_artifact.communication_capacity,
        saved_artifact.initial_state_distribution,
        saved_artifact.config_hash,
        saved_artifact.timestep,
    )
    updated = []
    replaced = False
    for existing_artifact in existing:
        existing_key = (
            existing_artifact.subtask_edge,
            existing_artifact.communication_capacity,
            existing_artifact.initial_state_distribution,
            existing_artifact.config_hash,
            existing_artifact.timestep,
        )
        if existing_key == replacement_key:
            updated.append(saved_artifact)
            replaced = True
        else:
            updated.append(existing_artifact)
    if not replaced:
        updated.append(saved_artifact)
    write_option_policy_manifest(root, updated)
    return saved_artifact


def load_option_policy_artifact(
    learner,
    artifact: LowLevelPolicyArtifact,
) -> LowLevelPolicyArtifact:
    """Load one artifact through the existing learner callback."""
    checkpoint_directory = Path(artifact.checkpoint_path)
    if not checkpoint_directory.is_dir():
        raise FileNotFoundError(
            f"Option-policy checkpoint directory does not exist: {checkpoint_directory}"
        )
    learner.load_models(str(checkpoint_directory))
    return artifact


def save_option_policy_weights_artifact(
    root: str | Path,
    artifact: LowLevelPolicyArtifact,
    weights: Mapping[str, Any],
) -> LowLevelPolicyArtifact:
    """Save a seed-specific state dictionary alongside the option manifest."""
    setting_directory = option_setting_directory(root, artifact)
    checkpoint_directory = setting_directory / str(artifact.timestep)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    weights_path = checkpoint_directory / "weights.th"
    th.save(dict(weights), weights_path)

    saved_artifact = replace(
        artifact,
        checkpoint_path=str(checkpoint_directory),
    )
    existing = list(read_option_policy_manifest(root, artifact.seed))
    replacement_key = (
        saved_artifact.subtask_edge,
        saved_artifact.communication_capacity,
        saved_artifact.initial_state_distribution,
        saved_artifact.config_hash,
        saved_artifact.timestep,
    )
    updated = []
    replaced = False
    for existing_artifact in existing:
        existing_key = (
            existing_artifact.subtask_edge,
            existing_artifact.communication_capacity,
            existing_artifact.initial_state_distribution,
            existing_artifact.config_hash,
            existing_artifact.timestep,
        )
        if existing_key == replacement_key:
            updated.append(saved_artifact)
            replaced = True
        else:
            updated.append(existing_artifact)
    if not replaced:
        updated.append(saved_artifact)
    write_option_policy_manifest(root, updated)
    return saved_artifact


def load_option_policy_weights_artifact(
    artifact: LowLevelPolicyArtifact,
) -> dict[str, Any]:
    """Load one saved state dictionary without mutating a learner."""
    weights_path = Path(artifact.checkpoint_path) / "weights.th"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Option-policy weights do not exist: {weights_path}")
    weights = th.load(weights_path, map_location="cpu")
    if not isinstance(weights, dict):
        raise TypeError(
            f"Option-policy weights must be a state dictionary: {weights_path}"
        )
    return weights


def find_latest_option_policy_artifact(
    root: str | Path,
    seed: int,
    subtask_edge: tuple[int, int],
    communication_capacity: float,
    initial_state_distribution: StateDistribution,
    config_hash: str | None = None,
) -> LowLevelPolicyArtifact | None:
    """Return the latest matching artifact for one statewise setting."""
    matching = []
    for artifact in read_option_policy_manifest(root, seed):
        if artifact.subtask_edge != tuple(subtask_edge):
            continue
        if not math.isclose(
            artifact.communication_capacity,
            float(communication_capacity),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            continue
        if artifact.initial_state_distribution != initial_state_distribution:
            continue
        if config_hash is not None and artifact.config_hash != config_hash:
            continue
        matching.append(artifact)
    if not matching:
        return None
    return max(matching, key=lambda artifact: artifact.timestep)


def find_option_policy_artifact(
    root: str | Path,
    seed: int,
    subtask_edge: tuple[int, int],
    communication_capacity: float,
    initial_state_distribution: StateDistribution,
    timestep: int,
    config_hash: str | None = None,
) -> LowLevelPolicyArtifact | None:
    """Return one exact-timestep artifact for a statewise setting."""
    if not isinstance(timestep, int) or timestep < 0:
        raise ValueError("Option-policy timestep must be a non-negative integer.")
    for artifact in read_option_policy_manifest(root, seed):
        if artifact.subtask_edge != tuple(subtask_edge):
            continue
        if not math.isclose(
            artifact.communication_capacity,
            float(communication_capacity),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            continue
        if artifact.initial_state_distribution != initial_state_distribution:
            continue
        if config_hash is not None and artifact.config_hash != config_hash:
            continue
        if artifact.timestep == timestep:
            return artifact
    return None
