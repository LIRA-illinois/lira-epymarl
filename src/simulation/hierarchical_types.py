"""Typed data contracts for hierarchical option calibration and synthesis."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

StateVector = tuple[Any, ...]
TaskEdge = tuple[int, int]


def normalize_state(value: object) -> StateVector:
    """Convert nested state containers into an immutable, hashable state."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        return tuple(
            (str(key), normalize_state(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        normalized = tuple(normalize_state(item) for item in value)
        if all(
            isinstance(item, (int, float, np.integer, np.floating))
            and not isinstance(item, tuple)
            for item in value
        ):
            return tuple(
                float(item) if isinstance(item, (float, np.floating)) else int(item)
                for item in value
            )
        return normalized
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return (int(value),)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return (value,)
    return (repr(value),)


@dataclass(frozen=True)
class StateDistribution:
    """A finite distribution over physical states."""

    states: tuple[StateVector, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        states = tuple(normalize_state(state) for state in self.states)
        probabilities = tuple(float(probability) for probability in self.probabilities)
        if len(states) != len(probabilities):
            raise ValueError("State distribution states and probabilities must align.")
        if not probabilities:
            object.__setattr__(self, "states", ())
            object.__setattr__(self, "probabilities", ())
            return
        if any(
            not math.isfinite(probability) or probability < 0.0
            for probability in probabilities
        ):
            raise ValueError(
                "State distribution probabilities must be finite and non-negative."
            )
        total = float(sum(probabilities))
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"State distribution probabilities must sum to 1.0, got {total}."
            )

        combined: dict[StateVector, float] = {}
        for state, probability in zip(states, probabilities):
            combined[state] = combined.get(state, 0.0) + probability
        ordered_states = sorted(combined, key=repr)
        object.__setattr__(self, "states", tuple(ordered_states))
        object.__setattr__(
            self,
            "probabilities",
            tuple(combined[state] for state in ordered_states),
        )

    @classmethod
    def singleton(cls, state: object) -> StateDistribution:
        return cls((normalize_state(state),), (1.0,))

    @classmethod
    def empty(cls) -> StateDistribution:
        return cls((), ())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> StateDistribution:
        states = value.get("states", ())
        probabilities = value.get("probs", value.get("probabilities", ()))
        if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
            raise TypeError("State distribution 'states' must be a sequence.")
        if not isinstance(probabilities, Sequence) or isinstance(
            probabilities, (str, bytes)
        ):
            raise TypeError("State distribution probabilities must be a sequence.")
        return cls(tuple(states), tuple(probabilities))

    def to_dict(self) -> dict[str, list]:
        return {
            "states": [list(state) for state in self.states],
            "probs": list(self.probabilities),
        }


@dataclass(frozen=True)
class OptionSettingKey:
    """A low-level option calibration setting."""

    subtask_edge: TaskEdge
    communication_capacity: float
    start_state: StateVector

    def __post_init__(self) -> None:
        if (
            not isinstance(self.subtask_edge, tuple)
            or len(self.subtask_edge) != 2
            or any(not isinstance(state, int) for state in self.subtask_edge)
        ):
            raise ValueError("Option setting subtask_edge must be a pair of integers.")
        if self.subtask_edge[0] == self.subtask_edge[1]:
            raise ValueError("Option setting subtask_edge cannot be a self-loop.")
        capacity = float(self.communication_capacity)
        if not math.isfinite(capacity) or capacity < 0.0:
            raise ValueError("Communication capacity must be finite and non-negative.")
        object.__setattr__(self, "communication_capacity", capacity)
        object.__setattr__(self, "start_state", normalize_state(self.start_state))


class CalibrationOutcomeKind(str, Enum):
    FAILURE = "failure"
    SUCCESS = "success"


@dataclass(frozen=True)
class CalibrationOutcome:
    """A single categorical outcome from one calibration episode."""

    kind: CalibrationOutcomeKind
    terminal_state: StateVector | None = None

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            object.__setattr__(self, "kind", CalibrationOutcomeKind(self.kind))
        if self.kind == CalibrationOutcomeKind.FAILURE:
            if self.terminal_state is not None:
                raise ValueError("Failure outcomes cannot have a terminal state.")
        elif self.terminal_state is None:
            raise ValueError("Success outcomes must have a terminal state.")
        else:
            object.__setattr__(
                self,
                "terminal_state",
                normalize_state(self.terminal_state),
            )

    @classmethod
    def failure(cls) -> CalibrationOutcome:
        return cls(CalibrationOutcomeKind.FAILURE)

    @classmethod
    def success(cls, terminal_state: object) -> CalibrationOutcome:
        if terminal_state is None:
            raise ValueError("Success outcomes must have a terminal state.")
        return cls(CalibrationOutcomeKind.SUCCESS, normalize_state(terminal_state))


@dataclass(frozen=True)
class CalibrationRow:
    """Independent calibration episodes for one statewise option setting."""

    setting: OptionSettingKey
    outcomes: tuple[CalibrationOutcome, ...]
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.seed, (int, np.integer)):
            raise TypeError("Calibration row seed must be an integer.")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        if not self.outcomes:
            raise ValueError("A calibration row must contain at least one outcome.")

    @property
    def success_rate(self) -> float:
        return sum(
            outcome.kind == CalibrationOutcomeKind.SUCCESS for outcome in self.outcomes
        ) / len(self.outcomes)

    @property
    def success_count(self) -> int:
        return sum(
            outcome.kind == CalibrationOutcomeKind.SUCCESS for outcome in self.outcomes
        )

    @property
    def failure_count(self) -> int:
        return sum(
            outcome.kind == CalibrationOutcomeKind.FAILURE for outcome in self.outcomes
        )

    @property
    def terminal_state_distribution(self) -> StateDistribution:
        successful_states = [
            outcome.terminal_state
            for outcome in self.outcomes
            if outcome.kind == CalibrationOutcomeKind.SUCCESS
        ]
        if not successful_states:
            return StateDistribution.empty()

        counts: dict[StateVector, int] = {}
        for state in successful_states:
            if state is not None:
                counts[state] = counts.get(state, 0) + 1
        states = tuple(counts)
        probabilities = tuple(
            count / len(successful_states) for count in counts.values()
        )
        return StateDistribution(states, probabilities)

    @property
    def outcome_distribution(self) -> dict[str, float]:
        return {
            "failure": 1.0 - self.success_rate,
            "success": self.success_rate,
        }


@dataclass(frozen=True)
class LowLevelPolicyArtifact:
    """A loadable snapshot of one seed-specific low-level option policy."""

    seed: int
    subtask_edge: TaskEdge
    communication_capacity: float
    initial_state_distribution: StateDistribution
    timestep: int
    checkpoint_path: str
    config_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.seed, (int, np.integer)):
            raise TypeError("Policy artifact seed must be an integer.")
        if (
            not isinstance(self.subtask_edge, tuple)
            or len(self.subtask_edge) != 2
            or any(not isinstance(state, int) for state in self.subtask_edge)
        ):
            raise ValueError("Policy artifact subtask_edge must be a pair of integers.")
        if self.subtask_edge[0] == self.subtask_edge[1]:
            raise ValueError("Policy artifact subtask_edge cannot be a self-loop.")
        if not isinstance(self.timestep, (int, np.integer)) or self.timestep < 0:
            raise ValueError("Policy artifact timestep must be a non-negative integer.")
        if not isinstance(self.checkpoint_path, str) or not self.checkpoint_path:
            raise ValueError("Policy artifact checkpoint_path must be non-empty.")
        if not isinstance(self.config_hash, str) or not self.config_hash:
            raise ValueError("Policy artifact config_hash must be non-empty.")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "timestep", int(self.timestep))
        object.__setattr__(
            self, "communication_capacity", float(self.communication_capacity)
        )


class PolicyScope(str, Enum):
    SHARED_ACROSS_CONTEXTS = "shared_across_contexts"
    PER_CONTEXT = "per_context"


class CertificateStatus(str, Enum):
    CERTIFIED = "certified"
    NO_CERTIFIED_CANDIDATE = "no_certified_candidate"
    INFEASIBLE = "infeasible"
    SEARCH_FAILED = "search_failed"


@dataclass(frozen=True)
class ContextKey:
    """The exact context identity used for policy and evaluation isolation."""

    context_id: str
    policy_scope: PolicyScope
    high_level_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.context_id, str) or not self.context_id:
            raise ValueError("Context key context_id must be non-empty.")
        if isinstance(self.policy_scope, str):
            object.__setattr__(self, "policy_scope", PolicyScope(self.policy_scope))
        if not isinstance(self.high_level_seed, (int, np.integer)):
            raise TypeError("Context key high_level_seed must be an integer.")
        object.__setattr__(self, "high_level_seed", int(self.high_level_seed))


@dataclass(frozen=True)
class EvaluationRowKey:
    """The exact key for one project or calibration evaluation row."""

    low_level_seed: int
    high_level_seed: int
    context_id: str
    subtask_edge: TaskEdge
    communication_capacity: float
    start_state: StateVector
    episode_index: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.subtask_edge, tuple)
            or len(self.subtask_edge) != 2
            or any(not isinstance(state, int) for state in self.subtask_edge)
        ):
            raise ValueError("Evaluation row subtask_edge must be a pair of integers.")
        if self.subtask_edge[0] == self.subtask_edge[1]:
            raise ValueError("Evaluation row subtask_edge cannot be a self-loop.")
        for field_name in ("low_level_seed", "high_level_seed", "episode_index"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, np.integer)):
                raise TypeError(f"Evaluation row key {field_name} must be an integer.")
            object.__setattr__(self, field_name, int(value))
        if self.episode_index < 0:
            raise ValueError("Evaluation row key episode_index cannot be negative.")
        if not isinstance(self.context_id, str) or not self.context_id:
            raise ValueError("Evaluation row key context_id must be non-empty.")
        object.__setattr__(
            self,
            "communication_capacity",
            float(self.communication_capacity),
        )
        object.__setattr__(self, "start_state", normalize_state(self.start_state))


@dataclass
class HighLevelCandidate:
    """A seed-specific synthesized high-level policy and its evaluation state."""

    context_key: ContextKey
    goal_probability_spec: float
    empirical_goal_probability: float
    empirical_cost: float
    status: CertificateStatus
    iteration: int
    task_policy: Any | None = None
    comms_budget_policy: Any | None = None
    induced_initial_state_distributions: Mapping[int, StateDistribution] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for field_name in (
            "goal_probability_spec",
            "empirical_goal_probability",
            "empirical_cost",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
            object.__setattr__(self, field_name, value)
        for field_name in ("goal_probability_spec", "empirical_goal_probability"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
        if self.empirical_cost < 0.0:
            raise ValueError("empirical_cost must be non-negative.")
        if isinstance(self.status, str):
            object.__setattr__(self, "status", CertificateStatus(self.status))
        if not isinstance(self.iteration, (int, np.integer)) or self.iteration < 0:
            raise ValueError("High-level candidate iteration must be non-negative.")
        object.__setattr__(self, "iteration", int(self.iteration))
        object.__setattr__(
            self,
            "induced_initial_state_distributions",
            dict(self.induced_initial_state_distributions),
        )


@dataclass(frozen=True)
class HeldOutResult:
    """One independent held-out project evaluation."""

    row_key: EvaluationRowKey
    project_success: bool
    communication_cost: float
    successful_terminal_state: StateVector | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.project_success, (bool, np.bool_)):
            raise TypeError("Held-out project_success must be boolean.")
        cost = float(self.communication_cost)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(
                "Held-out communication cost must be finite and non-negative."
            )
        object.__setattr__(self, "project_success", bool(self.project_success))
        object.__setattr__(self, "communication_cost", cost)
        if self.successful_terminal_state is not None:
            object.__setattr__(
                self,
                "successful_terminal_state",
                normalize_state(self.successful_terminal_state),
            )


@dataclass(frozen=True)
class HeldOutSummary:
    """Exact-key held-out summary without cross-seed or cross-context pooling."""

    row_key: EvaluationRowKey
    n_episodes: int
    success_count: int
    project_success_probability: float
    mean_communication_cost: float
    successful_terminal_state_distribution: StateDistribution
