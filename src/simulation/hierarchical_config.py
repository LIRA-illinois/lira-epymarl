"""Configuration contracts for unified hierarchical experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum


class HierarchicalPolicyScope(str, Enum):
    SHARED_ACROSS_CONTEXTS = "shared_across_contexts"
    PER_CONTEXT = "per_context"


class HierarchicalResumeMode(str, Enum):
    NONE = "none"
    LATEST = "latest"
    SPECIFIC_TIMESTEP = "specific_timestep"


class LowLevelMode(str, Enum):
    TRAIN = "train"
    CALIBRATION_ONLY = "calibration_only"
    EVALUATION_ONLY = "evaluation_only"


@dataclass(frozen=True)
class UnifiedHierarchicalConfig:
    task_sequence: tuple[tuple[int, int], ...]
    policy_scope: HierarchicalPolicyScope = (
        HierarchicalPolicyScope.SHARED_ACROSS_CONTEXTS
    )
    pg_values: tuple[float, ...] = (0.95,)
    epsilon_est: float = 0.05
    beta: float = 0.95
    epsilon_p: float | None = None
    n_max: int = 1
    n_calibration_episodes: int | None = None
    max_iterations: int = 20
    resume: HierarchicalResumeMode = HierarchicalResumeMode.LATEST
    low_level_mode: LowLevelMode = LowLevelMode.TRAIN
    held_out_episodes: int | None = None
    contexts: tuple[str, ...] = ()
    resume_timestep: int | None = None
    mock_evaluation_times: tuple[int, ...] = (
        20_000,
        40_000,
        60_000,
        80_000,
        100_000,
    )

    def __post_init__(self) -> None:
        edges = tuple((int(edge[0]), int(edge[1])) for edge in self.task_sequence)
        if not edges:
            raise ValueError("Unified hierarchical task_sequence cannot be empty.")
        if any(from_state == to_state for from_state, to_state in edges):
            raise ValueError(
                "Unified hierarchical task_sequence cannot contain self-loops."
            )
        if len(set(edges)) != len(edges):
            raise ValueError(
                "Unified hierarchical task_sequence cannot contain duplicate edges."
            )
        object.__setattr__(self, "task_sequence", edges)

        source_states = {from_state for from_state, _ in edges}
        destination_states = {to_state for _, to_state in edges}

        outgoing: dict[int, list[tuple[int, int]]] = {}
        indegree: dict[int, int] = {}
        states = set()
        for edge in edges:
            outgoing.setdefault(edge[0], []).append(edge)
            indegree.setdefault(edge[0], 0)
            indegree[edge[1]] = indegree.get(edge[1], 0) + 1
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
            raise ValueError("Unified hierarchical task_sequence must be acyclic.")
        root_states = source_states - destination_states
        if len(root_states) != 1:
            raise ValueError(
                "Unified hierarchical task_sequence must describe one rooted DAG."
            )

        if isinstance(self.policy_scope, str):
            object.__setattr__(
                self,
                "policy_scope",
                HierarchicalPolicyScope(self.policy_scope),
            )
        if isinstance(self.resume, str):
            object.__setattr__(
                self,
                "resume",
                HierarchicalResumeMode(self.resume),
            )
        if isinstance(self.low_level_mode, str):
            object.__setattr__(
                self,
                "low_level_mode",
                LowLevelMode(self.low_level_mode),
            )
        if (
            self.low_level_mode
            in (
                LowLevelMode.CALIBRATION_ONLY,
                LowLevelMode.EVALUATION_ONLY,
            )
            and self.resume == HierarchicalResumeMode.NONE
        ):
            raise ValueError(
                "Calibration-only and evaluation-only modes require "
                "a loaded option artifact."
            )

        pg_values = tuple(sorted({float(value) for value in self.pg_values}))
        if not pg_values:
            raise ValueError("Unified hierarchical pg_values cannot be empty.")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in pg_values
        ):
            raise ValueError(
                "Unified hierarchical pg_values must contain probabilities in [0, 1]."
            )
        object.__setattr__(self, "pg_values", pg_values)

        epsilon_est = float(self.epsilon_est)
        beta = float(self.beta)
        if not 0.0 < epsilon_est < 1.0:
            raise ValueError("Unified hierarchical epsilon_est must be in (0, 1).")
        if not 0.0 < beta < 1.0:
            raise ValueError("Unified hierarchical beta must be in (0, 1).")
        if not isinstance(self.n_max, int) or self.n_max < 1:
            raise TypeError("Unified hierarchical n_max must be a positive integer.")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise TypeError(
                "Unified hierarchical max_iterations must be a positive integer."
            )
        object.__setattr__(self, "epsilon_est", epsilon_est)
        object.__setattr__(self, "beta", beta)

        delta_p = 1.0 - (1.0 - epsilon_est) ** self.n_max
        if self.epsilon_p is None:
            epsilon_p = delta_p
        else:
            epsilon_p = float(self.epsilon_p)
            if not math.isfinite(epsilon_p) or epsilon_p < 0.0:
                raise ValueError(
                    "Unified hierarchical epsilon_p must be finite and non-negative."
                )
            if epsilon_p + 1e-12 < delta_p:
                raise ValueError(
                    f"Unified hierarchical epsilon_p={epsilon_p} must be at least "
                    f"Delta_p={delta_p}."
                )
        object.__setattr__(self, "epsilon_p", epsilon_p)

        if self.n_calibration_episodes is not None and (
            not isinstance(self.n_calibration_episodes, int)
            or self.n_calibration_episodes < 1
        ):
            raise TypeError(
                "Unified hierarchical n_calibration_episodes must be positive."
            )
        if self.held_out_episodes is not None and (
            not isinstance(self.held_out_episodes, int) or self.held_out_episodes < 1
        ):
            raise TypeError("Unified hierarchical held_out_episodes must be positive.")
        if self.resume_timestep is not None and (
            not isinstance(self.resume_timestep, int)
            or self.resume_timestep < 0
        ):
            raise TypeError(
                "Unified hierarchical resume_timestep must be a non-negative integer."
            )
        if (
            self.resume == HierarchicalResumeMode.SPECIFIC_TIMESTEP
            and self.resume_timestep is None
        ):
            raise ValueError(
                "Specific-timestep resume requires resume_timestep."
            )

        contexts = tuple(str(context) for context in self.contexts)
        if len(set(contexts)) != len(contexts):
            raise ValueError("Unified hierarchical contexts must be unique.")
        if (
            self.policy_scope == HierarchicalPolicyScope.PER_CONTEXT
            and not contexts
        ):
            raise ValueError(
                "Per-context unified hierarchical policy requires contexts."
            )
        object.__setattr__(self, "contexts", contexts)

        mock_evaluation_times = tuple(
            sorted({int(t_env) for t_env in self.mock_evaluation_times})
        )
        if len(mock_evaluation_times) < 5:
            raise ValueError(
                "Unified hierarchical mock_evaluation_times must contain "
                "at least five checkpoints."
            )
        if any(t_env < 0 for t_env in mock_evaluation_times):
            raise ValueError(
                "Unified hierarchical mock_evaluation_times cannot be negative."
            )
        object.__setattr__(
            self,
            "mock_evaluation_times",
            mock_evaluation_times,
        )

    @property
    def delta_p(self) -> float:
        return 1.0 - (1.0 - self.epsilon_est) ** self.n_max

    def validate_calibration_count(self, m_est: int) -> None:
        if not isinstance(m_est, int) or m_est < 1:
            raise TypeError("m_est must be a positive integer.")
        if (
            self.n_calibration_episodes is not None
            and self.n_calibration_episodes < m_est
        ):
            raise ValueError(
                f"n_calibration_episodes={self.n_calibration_episodes} "
                f"must be at least m_est={m_est}."
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> UnifiedHierarchicalConfig:
        task_sequence = value.get(
            "task_sequence",
            value.get("hl_task_sequence"),
        )
        if not isinstance(task_sequence, Sequence) or isinstance(
            task_sequence,
            (str, bytes),
        ):
            raise TypeError(
                "Unified hierarchical task_sequence must be a sequence of edges."
            )

        pg_values = value.get(
            "pg_values",
            value.get(
                "p_g_values",
                value.get("postprocess_success_rate_specs", (0.95,)),
            ),
        )
        if isinstance(pg_values, (int, float)):
            pg_values = (pg_values,)

        if "policy_seeds" in value:
            raise ValueError(
                "Unified hierarchical policy_seeds is no longer supported; "
                "use the outer experiment seed."
            )

        contexts = value.get("contexts", ())
        if isinstance(contexts, str):
            contexts = (contexts,)

        epsilon_p = value.get(
            "epsilon_p",
            value.get("project_reliability_margin"),
        )

        return cls(
            task_sequence=tuple(tuple(edge) for edge in task_sequence),
            policy_scope=HierarchicalPolicyScope(
                value.get(
                    "policy_scope",
                    HierarchicalPolicyScope.SHARED_ACROSS_CONTEXTS.value,
                )
            ),
            pg_values=tuple(pg_values),
            epsilon_est=float(
                value.get(
                    "epsilon_est",
                    value.get("epsilon_estimation", 0.05),
                )
            ),
            beta=float(value.get("beta", 0.95)),
            epsilon_p=(None if epsilon_p is None else float(epsilon_p)),
            n_max=int(
                value.get(
                    "n_max",
                    value.get("n_max_project_executions", 1),
                )
            ),
            n_calibration_episodes=(
                None
                if value.get(
                    "n_calibration_episodes",
                    value.get("n_estimation_episodes"),
                )
                is None
                else int(
                    value.get(
                        "n_calibration_episodes",
                        value.get("n_estimation_episodes"),
                    )
                )
            ),
            max_iterations=int(value.get("max_iterations", 20)),
            resume=HierarchicalResumeMode(value.get("resume", "latest")),
            low_level_mode=LowLevelMode(
                value.get("low_level_mode", "train")
            ),
            held_out_episodes=(
                None
                if value.get("held_out_episodes") is None
                else int(value["held_out_episodes"])
            ),
            contexts=tuple(contexts),
            resume_timestep=(
                None
                if value.get("resume_timestep") is None
                else int(value["resume_timestep"])
            ),
            mock_evaluation_times=tuple(
                value.get(
                    "mock_evaluation_times",
                    (20_000, 40_000, 60_000, 80_000, 100_000),
                )
            ),
        )

    def to_dict(self) -> dict:
        return {
            "task_sequence": [list(edge) for edge in self.task_sequence],
            "policy_scope": self.policy_scope.value,
            "pg_values": list(self.pg_values),
            "epsilon_est": self.epsilon_est,
            "beta": self.beta,
            "epsilon_p": self.epsilon_p,
            "n_max": self.n_max,
            "n_calibration_episodes": self.n_calibration_episodes,
            "max_iterations": self.max_iterations,
            "resume": self.resume.value,
            "low_level_mode": self.low_level_mode.value,
            "held_out_episodes": self.held_out_episodes,
            "contexts": list(self.contexts),
            "resume_timestep": self.resume_timestep,
            "mock_evaluation_times": list(self.mock_evaluation_times),
        }


@dataclass(frozen=True)
class UnifiedRunIdentity:
    experiment: str
    time_id: str
    scenario: str
    seed: int

    def __post_init__(self) -> None:
        for field_name in ("experiment", "time_id", "scenario"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"Unified run identity {field_name} must be non-empty."
                )
        if not isinstance(self.seed, int):
            raise TypeError("Unified run identity seed must be an integer.")

    @property
    def artifact_key(self) -> str:
        return f"{self.time_id}/{self.scenario}/seed_{self.seed}"


def get_unified_hierarchical_config(
    value: Mapping[str, object] | None,
) -> UnifiedHierarchicalConfig | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("Unified hierarchical config must be a mapping.")
    return UnifiedHierarchicalConfig.from_mapping(value)


def is_unified_hierarchical_scenario(scenario: Mapping[str, object]) -> bool:
    return any(
        isinstance(scenario.get(key), Mapping)
        for key in ("hierarchical", "hl_env", "hl_env_config")
    )


def extract_unified_hierarchical_config(
    scenario: Mapping[str, object],
) -> UnifiedHierarchicalConfig | None:
    for key in ("hierarchical", "hl_env", "hl_env_config"):
        value = scenario.get(key)
        if isinstance(value, Mapping):
            return get_unified_hierarchical_config(value)
    return None
