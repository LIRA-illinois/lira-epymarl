"""Stateless helpers used by Simulation and post-processing."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
import pandas as pd

DEFAULT_POSTPROCESS_SUCCESS_RATE_SPECS = tuple(np.round(np.arange(0.0, 1.0, 0.01), 2))


def episode_batch_reward_totals(episode_batch) -> np.ndarray:
    rewards = episode_batch["reward"].detach().cpu().numpy()
    filled = episode_batch["filled"].detach().cpu().numpy().squeeze(-1)
    episode_reward_totals = (rewards * filled[..., None]).sum(axis=1)
    if episode_reward_totals.ndim > 1:
        episode_reward_totals = episode_reward_totals.sum(axis=-1)
    return np.asarray(episode_reward_totals, dtype=float)


def training_reward_stabilized(
    episode_reward_history: list[tuple[int, float]],
    t_env: int,
    window: int,
    min_delta: float,
) -> bool:
    if window <= 0 or t_env < window:
        return False

    window_start = t_env - window
    midpoint = window_start + window / 2
    first_half = [
        episode_reward_total
        for timestamp, episode_reward_total in episode_reward_history
        if window_start <= timestamp < midpoint
    ]
    second_half = [
        episode_reward_total
        for timestamp, episode_reward_total in episode_reward_history
        if timestamp >= midpoint
    ]
    if len(first_half) < 5 or len(second_half) < 5:
        return False

    return abs(float(np.mean(second_half)) - float(np.mean(first_half))) <= min_delta


def success_rate_lcb(
    success_rate: float,
    n_episodes: int,
    confidence: float,
) -> float | None:
    if n_episodes <= 0 or not 0.0 <= success_rate <= 1.0:
        return None
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + z**2 / n_episodes
    center = success_rate + z**2 / (2.0 * n_episodes)
    margin = z * np.sqrt(
        success_rate * (1.0 - success_rate) / n_episodes + z**2 / (4.0 * n_episodes**2)
    )
    return float((center - margin) / denominator)


def map_name_from_config(config: dict) -> str:
    env_args = config.get("env_args", {}) or {}
    return str(env_args.get("map_name", config.get("env_args.map_name", "")))


def configured_message_budgets(value) -> list[float]:
    if isinstance(value, (list, tuple, set)):
        values = value
    elif isinstance(value, str):
        values = value.strip("[]").replace(",", " ").split()
    else:
        values = [value]
    try:
        return [float(item) for item in values]
    except (TypeError, ValueError):
        return []


def postprocess_success_rate_specs(value) -> list[float]:
    if value is None:
        values = DEFAULT_POSTPROCESS_SUCCESS_RATE_SPECS
    elif isinstance(value, (int, float, np.number)):
        values = [value]
    else:
        values = value
    try:
        specs = sorted({float(spec) for spec in values})
    except (TypeError, ValueError) as error:
        raise ValueError(
            "postprocess_success_rate_specs must be a non-empty sequence "
            "of probabilities."
        ) from error
    if not specs:
        raise ValueError("postprocess_success_rate_specs cannot be empty.")
    if any(not np.isfinite(spec) or not 0.0 <= spec <= 1.0 for spec in specs):
        raise ValueError("Every post-processing success-rate spec must be in [0, 1].")
    return specs


def task_edge_from_config(value) -> tuple[int, int]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"Could not parse high-level task {value!r}.") from error
    if not isinstance(value, (list, tuple, np.ndarray, Sequence)) or len(value) != 2:
        raise ValueError(
            "Each high-level task configuration must be a [from_state, to_state] pair."
        )
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid high-level task {value!r}.") from error


def load_test_returns(source_run) -> pd.DataFrame:
    metric_name = "test_return_mean"
    try:
        history = source_run.history(
            keys=["t_env", metric_name],
            pandas=True,
            samples=10000,
        )
    except (AttributeError, TypeError, ValueError):
        return pd.DataFrame(columns=["t_env", "test_return"])
    if history is None or history.empty or metric_name not in history:
        return pd.DataFrame(columns=["t_env", "test_return"])
    test_return_df = history[["t_env", metric_name]].dropna()
    return test_return_df.rename(columns={metric_name: "test_return"})


def wandb_project_path(api, project: str) -> str:
    if "/" in project:
        return project
    entity = getattr(api, "default_entity", None)
    if not entity:
        raise RuntimeError(
            f"Cannot resolve W&B entity for project {project!r}; "
            "specify it as 'entity/project'."
        )
    return f"{entity}/{project}"
