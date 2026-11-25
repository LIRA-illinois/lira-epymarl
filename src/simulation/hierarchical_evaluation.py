"""Held-out evaluation summaries keyed to seed and policy context."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.simulation.hierarchical_types import (
    HeldOutResult,
    HeldOutSummary,
    StateDistribution,
)


def summarize_held_out_results(
    results: Sequence[HeldOutResult],
) -> HeldOutSummary:
    """Pool episodes only within one exact seed/context/option/start-state key."""
    if not results:
        raise ValueError("Held-out results cannot be empty.")

    first_key = results[0].row_key
    episode_indices: set[int] = set()
    success_count = 0
    communication_cost = 0.0
    successful_states: dict[tuple, int] = {}

    for result in results:
        key = result.row_key
        if key.episode_index in episode_indices:
            raise ValueError(
                f"Duplicate held-out episode index {key.episode_index} "
                "for one exact evaluation key."
            )
        episode_indices.add(key.episode_index)
        if (
            key.low_level_seed != first_key.low_level_seed
            or key.high_level_seed != first_key.high_level_seed
            or key.context_id != first_key.context_id
            or key.subtask_edge != first_key.subtask_edge
            or key.communication_capacity != first_key.communication_capacity
            or key.start_state != first_key.start_state
        ):
            raise ValueError(
                "Held-out episode results span multiple seed/context/option keys."
            )
        if result.project_success:
            success_count += 1
            if result.successful_terminal_state is not None:
                state = result.successful_terminal_state
                successful_states[state] = successful_states.get(state, 0) + 1
        communication_cost += result.communication_cost

    n_episodes = len(results)
    states = tuple(successful_states)
    if states:
        successful_terminal_count = sum(successful_states.values())
        terminal_distribution = StateDistribution(
            states,
            tuple(
                successful_states[state] / successful_terminal_count for state in states
            ),
        )
    else:
        terminal_distribution = StateDistribution.empty()

    return HeldOutSummary(
        row_key=first_key,
        n_episodes=n_episodes,
        success_count=success_count,
        project_success_probability=success_count / n_episodes,
        mean_communication_cost=communication_cost / n_episodes,
        successful_terminal_state_distribution=terminal_distribution,
    )


def held_out_meets_screen(
    summary: HeldOutSummary,
    goal_probability_spec: float,
    epsilon_p: float,
) -> bool:
    """Return whether held-out empirical success clears the screening threshold."""
    threshold = float(goal_probability_spec) + float(epsilon_p)
    return summary.project_success_probability >= threshold


def aggregate_seed_frontier_artifacts(
    scenario_artifact_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate only completed seed frontier tables for one scenario."""
    root = Path(scenario_artifact_root)
    frontier_paths = sorted(root.glob("**/pg_frontier.csv"))
    if not frontier_paths:
        raise ValueError(
            f"No seed pg_frontier.csv artifacts found under {root}."
        )

    seed_frames = []
    seen_seeds: set[int] = set()
    for frontier_path in frontier_paths:
        frame = pd.read_csv(frontier_path)
        required = {"seed", "pg", "status"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Frontier artifact {frontier_path} is missing "
                f"{sorted(missing)}."
            )
        for seed in frame["seed"].dropna().astype(int).unique():
            if seed in seen_seeds:
                raise ValueError(
                    f"Duplicate seed {seed} frontier artifact."
                )
            seen_seeds.add(seed)
        seed_frames.append(frame.assign(source=str(frontier_path)))

    raw = pd.concat(seed_frames, ignore_index=True)
    numeric = raw.dropna(subset=["empirical_goal_probability", "empirical_cost"])
    summary_rows = []
    for pg, pg_raw in raw.groupby("pg", dropna=False):
        pg_numeric = numeric[numeric["pg"] == pg]
        status_counts = (
            pg_raw.groupby("status", dropna=False)
            .size()
            .to_dict()
        )
        summary_rows.append(
            {
                "pg": pg,
                "seed_count": pg_raw["seed"].nunique(),
                "success_median": (
                    pg_numeric["empirical_goal_probability"].median()
                    if not pg_numeric.empty
                    else float("nan")
                ),
                "success_min": (
                    pg_numeric["empirical_goal_probability"].min()
                    if not pg_numeric.empty
                    else float("nan")
                ),
                "success_max": (
                    pg_numeric["empirical_goal_probability"].max()
                    if not pg_numeric.empty
                    else float("nan")
                ),
                "cost_median": (
                    pg_numeric["empirical_cost"].median()
                    if not pg_numeric.empty
                    else float("nan")
                ),
                "cost_min": (
                    pg_numeric["empirical_cost"].min()
                    if not pg_numeric.empty
                    else float("nan")
                ),
                "cost_max": (
                    pg_numeric["empirical_cost"].max()
                    if not pg_numeric.empty
                    else float("nan")
                ),
                "status_counts": str(status_counts),
                "certified_seed_count": int(
                    status_counts.get("certified", 0)
                ),
                "infeasible_seed_count": int(
                    status_counts.get("infeasible", 0)
                ),
                "no_certified_candidate_seed_count": int(
                    status_counts.get("no_certified_candidate", 0)
                ),
                "search_failed_seed_count": int(
                    status_counts.get("search_failed", 0)
                ),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows)
    return raw, summary


def aggregate_seed_held_out_artifacts(
    scenario_artifact_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate exact held-out seed tables without mixing exact keys."""
    root = Path(scenario_artifact_root)
    held_out_paths = sorted(root.glob("**/held_out_results.csv"))
    if not held_out_paths:
        raise ValueError(
            f"No seed held_out_results.csv artifacts found under {root}."
        )

    frames = []
    seen_seeds: set[int] = set()
    for held_out_path in held_out_paths:
        frame = pd.read_csv(held_out_path)
        required = {
            "seed",
            "pg",
            "status",
            "n_episodes",
            "project_success_probability",
            "empirical_cost",
        }
        missing = required - set(frame.columns)
        if missing:
            continue
        for seed in frame["seed"].dropna().astype(int).unique():
            if seed in seen_seeds:
                raise ValueError(
                    f"Duplicate seed {seed} held-out artifact."
                )
            seen_seeds.add(seed)
        frames.append(frame.assign(source=str(held_out_path)))

    if not frames:
            raise ValueError(
                f"No seed-level held_out_results.csv artifacts found under {root}."
            )
    raw = pd.concat(frames, ignore_index=True)
    summary_rows = []
    for pg, pg_raw in raw.groupby("pg", dropna=False):
        numeric = pg_raw.dropna(
            subset=["project_success_probability", "empirical_cost"]
        )
        summary_rows.append(
            {
                "pg": pg,
                "seed_count": pg_raw["seed"].nunique(),
                "success_median": (
                    numeric["project_success_probability"].median()
                    if not numeric.empty
                    else float("nan")
                ),
                "success_min": (
                    numeric["project_success_probability"].min()
                    if not numeric.empty
                    else float("nan")
                ),
                "success_max": (
                    numeric["project_success_probability"].max()
                    if not numeric.empty
                    else float("nan")
                ),
                "cost_median": (
                    numeric["empirical_cost"].median()
                    if not numeric.empty
                    else float("nan")
                ),
                "cost_min": (
                    numeric["empirical_cost"].min()
                    if not numeric.empty
                    else float("nan")
                ),
                "cost_max": (
                    numeric["empirical_cost"].max()
                    if not numeric.empty
                    else float("nan")
                ),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows)
    return raw, summary


def write_seed_postprocessing(
    scenario_artifact_root: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write raw and sensitivity tables for one completed scenario."""
    root = Path(scenario_artifact_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frontier_raw, frontier_summary = aggregate_seed_frontier_artifacts(root)
    held_out_raw, held_out_summary = aggregate_seed_held_out_artifacts(root)

    paths = {
        "frontier_raw": str(output / "frontier_raw.csv"),
        "frontier_summary": str(output / "frontier_summary.csv"),
        "held_out_raw": str(output / "held_out_raw.csv"),
        "held_out_summary": str(output / "held_out_summary.csv"),
    }
    frontier_raw.to_csv(paths["frontier_raw"], index=False)
    frontier_summary.to_csv(paths["frontier_summary"], index=False)
    held_out_raw.to_csv(paths["held_out_raw"], index=False)
    held_out_summary.to_csv(paths["held_out_summary"], index=False)
    return paths
