import pandas as pd
import pytest

from src.simulation.hierarchical_evaluation import (
    aggregate_seed_frontier_artifacts,
    aggregate_seed_held_out_artifacts,
    held_out_meets_screen,
    summarize_held_out_results,
    write_seed_postprocessing,
)
from src.simulation.hierarchical_types import (
    EvaluationRowKey,
    HeldOutResult,
)


def _row_key(episode_index, low_level_seed=11, high_level_seed=11):
    return EvaluationRowKey(
        low_level_seed=low_level_seed,
        high_level_seed=high_level_seed,
        context_id="test-map",
        subtask_edge=(0, 1),
        communication_capacity=1.0,
        start_state=(0, 0),
        episode_index=episode_index,
    )


def test_summarize_held_out_results_preserves_exact_key():
    results = (
        HeldOutResult(
            row_key=_row_key(0),
            project_success=True,
            communication_cost=1.0,
            successful_terminal_state=(1, 0),
        ),
        HeldOutResult(
            row_key=_row_key(1),
            project_success=False,
            communication_cost=2.0,
        ),
        HeldOutResult(
            row_key=_row_key(2),
            project_success=True,
            communication_cost=1.5,
            successful_terminal_state=(1, 1),
        ),
    )

    summary = summarize_held_out_results(results)

    assert summary.n_episodes == 3
    assert summary.success_count == 2
    assert summary.project_success_probability == pytest.approx(2 / 3)
    assert summary.mean_communication_cost == pytest.approx(1.5)
    assert summary.successful_terminal_state_distribution.states == (
        (1, 0),
        (1, 1),
    )
    assert summary.successful_terminal_state_distribution.probabilities == (
        0.5,
        0.5,
    )
    assert held_out_meets_screen(summary, 0.6, 0.05)
    assert not held_out_meets_screen(summary, 0.7, 0.0)


def test_summarize_held_out_results_rejects_duplicate_episode():
    results = (
        HeldOutResult(
            row_key=_row_key(0),
            project_success=True,
            communication_cost=1.0,
        ),
        HeldOutResult(
            row_key=_row_key(0),
            project_success=False,
            communication_cost=1.0,
        ),
    )

    with pytest.raises(ValueError, match="Duplicate held-out episode"):
        summarize_held_out_results(results)


def test_aggregate_seed_frontier_artifacts_preserves_seed_status(tmp_path):
    seed_root = tmp_path / "seed_101"
    seed_root.mkdir()
    pd.DataFrame(
        [
            {
                "seed": 101,
                "pg": 0.55,
                "status": "certified",
                "empirical_goal_probability": 1.0,
                "empirical_cost": 0.0,
            },
            {
                "seed": 101,
                "pg": 0.65,
                "status": "infeasible",
                "empirical_goal_probability": None,
                "empirical_cost": None,
            },
        ]
    ).to_csv(seed_root / "pg_frontier.csv", index=False)

    raw, summary = aggregate_seed_frontier_artifacts(tmp_path)

    assert len(raw) == 2
    assert set(raw["status"]) == {"certified", "infeasible"}
    assert summary.loc[summary["pg"] == 0.55, "success_median"].item() == 1.0


def test_aggregate_seed_held_out_artifacts_and_writer(tmp_path):
    seed_root = tmp_path / "seed_101"
    seed_root.mkdir()
    pd.DataFrame(
        [
            {
                "seed": 101,
                "pg": 0.55,
                "status": "certified",
                "n_episodes": 20,
                "project_success_probability": 1.0,
                "empirical_cost": 0.0,
            },
            {
                "seed": 101,
                "pg": 0.65,
                "status": "certified",
                "n_episodes": 20,
                "project_success_probability": 0.95,
                "empirical_cost": 1.0,
            },
        ]
    ).to_csv(seed_root / "held_out_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "seed": 101,
                "pg": 0.55,
                "status": "certified",
                "empirical_goal_probability": 1.0,
                "empirical_cost": 0.0,
            }
        ]
    ).to_csv(seed_root / "pg_frontier.csv", index=False)

    held_out_raw, held_out_summary = aggregate_seed_held_out_artifacts(tmp_path)
    assert len(held_out_raw) == 2
    assert held_out_summary.loc[
        held_out_summary["pg"] == 0.55,
        "success_median",
    ].item() == 1.0

    output = tmp_path / "postprocessing"
    paths = write_seed_postprocessing(tmp_path, output)
    assert (output / "frontier_raw.csv").is_file()
    assert (output / "frontier_summary.csv").is_file()
    assert (output / "held_out_raw.csv").is_file()
    assert (output / "held_out_summary.csv").is_file()
    assert set(paths) == {
        "frontier_raw",
        "frontier_summary",
        "held_out_raw",
        "held_out_summary",
    }


def test_aggregate_seed_held_out_ignores_exact_candidate_tables(tmp_path):
    seed_root = tmp_path / "seed_101"
    candidate_dir = seed_root / "context_unified" / "pg_0.55"
    candidate_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "seed": 101,
                "pg": 0.55,
                "context_id": "unified-pg-0.55",
                "edge": "[0, 1]",
                "communication_capacity": 0.0,
                "start_state": "[0, 0]",
                "n_episodes": 20,
                "project_success_probability": 1.0,
            }
        ]
    ).to_csv(candidate_dir / "held_out_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "seed": 101,
                "pg": 0.55,
                "status": "certified",
                "n_episodes": 20,
                "project_success_probability": 1.0,
                "empirical_cost": 0.0,
            }
        ]
    ).to_csv(seed_root / "held_out_results.csv", index=False)

    raw, _ = aggregate_seed_held_out_artifacts(tmp_path)

    assert len(raw) == 1
    assert raw.iloc[0]["status"] == "certified"


def test_aggregate_seed_frontier_rejects_duplicate_seed(tmp_path):
    for subdir in ("scenario_a", "scenario_b"):
        seed_root = tmp_path / subdir / "seed_101"
        seed_root.mkdir(parents=True)
        pd.DataFrame(
            [{"seed": 101, "pg": 0.55, "status": "certified"}]
        ).to_csv(seed_root / "pg_frontier.csv", index=False)

    with pytest.raises(ValueError, match="Duplicate seed"):
        aggregate_seed_frontier_artifacts(tmp_path)


def test_summarize_held_out_results_rejects_cross_seed_pooling():
    results = (
        HeldOutResult(
            row_key=_row_key(0, low_level_seed=11),
            project_success=True,
            communication_cost=1.0,
        ),
        HeldOutResult(
            row_key=_row_key(1, low_level_seed=12),
            project_success=True,
            communication_cost=1.0,
        ),
    )

    with pytest.raises(ValueError, match="multiple seed/context/option"):
        summarize_held_out_results(results)
