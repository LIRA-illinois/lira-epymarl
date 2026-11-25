import pandas as pd

from src.simulation.context_utils import (
    evaluation_context_label,
    plot_combined_context_evaluation,
    plot_context_evaluation,
    plot_mixed_initial_distributions,
)


def _context_evaluation_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "hl_task": (0, 1),
                "msg_budget_per_agent": 0,
                "t_env": 100,
                "test_return_mean": 1.0,
                "test_return_std": 0.1,
                "test_task_completed_mean": 0.8,
                "test_ep_length_mean": 10.0,
                "test_n_episodes": 20,
            },
            {
                "hl_task": (1, 2),
                "msg_budget_per_agent": 1,
                "t_env": 100,
                "test_return_mean": 2.0,
                "test_return_std": 0.2,
                "test_task_completed_mean": 0.9,
                "test_ep_length_mean": 8.0,
                "test_n_episodes": 20,
            },
        ]
    )


def test_evaluation_context_label_includes_task_and_budget():
    assert (
        evaluation_context_label({"hl_task": [0, 1], "msg_budget_per_agent": 1.0})
        == "hl_task_0_1/msg_budget_per_agent_1"
    )


def test_context_plot_apis_save_metrics_for_all_contexts(tmp_path):
    rows = _context_evaluation_rows()

    image_paths = plot_context_evaluation(rows, str(tmp_path / "contexts"))
    combined_image_paths = plot_combined_context_evaluation(
        [("task 0-1", rows, None)],
        str(tmp_path / "combined"),
    )

    assert len(image_paths) == 4
    assert len(combined_image_paths) == 4
    assert all(path.endswith(".png") for path in image_paths)
    assert all(path.endswith(".png") for path in combined_image_paths.values())


def test_mixed_initial_distribution_plot_exports_png_pdf_and_csv(tmp_path):
    rows = pd.DataFrame.from_records(
        [
            {
                "seed": 101,
                "pg": 0.65,
                "context_id": "unified",
                "iteration": 0,
                "status": "no_certified_candidate",
                "state": 1,
                "start_state": "((1, 1), (1, 3), (1, 5))",
                "probability": 0.5,
            },
            {
                "seed": 101,
                "pg": 0.65,
                "context_id": "unified",
                "iteration": 0,
                "status": "no_certified_candidate",
                "state": 1,
                "start_state": "((2, 1), (2, 3), (2, 5))",
                "probability": 0.5,
            },
            {
                "seed": 101,
                "pg": 0.65,
                "context_id": "unified",
                "iteration": 1,
                "status": "certified",
                "state": 1,
                "start_state": "((1, 1), (1, 3), (1, 5))",
                "probability": 0.25,
            },
            {
                "seed": 101,
                "pg": 0.65,
                "context_id": "unified",
                "iteration": 1,
                "status": "certified",
                "state": 1,
                "start_state": "((2, 1), (2, 3), (2, 5))",
                "probability": 0.75,
            },
        ]
    )

    paths = plot_mixed_initial_distributions(
        rows,
        str(tmp_path),
        seed=101,
    )

    assert len(paths) == 3
    assert (
        tmp_path / "mixed_initial_distributions_seed_101_pg_0p65.png"
    ).is_file()
    assert (
        tmp_path / "mixed_initial_distributions_seed_101_pg_0p65.pdf"
    ).is_file()
    assert (tmp_path / "mixed_initial_distributions_seed_101.csv").is_file()


def test_mixed_initial_distribution_plot_uses_shared_probability_scale(
    tmp_path,
):
    rows = pd.DataFrame.from_records(
        [
            {
                "pg": pg,
                "iteration": iteration,
                "state": 1,
                "start_state": f"S{start_index}",
                "probability": (
                    0.10
                    + 0.01 * start_index
                    + 0.005 * pg
                    + 0.002 * iteration
                ),
            }
            for pg in (0.55, 0.65, 0.75)
            for iteration in (0, 1, 2)
            for start_index in range(8)
        ]
    )

    paths = plot_mixed_initial_distributions(
        rows,
        str(tmp_path),
        seed=101,
    )

    assert len(paths) == 7
    assert all(path.endswith(".png") or path.endswith(".pdf") or path.endswith(".csv") for path in paths)
