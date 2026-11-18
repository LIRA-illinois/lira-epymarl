import pandas as pd

from src.simulation.context_utils import (
    evaluation_context_label,
    plot_combined_context_evaluation,
    plot_context_evaluation,
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
