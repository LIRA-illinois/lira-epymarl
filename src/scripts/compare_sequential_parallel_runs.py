"""Compare exp50 batch1 and exp51 batch10 low-level results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_training_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing training evaluation CSV: {path}")
    frame = pd.read_csv(path)
    required = {
        "hl_task",
        "msg_budget_per_agent",
        "test_task_completed_mean",
        "t_env",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training CSV missing columns: {sorted(missing)}")
    return frame


def build_comparison(
    exp50_root: Path,
    exp51_root: Path,
) -> pd.DataFrame:
    sequential = load_training_csv(
        exp50_root / "real_training_eval_seed_101.csv"
    )
    parallel = load_training_csv(
        exp51_root / "real_training_eval_seed_101.csv"
    )
    sequential["low_level_run"] = "exp50_batch1"
    parallel["low_level_run"] = "exp51_batch10"
    return pd.concat([sequential, parallel], ignore_index=True)


def plot_comparison(
    comparison: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sequential_vs_parallel_training.csv"
    comparison.to_csv(csv_path, index=False)

    frame = comparison.copy()
    frame["edge"] = frame["hl_task"].astype(str)
    edge_budget_groups = (
        frame.groupby(["edge", "msg_budget_per_agent"], sort=True)
    )
    n_groups = max(1, len(edge_budget_groups))
    figure, axes = plt.subplots(
        nrows=n_groups,
        ncols=1,
        figsize=(7.2, 3.5 * n_groups),
        squeeze=False,
        sharex=True,
    )
    axes = axes[:, 0]
    for axis, ((edge, budget), group) in zip(
        axes,
        edge_budget_groups,
    ):
        for run_name, run_group in group.groupby("low_level_run"):
            run_group = run_group.sort_values("t_env")
            axis.plot(
                run_group["t_env"],
                run_group["test_task_completed_mean"],
                marker="o",
                label=run_name,
            )
        axis.set_title(f"Edge {edge}, communication {budget:g}")
        axis.set_ylabel("Empirical success")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True)
        axis.legend()
    axes[-1].set_xlabel("Environment steps")
    figure.tight_layout()

    png_path = output_dir / "sequential_vs_parallel_training.png"
    pdf_path = output_dir / "sequential_vs_parallel_training.pdf"
    figure.savefig(png_path, dpi=500)
    figure.savefig(pdf_path)
    return {
        "csv": str(csv_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp50-root",
        default="results/hierarchical_artifacts/exp50_real_1000/01/seed_101",
    )
    parser.add_argument(
        "--exp51-root",
        default="results/hierarchical_artifacts/exp51_real_batch10/01/seed_101",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hierarchical_artifacts/exp50_exp51_comparison",
    )
    args = parser.parse_args()

    comparison = build_comparison(
        Path(args.exp50_root),
        Path(args.exp51_root),
    )
    paths = plot_comparison(comparison, Path(args.output_dir))
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
