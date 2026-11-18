from collections.abc import Mapping, Sequence
from os import makedirs
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("agg")

CONTEXT_EVALUATION_METRICS = (
    "test_return_mean",
    "test_return_std",
    "test_task_completed_mean",
    "test_ep_length_mean",
)


def evaluation_context_label(log_stats: Mapping[str, object]) -> str:
    context_parts = []
    for key in ("hl_task", "subtask_idx", "msg_budget_per_agent"):
        if key not in log_stats:
            continue
        value = log_stats[key]
        if isinstance(value, (list, tuple)):
            value = "_".join(str(item) for item in value)
        elif isinstance(value, (float, np.floating)) and value.is_integer():
            value = int(value)
        context_parts.append(f"{key}_{value}")
    return "/".join(context_parts) or "default"


def context_sample_count_label(data: pd.DataFrame) -> str:
    if "test_n_episodes" not in data:
        return "episodes per (context, seed) unavailable"
    counts = sorted(
        {
            int(count)
            for count in pd.to_numeric(
                data["test_n_episodes"], errors="coerce"
            ).dropna()
        }
    )
    if not counts:
        return "episodes per (context, seed) unavailable"
    count_label = str(counts[0]) if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"
    return f"{count_label} episodes per (context, seed)"


def aggregate_eval_context_rows(
    data: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    records = []
    for context, rows in data.groupby(
        group_columns, dropna=False, sort=True, observed=True
    ):
        context_values = context if isinstance(context, tuple) else (context,)
        record = dict(zip(group_columns, context_values))
        counts = (
            pd.to_numeric(rows["test_n_episodes"], errors="coerce")
            if "test_n_episodes" in rows
            else pd.Series(1.0, index=rows.index)
        )
        weights = counts.fillna(0).to_numpy(dtype=float)
        total_count = float(weights.sum())
        if "test_n_episodes" in rows:
            record["test_n_episodes"] = int(total_count)

        numeric_columns = rows.select_dtypes(include="number").columns
        for column in numeric_columns:
            if column in group_columns or column == "test_n_episodes":
                continue
            values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            if not valid.any():
                record[column] = np.nan
                continue
            if column.endswith("_std"):
                mean_column = f"{column[:-4]}_mean"
                means = pd.to_numeric(
                    rows.get(mean_column, pd.Series(np.nan, index=rows.index)),
                    errors="coerce",
                ).to_numpy(dtype=float)
                valid &= np.isfinite(means)
                if valid.any() and total_count > 0:
                    combined_mean = float(
                        np.average(means[valid], weights=weights[valid])
                    )
                    variance = np.average(
                        values[valid] ** 2 + (means[valid] - combined_mean) ** 2,
                        weights=weights[valid],
                    )
                    record[column] = float(np.sqrt(variance))
                else:
                    record[column] = np.nan
            elif column.endswith("_mean") and total_count > 0:
                record[column] = float(
                    np.average(values[valid], weights=weights[valid])
                )
            else:
                record[column] = float(np.mean(values[valid]))
        records.append(record)
    return pd.DataFrame.from_records(records)


def context_columns(data: pd.DataFrame) -> list[str]:
    return [
        column
        for column in (
            "hl_task",
            "hl_start_state",
            "subtask_idx",
            "msg_budget_per_agent",
        )
        if column in data
    ]


def plot_context_evaluation(
    data: pd.DataFrame,
    output_dir: str,
    info_str: str = "",
) -> list[str]:
    """Plot evaluation metrics for each context and return the image paths."""
    df = data.copy()
    df["msg_budget_per_agent"] = pd.to_numeric(
        df["msg_budget_per_agent"], errors="coerce"
    )
    makedirs(output_dir, exist_ok=True)
    df["context_label"] = [
        evaluation_context_label(row) for row in df.to_dict(orient="records")
    ]

    image_paths = []
    for metric in CONTEXT_EVALUATION_METRICS:
        figure, axis = plt.subplots()
        legend_title = context_sample_count_label(df)
        if info_str:
            legend_title += f"\n{info_str}"

        for context_label, context_data in df.groupby(
            "context_label", sort=True, dropna=False
        ):
            label = (
                context_label.replace("hl_task_", "Task ")
                .replace("subtask_idx_", "Subtask ")
                .replace("msg_budget_per_agent_", "Comms ")
                .replace("/", ", ")
            )
            axis.plot(
                context_data["t_env"],
                context_data[metric],
                marker="o",
                alpha=1.0,
                label=label,
            )
            if metric == "test_task_completed_mean":
                axis.set_ylim(-0.05, 1.05)

        axis.set_xlabel("t_env")
        axis.set_ylabel(metric)
        axis.set_title(metric)
        axis.legend(title=legend_title)
        axis.grid(True)

        image_path = join(output_dir, f"comms_eval_{metric}.png")
        figure.tight_layout()
        figure.savefig(image_path)
        plt.close(figure)
        image_paths.append(image_path)

    return image_paths


def plot_combined_context_evaluation(
    groups: Sequence[tuple[str, pd.DataFrame, object]],
    output_dir: str,
) -> dict[str, str]:
    """Plot context metrics as vertically stacked task panels."""
    if not groups:
        return {}
    makedirs(output_dir, exist_ok=True)

    image_paths = {}
    for metric in CONTEXT_EVALUATION_METRICS:
        figure, axes = plt.subplots(
            nrows=len(groups),
            ncols=1,
            figsize=(8, max(3.5 * len(groups), 4.0)),
            squeeze=False,
            sharex=True,
        )
        axes = axes[:, 0]

        for axis, (task_label, data, _) in zip(axes, groups):
            budgets = sorted(data["msg_budget_per_agent"].dropna().unique())
            for budget in budgets:
                budget_data = data[data["msg_budget_per_agent"] == budget]
                sample_count = context_sample_count_label(budget_data)
                axis.plot(
                    budget_data["t_env"],
                    budget_data[metric],
                    marker="o",
                    label=f"Comms: {budget} ({sample_count})",
                )

            axis.set_title(task_label, loc="left")
            axis.set_ylabel(metric)
            axis.grid(True)
            if metric == "test_task_completed_mean":
                axis.set_ylim(-0.05, 1.05)
            axis.legend()

        axes[-1].set_xlabel("t_env")
        figure.suptitle(metric)
        figure.tight_layout()
        image_path = join(output_dir, f"comms_eval_combined_{metric}.png")
        figure.savefig(image_path)
        plt.close(figure)
        image_paths[metric] = image_path

    return image_paths
