from collections.abc import Mapping, Sequence
from os import makedirs
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from src.utils.navigation_rendering import load_navigation_mdp

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


def plot_success_vs_comms_by_subtask(
    data: pd.DataFrame,
    output_dir: str,
) -> list[str]:
    """Plot success vs communication capacity for each seed/subtask panel."""
    df = data.copy()
    if "test_task_completed_mean" not in df:
        raise ValueError("Success-vs-comms data requires test_task_completed_mean.")
    if "hl_task" not in df:
        df["hl_task"] = df.get("subtask_idx", "")
    df["hl_task"] = df["hl_task"].apply(
        lambda value: tuple(value) if isinstance(value, (list, tuple)) else value
    )
    makedirs(output_dir, exist_ok=True)

    seeds = sorted(df["seed"].dropna().unique()) if "seed" in df else [None]
    image_paths = []
    for seed in seeds:
        seed_df = df[df["seed"] == seed].copy() if seed is not None else df.copy()
        subtasks = sorted(
            seed_df["hl_task"].dropna().unique(),
            key=lambda value: str(value),
        )
        if not subtasks:
            continue
        figure, axes = plt.subplots(
            nrows=len(subtasks),
            ncols=1,
            figsize=(6.4, 3.2 * len(subtasks)),
            squeeze=False,
            sharex=True,
        )
        axes = axes[:, 0]
        for axis, subtask in zip(axes, subtasks):
            subtask_df = seed_df[
                seed_df["hl_task"].apply(lambda value, target=subtask: value == target)
            ]
            checkpoints = sorted(subtask_df["t_env"].dropna().unique())
            for t_env in checkpoints:
                checkpoint_df = subtask_df[subtask_df["t_env"] == t_env].sort_values(
                    "msg_budget_per_agent"
                )
                axis.plot(
                    checkpoint_df["msg_budget_per_agent"],
                    checkpoint_df["test_task_completed_mean"],
                    marker="o",
                    label=f"t_env={int(t_env)}",
                )
            axis.set_ylim(-0.05, 1.05)
            axis.set_ylabel("Success rate")
            axis.set_title(f"Subtask {subtask}", loc="left")
            axis.grid(True)
            axis.legend(fontsize="small")
        axes[-1].set_xlabel("Communication capacity")
        figure.suptitle(
            f"Success vs communication (seed {seed})"
            if seed is not None
            else "Success vs communication"
        )
        figure.tight_layout()
        seed_token = f"seed_{seed}" if seed is not None else "seed"
        image_path = join(
            output_dir,
            f"success_vs_comms_{seed_token}.png",
        )
        figure.savefig(image_path)
        plt.close(figure)
        image_paths.append(image_path)
    return image_paths


def merge_state_distributions(
    distributions: Sequence[dict],
    weights: Sequence[float],
) -> dict:
    """Merge weighted `{"states","probs"}` distributions."""
    if len(distributions) != len(weights):
        raise ValueError("State distributions and weights must have the same length.")
    total_weight = float(sum(weights))
    if total_weight <= 0:
        weights = [1.0] * len(weights)
        total_weight = float(len(weights))

    state_probabilities: dict[tuple, float] = {}
    for distribution, weight in zip(distributions, weights, strict=True):
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(
                "State-distribution weights must be finite and non-negative."
            )
        states = distribution.get("states", [])
        probabilities = distribution.get("probs", [])
        if len(states) != len(probabilities):
            raise ValueError("Distribution states and probabilities must align.")
        for state, probability in zip(states, probabilities, strict=True):
            if isinstance(state, (list, tuple)) and all(
                isinstance(item, (list, tuple)) for item in state
            ):
                state_key = tuple(
                    tuple(
                        int(item) if isinstance(item, (int, np.integer)) else item
                        for item in position
                    )
                    for position in state
                )
            elif isinstance(state, (list, tuple)):
                state_key = tuple(
                    int(item) if isinstance(item, (int, np.integer)) else item
                    for item in state
                )
            else:
                state_key = (state,)
            state_probabilities[state_key] = state_probabilities.get(
                state_key, 0.0
            ) + weight * float(probability)

    if not state_probabilities:
        raise ValueError(
            "No successful terminal states available to build the next "
            "subtask's spawn distribution; predecessor subtask(s) never succeeded."
        )

    states = sorted(state_probabilities)
    probabilities = [state_probabilities[state] / total_weight for state in states]
    rendered_states = []
    for state in states:
        rendered_states.append(list(state))
    return {
        "states": rendered_states,
        "probs": probabilities,
    }


def plot_high_level_environment(axis, source_run, map_name: str) -> None:
    """Render high-level navigation states and transitions on an axis."""
    mdp = load_navigation_mdp(source_run.config, logger=None)
    states = mdp.get("states", [])
    transitions = mdp.get("transitions", [])
    if not states or not transitions:
        axis.text(0.5, 0.5, "high-level graph unavailable", ha="center", va="center")
        axis.axis("off")
        return

    levels = {states[0]: 0}
    for _ in states:
        for transition in transitions:
            start = transition["from_state"]
            end = transition["to_state"]
            if start in levels:
                levels[end] = max(levels.get(end, 0), levels[start] + 1)
    positions = {}
    for level in sorted(set(levels.values())):
        level_states = [state for state in levels if levels[state] == level]
        offset = (len(level_states) - 1) / 2
        positions.update(
            {state: (level, offset - index) for index, state in enumerate(level_states)}
        )

    for transition in transitions:
        start = positions.get(transition["from_state"])
        end = positions.get(transition["to_state"])
        if start is not None and end is not None:
            axis.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="->",
                    mutation_scale=12,
                    linewidth=1.2,
                    color="0.35",
                    connectionstyle="arc3,rad=0.08",
                )
            )

    goal_states = set(mdp.get("goal_states", []))
    for state, position in positions.items():
        axis.scatter(
            *position,
            s=500,
            color="tab:green" if state in goal_states else "white",
            edgecolor="0.2",
            zorder=2,
        )
        axis.text(*position, state, ha="center", va="center", zorder=3)
    axis.set_title(f"high-level env: {map_name}", fontsize="small")
    axis.set_xlim(-0.5, max(levels.values()) + 0.5)
    axis.set_ylim(
        min(position[1] for position in positions.values()) - 0.7,
        1.0,
    )
    axis.axis("off")


def plot_mixed_initial_distributions(
    data: pd.DataFrame,
    output_dir: str,
    seed: int | None = None,
) -> list[str]:
    """Plot candidate-induced start-state distributions across synthesis iterations."""
    required = {
        "pg",
        "iteration",
        "state",
        "start_state",
        "probability",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "Mixed initial-distribution data is missing required columns: "
            f"{sorted(missing)}."
        )
    if data.empty:
        return []

    makedirs(output_dir, exist_ok=True)
    seed_token = f"seed_{seed}" if seed is not None else "all_seeds"
    csv_path = join(
        output_dir,
        f"mixed_initial_distributions_{seed_token}.csv",
    )
    data = (
        data.groupby(
            ["pg", "iteration", "state", "start_state"],
            as_index=False,
            dropna=False,
        )["probability"]
        .mean()
        .sort_values(["pg", "iteration", "state", "start_state"])
    )
    data.to_csv(csv_path, index=False)

    pg_values = sorted(data["pg"].dropna().unique())
    states = sorted(data["state"].dropna().unique())
    if not pg_values or not states:
        return [csv_path]

    start_states = sorted(data["start_state"].dropna().unique())
    start_labels = {
        start_state: chr(ord("A") + index)
        for index, start_state in enumerate(start_states)
    }
    if len(start_states) > 26:
        start_labels = {
            start_state: f"S{index + 1}"
            for index, start_state in enumerate(start_states)
        }

    image_paths = [csv_path]
    all_positive = data["probability"].to_numpy(dtype=float)
    all_positive = all_positive[all_positive > 0.0]
    if all_positive.size:
        probability_min = float(np.min(all_positive))
        probability_max = float(np.max(all_positive))
    else:
        probability_min, probability_max = 0.0, 1.0
    if np.isclose(probability_min, probability_max):
        probability_min = max(0.0, probability_min - 0.01)
        probability_max = probability_max + 0.01

    for pg in pg_values:
        pg_data = data[data["pg"] == pg]
        figure, axes = plt.subplots(
            nrows=len(states),
            ncols=1,
            figsize=(11.0, 2.2 * len(states)),
            squeeze=False,
            sharex=True,
        )
        for row_index, state in enumerate(states):
            axis = axes[row_index, 0]
            state_data = pg_data[pg_data["state"] == state]
            state_data = state_data.assign(
                start_label=state_data["start_state"].map(start_labels)
            )
            iterations = sorted(state_data["iteration"].dropna().unique())
            labels = [
                start_labels[start_state]
                for start_state in start_states
                if start_state in set(state_data["start_state"])
            ]
            matrix = state_data.pivot_table(
                index="iteration",
                columns="start_label",
                values="probability",
                aggfunc="mean",
                fill_value=0.0,
            )
            matrix = matrix.reindex(index=iterations)
            matrix = matrix.reindex(
                columns=labels,
                fill_value=0.0,
            )
            image = axis.imshow(
                matrix.to_numpy(dtype=float),
                aspect="auto",
                origin="upper",
                vmin=probability_min,
                vmax=probability_max,
                cmap="viridis",
            )
            for row_index, iteration_row in enumerate(matrix.to_numpy(dtype=float)):
                for column_index, probability in enumerate(iteration_row):
                    axis.text(
                        column_index,
                        row_index,
                        f"{probability:.3f}",
                        ha="center",
                        va="center",
                        fontsize="x-small",
                        color=(
                            "white"
                            if probability
                            > (
                                probability_min
                                + 0.55
                                * (probability_max - probability_min)
                            )
                            else "black"
                        ),
                    )
            axis.set_xticks(range(len(matrix.columns)))
            axis.set_xticklabels(matrix.columns)
            axis.set_yticks(range(len(matrix.index)))
            axis.set_yticklabels([str(int(value)) for value in matrix.index])
            axis.set_ylabel("Iteration")
            axis.set_title(f"Source state {state}", loc="left")
        axes[-1, 0].set_xlabel("Start-state combination")
        mapping_text = "  ".join(
            f"{start_labels[start_state]}={start_state}"
            for start_state in start_states
        )
        figure.suptitle(
            f"Mixed start-state distributions, $p_g={float(pg):.2f}$",
            fontsize=14,
            y=0.995,
        )
        figure.text(
            0.5,
            0.01,
            mapping_text,
            ha="center",
            va="bottom",
            fontsize="x-small",
            wrap=True,
        )
        figure.subplots_adjust(bottom=0.18)
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="State probability",
            fraction=0.02,
            pad=0.02,
        )
        pg_token = f"{float(pg):.2f}".replace(".", "p")
        png_path = join(
            output_dir,
            f"mixed_initial_distributions_{seed_token}_pg_{pg_token}.png",
        )
        pdf_path = join(
            output_dir,
            f"mixed_initial_distributions_{seed_token}_pg_{pg_token}.pdf",
        )
        figure.savefig(png_path, dpi=500)
        figure.savefig(pdf_path)
        plt.close(figure)
        image_paths.extend([png_path, pdf_path])
    return image_paths


def plot_pg_iteration_verification(
    data: pd.DataFrame,
    output_dir: str,
    seed: int | None = None,
) -> list[str]:
    """Plot empirical success and policy cost across synthesis iterations."""
    required = {"pg", "iteration", "empirical_goal_probability", "empirical_cost"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Iteration verification data is missing required columns: "
            f"{sorted(missing)}."
        )

    makedirs(output_dir, exist_ok=True)
    seed_token = f"seed_{seed}" if seed is not None else "all_seeds"
    data = data.sort_values(["pg", "iteration"])
    csv_path = join(output_dir, f"pg_iteration_verification_{seed_token}.csv")
    data.to_csv(csv_path, index=False)

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7.2, 6.4),
        sharex=True,
    )
    for pg, pg_data in data.groupby("pg", sort=True):
        axes[0].plot(
            pg_data["iteration"],
            pg_data["empirical_goal_probability"],
            marker="o",
            label=f"$p_g={float(pg):.2f}$",
        )
        axes[1].plot(
            pg_data["iteration"],
            pg_data["empirical_cost"],
            marker="o",
            label=f"$p_g={float(pg):.2f}$",
        )
    axes[0].axhline(1.0, color="0.75", linestyle="--", linewidth=1)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Empirical success")
    axes[0].set_title("Iterative high-level policy verification")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Policy communication cost")
    axes[1].grid(True)
    axes[1].legend()
    figure.tight_layout()

    png_path = join(output_dir, f"pg_iteration_verification_{seed_token}.png")
    pdf_path = join(output_dir, f"pg_iteration_verification_{seed_token}.pdf")
    figure.savefig(png_path, dpi=500)
    figure.savefig(pdf_path)
    plt.close(figure)
    return [png_path, pdf_path, csv_path]


def plot_success_vs_cost_frontier(
    data: pd.DataFrame,
    output_dir: str,
    seed: int | None = None,
) -> list[str]:
    """Plot mission-success versus normalized communication cost."""
    required = {
        "pg",
        "ours_cost",
        "baseline_cost",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Frontier data is missing required columns: {sorted(missing)}."
        )

    makedirs(output_dir, exist_ok=True)
    frontier = data.sort_values("pg")
    seed_token = f"seed_{seed}" if seed is not None else "all_seeds"
    image_paths = []

    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.plot(
        frontier["pg"],
        frontier["baseline_cost"],
        marker="o",
        markersize=8,
        color="darkorange",
        linewidth=2,
        alpha=0.85,
        label="Fixed Comms. Baseline",
    )
    axis.plot(
        frontier["pg"],
        frontier["ours_cost"],
        marker="^",
        markersize=7,
        color="black",
        linewidth=2,
        alpha=0.9,
        label="Ours",
    )
    if "task_greedy_cost" in frontier:
        axis.plot(
            frontier["pg"],
            frontier["task_greedy_cost"],
            marker="s",
            markersize=7,
            color="tab:blue",
            linewidth=2,
            alpha=0.9,
            label="Task Greedy",
        )
    if "comms_greedy_cost" in frontier:
        axis.plot(
            frontier["pg"],
            frontier["comms_greedy_cost"],
            marker="D",
            markersize=7,
            color="tab:green",
            linewidth=2,
            alpha=0.9,
            label="Comms Greedy",
        )
    axis.set_xlabel("Mission Success Prob., $p_c$")
    axis.set_ylabel("Communication Cost")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()

    png_path = join(output_dir, f"success_vs_cost_{seed_token}.png")
    pdf_path = join(output_dir, f"success_vs_cost_{seed_token}.pdf")
    csv_path = join(output_dir, f"success_vs_cost_{seed_token}.csv")
    figure.savefig(png_path, dpi=500)
    figure.savefig(pdf_path)
    plt.close(figure)
    frontier.to_csv(csv_path, index=False)
    image_paths.extend([png_path, pdf_path, csv_path])
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
