"""Generate mock success and terminal-state data for the two-goal branching env."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.envs.mock_hierarchical_env import MockHierarchicalEnv


REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_mock_data(
    *,
    navigation_config: Path,
    n_episodes: int,
    seed: int,
    team_delay_prob: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = yaml.safe_load(navigation_config.read_text(encoding="utf-8"))
    tasks = config["tasks"]
    transitions = [
        (int(task["from_state"]), int(task["to_state"]))
        for task in tasks
    ]
    capacities = (0.0, 1.0, 2.0)

    env = MockHierarchicalEnv(
        seed=seed,
        n_agents=3,
        n_comms_values=3,
        navigation_config=str(navigation_config),
        msg_budget_per_agent=capacities,
        max_episode_steps=1,
        team_delay_prob=team_delay_prob,
    )
    rows = []
    terminal_rows = []

    try:
        for edge_index, edge in enumerate(transitions):
            task = next(
                task
                for task in tasks
                if (
                    int(task["from_state"]) == edge[0]
                    and int(task["to_state"]) == edge[1]
                )
            )
            init_state_dist = task.get("init_state_dist")
            if init_state_dist:
                start_states = tuple(
                    tuple(
                        int(value)
                        for position in joint_state
                        for value in position
                    )
                    for joint_state in init_state_dist["states"]
                )
                start_probs = tuple(
                    float(probability)
                    for probability in init_state_dist["probs"]
                )
            else:
                start_states = ((edge[0], 0),)
                start_probs = (1.0,)

            for capacity in capacities:
                for start_index, start_state in enumerate(start_states):
                    reset_dist = {
                        "states": [
                            [
                                list(
                                    start_state[offset:offset + 2]
                                )
                                for offset in range(
                                    0,
                                    len(start_state),
                                    2,
                                )
                            ]
                        ],
                        "probs": [1.0],
                    }
                    successes = 0
                    terminals = Counter()
                    episode_success_rates = []

                    for episode_index in range(n_episodes):
                        env.reset(
                            seed=seed * 100_000
                            + edge_index * 10_000
                            + int(capacity * 1000)
                            + start_index * 100
                            + episode_index,
                            options={
                                "hl_task": list(edge),
                                "msg_budget_per_agent": capacity,
                                "navigation_init_state_dist": reset_dist,
                            },
                        )
                        _, _, _, _, info = env.step(
                            env.action_space.sample()
                        )
                        episode_success_rates.append(
                            float(info["mock_success_rate"])
                        )
                        if info["task_completed"]:
                            successes += 1
                            terminal_state = tuple(
                                int(value)
                                for value in info["final_state"]
                            )
                            terminals[terminal_state] += 1

                    terminal_states = tuple(terminals)
                    terminal_probs = tuple(
                        terminals[state] / successes
                        if successes
                        else 0.0
                        for state in terminal_states
                    )
                    success_rate = successes / n_episodes
                    mean_success_rate = sum(episode_success_rates) / n_episodes
                    rows.append(
                        {
                            "edge": f"{edge[0]}->{edge[1]}",
                            "from_state": edge[0],
                            "to_state": edge[1],
                            "team_delay_prob": team_delay_prob,
                            "team_delay_prob": team_delay_prob,
                            "communication_capacity": capacity,
                            "start_state": str(start_state),
                            "start_state_probability": start_probs[start_index],
                            "n_episodes": n_episodes,
                            "success_count": successes,
                            "success_rate": success_rate,
                            "mock_success_rate": mean_success_rate,
                        }
                    )
                    for terminal_index, terminal_state in enumerate(
                        terminal_states
                    ):
                        terminal_rows.append(
                            {
                                "edge": f"{edge[0]}->{edge[1]}",
                                "from_state": edge[0],
                                "to_state": edge[1],
                                "communication_capacity": capacity,
                                "start_state": str(start_state),
                                "n_successful_episodes": successes,
                                "terminal_state": str(terminal_state),
                                "terminal_state_index": terminal_index,
                                "terminal_count": terminals[terminal_state],
                                "terminal_probability": (
                                    terminal_probs[terminal_index]
                                ),
                            }
                        )
    finally:
        env.close()

    return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(
        terminal_rows
    )


def plot_success_vs_capacity(
    success_df: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for edge, edge_data in success_df.groupby("edge", sort=True):
        edge_data = edge_data.sort_values("communication_capacity")
        axis.plot(
            edge_data["communication_capacity"],
            edge_data["success_rate"],
            marker="o",
            label=f"Task {edge}",
        )
    axis.set_xlabel("Communication capacity")
    axis.set_ylabel("Mock success rate")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=500)
    plt.close(figure)


def generate_and_plot_mixed_initial_mock(
    *,
    output_path: Path,
    seed: int = 101,
) -> pd.DataFrame:
    """Generate exp55-style mock mixed-distribution rows and plot heatmaps."""
    del seed
    pg_values = (0.55, 0.65, 0.75)
    states = tuple(range(5))
    start_labels = tuple(chr(ord("A") + index) for index in range(8))
    rows = []
    rng = np.random.default_rng(101)

    for pg in pg_values:
        for iteration in range(10):
            progress = iteration / 9.0
            for state in states:
                raw = np.exp(
                    rng.normal(
                        loc=0.10 + 0.02 * state + 0.08 * progress,
                        scale=0.05,
                        size=len(start_labels),
                    )
                )
                probabilities = raw / raw.sum()
                for label, probability in zip(
                    start_labels,
                    probabilities,
                    strict=True,
                ):
                    rows.append(
                        {
                            "pg": pg,
                            "iteration": iteration,
                            "state": state,
                            "start_state": label,
                            "probability": float(probability),
                        }
                    )

    data = pd.DataFrame.from_records(rows)
    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(16.0, 8.5),
        squeeze=False,
        constrained_layout=True,
    )
    all_positive = data["probability"].to_numpy(dtype=float)
    all_positive = all_positive[all_positive > 0.0]
    if all_positive.size:
        shared_vmin = float(all_positive.min())
        shared_vmax = float(all_positive.max())
    else:
        shared_vmin, shared_vmax = 0.0, 1.0
    if np.isclose(shared_vmin, shared_vmax):
        shared_vmin = max(0.0, shared_vmin - 0.01)
        shared_vmax = shared_vmax + 0.01
    for pg_index, pg in enumerate(pg_values):
        axis = axes[0, pg_index]
        pg_data = data[data["pg"] == pg]
        matrix = pg_data.pivot_table(
            index="state",
            columns="start_state",
            values="probability",
            aggfunc="mean",
            fill_value=0.0,
        ).reindex(
            index=list(states),
            columns=list(start_labels),
        )
        image = axis.imshow(
            matrix.to_numpy(dtype=float),
            aspect="auto",
            origin="upper",
            vmin=shared_vmin,
            vmax=shared_vmax,
            cmap="viridis",
        )
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                probability = float(matrix.iloc[row_index, column_index])
                axis.text(
                    column_index,
                    row_index,
                    f"{probability:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=(
                        "white"
                        if probability > shared_vmin + 0.55 * (shared_vmax - shared_vmin)
                        else "black"
                    ),
                )
        axis.set_xticks(range(len(start_labels)))
        axis.set_xticklabels(start_labels)
        axis.set_yticks(range(len(states)))
        axis.set_yticklabels([str(state) for state in states])
        axis.set_xlabel("A–H joint-state combination")
        axis.set_ylabel("Source state")
        axis.set_title(f"$p_g={pg:.2f}$")
        figure.colorbar(
            image,
            ax=axis,
            label="State probability",
            fraction=0.046,
            pad=0.04,
        )
    figure.suptitle(
        "Mock initial joint-state distributions across synthesis",
        fontsize=16,
    )
    mapping_text = "A-H: all 8 combinations of each agent's two hallway levels"
    figure.text(
        0.5,
        -0.02,
        mapping_text,
        ha="center",
        va="bottom",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=500, bbox_inches="tight")
    plt.close(figure)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--navigation-config",
        type=Path,
        default=(
            REPO_ROOT
            / "submodules/gym-multigrid/gym_multigrid/envs/maps/"
            "team_navigation/navigation_branching_two_goals_small_hall.yaml"
        ),
    )
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--team-delay-prob", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "scratch/two_goal_mock_data",
    )
    args = parser.parse_args()

    success_df, terminal_df = generate_mock_data(
        navigation_config=args.navigation_config,
        n_episodes=args.n_episodes,
        seed=args.seed,
        team_delay_prob=args.team_delay_prob,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    success_df.to_csv(
        args.output_dir / "mock_success_vs_capacity.csv",
        index=False,
    )
    terminal_df.to_csv(
        args.output_dir / "mock_terminal_state_distributions.csv",
        index=False,
    )
    plot_success_vs_capacity(
        success_df,
        args.output_dir / "mock_success_vs_capacity.png",
    )
    generate_and_plot_mixed_initial_mock(
        output_path=(
            args.output_dir
            / "mock_mixed_initial_distributions.png"
        ),
        seed=args.seed,
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
