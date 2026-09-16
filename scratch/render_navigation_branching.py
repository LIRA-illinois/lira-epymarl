"""Render all navigation tasks from a branching navigation YAML file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def render_navigation_tasks(
    config_path: Path,
    output_path: Path,
    seed: int,
    dpi: int,
) -> None:
    sys.path.insert(0, str(REPO_ROOT / "submodules" / "gym-multigrid"))
    from gym_multigrid.envs.team_navigation import TeamNavigationEnv

    config = yaml.safe_load(config_path.read_text())
    tasks = config["tasks"]
    transitions = [(task["from_state"], task["to_state"]) for task in tasks]

    env = TeamNavigationEnv(
        map_name="3ga_1r_small_hall",
        n_agents=len(tasks[0]["goal_positions"]),
        navigation_tasks=tasks,
    )

    frames = []
    try:
        for transition in transitions:
            env.reset(
                seed=seed,
                options={"navigation_task_transition": transition},
            )
            frame = env.render()
            if frame is None:
                raise RuntimeError(
                    f"Environment returned no image for transition {transition}."
                )
            frames.append((transition, frame))
    finally:
        env.close()

    columns = 2
    rows = (len(frames) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6 * columns, 3 * rows),
        dpi=dpi,
        squeeze=False,
    )

    for ax, (transition, frame) in zip(axes.flat, frames):
        ax.imshow(frame)
        ax.set_title(f"Task {transition[0]} -> {transition[1]}")
        ax.axis("off")

    for ax in axes.flat[len(frames) :]:
        ax.axis("off")

    fig.suptitle("Navigation branching goal positions", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "submodules/gym-multigrid/gym_multigrid/envs/maps/navigation_branching_small_hall.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "tmp/navigation_branching_small_hall.png",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    render_navigation_tasks(args.config, args.output, args.seed, args.dpi)
    print(f"Saved image to {args.output}")
