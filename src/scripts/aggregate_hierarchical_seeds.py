"""Aggregate completed seed-level hierarchical artifact tables."""

from __future__ import annotations

import argparse
import json

from src.simulation.hierarchical_evaluation import write_seed_postprocessing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_artifact_root")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = (
        args.output_dir
        or f"{args.scenario_artifact_root}/postprocessing"
    )
    paths = write_seed_postprocessing(
        args.scenario_artifact_root,
        output_dir,
    )
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
