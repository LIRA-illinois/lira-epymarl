import argparse

from src.experiments.ray_cluster import RayCluster


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sync offline W&B runs from configured lab computers that are "
            "currently live in Ray."
        )
    )
    parser.add_argument(
        "--config",
        default="experiments/lab_hosts.yaml",
        help="Lab host inventory (default: experiments/lab_hosts.yaml)",
    )
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Ray cluster address (default: auto)",
    )
    parser.add_argument(
        "--host",
        action="append",
        dest="hosts",
        help="Only sync this configured host; may be specified more than once",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching offline runs without uploading them",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=5,
        help="Maximum simultaneous uploads across the cluster (default: 5)",
    )
    args = parser.parse_args()
    RayCluster.sync_offline_wandb_runs(
        args.config,
        args.ray_address,
        selected_hosts=args.hosts,
        dry_run=args.dry_run,
        parallelism=args.parallelism,
    )


if __name__ == "__main__":
    main()
