import argparse

from src.experiments.ray_cluster import RayCluster


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast-forward the checkouts on currently live lab Ray nodes."
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
    args = parser.parse_args()
    RayCluster.update_running_hosts(args.config, args.ray_address)


if __name__ == "__main__":
    main()
