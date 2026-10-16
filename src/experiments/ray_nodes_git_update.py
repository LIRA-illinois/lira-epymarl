import argparse
import subprocess

from src.experiments.ray_lab_fleet import load_lab_hosts


def _update_checkout(project_dir: str) -> dict:
    setup_lines = [
        "set -euo pipefail",
        "git rev-parse --show-toplevel >/dev/null",
        "git diff --quiet && git diff --cached --quiet || { echo 'Root repository has tracked local changes; refusing to update' >&2; exit 1; }",
        "git submodule foreach --recursive 'git diff --quiet && git diff --cached --quiet || { echo \"Submodule has tracked local changes; refusing to update\" >&2; exit 1; }'",
        "git pull --ff-only",
        "git submodule sync --recursive",
        "git submodule update --init --recursive",
    ]
    try:
        completed = subprocess.run(
            ["bash", "-c", "\n".join(setup_lines)],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        return {"returncode": 1, "output": str(error)}

    return {
        "returncode": completed.returncode,
        "output": completed.stdout + completed.stderr,
    }


def update_running_hosts(config_path: str, ray_address: str) -> None:
    try:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    except ImportError as error:
        raise RuntimeError(
            "Lab fleet updates require Ray. Install it with `poetry install --with ray`."
        ) from error

    hosts = load_lab_hosts(config_path)
    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(address=ray_address)

    try:
        nodes_by_ip = {}
        for node in ray.nodes():
            if node.get("Alive"):
                nodes_by_ip.setdefault(node["NodeManagerAddress"], []).append(node)

        running_hosts = []
        for host in hosts:
            matching_nodes = nodes_by_ip.get(host["ray_node_ip"], [])
            if not matching_nodes:
                print(
                    f"[{host['name']}] skipped: no live Ray node at {host['ray_node_ip']}"
                )
                continue
            if len(matching_nodes) != 1:
                raise RuntimeError(
                    f"Expected one live Ray node at {host['ray_node_ip']} for "
                    f"{host['name']}, found {len(matching_nodes)}"
                )
            running_hosts.append((host, matching_nodes[0]))

        if not running_hosts:
            raise RuntimeError("No configured lab computers are currently live in Ray")

        update_task = ray.remote(_update_checkout)
        references = [
            update_task.options(
                num_cpus=0,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False,
                ),
            ).remote(host["project_dir"])
            for host, node in running_hosts
        ]
        outcomes = ray.get(references)

        failures = []
        for (host, _), outcome in zip(running_hosts, outcomes):
            output = outcome["output"].strip()
            if outcome["returncode"]:
                failures.append(host["name"])
                print(f"[{host['name']}] update failed:\n{output}")
            else:
                print(f"[{host['name']}] checkout updated with fast-forward pull")
                if output:
                    print(output)

        if failures:
            raise RuntimeError("Update failed on: " + ", ".join(failures))
    finally:
        if owns_ray:
            ray.shutdown()


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
    update_running_hosts(args.config, args.ray_address)


if __name__ == "__main__":
    main()
