import os
import re
import subprocess
import webbrowser
from hashlib import sha256
from os.path import isabs, join

import yaml


def load_lab_hosts(config_path: str) -> list[dict]:
    with open(config_path, "r", encoding="utf8") as f:
        config = yaml.safe_load(f)

    hosts = config.get("hosts") if isinstance(config, dict) else None
    if not isinstance(hosts, list) or not hosts:
        raise ValueError(f"{config_path} must contain a non-empty hosts list")

    host_names = set()
    node_ips = set()
    required_fields = {
        "name",
        "ray_node_ip",
        "project_dir",
        "n_runners",
        "gpus",
    }
    for host in hosts:
        if not isinstance(host, dict):
            raise TypeError("Each lab host must be a mapping")
        missing_fields = required_fields - host.keys()
        if missing_fields:
            raise ValueError(
                f"Lab host is missing required fields: {sorted(missing_fields)}"
            )

        name = host["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Lab host names may contain only letters, digits, _ and -")
        if name in host_names:
            raise ValueError(f"Duplicate lab host name: {name}")
        host_names.add(name)

        node_ip = host["ray_node_ip"]
        if (
            not isinstance(node_ip, str)
            or not node_ip
            or any(c.isspace() for c in node_ip)
        ):
            raise ValueError(f"Invalid ray_node_ip for lab host {name}")
        if node_ip in node_ips:
            raise ValueError(f"Duplicate Ray node IP: {node_ip}")
        node_ips.add(node_ip)

        project_dir = host["project_dir"]
        if not isinstance(project_dir, str) or not isabs(project_dir):
            raise ValueError(
                f"project_dir for lab host {name} must be an absolute path"
            )

        n_runners = host["n_runners"]
        if (
            not isinstance(n_runners, int)
            or isinstance(n_runners, bool)
            or n_runners <= 0
        ):
            raise ValueError(f"n_runners for lab host {name} must be positive")

        gpus = host["gpus"]
        if (
            not isinstance(gpus, list)
            or any(
                not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0
                for gpu in gpus
            )
            or len(gpus) != len(set(gpus))
        ):
            raise ValueError(
                f"gpus for lab host {name} must be unique, non-negative integers"
            )

    return hosts


def distribute_commands(commands, hosts: list[dict]) -> list[list[list[str]]]:
    queues = [[[] for _ in range(host["n_runners"])] for host in hosts]
    runner_slots = [
        (host_idx, runner_idx)
        for host_idx, host in enumerate(hosts)
        for runner_idx in range(host["n_runners"])
    ]
    if not runner_slots:
        raise ValueError("At least one lab runner is required")

    for command_idx, command in enumerate(commands):
        host_idx, runner_idx = runner_slots[command_idx % len(runner_slots)]
        queues[host_idx][runner_idx].append(str(command))
    return queues


def _prepare_node(project_dir: str, gpus: list[int], sync_wandb: bool) -> dict:
    setup_lines = [
        "set -euo pipefail",
        "test -f .venv/bin/activate",
        "git rev-parse --show-toplevel >/dev/null",
        "git diff --quiet && git diff --cached --quiet || { echo 'Root repository has tracked local changes; refusing to update' >&2; exit 1; }",
        "git submodule foreach --recursive 'git diff --quiet && git diff --cached --quiet || { echo \"Submodule has tracked local changes\" >&2; exit 1; }'",
        "git pull --ff-only",
        "git submodule sync --recursive",
        "git submodule update --init --recursive",
        "source .venv/bin/activate",
        "python --version",
    ]
    if sync_wandb:
        setup_lines.append("command -v wandb >/dev/null")

    completed = subprocess.run(
        ["bash", "-c", "\n".join(setup_lines)],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Repository/setup preflight failed in {project_dir}:\n"
            f"{completed.stdout}{completed.stderr}"
        )

    available_gpus = []
    if gpus:
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        available_gpus = [
            int(gpu) for gpu in gpu_result.stdout.splitlines() if gpu.strip()
        ]
        missing_gpus = set(gpus) - set(available_gpus)
        if missing_gpus:
            raise RuntimeError(
                f"{project_dir} does not have configured GPU IDs: "
                f"{sorted(missing_gpus)} (available: {sorted(available_gpus)})"
            )

    lock_path = join(project_dir, "poetry.lock")
    if not os.path.isfile(lock_path):
        raise RuntimeError(f"Missing poetry.lock in {project_dir}")
    with open(lock_path, "rb") as lock_file:
        lockfile_sha256 = sha256(lock_file.read()).hexdigest()

    wandb_dir = join(project_dir, "results", "wandb")
    baseline_runs = []
    if os.path.isdir(wandb_dir):
        baseline_runs = sorted(
            name
            for name in os.listdir(wandb_dir)
            if "-run-" in name and os.path.isdir(join(wandb_dir, name))
        )

    environment_code = (
        "import platform, sys; "
        "print(sys.version_info.major); "
        "print(platform.python_version())"
    )
    environment_result = subprocess.run(
        [join(project_dir, ".venv", "bin", "python"), "-c", environment_code],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    python_major, python_version = environment_result.stdout.splitlines()

    print(completed.stdout, end="")
    return {
        "available_gpus": available_gpus,
        "baseline_runs": baseline_runs,
        "venv": {
            "python_major": int(python_major),
            "python_version": python_version,
            "lockfile_sha256": lockfile_sha256,
        },
    }


def _run_queue(project_dir: str, gpu: int | None, commands: list[str]) -> list[dict]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu) if gpu is not None else ""
    outcomes = []
    for command in commands:
        print(f"Starting command on {project_dir}: {command}")
        result = subprocess.run(
            ["bash", "-c", f"source .venv/bin/activate && {command}"],
            cwd=project_dir,
            env=environment,
            check=False,
        )
        outcomes.append({"command": command, "returncode": result.returncode})
    return outcomes


def _sync_node(project_dir: str, baseline_runs: list[str]) -> list[dict]:
    wandb_dir = join(project_dir, "results", "wandb")
    if not os.path.isdir(wandb_dir):
        return []

    baseline = set(baseline_runs)
    new_runs = sorted(
        join(wandb_dir, name)
        for name in os.listdir(wandb_dir)
        if "-run-" in name
        and name not in baseline
        and os.path.isdir(join(wandb_dir, name))
    )
    outcomes = []
    for run_dir in new_runs:
        result = subprocess.run(
            [join(project_dir, ".venv", "bin", "wandb"), "sync", run_dir],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        outcomes.append({
            "run_dir": run_dir,
            "returncode": result.returncode,
            "output": result.stdout + result.stderr,
        })
    return outcomes


class RayLabFleet:
    def __init__(
        self,
        config_path: str,
        experiment: str,
        ray_address: str,
        sync_wandb: bool = True,
        open_ray_dashboard: bool = False,
    ) -> None:
        self.hosts = load_lab_hosts(config_path)
        self.experiment = experiment
        self.ray_address = ray_address
        self.sync_wandb = sync_wandb
        self.open_ray_dashboard = open_ray_dashboard

    def run(self, commands) -> None:
        try:
            import ray
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        except ImportError as error:
            raise RuntimeError(
                "Ray lab-fleet execution requires the optional 'ray' dependency. "
                "Install it with `poetry install --with ray`."
            ) from error

        queues = distribute_commands(commands, self.hosts)
        owns_ray = not ray.is_initialized()
        if owns_ray:
            ray_context = ray.init(address=self.ray_address)
        else:
            ray_context = ray.init()
        dashboard_url = ray_context.dashboard_url
        if self.open_ray_dashboard and not dashboard_url:
            if owns_ray:
                ray.shutdown()
            raise RuntimeError(
                "The Ray dashboard is unavailable. Start the Ray head with "
                "dashboard support enabled, then retry."
            )

        try:
            node_by_ip = {}
            for node in ray.nodes():
                if node.get("Alive"):
                    node_by_ip.setdefault(node["NodeManagerAddress"], []).append(node)

            host_nodes = {}
            for host in self.hosts:
                matching_nodes = node_by_ip.get(host["ray_node_ip"], [])
                if len(matching_nodes) != 1:
                    raise ValueError(
                        f"Expected one live Ray node at {host['ray_node_ip']} for "
                        f"{host['name']}, found {len(matching_nodes)}"
                    )
                node = matching_nodes[0]
                cpu_capacity = node.get("Resources", {}).get("CPU", 0)
                if host["n_runners"] > cpu_capacity:
                    raise ValueError(
                        f"{host['name']} requests {host['n_runners']} runners but "
                        f"its Ray node advertises {cpu_capacity:g} CPUs"
                    )
                host_nodes[host["name"]] = node

            def node_strategy(host):
                node = host_nodes[host["name"]]
                return NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False,
                )

            prepare_task = ray.remote(_prepare_node)
            prepare_refs = [
                prepare_task.options(
                    num_cpus=1,
                    scheduling_strategy=node_strategy(host),
                ).remote(host["project_dir"], host["gpus"], self.sync_wandb)
                for host in self.hosts
            ]
            prepared = ray.get(prepare_refs)
            for host, setup in zip(self.hosts, prepared):
                print(
                    f"[{host['name']}] updated checkout; available GPUs "
                    f"{setup['available_gpus']}"
                )
            self._ensure_matching_environments(prepared)

            run_task = ray.remote(max_retries=0)(_run_queue)
            run_refs = []
            for host_idx, host in enumerate(self.hosts):
                strategy = node_strategy(host)
                for runner_idx, queue in enumerate(queues[host_idx]):
                    if not queue:
                        continue
                    gpu_ids = host["gpus"]
                    gpu = gpu_ids[runner_idx % len(gpu_ids)] if gpu_ids else None
                    run_ref = run_task.options(
                        num_cpus=1,
                        scheduling_strategy=strategy,
                    ).remote(host["project_dir"], gpu, queue)
                    run_refs.append((host, queue, run_ref))

            if self.open_ray_dashboard and run_refs and dashboard_url:
                self._open_dashboard(dashboard_url)

            run_results = ray.get([reference for _, _, reference in run_refs])
            failures = []
            for (host, queue, _), outcomes in zip(run_refs, run_results):
                for command, outcome in zip(queue, outcomes):
                    if outcome["returncode"]:
                        failures.append(
                            f"{host['name']}: {command} exited with "
                            f"{outcome['returncode']}"
                        )

            if self.sync_wandb:
                sync_task = ray.remote(_sync_node)
                sync_refs = [
                    sync_task.options(
                        num_cpus=1,
                        scheduling_strategy=node_strategy(host),
                    ).remote(host["project_dir"], setup["baseline_runs"])
                    for host, setup in zip(self.hosts, prepared)
                ]
                sync_results = ray.get(sync_refs)
                for host, outcomes in zip(self.hosts, sync_results):
                    for outcome in outcomes:
                        if outcome["returncode"]:
                            failures.append(
                                f"{host['name']}: W&B sync failed for "
                                f"{outcome['run_dir']}: {outcome['output']}"
                            )
                        else:
                            print(f"[{host['name']}] synced {outcome['run_dir']}")

            if failures:
                raise RuntimeError("Lab-fleet failures:\n" + "\n".join(failures))
        finally:
            if owns_ray:
                ray.shutdown()

    def _ensure_matching_environments(self, preparations: list[dict]) -> None:
        reference_host = self.hosts[0]
        reference_environment = preparations[0]["venv"]
        mismatches = []

        for host, preparation in zip(self.hosts[1:], preparations[1:]):
            environment = preparation["venv"]
            if (
                environment["python_major"]
                == reference_environment["python_major"]
                and environment["lockfile_sha256"]
                == reference_environment["lockfile_sha256"]
            ):
                continue
            differences = []
            if environment["python_major"] != reference_environment["python_major"]:
                differences.append(f"Python major {environment['python_major']}")
            if environment["lockfile_sha256"] != reference_environment["lockfile_sha256"]:
                differences.append("poetry.lock differs")
            mismatches.append(
                f"{host['name']} ({'; '.join(differences)})"
            )

        if mismatches:
            raise RuntimeError(
                "Lab computer Python major versions or Poetry lockfiles differ from "
                f"{reference_host['name']} (Python major "
                f"{reference_environment['python_major']}; Python "
                f"{reference_environment['python_version']}; poetry.lock SHA-256 "
                f"{reference_environment['lockfile_sha256']}). Ensure all hosts use "
                "the same lockfile, then retry. Differences: " + "; ".join(mismatches)
            )
        print("All lab computers have compatible Python majors and the same poetry.lock")

    def _open_dashboard(self, dashboard_url: str) -> None:
        print(f"Ray dashboard: {dashboard_url}")
        webbrowser.open(dashboard_url)
