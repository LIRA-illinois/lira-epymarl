import os
import re
import subprocess
import webbrowser
from collections import deque
from contextlib import contextmanager
from hashlib import sha256
from os.path import isabs, join

import yaml

_RUNNER_CONFIG_POLL_INTERVAL_SECONDS = 5


class RayCluster:
    @classmethod
    @contextmanager
    def _ray_session(cls, ray_address: str):
        try:
            import ray
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        except ImportError as error:
            raise RuntimeError(
                "Ray cluster operations require the optional 'ray' dependency. "
                "Install it with `poetry install --with ray`."
            ) from error

        owns_ray = not ray.is_initialized()
        ray_context = (
            ray.init(address=ray_address) if owns_ray else ray.init()
        )
        try:
            yield ray, ray_context, NodeAffinitySchedulingStrategy
        finally:
            if owns_ray:
                ray.shutdown()

    @staticmethod
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

    @staticmethod
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


    @staticmethod
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


    @staticmethod
    def _run_command_queues(
        ray,
        run_task,
        hosts: list[dict],
        commands: deque[str],
        host_nodes: dict[str, dict],
        node_strategy,
        config_path: str,
        poll_interval: float = _RUNNER_CONFIG_POLL_INTERVAL_SECONDS,
    ) -> list[str]:
        if poll_interval <= 0:
            raise ValueError("Runner config poll interval must be positive")

        failures = []
        pending = {}
        active_counts = [0 for _ in hosts]
        gpu_indices = [0 for _ in hosts]
        next_host_idx = 0
        last_config_error = None

        def refresh_runner_targets() -> bool:
            nonlocal last_config_error
            try:
                configured_hosts = RayCluster.load_lab_hosts(config_path)
                if len(configured_hosts) != len(hosts):
                    raise ValueError(
                        "The configured host list cannot change while a search is running"
                    )

                for host, configured_host in zip(hosts, configured_hosts):
                    current_settings = {
                        key: value for key, value in host.items() if key != "n_runners"
                    }
                    configured_settings = {
                        key: value
                        for key, value in configured_host.items()
                        if key != "n_runners"
                    }
                    if current_settings != configured_settings:
                        raise ValueError(
                            "Only n_runners may change while a search is running"
                        )
                    cpu_capacity = host_nodes[host["name"]]["Resources"].get("CPU", 0)
                    if configured_host["n_runners"] > cpu_capacity:
                        raise ValueError(
                            f"{host['name']} requests {configured_host['n_runners']} "
                            f"runners but its Ray node advertises "
                            f"{cpu_capacity:g} CPUs"
                        )
            except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
                error_message = str(error)
                if error_message != last_config_error:
                    print(
                        "[ray-cluster] Could not apply live runner settings; "
                        f"retaining the last valid limits: {error_message}"
                    )
                last_config_error = error_message
                return False

            last_config_error = None
            changed = False
            for host, configured_host in zip(hosts, configured_hosts):
                old_target = host["n_runners"]
                new_target = configured_host["n_runners"]
                if old_target != new_target:
                    host["n_runners"] = new_target
                    changed = True
                    print(
                        f"[{host['name']}] runner target changed from "
                        f"{old_target} to {new_target}"
                    )
            return changed

        def submit_command(host_idx: int) -> None:
            host = hosts[host_idx]
            command = commands[0]
            gpu_ids = host["gpus"]
            gpu = gpu_ids[gpu_indices[host_idx] % len(gpu_ids)] if gpu_ids else None
            memory_request = (
                host_nodes[host["name"]]["Resources"]["memory"] / host["n_runners"]
            )
            run_ref = run_task.options(
                num_cpus=1,
                memory=memory_request,
                scheduling_strategy=node_strategy(host),
            ).remote(host["project_dir"], gpu, [command])
            commands.popleft()
            pending[run_ref] = (host_idx, command)
            active_counts[host_idx] += 1
            gpu_indices[host_idx] += 1

        def submit_available_commands() -> None:
            nonlocal next_host_idx
            while commands:
                available_host_idx = None
                for offset in range(len(hosts)):
                    host_idx = (next_host_idx + offset) % len(hosts)
                    if active_counts[host_idx] < hosts[host_idx]["n_runners"]:
                        available_host_idx = host_idx
                        break
                if available_host_idx is None:
                    break
                submit_command(available_host_idx)
                next_host_idx = (available_host_idx + 1) % len(hosts)

        def print_progress() -> None:
            host_counts = ", ".join(
                f"{host['name']}={active_counts[host_idx]}/{host['n_runners']}"
                for host_idx, host in enumerate(hosts)
            )
            print(
                f"[ray-cluster] in-flight tasks: {host_counts}; "
                f"queued commands: {len(commands)}"
            )

        refresh_runner_targets()
        submit_available_commands()
        print_progress()
        while pending:
            ready_refs, _ = ray.wait(
                list(pending),
                num_returns=1,
                timeout=poll_interval,
            )
            if ready_refs:
                run_ref = ready_refs[0]
                host_idx, command = pending.pop(run_ref)
                host = hosts[host_idx]
                active_counts[host_idx] -= 1
                try:
                    outcomes = ray.get(run_ref)
                except ray.exceptions.RayError as error:
                    failures.append(
                        f"{host['name']}: {command} Ray task failed: {error}"
                    )
                else:
                    if len(outcomes) != 1:
                        raise RuntimeError(
                            f"Expected one outcome for {host['name']} command {command}"
                        )
                    if outcomes[0]["returncode"]:
                        failures.append(
                            f"{host['name']}: {command} exited with "
                            f"{outcomes[0]['returncode']}"
                        )

            targets_changed = refresh_runner_targets()
            submit_available_commands()
            if ready_refs or targets_changed:
                print_progress()

        return failures


    @staticmethod
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

    @staticmethod
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

    @classmethod
    def update_running_hosts(cls, config_path: str, ray_address: str) -> None:
        hosts = cls.load_lab_hosts(config_path)
        with cls._ray_session(ray_address) as (
            ray,
            _,
            NodeAffinitySchedulingStrategy,
        ):
            nodes_by_ip = {}
            for node in ray.nodes():
                if node.get("Alive"):
                    nodes_by_ip.setdefault(node["NodeManagerAddress"], []).append(node)

            running_hosts = []
            for host in hosts:
                matching_nodes = nodes_by_ip.get(host["ray_node_ip"], [])
                if not matching_nodes:
                    print(
                        f"[{host['name']}] skipped: no live Ray node at "
                        f"{host['ray_node_ip']}"
                    )
                    continue
                if len(matching_nodes) != 1:
                    raise RuntimeError(
                        f"Expected one live Ray node at {host['ray_node_ip']} for "
                        f"{host['name']}, found {len(matching_nodes)}"
                    )
                running_hosts.append((host, matching_nodes[0]))

            if not running_hosts:
                raise RuntimeError(
                    "No configured lab computers are currently live in Ray"
                )

            from src.experiments.ray_nodes_git_update import _update_checkout

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

    def __init__(
        self,
        config_path: str,
        ray_address: str,
        sync_wandb: bool = True,
        open_ray_dashboard: bool = False,
    ) -> None:
        self.config_path = config_path
        self.hosts = self.load_lab_hosts(config_path)
        self.ray_address = ray_address
        self.sync_wandb = sync_wandb
        self.open_ray_dashboard = open_ray_dashboard

    def run(self, commands) -> None:
        self.hosts = self.load_lab_hosts(self.config_path)
        command_queue = deque(str(command) for command in commands)
        with self._ray_session(self.ray_address) as (
            ray,
            ray_context,
            NodeAffinitySchedulingStrategy,
        ):
            dashboard_url = ray_context.dashboard_url
            if self.open_ray_dashboard and not dashboard_url:
                raise RuntimeError(
                    "The Ray dashboard is unavailable. Start the Ray head with "
                    "dashboard support enabled, then retry."
                )

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
                memory_capacity = node.get("Resources", {}).get("memory")
                if (
                    not isinstance(memory_capacity, (int, float))
                    or memory_capacity <= 0
                ):
                    raise ValueError(
                        f"{host['name']} does not advertise a positive Ray memory "
                        "resource; memory-aware command scheduling is unavailable"
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

            has_commands = bool(command_queue)
            if self.open_ray_dashboard and has_commands and dashboard_url:
                self._open_dashboard(dashboard_url)

            failures = self._run_command_queues(
                ray,
                run_task,
                self.hosts,
                command_queue,
                host_nodes,
                node_strategy,
                self.config_path,
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
                for host, sync_ref in zip(self.hosts, sync_refs):
                    try:
                        outcomes = ray.get(sync_ref)
                    except ray.exceptions.RayError as error:
                        failures.append(
                            f"{host['name']}: W&B sync Ray task failed: {error}"
                        )
                        continue
                    for outcome in outcomes:
                        if outcome["returncode"]:
                            failures.append(
                                f"{host['name']}: W&B sync failed for "
                                f"{outcome['run_dir']}: {outcome['output']}"
                            )
                        else:
                            print(f"[{host['name']}] synced {outcome['run_dir']}")

            if failures:
                raise RuntimeError("Ray cluster failures:\n" + "\n".join(failures))

    def _ensure_matching_environments(self, preparations: list[dict]) -> None:
        reference_host = self.hosts[0]
        reference_environment = preparations[0]["venv"]
        mismatches = []

        for host, preparation in zip(self.hosts[1:], preparations[1:]):
            environment = preparation["venv"]
            if (
                environment["python_major"] == reference_environment["python_major"]
                and environment["lockfile_sha256"]
                == reference_environment["lockfile_sha256"]
            ):
                continue
            differences = []
            if environment["python_major"] != reference_environment["python_major"]:
                differences.append(f"Python major {environment['python_major']}")
            if (
                environment["lockfile_sha256"]
                != reference_environment["lockfile_sha256"]
            ):
                differences.append("poetry.lock differs")
            mismatches.append(f"{host['name']} ({'; '.join(differences)})")

        if mismatches:
            raise RuntimeError(
                "Lab computer Python major versions or Poetry lockfiles differ from "
                f"{reference_host['name']} (Python major "
                f"{reference_environment['python_major']}; Python "
                f"{reference_environment['python_version']}; poetry.lock SHA-256 "
                f"{reference_environment['lockfile_sha256']}). Ensure all hosts use "
                "the same lockfile, then retry. Differences: " + "; ".join(mismatches)
            )
        print(
            "All lab computers have compatible Python majors and the same poetry.lock"
        )

    def _open_dashboard(self, dashboard_url: str) -> None:
        print(f"Ray dashboard: {dashboard_url}")
        webbrowser.open(dashboard_url)


# Keep Ray's module-level import names stable so older workers can run preflight.
def _prepare_node(project_dir: str, gpus: list[int], sync_wandb: bool) -> dict:
    return RayCluster._prepare_node(project_dir, gpus, sync_wandb)


def _run_queue(project_dir: str, gpu: int | None, commands: list[str]) -> list[dict]:
    return RayCluster._run_queue(project_dir, gpu, commands)


def _sync_node(project_dir: str, baseline_runs: list[str]) -> list[dict]:
    return RayCluster._sync_node(project_dir, baseline_runs)
