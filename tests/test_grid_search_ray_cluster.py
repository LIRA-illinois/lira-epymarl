import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.experiments import ray_cluster, ray_nodes_git_update
from src.experiments.ray_cluster import RayCluster


def test_ray_remote_entrypoints_keep_module_level_names():
    assert ray_cluster._prepare_node.__qualname__ == "_prepare_node"
    assert ray_cluster._run_queue.__qualname__ == "_run_queue"
    assert ray_cluster._sync_node.__qualname__ == "_sync_node"
    assert ray_nodes_git_update._update_checkout.__qualname__ == "_update_checkout"


def test_load_lab_hosts_reads_runner_and_gpu_settings(tmp_path):
    config_path = tmp_path / "lab_hosts.yaml"
    config_path.write_text(
        "hosts: [{name: lab-a, ray_node_ip: 10.0.0.1, project_dir: /home/user/project, n_runners: 2, gpus: [0, 2]}]",
        encoding="utf8",
    )

    hosts = RayCluster.load_lab_hosts(str(config_path))

    assert hosts[0]["n_runners"] == 2
    assert hosts[0]["gpus"] == [0, 2]


def test_load_lab_hosts_requires_absolute_project_path(tmp_path):
    config_path = tmp_path / "lab_hosts.yaml"
    config_path.write_text(
        "hosts: [{name: lab-a, ray_node_ip: 10.0.0.1, project_dir: project, n_runners: 1, gpus: [0]}]",
        encoding="utf8",
    )

    with pytest.raises(ValueError, match="absolute path"):
        RayCluster.load_lab_hosts(str(config_path))


@pytest.mark.parametrize("already_initialized", [False, True])
def test_ray_session_only_shuts_down_ray_it_started(
    monkeypatch, already_initialized
):
    calls = []
    fake_ray = ModuleType("ray")
    fake_ray.is_initialized = lambda: already_initialized

    def fake_init(address=None):
        calls.append(("init", address))
        return SimpleNamespace(dashboard_url="http://localhost:8265")

    fake_ray.init = fake_init
    fake_ray.shutdown = lambda: calls.append(("shutdown", None))

    fake_util = ModuleType("ray.util")
    fake_scheduling = ModuleType("ray.util.scheduling_strategies")

    class FakeNodeAffinitySchedulingStrategy:
        pass

    fake_scheduling.NodeAffinitySchedulingStrategy = FakeNodeAffinitySchedulingStrategy
    fake_util.scheduling_strategies = fake_scheduling
    fake_ray.util = fake_util
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.util", fake_util)
    monkeypatch.setitem(
        sys.modules,
        "ray.util.scheduling_strategies",
        fake_scheduling,
    )

    with RayCluster._ray_session("ray-address") as (ray, context, strategy):
        assert ray is fake_ray
        assert context.dashboard_url == "http://localhost:8265"
        assert strategy is FakeNodeAffinitySchedulingStrategy

    expected_calls = [("init", None if already_initialized else "ray-address")]
    if not already_initialized:
        expected_calls.append(("shutdown", None))
    assert calls == expected_calls


def test_distribute_commands_round_robins_across_hosts_by_runner_capacity():
    hosts = [
        {"name": "lab-a", "n_runners": 2},
        {"name": "lab-b", "n_runners": 1},
    ]

    queues = RayCluster.distribute_commands(
        ["run-1", "run-2", "run-3", "run-4"], hosts
    )

    assert queues == [["run-1", "run-2", "run-4"], ["run-3"]]


def test_runner_task_sets_requested_gpu_and_returns_command_status(monkeypatch):
    captured = {}

    def fake_run(command, cwd, env, check):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["visible_gpu"] = env["CUDA_VISIBLE_DEVICES"]
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr("src.experiments.ray_cluster.subprocess.run", fake_run)
    outcomes = RayCluster._run_queue(
        "/home/user/project",
        gpu=5,
        commands=["python src/main.py --debug"],
    )

    assert captured["cwd"] == "/home/user/project"
    assert captured["visible_gpu"] == "5"
    assert "source .venv/bin/activate" in captured["command"][-1]
    assert outcomes[0]["returncode"] == 0


def test_command_queues_apply_live_runner_changes_and_preserve_failures(tmp_path):
    class RayTaskFailure(Exception):
        pass

    class FakeRef:
        def __init__(self, command):
            self.command = command

    class FakeRay:
        exceptions = SimpleNamespace(RayError=RayTaskFailure)

        def __init__(self):
            self.wait_calls = []

        def get(self, ref):
            if ref.command == "oom":
                write_config(1)
                raise RayTaskFailure("worker died")
            if ref.command == "success":
                write_config(3)
            return [{"returncode": 1 if ref.command == "nonzero" else 0}]

        def wait(self, refs, num_returns, timeout):
            self.wait_calls.append([ref.command for ref in refs])
            return refs[:num_returns], refs[num_returns:]

    class FakeRunTask:
        def __init__(self):
            self.options_seen = []
            self.commands = []

        def options(self, **options):
            self.options_seen.append(options)
            return self

        def remote(self, project_dir, gpu, commands):
            assert len(commands) == 1
            self.commands.append((project_dir, gpu, commands[0]))
            return FakeRef(commands[0])

    config_path = tmp_path / "lab_hosts.yaml"

    def write_config(n_runners):
        config_path.write_text(
            "hosts: [{name: lab-a, ray_node_ip: 10.0.0.1, "
            "project_dir: /home/user/project, "
            f"n_runners: {n_runners}, gpus: [0, 2]}}]",
            encoding="utf8",
        )

    hosts = [
        {
            "name": "lab-a",
            "ray_node_ip": "10.0.0.1",
            "project_dir": "/home/user/project",
            "n_runners": 2,
            "gpus": [0, 2],
        }
    ]
    queues = [["oom", "nonzero", "success", "extra-1", "extra-2", "extra-3"]]
    host_nodes = {"lab-a": {"Resources": {"CPU": 4, "memory": 1000}}}
    task = FakeRunTask()
    write_config(2)
    ray = FakeRay()

    failures = RayCluster._run_command_queues(
        ray,
        task,
        hosts,
        queues,
        host_nodes,
        lambda host: f"node-for-{host['name']}",
        str(config_path),
    )

    assert task.commands == [
        ("/home/user/project", 0, "oom"),
        ("/home/user/project", 2, "nonzero"),
        ("/home/user/project", 0, "success"),
        ("/home/user/project", 2, "extra-1"),
        ("/home/user/project", 0, "extra-2"),
        ("/home/user/project", 2, "extra-3"),
    ]
    assert [options["memory"] for options in task.options_seen] == [
        500,
        500,
        1000,
        1000 / 3,
        1000 / 3,
        1000 / 3,
    ]
    assert all(options["num_cpus"] == 1 for options in task.options_seen)
    assert ray.wait_calls[1] == ["nonzero"]
    assert failures == [
        "lab-a: oom Ray task failed: worker died",
        "lab-a: nonzero exited with 1",
    ]


def test_checkout_update_fast_forwards_without_resetting_or_cleaning(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_run(command, cwd, capture_output, text, check):
        captured["command"] = command
        captured["cwd"] = cwd
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "Already up to date.\n", "stderr": ""},
        )()

    monkeypatch.setattr("src.experiments.ray_cluster.subprocess.run", fake_run)

    outcome = RayCluster._update_checkout(str(tmp_path))
    shell_script = captured["command"][-1]

    assert captured["cwd"] == str(tmp_path)
    assert "git pull --ff-only" in shell_script
    assert "git submodule update --init --recursive" in shell_script
    assert "git reset" not in shell_script
    assert "git clean" not in shell_script
    assert outcome["returncode"] == 0


def test_preflight_accepts_python_minor_and_tooling_differences(capsys):
    cluster = RayCluster.__new__(RayCluster)
    cluster.hosts = [{"name": "lab-a"}, {"name": "lab-b"}]
    cluster._ensure_matching_environments([
        {
            "venv": {
                "python_major": 3,
                "python_version": "3.12.15",
                "lockfile_sha256": "shared-lock",
                "packages": ("poetry==2.5.1",),
            }
        },
        {
            "venv": {
                "python_major": 3,
                "python_version": "3.13.1",
                "lockfile_sha256": "shared-lock",
                "packages": ("poetry==2.5.2",),
            }
        },
    ])

    assert "compatible Python majors" in capsys.readouterr().out


def test_preflight_rejects_different_lockfiles():
    cluster = RayCluster.__new__(RayCluster)
    cluster.hosts = [{"name": "lab-a"}, {"name": "lab-b"}]

    with pytest.raises(
        RuntimeError, match="Python major versions or Poetry lockfiles differ"
    ):
        cluster._ensure_matching_environments([
            {
                "venv": {
                    "python_major": 3,
                    "python_version": "3.12.15",
                    "lockfile_sha256": "lock-a",
                }
            },
            {
                "venv": {
                    "python_major": 3,
                    "python_version": "3.12.15",
                    "lockfile_sha256": "lock-b",
                }
            },
        ])


def test_preflight_rejects_different_python_major_versions():
    cluster = RayCluster.__new__(RayCluster)
    cluster.hosts = [{"name": "lab-a"}, {"name": "lab-b"}]

    with pytest.raises(RuntimeError, match="Python major 4"):
        cluster._ensure_matching_environments([
            {
                "venv": {
                    "python_major": 3,
                    "python_version": "3.12.15",
                    "lockfile_sha256": "shared-lock",
                }
            },
            {
                "venv": {
                    "python_major": 4,
                    "python_version": "4.0.0",
                    "lockfile_sha256": "shared-lock",
                }
            },
        ])


def test_wandb_sync_only_selects_new_run_folders(tmp_path, monkeypatch):
    wandb_dir = tmp_path / "results" / "wandb"
    (wandb_dir / "offline-run-old").mkdir(parents=True)
    (wandb_dir / "offline-run-new").mkdir()
    sync_calls = []

    def fake_run(command, cwd, capture_output, text, check):
        sync_calls.append(command[-1])
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("src.experiments.ray_cluster.subprocess.run", fake_run)

    outcomes = RayCluster._sync_node(str(tmp_path), ["offline-run-old"])

    assert sync_calls == [str(wandb_dir / "offline-run-new")]
    assert len(outcomes) == 1
    assert outcomes[0]["returncode"] == 0


def test_dashboard_opens_when_requested(monkeypatch):
    opened_urls = []
    monkeypatch.setattr(
        "src.experiments.ray_cluster.webbrowser.open",
        lambda url: opened_urls.append(url),
    )
    cluster = RayCluster.__new__(RayCluster)

    cluster._open_dashboard("http://127.0.0.1:8265")

    assert opened_urls == ["http://127.0.0.1:8265"]
