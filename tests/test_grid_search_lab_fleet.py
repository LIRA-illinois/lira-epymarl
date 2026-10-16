import pytest

from experiments.ray_nodes_git_update import _update_checkout
from src.experiments.ray_lab_fleet import (
    RayLabFleet,
    _run_queue,
    _sync_node,
    distribute_commands,
    load_lab_hosts,
)


def test_load_lab_hosts_reads_runner_and_gpu_settings(tmp_path):
    config_path = tmp_path / "lab_hosts.yaml"
    config_path.write_text(
        "hosts: [{name: lab-a, ray_node_ip: 10.0.0.1, project_dir: /home/user/project, n_runners: 2, gpus: [0, 2]}]",
        encoding="utf8",
    )

    hosts = load_lab_hosts(str(config_path))

    assert hosts[0]["n_runners"] == 2
    assert hosts[0]["gpus"] == [0, 2]


def test_load_lab_hosts_requires_absolute_project_path(tmp_path):
    config_path = tmp_path / "lab_hosts.yaml"
    config_path.write_text(
        "hosts: [{name: lab-a, ray_node_ip: 10.0.0.1, project_dir: project, n_runners: 1, gpus: [0]}]",
        encoding="utf8",
    )

    with pytest.raises(ValueError, match="absolute path"):
        load_lab_hosts(str(config_path))


def test_distribute_commands_round_robins_across_host_runner_slots():
    hosts = [
        {"name": "lab-a", "n_runners": 2},
        {"name": "lab-b", "n_runners": 1},
    ]

    queues = distribute_commands(["run-1", "run-2", "run-3", "run-4"], hosts)

    assert queues == [[["run-1", "run-4"], ["run-2"]], [["run-3"]]]


def test_runner_task_sets_requested_gpu_and_returns_command_status(monkeypatch):
    captured = {}

    def fake_run(command, cwd, env, check):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["visible_gpu"] = env["CUDA_VISIBLE_DEVICES"]
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr("src.experiments.ray_lab_fleet.subprocess.run", fake_run)
    outcomes = _run_queue(
        "/home/user/project",
        gpu=5,
        commands=["python src/main.py --debug"],
    )

    assert captured["cwd"] == "/home/user/project"
    assert captured["visible_gpu"] == "5"
    assert "source .venv/bin/activate" in captured["command"][-1]
    assert outcomes[0]["returncode"] == 0


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

    monkeypatch.setattr("src.experiments.update_lab_fleet.subprocess.run", fake_run)

    outcome = _update_checkout(str(tmp_path))
    shell_script = captured["command"][-1]

    assert captured["cwd"] == str(tmp_path)
    assert "git pull --ff-only" in shell_script
    assert "git submodule update --init --recursive" in shell_script
    assert "git reset" not in shell_script
    assert "git clean" not in shell_script
    assert outcome["returncode"] == 0


def test_preflight_accepts_python_minor_and_tooling_differences(capsys):
    fleet = RayLabFleet.__new__(RayLabFleet)
    fleet.hosts = [{"name": "lab-a"}, {"name": "lab-b"}]
    fleet._ensure_matching_environments([
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
    fleet = RayLabFleet.__new__(RayLabFleet)
    fleet.hosts = [{"name": "lab-a"}, {"name": "lab-b"}]

    with pytest.raises(
        RuntimeError, match="Python major versions or Poetry lockfiles differ"
    ):
        fleet._ensure_matching_environments([
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
    fleet = RayLabFleet.__new__(RayLabFleet)
    fleet.hosts = [{"name": "lab-a"}, {"name": "lab-b"}]

    with pytest.raises(RuntimeError, match="Python major 4"):
        fleet._ensure_matching_environments([
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

    monkeypatch.setattr("src.experiments.ray_lab_fleet.subprocess.run", fake_run)

    outcomes = _sync_node(str(tmp_path), ["offline-run-old"])

    assert sync_calls == [str(wandb_dir / "offline-run-new")]
    assert len(outcomes) == 1
    assert outcomes[0]["returncode"] == 0


def test_dashboard_opens_when_requested(monkeypatch):
    opened_urls = []
    monkeypatch.setattr(
        "src.experiments.ray_lab_fleet.webbrowser.open",
        lambda url: opened_urls.append(url),
    )
    fleet = RayLabFleet.__new__(RayLabFleet)

    fleet._open_dashboard("http://127.0.0.1:8265")

    assert opened_urls == ["http://127.0.0.1:8265"]
