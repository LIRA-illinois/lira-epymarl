from unittest.mock import Mock

import pytest

from src.main import experiment_main
from src.runners.shared_memory_parallel_runner import (
    SharedMemoryParallelRunner,
    _get_env_attr,
)


class _FakeConnection:
    def __init__(self, response):
        self.response = response
        self.sent = None

    def send(self, message):
        self.sent = message

    def recv(self):
        return self.response


def test_get_env_attr_reads_from_worker_and_returns_default():
    connection = _FakeConnection(False)
    runner = SharedMemoryParallelRunner.__new__(SharedMemoryParallelRunner)
    runner.parent_conns = (connection,)

    assert runner.get_env_attr("terminate_on_task_completed") is False
    assert connection.sent == ("get_attr", ("terminate_on_task_completed", None))


def test_get_env_attr_uses_default_for_missing_attribute():
    assert _get_env_attr(object(), "missing", default=False) is False


def test_reset_sends_distinct_worker_options_and_policy_contexts():
    from types import SimpleNamespace

    args = SimpleNamespace(
        batch_size_run=3,
        condition_on_msg_budget=True,
        condition_on_subtask=True,
        subtask_contexts=[[0, 1], [1, 2]],
        n_agents=2,
        msg_budget_per_agent=[0, 1],
        env_args={},
    )
    runner = SharedMemoryParallelRunner.__new__(SharedMemoryParallelRunner)
    runner.args = args
    runner.batch_size = 3
    runner.new_batch = Mock()
    runner.parent_conns = tuple(
        _FakeConnection("reset") for _ in range(runner.batch_size)
    )
    runner._write_pre_transition_data = Mock()

    worker_options = [
        {"hl_task": [0, 1], "msg_budget_per_agent": 0},
        {"hl_task": [1, 2], "msg_budget_per_agent": 1},
        {"hl_task": [0, 1], "msg_budget_per_agent": 1},
    ]
    runner.reset(options=worker_options)

    assert [connection.sent for connection in runner.parent_conns] == [
        ("reset", worker_option) for worker_option in worker_options
    ]
    assert len(runner._policy_contexts) == 3
    assert not (
        runner._policy_contexts[0] == runner._policy_contexts[1]
    ).all()


def test_mixed_navigation_sampling_is_independent_per_worker(monkeypatch):
    import numpy as np

    from src.simulation import unified_hierarchical_runner as runner_module

    def fake_navigation_reset(_args, edge, capacity):
        del capacity
        return {
            "hl_start_state": edge[0],
            "hl_task": list(edge),
            "navigation_task_transition": list(edge),
            "navigation_init_state_dist": {
                "states": [[[edge[0], 0]], [[edge[0], 1]]],
                "probs": [0.75, 0.25],
            },
            "msg_budget_per_agent": 0,
        }

    monkeypatch.setattr(
        runner_module,
        "_navigation_reset_options",
        fake_navigation_reset,
    )
    rng = np.random.default_rng(101)
    options = runner_module._sample_mixed_navigation_reset_options(
        object(),
        ((0, 1), (0, 2), (1, 3)),
        (0.0, 1.0),
        batch_size=20,
        rng=rng,
    )

    assert len({tuple(option["hl_task"]) for option in options}) > 1
    assert len({
        option["msg_budget_per_agent"] for option in options
    }) > 1
    singleton_states = {
        tuple(
            tuple(position)
            for position in option["navigation_init_state_dist"]["states"][0]
        )
        for option in options
    }
    assert len(singleton_states) > 1
    assert all(
        option["navigation_init_state_dist"]["probs"] == [1.0]
        for option in options
    )


def test_mixed_navigation_sampling_honors_equal_initial_state_probabilities(monkeypatch):
    import numpy as np

    from src.simulation import unified_hierarchical_runner as runner_module

    def fake_navigation_reset(_args, edge, capacity):
        del capacity
        return {
            "hl_start_state": edge[0],
            "hl_task": list(edge),
            "navigation_task_transition": list(edge),
            "navigation_init_state_dist": {
                "states": [
                    [[edge[0], 1], [edge[0], 3], [edge[0], 5]],
                    [[edge[0] + 1, 1], [edge[0] + 1, 3], [edge[0] + 1, 5]],
                ],
                "probs": [0.5, 0.5],
            },
            "msg_budget_per_agent": 0,
        }

    monkeypatch.setattr(
        runner_module,
        "_navigation_reset_options",
        fake_navigation_reset,
    )
    rng = np.random.default_rng(2026)
    options = runner_module._sample_mixed_navigation_reset_options(
        object(),
        ((0, 1),),
        (0.0, 1.0),
        batch_size=4000,
        rng=rng,
    )
    chosen_columns = [
        tuple(
            tuple(position)
            for position in option["navigation_init_state_dist"]["states"][0]
        )
        for option in options
    ]
    support_counts = {
        support: chosen_columns.count(support)
        for support in set(chosen_columns)
    }
    proportions = [
        count / len(chosen_columns)
        for count in support_counts.values()
    ]

    assert set(support_counts) and len(support_counts) == 2
    assert all(0.47 <= proportion <= 0.53 for proportion in proportions)


def test_start_recording_isolated_by_task_prefix(tmp_path):
    connection = _FakeConnection("recording")
    logger = Mock()
    logger.dir = str(tmp_path)
    runner = SharedMemoryParallelRunner.__new__(SharedMemoryParallelRunner)
    runner.t_env = 500_000
    runner.logger = logger
    runner.parent_conns = (connection,)

    runner.start_recording(
        n_test_replays_save=2,
        video_prefix="replay-task-3-4-msg-0.00",
        t_env=500_000,
    )

    command, payload = connection.sent
    assert command == "start_recording"
    assert payload["video_folder"].endswith(
        "replays/t_500000/replay-task-3-4-msg-0_00"
    )
    assert (tmp_path / "replays" / "t_500000" / "replay-task-3-4-msg-0_00").is_dir()


def test_close_env_releases_shared_memory_once():
    connection = Mock()
    process = Mock()
    process.is_alive.return_value = False
    shared_memory = Mock()
    runner = SharedMemoryParallelRunner.__new__(SharedMemoryParallelRunner)
    runner._closed = False
    runner.parent_conns = (connection,)
    runner.ps = (process,)
    runner._shms = {"state": shared_memory}

    runner.close_env()
    runner.close_env()

    connection.send.assert_called_once_with(("close", None))
    process.join.assert_called_once_with(timeout=2)
    shared_memory.close.assert_called_once_with()
    shared_memory.unlink.assert_called_once_with()


def test_experiment_main_closes_shared_memory_runner_after_failure(monkeypatch):
    runner = SharedMemoryParallelRunner.__new__(SharedMemoryParallelRunner)
    runner.close_env = Mock()

    class _FailingSimulation:
        def __init__(self, config, logger):
            self.runner = runner

        def run(self):
            raise KeyboardInterrupt

        def finish(self):
            raise AssertionError("finish should not run after an interruption")

    monkeypatch.setattr("src.main.Simulation", _FailingSimulation)

    with pytest.raises(KeyboardInterrupt):
        experiment_main({"seed": 1, "env_args": {"seed": 1}}, Mock())

    runner.close_env.assert_called_once_with()


def test_experiment_main_normalizes_hierarchical_config_before_simulation(monkeypatch):
    captured = {}

    class _Simulation:
        def __init__(self, config, logger):
            captured["config"] = config
            self.runner = object()

        def run(self):
            captured["ran"] = True

        def finish(self):
            captured["finished"] = True

    monkeypatch.setattr("src.main.Simulation", _Simulation)

    experiment_main(
        {
            "seed": 101,
            "env_args": {},
            "msg_budget_per_agent": [0, 1],
            "hierarchical": {
                "task_sequence": [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]],
                "pg_values": [0.55, 0.65],
                "epsilon_est": 0.05,
                "beta": 0.95,
                "epsilon_p": 0.20,
                "n_max": 4,
                "n_calibration_episodes": 100,
                "held_out_episodes": 20,
                "max_iterations": 3,
                "resume": "none",
            },
        },
        Mock(),
    )

    assert captured["config"]["hierarchical"]["task_sequence"] == [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [3, 4],
    ]
    assert captured["config"]["msg_budget_per_agent"] == [0, 1]
    assert captured["ran"] is True
    assert captured["finished"] is True
