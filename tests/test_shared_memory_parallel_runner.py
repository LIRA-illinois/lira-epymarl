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
