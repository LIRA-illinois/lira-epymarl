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
