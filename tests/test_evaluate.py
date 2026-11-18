from types import SimpleNamespace

from src.simulation.evaluate import run_eval_episodes


class _EvaluationRunner:
    def __init__(self):
        self.batch_size = 1
        self.logger = SimpleNamespace(info=lambda *_args: None)
        self.env = SimpleNamespace(terminate_on_task_completed=False)
        self.test_returns = []
        self.test_stats = {}

    def run(self, test_mode, return_log_stats, reset_options=None):
        if return_log_stats:
            self.test_returns.clear()
            self.test_stats.clear()
            return {"log_stats": {"test_n_episodes": 1}}
        self.test_returns.append(1)
        self.test_stats["n_episodes"] = self.test_stats.get("n_episodes", 0) + 1
        return {}

    def start_recording(self, **_kwargs):
        pass

    def stop_recording(self, **_kwargs):
        pass


def test_saved_replays_do_not_pollute_later_evaluation_stats():
    runner = _EvaluationRunner()
    args = SimpleNamespace(
        save_test_replays=True,
        n_test_replays_save=1,
        unique_policy_per_msg_budget=False,
    )

    result = run_eval_episodes(
        args=args,
        runner=runner,
        n_eval_eps=1,
        t_env=500_000,
    )

    assert result["log_stats"]["test_n_episodes"] == 1
    assert runner.test_returns == []
    assert runner.test_stats == {}
