from types import SimpleNamespace

from src.simulation.evaluate import run_eval_episodes


class _EvaluationRunner:
    def __init__(self):
        self.batch_size = 1
        self.logger = SimpleNamespace(info=lambda *_args: None)
        self.env = SimpleNamespace(terminate_on_task_completed=False)
        self.test_returns = []
        self.test_stats = {}
        self.mac = SimpleNamespace(msg_budget_per_agent=-1)

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


def test_saved_replay_recording_uses_selected_episode_prefix():
    runner = _EvaluationRunner()
    runner.recording_calls = []

    def start_recording(**kwargs):
        runner.recording_calls.append(("start", kwargs))

    def stop_recording(**kwargs):
        runner.recording_calls.append(("stop", kwargs))

    runner.start_recording = start_recording
    runner.stop_recording = stop_recording

    args = SimpleNamespace(
        save_test_replays=True,
        n_test_replays_save=2,
        unique_policy_per_msg_budget=False,
        n_agents=3,
    )
    result = run_eval_episodes(
        args=args,
        runner=runner,
        n_eval_eps=1,
        t_env=500_000,
        video_prefix="replay",
        reset_options={
            "hl_start_state": 0,
            "msg_budget_per_agent": 1.0,
        },
    )

    assert result["log_stats"]["test_n_episodes"] == 1
    assert runner.recording_calls[0][0] == "start"
    assert runner.recording_calls[0][1]["video_prefix"] == (
        "replay-task_0-msg_budget_1.00"
    )
    assert runner.recording_calls[0][1]["n_test_replays_save"] == 2
    assert runner.recording_calls[-1][0] == "stop"


def test_collect_statewise_calibration_overrides_full_spawn_distribution():
    from src.simulation.evaluate import collect_statewise_calibration_row
    from src.simulation.hierarchical_types import OptionSettingKey

    class _CaptureRunner:
        def __init__(self):
            self.batch_size = 1
            self.t_env = 10
            self.test_returns = []
            self.test_stats = {}
            self.env = SimpleNamespace(terminate_on_task_completed=False)
            self.mac = SimpleNamespace(msg_budget_per_agent=-1)
            self.reset_options_seen = []

        def run(self, test_mode, return_log_stats, reset_options=None):
            del test_mode, return_log_stats
            self.reset_options_seen.append(reset_options)
            return {
                "final_infos": [
                    {
                        "task_completed": True,
                        "final_state": ((1, 0),),
                    }
                ]
            }

    runner = _CaptureRunner()
    setting = OptionSettingKey(
        subtask_edge=(0, 1),
        communication_capacity=1.0,
        start_state=(0, 1),
    )
    collect_statewise_calibration_row(
        runner,
        setting=setting,
        seed=101,
        n_eval_eps=1,
        reset_options={
            "navigation_init_state_dist": {
                "states": [[[0, 0]], [[0, 1]]],
                "probs": [0.5, 0.5],
            }
        },
    )

    assert runner.reset_options_seen[0]["navigation_init_state_dist"] == {
        "states": [[[0, 1]]],
        "probs": [1.0],
    }


def test_collect_statewise_calibration_can_disable_replays():
    from src.simulation.evaluate import collect_statewise_calibration_row
    from src.simulation.hierarchical_types import OptionSettingKey

    class _NoReplayRunner:
        batch_size = 1

        def __init__(self):
            self.args = SimpleNamespace(
                save_test_replays=True,
                n_test_replays_save=2,
            )
            self.t_env = 10
            self.test_returns = []
            self.test_stats = {}
            self.env = SimpleNamespace(terminate_on_task_completed=False)
            self.mac = SimpleNamespace(msg_budget_per_agent=-1)
            self.recording_calls = []

        def run(self, test_mode, return_log_stats, reset_options=None):
            del test_mode, return_log_stats, reset_options
            return {
                "final_infos": [
                    {
                        "task_completed": True,
                        "final_state": ((1, 0),),
                    }
                ]
            }

        def start_recording(self, **kwargs):
            self.recording_calls.append(("start", kwargs))

        def stop_recording(self, **kwargs):
            self.recording_calls.append(("stop", kwargs))

    runner = _NoReplayRunner()
    collect_statewise_calibration_row(
        runner,
        setting=OptionSettingKey(
            subtask_edge=(0, 1),
            communication_capacity=1.0,
            start_state=(0, 0),
        ),
        seed=101,
        n_eval_eps=1,
        record_replays=False,
    )

    assert runner.recording_calls == []
