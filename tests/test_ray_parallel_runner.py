import os
from types import SimpleNamespace

import pytest
import torch as th

from src.components.episode_buffer import EpisodeBatch
from src.controllers.basic_controller import BasicMAC
from src.runners.ray_parallel_runner import RayParallelRunner, _cpu_policy_copy


def _make_batch(scheme, values):
    batch = EpisodeBatch(
        scheme=scheme,
        groups={},
        batch_size=len(values),
        max_seq_length=2,
        device="cpu",
    )
    batch.update({"state": [[value] for value in values]}, ts=0)
    return batch


def test_policy_snapshot_moves_models_and_args_to_cpu():
    args = SimpleNamespace(
        n_agents=1,
        n_actions=2,
        obs_last_action=False,
        obs_agent_id=False,
        agent="rnn",
        agent_output_type="q",
        hidden_dim=8,
        use_rnn=True,
        action_selector="epsilon_greedy",
        epsilon_start=1.0,
        epsilon_finish=0.05,
        epsilon_anneal_time=100,
        evaluation_epsilon=0.05,
        device="cuda",
        use_cuda=True,
    )
    scheme = {
        "obs": {"vshape": 2, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
    }
    policy = BasicMAC(scheme, {"agents": 1}, args)

    snapshot = _cpu_policy_copy(policy)

    assert snapshot.args.device == "cpu"
    assert snapshot.args.use_cuda is False
    assert all(parameter.device.type == "cpu" for parameter in snapshot.agent.parameters())
    assert policy.args.device == "cuda"


def test_policy_snapshot_detaches_nonleaf_cached_tensors():
    class Policy(th.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = th.nn.Linear(2, 2)
            self.cached_weight = self.layer.weight * 2

    policy = Policy()
    snapshot = _cpu_policy_copy(policy)

    assert snapshot.cached_weight.grad_fn is None
    assert snapshot.cached_weight.data_ptr() != policy.cached_weight.data_ptr()
    assert snapshot.layer.weight.data_ptr() != policy.layer.weight.data_ptr()


def test_episode_batches_merge_into_driver_batch():
    scheme = {"state": {"vshape": (1,)}}
    runner = RayParallelRunner.__new__(RayParallelRunner)
    runner.node_batch_sizes = [2, 1]
    runner.new_batch = lambda: EpisodeBatch(
        scheme=scheme,
        groups={},
        batch_size=3,
        max_seq_length=2,
        device="cpu",
    )

    batch = runner._merge_batches(
        [
            {"batch": _make_batch(scheme, [4.0, 5.0])},
            {"batch": _make_batch(scheme, [9.0])},
        ]
    )

    assert batch["state"][:, 0, 0].tolist() == [4.0, 5.0, 9.0]
    assert batch["filled"][:, 0, 0].tolist() == [1, 1, 1]


def test_parallel_ray_requires_measured_memory_reservation():
    args = SimpleNamespace(
        batch_size_run=1,
        ray_cpus_per_env=1,
        ray_memory_per_env_gb=None,
    )

    with pytest.raises(ValueError, match="ray_memory_per_env_gb"):
        RayParallelRunner(args, logger=None)


@pytest.mark.skipif(
    os.environ.get("RUN_RAY_INTEGRATION") != "1",
    reason="set RUN_RAY_INTEGRATION=1 to start a local Ray runtime",
)
def test_local_ray_runs_and_merges_multiple_environment_pools(monkeypatch):
    from src.main import main

    observed_batches = []
    run = RayParallelRunner.run

    def checked_run(runner, *args, **kwargs):
        previous_t_env = runner.t_env
        result = run(runner, *args, **kwargs)
        assert len(runner.workers) == 2
        if not kwargs.get("test_mode", args[0] if args else False):
            assert result.batch_size == 2
            assert runner.env_steps_this_run > 0
            assert runner.t_env - previous_t_env == runner.env_steps_this_run
            assert int(result["filled"].sum()) - result.batch_size == (
                runner.env_steps_this_run
            )
            observed_batches.append(result.batch_size)
        return result

    monkeypatch.setattr(RayParallelRunner, "run", checked_run)
    main(
        [
            "src/main.py",
            "--config=maic",
            "--env-config=team-nav-v0",
            "with",
            "seed=101",
            "runner=parallel_ray",
            "ray_memory_per_env_gb=0.5",
            "ray_node_overhead_memory_gb=0.5",
            "ray_envs_per_node=1",
            "ray_worker_start_timeout_s=120",
            "batch_size_run=2",
            "use_cuda=False",
            "use_wandb=False",
            "use_sacred=False",
            "save_model=False",
            "t_max=24",
            "batch_size=1",
            "buffer_size=64",
            "test_nepisode=2",
            "test_interval=10000",
            "log_interval=1",
            "runner_log_interval=1",
            "epsilon_anneal_time=48",
            "correlated_exploration=True",
            "correlated_exploration_prob=0.75",
            "unique_policy_per_msg_budget=True",
            "msg_budget_per_agent=[1]",
            "msg_filter_type=hard_topk",
            "msg_drop_rate=0.0",
            "hl_task=[0 1]",
            "env_args.map_name=3ga_1r_very_small_hall",
            "env_args.navigation_config=submodules/gym-multigrid/gym_multigrid/envs/maps/team_navigation/navigation_hallway_cm_smoke.yaml",
            "env_args.hierarchical=True",
            "env_args.terminate_on_task_completed=True",
            "env_args.max_episode_steps=6",
            "env_args.n_agents=3",
            "env_args.goal_type=simultaneous_arrival",
            "env_args.reward_config.simultaneous_goal_reward_type=per_step_arrival_group_size",
            "env_args.agent_0_delay_prob=0.0",
        ]
    )

    assert observed_batches
