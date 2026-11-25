from types import SimpleNamespace

import pandas as pd
import pytest

from src.simulation.hierarchical_config import (
    HierarchicalPolicyScope,
    HierarchicalResumeMode,
    UnifiedHierarchicalConfig,
    UnifiedRunIdentity,
    extract_unified_hierarchical_config,
)
from src.simulation.hierarchical_types import (
    CalibrationOutcome,
    CalibrationRow,
    CertificateStatus,
    ContextKey,
    HighLevelCandidate,
    OptionSettingKey,
    PolicyScope,
)
from src.simulation.run import Simulation, UnifiedHierarchicalSeedRunner


def test_wandb_logging_helpers_log_tables_and_pngs(tmp_path):
    from src.simulation.unified_hierarchical_runner import (
        _log_wandb_plot_paths,
        _log_wandb_table,
    )

    calls = []
    logger = SimpleNamespace(
        log_table=lambda **kwargs: calls.append(("table", kwargs)),
        log_image=lambda path, t, key: calls.append(
            ("image", path, t, key)
        ),
    )
    frame = pd.DataFrame([{"seed": 101, "pg": 0.55}])
    png_path = tmp_path / "success_vs_cost_seed_101.png"
    png_path.write_bytes(b"png")

    _log_wandb_table(logger, "high_level/frontier", frame, 100)
    _log_wandb_plot_paths(
        logger,
        [str(png_path), str(tmp_path / "success_vs_cost_seed_101.pdf")],
        100,
        "high_level/seed_101",
    )

    assert calls[0][0] == "table"
    assert calls[0][1]["key"] == "high_level/frontier"
    assert calls[1][0:4] == (
        "image",
        str(png_path),
        100,
        "high_level/seed_101/success_vs_cost_seed_101",
    )


def test_unified_hierarchical_config_validates_merge_dag_and_margin():
    config = UnifiedHierarchicalConfig(
        task_sequence=((0, 1), (0, 2), (1, 3), (2, 3), (3, 4)),
        policy_scope=HierarchicalPolicyScope.SHARED_ACROSS_CONTEXTS,
        pg_values=(0.65, 0.60),
        epsilon_est=0.05,
        beta=0.95,
        epsilon_p=0.20,
        n_max=4,
        n_calibration_episodes=100,
        max_iterations=3,
        resume=HierarchicalResumeMode.NONE,
    )

    assert config.pg_values == (0.60, 0.65)
    assert config.delta_p == pytest.approx(1 - 0.95**4)
    assert config.policy_scope == HierarchicalPolicyScope.SHARED_ACROSS_CONTEXTS


def test_low_level_modes_require_artifact_resume():
    for mode in ("calibration_only", "evaluation_only"):
        with pytest.raises(ValueError, match="loaded option artifact"):
            UnifiedHierarchicalConfig(
                task_sequence=((0, 1),),
                resume=HierarchicalResumeMode.NONE,
                low_level_mode=mode,
                epsilon_p=0.20,
            )

    for mode in ("train", "calibration_only", "evaluation_only"):
        config = UnifiedHierarchicalConfig(
            task_sequence=((0, 1),),
            resume=(
                HierarchicalResumeMode.LATEST
                if mode != "train"
                else HierarchicalResumeMode.NONE
            ),
            low_level_mode=mode,
            epsilon_p=0.20,
        )
        assert config.to_dict()["low_level_mode"] == mode


def test_unified_hierarchical_config_validates_specific_resume():
    config = UnifiedHierarchicalConfig(
        task_sequence=((0, 1),),
        resume=HierarchicalResumeMode.SPECIFIC_TIMESTEP,
        resume_timestep=50,
        epsilon_p=0.20,
    )

    assert config.resume_timestep == 50
    assert config.to_dict()["resume_timestep"] == 50

    with pytest.raises(ValueError, match="resume_timestep"):
        UnifiedHierarchicalConfig(
            task_sequence=((0, 1),),
            resume=HierarchicalResumeMode.SPECIFIC_TIMESTEP,
        )


def test_per_context_policy_requires_explicit_contexts():
    with pytest.raises(ValueError, match="requires contexts"):
        UnifiedHierarchicalConfig(
            task_sequence=((0, 1),),
            policy_scope=HierarchicalPolicyScope.PER_CONTEXT,
            epsilon_p=0.20,
        )

    config = UnifiedHierarchicalConfig(
        task_sequence=((0, 1),),
        policy_scope=HierarchicalPolicyScope.PER_CONTEXT,
        contexts=("map-a", "map-b"),
        epsilon_p=0.20,
    )
    assert config.contexts == ("map-a", "map-b")


def test_unified_hierarchical_config_rejects_cycle_and_small_margin():
    with pytest.raises(ValueError, match="acyclic"):
        UnifiedHierarchicalConfig(
            task_sequence=((0, 1), (1, 0)),
            epsilon_p=0.10,
        )

    with pytest.raises(ValueError, match="epsilon_p"):
        UnifiedHierarchicalConfig(
            task_sequence=((0, 1),),
            epsilon_est=0.05,
            n_max=4,
            epsilon_p=0.01,
        )


def test_unified_hierarchical_config_from_mapping_and_dag_validation():
    config = extract_unified_hierarchical_config(
        {
            "hierarchical": {
                "task_sequence": [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]],
                "policy_scope": "shared_across_contexts",
                "pg_values": [0.60, 0.65],
                "epsilon_est": 0.05,
                "beta": 0.95,
                "epsilon_p": 0.20,
                "n_max": 4,
                "n_calibration_episodes": 100,
                "max_iterations": 3,
                "resume": "none",
            }
        }
    )

    assert config is not None
    assert config.task_sequence == (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (3, 4),
    )
    assert config.pg_values == (0.60, 0.65)


def test_unified_hierarchical_config_validation_against_m_est():
    config = UnifiedHierarchicalConfig(
        task_sequence=((0, 1),),
        n_calibration_episodes=100,
        epsilon_p=0.10,
    )

    config.validate_calibration_count(100)
    with pytest.raises(ValueError, match="n_calibration_episodes"):
        config.validate_calibration_count(101)


def test_unified_run_identity_has_seed_isolated_artifact_key():
    identity = UnifiedRunIdentity(
        experiment="exp49",
        time_id="exp49_test",
        scenario="01",
        seed=101,
    )

    assert identity.artifact_key == "exp49_test/01/seed_101"


def test_unified_run_dispatches_through_seed_runner(tmp_path, monkeypatch):
    simulation = object.__new__(Simulation)
    simulation.args = SimpleNamespace(
        local_results_path=str(tmp_path),
        time_id="exp49_test",
        unique_token="exp49_token",
        seed=101,
    )
    simulation.runner = SimpleNamespace()
    simulation.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
    )

    config = UnifiedHierarchicalConfig(
        task_sequence=((0, 1), (0, 2), (1, 3), (2, 3), (3, 4)),
        pg_values=(0.60, 0.65),
        epsilon_est=0.05,
        beta=0.95,
        epsilon_p=0.20,
        n_max=4,
        n_calibration_episodes=100,
        max_iterations=3,
        resume=HierarchicalResumeMode.NONE,
    )

    dispatched = {}

    def fake_run(self, hierarchical_config, artifact_root):
        dispatched["config"] = hierarchical_config
        dispatched["artifact_root"] = artifact_root

    monkeypatch.setattr(UnifiedHierarchicalSeedRunner, "run", fake_run)
    simulation._run_unified_hierarchical(config)

    assert dispatched["config"] is config
    assert dispatched["artifact_root"].endswith(
        "hierarchical_artifacts/exp49_test/seed_101"
    ) or dispatched["artifact_root"].endswith(
        "hierarchical_artifacts/exp49_test/seed_101/"
    )


def test_run_sanity_check_normalizes_hierarchical_alias_and_margin():
    simulation = object.__new__(Simulation)
    config = {
        "use_cuda": False,
        "test_nepisode": 2,
        "batch_size_run": 1,
        "hl_env": {
            "task_sequence": [[0, 1], [1, 2]],
            "pg_values": [0.60],
            "epsilon_est": 0.05,
            "n_max": 2,
            "epsilon_p": 0.10,
        },
    }

    normalized = simulation._args_sanity_check(
        config,
        SimpleNamespace(warning=lambda *args, **kwargs: None),
    )

    assert "hl_env" not in normalized
    assert normalized["hierarchical"]["task_sequence"] == [[0, 1], [1, 2]]
    assert normalized["hierarchical"]["epsilon_p"] == 0.10


def test_real_candidate_held_out_runs_one_episode_per_selected_option():
    from src.simulation.unified_hierarchical_runner import (
        _evaluate_real_candidate_held_out,
    )

    class _Mac:
        msg_budget_per_agent = 1.0

        def __init__(self):
            self.agent = SimpleNamespace(
                load_state_dict=lambda weights: None,
            )

    class _Runner:
        batch_size = 1

        def __init__(self):
            self.test_returns = []
            self.test_stats = {}
            self.mac = _Mac()

        def get_env_attr(self, name):
            return False

        def set_env_attr(self, name, value):
            return None

        def run(self, test_mode, return_log_stats, reset_options):
            final_info = {
                "task_completed": True,
                "final_state": ((1, 1), (1, 3), (1, 5)),
            }
            return {
                "final_info": final_info,
                "final_infos": [final_info],
            }

    runner_self = SimpleNamespace(
        args=SimpleNamespace(
            env_args={
                "navigation_config": (
                    "submodules/gym-multigrid/gym_multigrid/envs/maps/"
                    "team_navigation/navigation_branching_small_hall.yaml"
                )
            },
        ),
        runner=_Runner(),
    )
    candidate = HighLevelCandidate(
        context_key=ContextKey(
            "held-out-test",
            PolicyScope.SHARED_ACROSS_CONTEXTS,
            101,
        ),
        goal_probability_spec=0.60,
        empirical_goal_probability=1.0,
        empirical_cost=1.0,
        status=CertificateStatus.CERTIFIED,
        iteration=0,
        task_policy=pd.DataFrame.from_records(
            [
                {
                    "state": 0,
                    "next_state": 1,
                    "occupancy": 1.0,
                    "action_probability": 1.0,
                }
            ]
        ),
        comms_budget_policy=pd.DataFrame.from_records(
            [
                {
                    "state": 0,
                    "next_state": 1,
                    "msg_budget_per_agent": 1.0,
                    "action_probability": 1.0,
                }
            ]
        ),
    )
    calibration_rows = (
        CalibrationRow(
            setting=OptionSettingKey((0, 1), 1.0, (0, 0)),
            outcomes=(CalibrationOutcome.success((1, 0)),),
            seed=101,
        ),
    )

    held_out_results, held_out_summaries = (
        _evaluate_real_candidate_held_out(
            runner_self,
            seed=101,
            pg=0.60,
            candidate=candidate,
            task_edges=((0, 1),),
            capacities=(1.0,),
            calibration_rows=calibration_rows,
            policy_bank={
                (0, 1, 1.0, 101): {"agent.weight": object()},
            },
            root_state=0,
            goal_states=(1,),
            n_episodes=3,
        )
    )

    assert len(held_out_results) == 3
    assert all(result.project_success for result in held_out_results)
    assert all(
        result.row_key.episode_index == index
        for index, result in enumerate(held_out_results)
    )
    assert held_out_summaries[0]["n_episodes"] == 3


def test_pg_iteration_verification_plot(tmp_path):
    from src.simulation.context_utils import plot_pg_iteration_verification

    iteration_df = pd.DataFrame(
        [
            {
                "pg": 0.55,
                "iteration": 0,
                "empirical_goal_probability": 0.60,
                "empirical_cost": 1.0,
            },
            {
                "pg": 0.55,
                "iteration": 1,
                "empirical_goal_probability": 0.80,
                "empirical_cost": 0.50,
            },
            {
                "pg": 0.65,
                "iteration": 0,
                "empirical_goal_probability": 0.70,
                "empirical_cost": 1.0,
            },
        ]
    )
    paths = plot_pg_iteration_verification(
        iteration_df,
        str(tmp_path),
        seed=101,
    )

    assert len(paths) == 3
    assert all((tmp_path / path.rsplit("/", 1)[-1]).is_file() for path in paths)


def test_navigation_reset_options_use_real_joint_position_distributions():
    from src.simulation.unified_hierarchical_runner import (
        _navigation_reset_options,
    )

    args = SimpleNamespace(
        env_args={
            "navigation_config": (
                "submodules/gym-multigrid/gym_multigrid/envs/maps/"
                "team_navigation/navigation_branching_small_hall.yaml"
            )
        }
    )

    for edge in ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4)):
        reset_options = _navigation_reset_options(args, edge, 1.0)
        dist = reset_options["navigation_init_state_dist"]
        assert reset_options["hl_task"] == list(edge)
        assert reset_options["navigation_task_transition"] == list(edge)
        assert dist["states"]
        assert all(
            isinstance(position, list)
            for joint_state in dist["states"]
            for position in joint_state
        )
        assert sum(dist["probs"]) == pytest.approx(1.0)

    tiny_args = SimpleNamespace(
        env_args={
            "navigation_config": (
                "submodules/gym-multigrid/gym_multigrid/envs/maps/"
                "team_navigation/navigation_hallway_cm_smoke.yaml"
            )
        }
    )
    successor_reset = _navigation_reset_options(tiny_args, (1, 2), 1.0)
    assert successor_reset["navigation_init_state_dist"]["states"] == [
        [[1, 1], [1, 3], [1, 5]]
    ]


def test_mock_unified_seed_runner_uses_runner_calibration_and_writes_frontier(
    tmp_path,
):
    class _MockRunner:
        batch_size = 1
        t_env = 100

        def __init__(self):
            self.test_returns = []
            self.test_stats = {}
            self.env = SimpleNamespace(terminate_on_task_completed=False)

        def get_env_attr(self, name):
            if name == "terminate_on_task_completed":
                return getattr(self.env, name)
            raise AttributeError(name)

        def set_env_attr(self, name, value):
            setattr(self.env, name, value)

        def run(self, test_mode, return_log_stats, reset_options):
            edge = tuple(reset_options["hl_task"])
            capacity = float(reset_options["msg_budget_per_agent"])
            success = 0.75 if capacity == 1.0 else 0.55
            success = success + edge[1] * 0.01
            task_completed = success >= 0.70
            final_info = {
                "task_completed": task_completed,
                "final_state": ((edge[1], 0),) if task_completed else None,
            }
            return {
                "final_info": final_info,
                "final_infos": [final_info],
            }

    simulation = object.__new__(Simulation)
    simulation.args = SimpleNamespace(
        env="mock-hierarchical-v0",
        env_args={"mock_low_level": True},
        msg_budget_per_agent=[0.0, 1.0],
        test_nepisode=2,
        seed=101,
    )
    simulation.runner = _MockRunner()
    simulation.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
    )
    config = UnifiedHierarchicalConfig(
        task_sequence=((0, 1), (0, 2), (1, 3), (2, 3), (3, 4)),
        pg_values=(0.60, 0.65),
        epsilon_est=0.05,
        beta=0.95,
        epsilon_p=0.20,
        n_max=4,
        n_calibration_episodes=1200,
        held_out_episodes=2,
        max_iterations=2,
        resume=HierarchicalResumeMode.NONE,
    )

    simulation._run_mock_unified_hierarchical(
        config,
        str(tmp_path),
    )

    assert (tmp_path / "mock_unified_summary.json").is_file()
    assert (tmp_path / "pg_frontier.csv").is_file()
    assert (tmp_path / "option_calibration" / "seed_101" / "calibration.json").is_file()
    assert (tmp_path / "option_library" / "seed_101" / "manifest.json").is_file()


def test_collect_statewise_calibration_uses_parallel_batch_without_replays():
    class _ParallelRunner:
        batch_size = 10
        t_env = 500
        test_returns = []
        test_stats = {}
        mac = SimpleNamespace(msg_budget_per_agent=-1)
        env = SimpleNamespace(terminate_on_task_completed=False)

        def __init__(self):
            self.calls = 0

        def run(self, test_mode, return_log_stats, reset_options):
            del test_mode, return_log_stats
            self.calls += 1
            return {
                "final_infos": [
                    {
                        "task_completed": True,
                        "final_state": ((1, 0),),
                    }
                    for _ in range(10)
                ]
            }

    from src.simulation.evaluate import collect_statewise_calibration_row
    runner = _ParallelRunner()
    row = collect_statewise_calibration_row(
        runner,
        setting=OptionSettingKey(
            subtask_edge=(0, 1),
            communication_capacity=1.0,
            start_state=(0, 0),
        ),
        seed=101,
        n_eval_eps=100,
        record_replays=False,
    )

    assert runner.calls == 10
    assert len(row.outcomes) == 100


def test_real_unified_seed_runner_trains_freezes_calibrates_and_synthesizes(
    tmp_path,
    monkeypatch,
):
    class _WeightsMac:
        def __init__(self):
            self.comms_agent = SimpleNamespace(policy=None)
            self.agent = SimpleNamespace(
                state_dict=lambda: {"agent.weight": object()}
            )
            self.msg_budget_per_agent = 1.0

    class _Runner:
        t_env = 100
        batch_size = 1
        mac = _WeightsMac()

        def __init__(self):
            transitions = pd.DataFrame.from_records(
                [
                    {
                        "state": 0,
                        "state_type": "normal",
                        "action": (1, 0.0),
                        "next_state": 1,
                        "prob": 0.75,
                    },
                    {
                        "state": 0,
                        "state_type": "normal",
                        "action": (1, 0.0),
                        "next_state": -1,
                        "prob": 0.25,
                    },
                    {
                        "state": 0,
                        "state_type": "normal",
                        "action": (1, 1.0),
                        "next_state": 1,
                        "prob": 0.95,
                    },
                    {
                        "state": 0,
                        "state_type": "normal",
                        "action": (1, 1.0),
                        "next_state": -1,
                        "prob": 0.05,
                    },
                    {
                        "state": 1,
                        "state_type": "goal",
                        "action": (1, 0.0),
                        "next_state": 1,
                        "prob": 1.0,
                    },
                ]
            )
            hlmdp = SimpleNamespace(
                transition_probs=transitions,
                fail_state=-1,
                init_state=0,
                goal_states={1},
            )
            hlmdp._transition_probs = transitions
            self.env = SimpleNamespace(hlmdp=hlmdp)

        def run(
            self,
            test_mode,
            return_log_stats=False,
            reset_options=None,
        ):
            if not test_mode:
                self.t_env += 100
            edge = tuple(reset_options["hl_task"])
            final_info = {
                "task_completed": True,
                "final_state": ((edge[1], 0),),
            }
            return {
                "batch": SimpleNamespace(
                    batch_size=1,
                    max_t_filled=lambda: 1,
                ),
                "final_info": final_info,
                "final_infos": [final_info],
            }

        def close_env(self):
            return None

    simulation = object.__new__(Simulation)
    simulation.args = SimpleNamespace(
        env="team-nav-v0",
        env_args={
            "mock_low_level": False,
            "navigation_config": (
                "submodules/gym-multigrid/gym_multigrid/envs/maps/"
                "team_navigation/navigation_branching_small_hall.yaml"
            ),
        },
        n_agents=3,
        msg_budget_per_agent=[0.0, 1.0],
        test_nepisode=1200,
        t_max=1200,
        seed=101,
        local_results_path=str(tmp_path),
    )
    simulation.runner = _Runner()
    simulation.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
    )

    class _Buffer:
        def insert_episode_batch(self, batch):
            return None

        def can_sample(self, batch_size):
            return False

    class _Learner:
        def train(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        "src.simulation.unified_hierarchical_runner.build_sim",
        lambda args, logger: (
            args,
            _Runner(),
            _Buffer(),
            _Learner(),
        ),
    )

    candidate_calls = []

    def train_candidate(
        reset_options,
        budget,
        policy_seed,
        n_eval_eps,
        setting,
        include_calibration_row,
        option_snapshot_callback=None,
    ):
        del include_calibration_row
        edge = tuple(reset_options["hl_task"])
        candidate_calls.append((edge, budget, policy_seed, n_eval_eps))
        from src.simulation.hierarchical_types import (
            CalibrationOutcome,
            CalibrationRow,
        )
        setting_key = setting
        success_count = round(
            (0.95 if budget == 1.0 else 0.75) * n_eval_eps
        )
        outcomes = tuple(
            CalibrationOutcome.success((edge[1], 0))
            if index < success_count
            else CalibrationOutcome.failure()
            for index in range(n_eval_eps)
        )
        row = CalibrationRow(setting=setting_key, outcomes=outcomes, seed=policy_seed)
        if option_snapshot_callback is not None:
            option_snapshot_callback(50, row)
            option_snapshot_callback(100, row)
        return (
            {"states": [[edge[1], 0]], "probs": [1.0]},
            row.success_rate,
            {"agent.weight": object()},
            row,
        )

    simulation._train_dependent_candidate = train_candidate
    simulation._evaluate_dependent_cm_policy = lambda *args, **kwargs: None

    def fake_solve_policy(args, hlmdp, mixed_success_rates, threshold):
        del args, hlmdp, threshold
        task_rows = [
            {
                "state": edge[0],
                "next_state": edge[1],
                "occupancy": 1.0,
                "action_probability": 1.0,
            }
            for edge in ((0, 1),)
        ]
        capacity_rows = [
            {
                "state": 0,
                "next_state": 1,
                "msg_budget_per_agent": capacity,
                "action_probability": (1.0 if capacity == 1.0 else 0.0),
            }
            for capacity in (0.0, 1.0)
        ]
        return SimpleNamespace(
            task_policy=pd.DataFrame.from_records(task_rows),
            comms_budget_policy=pd.DataFrame.from_records(capacity_rows),
            objective_value=0.5,
            goal_reach_probability=0.95,
        )

    monkeypatch.setattr(
        "src.simulation.unified_hierarchical_runner._solve_mixed_project_policy",
        fake_solve_policy,
    )

    config = UnifiedHierarchicalConfig(
        task_sequence=((0, 1),),
        pg_values=(0.60,),
        epsilon_est=0.05,
        beta=0.95,
        epsilon_p=0.20,
        n_max=4,
        n_calibration_episodes=1200,
        max_iterations=2,
        resume=HierarchicalResumeMode.NONE,
    )

    runner = UnifiedHierarchicalSeedRunner(simulation)
    summary = runner.run_real_seed(config, str(tmp_path))

    assert candidate_calls == []
    assert summary["status"] == "real_seed_runner_completed"
    assert (tmp_path / "unified_seed_summary.json").is_file()
    assert (tmp_path / "option_calibration" / "seed_101" / "calibration.json").is_file()
    assert (tmp_path / "seed_101" / "pg_frontier.csv").is_file()
    assert (tmp_path / "option_library" / "seed_101").is_dir()
    assert (tmp_path / "seed_101" / "run_phase.json").is_file()
    assert (tmp_path / "seed_101" / "held_out_results.csv").is_file()
    assert (tmp_path / "real_training_eval_seed_101.csv").is_file()
    assert (tmp_path / "seed_101" / "pg_frontier.csv").is_file()
    assert (
        tmp_path
        / "seed_101"
        / "context_unified"
        / "pg_0.60"
        / "candidate_manifest.json"
    ).is_file()
