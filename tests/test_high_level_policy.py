from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import wandb
from gym_multigrid.envs.mdp import ProjectMDP

from src.envs.basic_gymnasium_wrappers import HLMDPEnvWrapper
from src.modules.agents.ilp_model import ILPModel, InfeasiblePolicyError
from src.simulation.context_utils import aggregate_eval_context_rows
from src.simulation.run import Simulation


def _evaluation(
    edge: tuple[int, int],
    budget: int,
    success_rate: float,
    *,
    run_id: str = "run-1",
    t_env: int = 100,
) -> dict:
    return {
        "hl_start_state": edge[0],
        "hl_task": list(edge),
        "msg_budget_per_agent": budget,
        "test_task_completed_mean": success_rate,
        "test_n_episodes": 20,
        "t_env": t_env,
        "run_id": run_id,
    }


def _chain_mdp() -> ProjectMDP:
    return ProjectMDP(
        msg_budget_per_agent=[0, 1],
        states=[0, 1, 2],
        transitions=[(0, 1), (1, 2)],
        initial_state=0,
        goal_states=[2],
        add_post_goal_state=False,
    )


def test_transition_probabilities_use_latest_pooled_success_rates():
    mdp = _chain_mdp()
    rows = [
        _evaluation((0, 1), 0, 0.1, t_env=50),
        _evaluation((0, 1), 0, 0.4, t_env=100),
        _evaluation((0, 1), 0, 0.6, run_id="run-2", t_env=80),
        _evaluation((0, 1), 1, 0.9),
        _evaluation((1, 2), 0, 0.7),
        _evaluation((1, 2), 1, 0.95),
    ]

    mdp.transition_probs = pd.DataFrame.from_records(rows)
    transitions = mdp.transition_probs
    selected = transitions.loc[
        transitions.apply(
            lambda row: (
                row.state == 0
                and row.action == (1, 0)
                and row.next_state in (1, mdp.fail_state)
            ),
            axis=1,
        )
    ].set_index("next_state")["prob"]

    assert np.isclose(selected[1], 0.5)
    assert np.isclose(selected[mdp.fail_state], 0.5)


def test_ilp_optimizes_capacity_and_exposes_action_policy():
    mdp = _chain_mdp()
    evaluations = pd.DataFrame.from_records([
        _evaluation((0, 1), 0, 0.8),
        _evaluation((0, 1), 1, 0.95),
        _evaluation((1, 2), 0, 0.8),
        _evaluation((1, 2), 1, 0.95),
    ])
    mdp.transition_probs = evaluations
    optimizer = ILPModel(SimpleNamespace())

    try:
        solution = optimizer.optimize_policy(mdp, success_rate_spec=0.9)

        assert solution.goal_reach_probability >= 0.9
        assert np.isclose(solution.goal_reach_probability, 0.95**2)
        assert np.isclose(solution.objective_value, (1.0 + 0.95) / 2.0)
        selected_budgets = solution.comms_budget_policy.loc[
            solution.comms_budget_policy.action_probability > 0.5
        ]
        assert set(zip(selected_budgets.state, selected_budgets.next_state)) == {
            (0, 1),
            (1, 2),
        }
        assert set(selected_budgets.msg_budget_per_agent) == {1.0}

        class StateBatch:
            def __getitem__(self, key):
                assert key == "hl_state"
                return torch.tensor([[[0]]])

        action = optimizer.select_actions(StateBatch(), t_ep=0, t_env=0, test_mode=True)
        assert action == {"chosen_next_state": 1, "comms_budget": 1.0}
    finally:
        optimizer.dispose()


def test_ilp_extracts_stochastic_task_policy_for_reliability_tradeoff():
    mdp = ProjectMDP(
        msg_budget_per_agent=[0, 1],
        states=[0, 1, 2, 3],
        transitions=[(0, 1), (0, 2), (1, 3), (2, 3)],
        initial_state=0,
        goal_states=[3],
        add_post_goal_state=False,
    )
    probabilities = {
        ((0, 1), 0): 0.5,
        ((0, 1), 1): 0.9,
        ((0, 2), 0): 0.1,
        ((0, 2), 1): 1.0,
        ((1, 3), 0): 1.0,
        ((1, 3), 1): 1.0,
        ((2, 3), 0): 1.0,
        ((2, 3), 1): 1.0,
    }
    mdp.transition_probs = pd.DataFrame.from_records([
        _evaluation(edge, budget, success_rate)
        for (edge, budget), success_rate in probabilities.items()
    ])
    optimizer = ILPModel(SimpleNamespace())

    try:
        solution = optimizer.optimize_policy(mdp, success_rate_spec=0.9)
        root_policy = solution.task_policy.loc[
            solution.task_policy.state == 0
        ].set_index("next_state")["action_probability"]

        assert np.isclose(solution.goal_reach_probability, 0.9)
        assert np.isclose(root_policy[1], 0.2)
        assert np.isclose(root_policy[2], 0.8)
        assert np.isclose(solution.objective_value, 0.4)
    finally:
        optimizer.dispose()


def test_ilp_can_solve_with_one_fixed_communication_budget():
    mdp = _chain_mdp()
    mdp.transition_probs = pd.DataFrame.from_records([
        _evaluation((0, 1), 0, 0.8),
        _evaluation((0, 1), 1, 0.95),
        _evaluation((1, 2), 0, 0.8),
        _evaluation((1, 2), 1, 0.95),
    ])

    infeasible_optimizer = ILPModel(SimpleNamespace())
    try:
        try:
            infeasible_optimizer.optimize_policy(
                mdp, success_rate_spec=0.9, fixed_comms_budget=0
            )
        except InfeasiblePolicyError:
            pass
        else:
            raise AssertionError("Budget zero should not meet a 0.9 success spec.")
    finally:
        infeasible_optimizer.dispose()

    optimizer = ILPModel(SimpleNamespace())
    try:
        solution = optimizer.optimize_policy(
            mdp, success_rate_spec=0.9, fixed_comms_budget=1
        )

        assert solution.goal_reach_probability >= 0.9
        assert np.isclose(solution.objective_value, 0.975)
        selected_budgets = solution.comms_budget_policy.loc[
            solution.comms_budget_policy.action_probability > 0.5
        ]
        assert set(selected_budgets.msg_budget_per_agent) == {1.0}
    finally:
        optimizer.dispose()


def test_aggregate_high_level_policy_sweep_writes_cost_curves(tmp_path):
    mdp = _chain_mdp()
    eval_data = pd.DataFrame.from_records([
        _evaluation((0, 1), 0, 0.8),
        _evaluation((0, 1), 1, 0.95),
        _evaluation((1, 2), 0, 0.8),
        _evaluation((1, 2), 1, 0.95),
    ])
    simulation = object.__new__(Simulation)
    simulation.args = SimpleNamespace(
        postprocess_success_rate_specs=[0.9],
        success_rate_spec=0.9,
        time_id="test",
        unique_token="aggregate",
        local_results_path=str(tmp_path),
    )
    simulation.runner = SimpleNamespace(env=SimpleNamespace(hlmdp=mdp))
    simulation.logger = SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        step_metric="t_env",
        log_suffix="",
    )

    simulation._make_high_level_success_vs_cost_plot(
        eval_data, map_name="test_map", t=100
    )

    output_dir = (
        tmp_path / "high_level_policy_analysis" / "test_postprocess" / "test_map"
    )
    ours = pd.read_csv(output_dir / "success_vs_cost_our_formulation.csv")
    baseline = pd.read_csv(output_dir / "success_vs_cost_baseline_formulation.csv")
    assert ours.loc[0, "feasible"]
    assert baseline.loc[0, "feasible"]
    assert np.isclose(ours.loc[0, "total_comms_cost"], 0.975)
    assert np.isclose(baseline.loc[0, "fixed_comms_budget"], 1.0)
    assert (output_dir / "success_vs_cost.png").is_file()
    assert (output_dir / "success_vs_cost.pdf").is_file()
    assert (output_dir / "policy" / "task_policy.csv").is_file()
    assert (output_dir / "policy" / "comms_budget_policy.csv").is_file()


def test_training_evaluation_metrics_are_logged_per_context():
    simulation = object.__new__(Simulation)
    logged_stats = []
    simulation.args = SimpleNamespace()
    simulation.runner = SimpleNamespace(t_env=500_000)
    simulation.logger = SimpleNamespace(
        log_stat=lambda *args: logged_stats.append(args)
    )

    simulation._log_context_evaluation_stats([
        {
            "hl_task": [0, 1],
            "msg_budget_per_agent": 0,
            "test_task_completed_mean": 0.4,
            "test_n_episodes": 50,
            "successful_terminal_state_distribution": {"states": [], "probs": []},
        },
        {
            "hl_task": [1, 3],
            "msg_budget_per_agent": 0,
            "test_task_completed_mean": 0.8,
            "test_n_episodes": 50,
        },
    ])

    assert logged_stats == [
        ("test_task_completed_mean/hl_task_0_1/msg_budget_per_agent_0", 0.4, 500_000),
        ("test_n_episodes/hl_task_0_1/msg_budget_per_agent_0", 50.0, 500_000),
        ("test_task_completed_mean/hl_task_1_3/msg_budget_per_agent_0", 0.8, 500_000),
        ("test_n_episodes/hl_task_1_3/msg_budget_per_agent_0", 50.0, 500_000),
    ]


def test_training_evaluation_sweeps_and_logs_every_context(monkeypatch):
    simulation = object.__new__(Simulation)
    logged_stats = []
    tables = {}
    simulation.args = SimpleNamespace(
        condition_on_msg_budget=True,
        condition_on_subtask=True,
        msg_budget_per_agent=[0, 1, 2],
        subtask_contexts=[[0, 1], [1, 3]],
    )
    simulation.runner = SimpleNamespace(
        batch_size=10,
        t_env=500_000,
        get_env_rng_states=lambda: [],
        set_env_rng_states=lambda _states: None,
    )
    simulation.logger = SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        log_stat=lambda *args: logged_stats.append(args),
        log_table=lambda key, value, t: tables.__setitem__(key, value),
        data_tables=tables,
    )
    simulation._make_comms_eval_plots = lambda *_args, **_kwargs: None
    evaluated_contexts = []

    def fake_run_eval_episodes(**kwargs):
        reset_options = kwargs["reset_options"]
        evaluated_contexts.append(reset_options)
        return {
            "log_stats": {
                **reset_options,
                "test_task_completed_mean": 0.5,
                "test_n_episodes": 50,
                "t_env": 500_000,
            }
        }

    monkeypatch.setattr(
        "src.simulation.run.run_eval_episodes", fake_run_eval_episodes
    )

    simulation.evaluate(n_eval_eps=5)

    assert len(evaluated_contexts) == 6
    assert {
        (tuple(context["hl_task"]), context["msg_budget_per_agent"])
        for context in evaluated_contexts
    } == {
        ((0, 1), 0),
        ((0, 1), 1),
        ((0, 1), 2),
        ((1, 3), 0),
        ((1, 3), 1),
        ((1, 3), 2),
    }
    assert sum(
        key.startswith("test_task_completed_mean/") for key, _, _ in logged_stats
    ) == 6


def test_postprocessing_aggregates_samples_within_each_context_and_seed():
    rows = pd.DataFrame.from_records([
        {
            "hl_task": (0, 1),
            "msg_budget_per_agent": 0,
            "t_env_rounded": 500_000,
            "test_task_completed_mean": 0.4,
            "test_task_completed_std": 0.49,
            "test_n_episodes": 50,
        },
        {
            "hl_task": (0, 1),
            "msg_budget_per_agent": 0,
            "t_env_rounded": 500_000,
            "test_task_completed_mean": 0.8,
            "test_task_completed_std": 0.4,
            "test_n_episodes": 50,
        },
        {
            "hl_task": (1, 3),
            "msg_budget_per_agent": 0,
            "t_env_rounded": 500_000,
            "test_task_completed_mean": 0.6,
            "test_task_completed_std": 0.49,
            "test_n_episodes": 50,
        },
    ])

    aggregated = aggregate_eval_context_rows(
        rows,
        ["t_env_rounded", "hl_task", "msg_budget_per_agent"],
    )

    task_01 = aggregated.loc[aggregated["hl_task"] == (0, 1)].iloc[0]
    task_13 = aggregated.loc[aggregated["hl_task"] == (1, 3)].iloc[0]
    assert task_01["test_n_episodes"] == 100
    assert np.isclose(task_01["test_task_completed_mean"], 0.6)
    assert task_13["test_n_episodes"] == 50
    assert np.isclose(task_13["test_task_completed_mean"], 0.6)


def test_independent_high_level_policies_are_synthesized_per_seed(tmp_path):
    simulation = object.__new__(Simulation)
    simulation.args = SimpleNamespace(
        postprocess_success_rate_specs=[0.9],
        success_rate_spec=0.9,
        time_id="per_seed",
        unique_token="aggregate",
        local_results_path=str(tmp_path),
    )
    simulation.runner = SimpleNamespace(env=SimpleNamespace(hlmdp=_chain_mdp()))
    simulation.logger = SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        step_metric="t_env",
        log_suffix="",
    )

    seed_evaluations = {
        11: [
            _evaluation(edge, budget, 0.5 if budget == 0 else 0.95)
            for edge in [(0, 1), (1, 2)]
            for budget in [0, 1]
        ],
        12: [
            _evaluation(edge, budget, 0.95 if budget == 0 else 0.99)
            for edge in [(0, 1), (1, 2)]
            for budget in [0, 1]
        ],
    }

    for seed, rows in seed_evaluations.items():
        simulation._make_high_level_success_vs_cost_plot(
            pd.DataFrame.from_records(rows),
            map_name="test_map",
            t=100,
            seed=seed,
        )

    output_root = tmp_path / "high_level_policy_analysis" / "per_seed_postprocess"
    seed_11_policy = pd.read_csv(
        output_root / "test_map" / "seed_11" / "policy" / "comms_budget_policy.csv"
    )
    seed_12_policy = pd.read_csv(
        output_root / "test_map" / "seed_12" / "policy" / "comms_budget_policy.csv"
    )
    selected_11 = seed_11_policy.loc[
        seed_11_policy.action_probability > 0.5, "msg_budget_per_agent"
    ]
    selected_12 = seed_12_policy.loc[
        seed_12_policy.action_probability > 0.5, "msg_budget_per_agent"
    ]

    assert set(selected_11) == {1.0}
    assert set(selected_12) == {0.0}


def test_loaded_evaluation_artifact_keeps_source_seed(monkeypatch):
    run = SimpleNamespace(
        id="seeded-run",
        entity="entity",
        project="project",
        config={
            "scenario": "01",
            "seed": "17",
            "test_interval": 100,
            "hl_task": [0, 1],
        },
    )
    table = pd.DataFrame.from_records([
        {
            "t_env": 100,
            "test_return_mean": 0.5,
            "hl_task": [1, 3],
            "hl_start_state": 1,
        }
    ])
    artifact = SimpleNamespace(
        get=lambda _name: SimpleNamespace(get_dataframe=lambda: table.copy())
    )
    api = SimpleNamespace(
        runs=lambda *_args, **_kwargs: [run],
        artifact=lambda **_kwargs: artifact,
    )
    monkeypatch.setattr("src.simulation.run.wandb.Api", lambda: api)

    simulation = object.__new__(Simulation)
    simulation.args = SimpleNamespace(
        wandb_project="entity/project",
        time_id="test-time",
        test_interval=100,
    )
    simulation.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)

    loaded, loaded_runs = simulation._load_experiment_table_artifact(return_runs=True)

    assert loaded["seed"].tolist() == [17]
    assert loaded["hl_task"].tolist() == [(1, 3)]
    assert loaded["hl_start_state"].tolist() == [1]
    assert loaded_runs == [run]


def test_loaded_evaluation_table_falls_back_to_wandb_media_file(
    monkeypatch, tmp_path
):
    table_file = tmp_path / "eval_stats.table.json"
    table_file.write_text(
        '{"columns":["t_env","test_return_mean","msg_budget_per_agent"],"data":[[100,0.5,0]]}',
        encoding="utf-8",
    )
    run = SimpleNamespace(
        id="table-run",
        entity="entity",
        project="project",
        config={
            "scenario": "01",
            "seed": 17,
            "test_interval": 100,
            "msg_budget_per_agent": [0, 1],
        },
        summary={
            "eval_stats": {
                "_type": "table-file",
                "path": "media/table/eval_stats.table.json",
            }
        },
        file=lambda _path: SimpleNamespace(
            download=lambda **_kwargs: SimpleNamespace(name=str(table_file))
        ),
    )

    def missing_artifact(**_kwargs):
        raise wandb.CommError("run-table artifact not found")

    api = SimpleNamespace(
        runs=lambda *_args, **_kwargs: [run],
        artifact=missing_artifact,
    )
    monkeypatch.setattr("src.simulation.run.wandb.Api", lambda: api)

    simulation = object.__new__(Simulation)
    simulation.args = SimpleNamespace(
        wandb_project="entity/project",
        time_id="test-time",
        test_interval=100,
    )
    simulation.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)

    loaded = simulation._load_experiment_table_artifact()

    assert loaded[
        ["t_env", "test_return_mean", "msg_budget_per_agent", "seed"]
    ].to_dict("records") == [
        {
            "t_env": 100,
            "test_return_mean": 0.5,
            "msg_budget_per_agent": 0,
            "seed": 17,
        }
    ]


def test_high_level_success_cost_plot_skips_incomplete_transition_data():
    class IncompleteMdp:
        @property
        def transition_probs(self):
            return None

        @transition_probs.setter
        def transition_probs(self, _value):
            raise ValueError("missing task/budget pairs")

    messages = []
    simulation = object.__new__(Simulation)
    simulation.runner = SimpleNamespace(env=SimpleNamespace(hlmdp=IncompleteMdp()))
    simulation.logger = SimpleNamespace(info=messages.append)

    simulation._make_high_level_success_vs_cost_plot(
        pd.DataFrame(), map_name="test-map", t=100, seed=17
    )

    assert len(messages) == 1
    assert "evaluation data is incomplete" in messages[0]
    assert "missing task/budget pairs" in messages[0]


def test_wandb_project_name_uses_authenticated_default_entity():
    api = SimpleNamespace(default_entity="example-entity")

    assert Simulation._wandb_project_path(api, "comm_experiments") == (
        "example-entity/comm_experiments"
    )
    assert Simulation._wandb_project_path(api, "other-entity/project") == (
        "other-entity/project"
    )


def test_aggregate_postprocessing_keeps_seeds_separate_and_skips_incomplete(
    monkeypatch,
):
    edges = [(0, 1), (1, 2)]
    runs = []
    rows = []
    for seed in [11, 12]:
        for scenario, edge in enumerate(edges, start=1):
            config = {
                "scenario": str(scenario),
                "seed": seed,
                "hl_task": edge,
                "env_args.map_name": "test_map",
                "msg_budget_per_agent": [0, 1],
            }
            runs.append(
                SimpleNamespace(
                    id=f"seed-{seed}-scenario-{scenario}",
                    config=config,
                    entity="entity",
                    project="project",
                )
            )
            if seed == 11 and edge == (1, 2):
                continue
            for eval_edge in edges:
                for budget in [0, 1]:
                    rows.append({
                        "scenario": scenario,
                        "seed": seed,
                        "map_name": "test_map",
                        "hl_task": eval_edge,
                        "msg_budget_per_agent": budget,
                        "t_env": 100,
                        "t_env_rounded": 100,
                        "test_n_episodes": 50,
                        "test_return_mean": 1.0,
                        "test_return_std": 0.0,
                        "test_task_completed_mean": 0.95,
                        "test_ep_length_mean": 5.0,
                    })
        runs.append(
            SimpleNamespace(
                id=f"seed-{seed}-flat-scenario",
                config={
                    "scenario": "03",
                    "seed": seed,
                    "env_args.map_name": "test_map",
                    "msg_budget_per_agent": [0, 1],
                },
                entity="entity",
                project="project",
            )
        )
        rows.append({
            "scenario": 3,
            "seed": seed,
            "map_name": "test_map",
            "msg_budget_per_agent": 0,
            "t_env": 100,
            "t_env_rounded": 100,
            "test_n_episodes": 50,
            "test_return_mean": float(seed == 12),
            "test_return_std": 0.0,
            "test_task_completed_mean": 0.0,
            "test_ep_length_mean": 5.0,
        })

    simulation = object.__new__(Simulation)
    simulation.args = SimpleNamespace(
        evaluate=False,
        post_processing="aggregate_runs",
        time_id="aggregate-test",
        wandb_project="entity/project",
        experiment="test",
    )
    log_messages = []
    simulation.logger = SimpleNamespace(info=log_messages.append)
    simulation._load_experiment_table_artifact = lambda **_kwargs: (
        pd.DataFrame.from_records(rows),
        runs,
    )
    policy_inputs = []
    aggregate_plot_inputs = []
    combined_plot_inputs = []
    simulation._make_comms_eval_plots = lambda table, **_kwargs: (
        aggregate_plot_inputs.append(table)
    )
    simulation._make_combined_comms_eval_plots = lambda groups, **kwargs: (
        combined_plot_inputs.append((groups, kwargs))
    )
    simulation._make_task_success_frame_plot = lambda *_args, **_kwargs: None
    simulation._make_high_level_success_vs_cost_plot = lambda eval_data, **kwargs: (
        policy_inputs.append((kwargs["seed"], set(eval_data["seed"])))
    )
    postprocess_run = SimpleNamespace(finish=lambda: None)
    monkeypatch.setattr(
        "src.simulation.run.wandb.Api",
        lambda: SimpleNamespace(runs=lambda *_args, **_kwargs: []),
    )
    monkeypatch.setattr(
        "src.simulation.run.wandb.init", lambda **_kwargs: postprocess_run
    )
    monkeypatch.setattr("src.simulation.run.wandb.Table", lambda dataframe: dataframe)

    simulation.run()

    assert policy_inputs == [(12, {12})]
    assert any(
        "seed 11" in message and "missing results" in message
        for message in log_messages
    )
    assert any(
        set(frame["test_return_mean"]) == {0.5} for frame in aggregate_plot_inputs
    )
    assert len(combined_plot_inputs) == 1
    task_groups, plot_options = combined_plot_inputs[0]
    assert {label for label, _, _ in task_groups} == {"task 0-1", "task 1-2"}
    assert all(
        set(frame["test_n_episodes"]) == {100}
        for _, frame, _ in task_groups
    )
    assert plot_options["condition_label"] == "model_config_02"


def test_navigation_graph_uses_declared_goal_without_synthetic_task():
    navigation_config = (
        Path(__file__).parents[1]
        / "submodules"
        / "gym-multigrid"
        / "gym_multigrid"
        / "envs"
        / "maps"
        / "team_navigation"
        / "navigation_branching_small_hall.yaml"
    )
    env = HLMDPEnvWrapper({
        "key": "team-nav-v0",
        "seed": 1,
        "map_name": "3ga_1r_small_hall",
        "navigation_config": str(navigation_config),
        "n_agents": 3,
        "max_episode_steps": 15,
        "msg_budget_per_agent": [0, 1],
    })

    try:
        assert env.hlmdp.goal_states == {4}
        assert (4, 5) not in env.hlmdp.tasks
    finally:
        env.close()


def test_navigation_task_with_asynchronous_arrival_transitions_to_fail():
    navigation_config = (
        Path(__file__).parents[1]
        / "submodules"
        / "gym-multigrid"
        / "gym_multigrid"
        / "envs"
        / "maps"
        / "team_navigation"
        / "navigation_branching_small_hall.yaml"
    )
    env = HLMDPEnvWrapper({
        "key": "team-nav-v0",
        "seed": 1,
        "map_name": "3ga_1r_small_hall",
        "navigation_config": str(navigation_config),
        "n_agents": 3,
        "max_episode_steps": 15,
        "msg_budget_per_agent": [0, 1],
    })

    try:
        env.reset(options={"hl_task": [0, 1]})
        actions = env.get_wrapper_attr("actions")
        hl_actions = {"chosen_next_state": 1, "comms_budget": 0}

        for _ in range(3):
            _, _, terminated, _, _ = env.step({
                "hl_actions": hl_actions,
                "env_actions": [actions.RIGHT, actions.STAY, actions.STAY],
            })
            assert not terminated

        for _ in range(3):
            _, _, terminated, _, info = env.step({
                "hl_actions": hl_actions,
                "env_actions": [actions.STAY, actions.RIGHT, actions.RIGHT],
            })

        assert terminated
        assert not info["task_completed"]
        assert info["project_failed"]
        assert env.hlmdp.agent.state == env.hlmdp.fail_state

        env.reset(options={"hl_task": [0, 1]})
        for _ in range(3):
            _, _, terminated, _, info = env.step({
                "hl_actions": hl_actions,
                "env_actions": [actions.RIGHT, actions.RIGHT, actions.RIGHT],
            })

        assert terminated
        assert info["task_completed"]
        assert not info["project_failed"]
        assert env.hlmdp.agent.state == 1
    finally:
        env.close()
