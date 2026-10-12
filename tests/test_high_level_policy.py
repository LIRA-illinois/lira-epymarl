from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from gym_multigrid.envs.mdp import ProjectMDP

from src.envs.basic_gymnasium_wrappers import HLMDPEnvWrapper
from src.modules.agents.ilp_model import ILPModel, InfeasiblePolicyError
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
            lambda row: row.state == 0
            and row.action == (1, 0)
            and row.next_state in (1, mdp.fail_state),
            axis=1,
        )
    ].set_index("next_state")["prob"]

    assert np.isclose(selected[1], 0.5)
    assert np.isclose(selected[mdp.fail_state], 0.5)


def test_ilp_optimizes_capacity_and_exposes_action_policy():
    mdp = _chain_mdp()
    evaluations = pd.DataFrame.from_records(
        [
            _evaluation((0, 1), 0, 0.8),
            _evaluation((0, 1), 1, 0.95),
            _evaluation((1, 2), 0, 0.8),
            _evaluation((1, 2), 1, 0.95),
        ]
    )
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
        assert set(
            zip(selected_budgets.state, selected_budgets.next_state)
        ) == {(0, 1), (1, 2)}
        assert set(selected_budgets.msg_budget_per_agent) == {1.0}

        class StateBatch:
            def __getitem__(self, key):
                assert key == "hl_state"
                return torch.tensor([[[0]]])

        action = optimizer.select_actions(
            StateBatch(), t_ep=0, t_env=0, test_mode=True
        )
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
    mdp.transition_probs = pd.DataFrame.from_records(
        [
            _evaluation(edge, budget, success_rate)
            for (edge, budget), success_rate in probabilities.items()
        ]
    )
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
    mdp.transition_probs = pd.DataFrame.from_records(
        [
            _evaluation((0, 1), 0, 0.8),
            _evaluation((0, 1), 1, 0.95),
            _evaluation((1, 2), 0, 0.8),
            _evaluation((1, 2), 1, 0.95),
        ]
    )

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
    eval_data = pd.DataFrame.from_records(
        [
            _evaluation((0, 1), 0, 0.8),
            _evaluation((0, 1), 1, 0.95),
            _evaluation((1, 2), 0, 0.8),
            _evaluation((1, 2), 1, 0.95),
        ]
    )
    simulation = object.__new__(Simulation)
    simulation.args = SimpleNamespace(
        postprocess_success_rate_specs=[0.9],
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
    env = HLMDPEnvWrapper(
        {
            "key": "team-nav-v0",
            "seed": 1,
            "map_name": "3ga_1r_small_hall",
            "navigation_config": str(navigation_config),
            "n_agents": 3,
            "max_episode_steps": 15,
            "msg_budget_per_agent": [0, 1],
        }
    )

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
    env = HLMDPEnvWrapper(
        {
            "key": "team-nav-v0",
            "seed": 1,
            "map_name": "3ga_1r_small_hall",
            "navigation_config": str(navigation_config),
            "n_agents": 3,
            "max_episode_steps": 15,
            "msg_budget_per_agent": [0, 1],
        }
    )

    try:
        env.reset(options={"hl_task": [0, 1]})
        actions = env.get_wrapper_attr("actions")
        hl_actions = {"chosen_next_state": 1, "comms_budget": 0}

        for _ in range(3):
            _, _, terminated, _, _ = env.step(
                {
                    "hl_actions": hl_actions,
                    "env_actions": [actions.RIGHT, actions.STAY, actions.STAY],
                }
            )
            assert not terminated

        for _ in range(3):
            _, _, terminated, _, info = env.step(
                {
                    "hl_actions": hl_actions,
                    "env_actions": [actions.STAY, actions.RIGHT, actions.RIGHT],
                }
            )

        assert terminated
        assert not info["task_completed"]
        assert info["project_failed"]
        assert env.hlmdp.agent.state == env.hlmdp.fail_state

        env.reset(options={"hl_task": [0, 1]})
        for _ in range(3):
            _, _, terminated, _, info = env.step(
                {
                    "hl_actions": hl_actions,
                    "env_actions": [actions.RIGHT, actions.RIGHT, actions.RIGHT],
                }
            )

        assert terminated
        assert info["task_completed"]
        assert not info["project_failed"]
        assert env.hlmdp.agent.state == 1
    finally:
        env.close()
