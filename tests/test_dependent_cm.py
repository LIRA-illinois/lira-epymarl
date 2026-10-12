from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from gym_multigrid.envs.mdp import ProjectMDP
from gym_multigrid.utils.subtasks import PositionDist

from src.controllers.factored.comms_controller import CommsMAC
from src.envs.basic_gymnasium_wrappers import HLMDPEnvWrapper
from src.experiments.grid_search_experiment import GridSearch
from src.modules.agents import ilp_model
from src.simulation.run import Simulation


def test_merge_state_distributions_uses_incoming_success_weights():
    merged = Simulation._merge_state_dists(
        [
            {"states": [[[1, 1]]], "probs": [1.0]},
            {
                "states": [[[1, 1]], [[2, 2]]],
                "probs": [0.25, 0.75],
            },
        ],
        [3.0, 1.0],
    )

    assert merged["states"] == [[(1, 1)], [(2, 2)]]
    assert np.allclose(merged["probs"], [0.8125, 0.1875])
    PositionDist(
        states=tuple(
            tuple(tuple(position) for position in state) for state in merged["states"]
        ),
        probs=tuple(merged["probs"]),
    )


class _Batch:
    batch_size = 1

    def __init__(self, state):
        self.state = state

    def __getitem__(self, key):
        assert key == "hl_state"
        return torch.full((1, 8, 1), self.state)


class _LowLevelMAC:
    def __init__(self):
        self.loaded_policies = []
        self.hidden_state_resets = 0
        self.msg_budget_per_agent = None

    def load_models(self, policy):
        self.loaded_policies.append(policy)

    def init_hidden(self, batch_size):
        assert batch_size == 1
        self.hidden_state_resets += 1

    def select_actions(self, *args):
        return np.zeros((1, 2), dtype=int)


class _HighLevelAgent:
    def __init__(self, actions, task_policy):
        self.actions = iter(actions)
        self.task_policy = task_policy
        self.policy = SimpleNamespace(task_policy=task_policy)
        self.calls = 0

    def select_actions(self, *args, **kwargs):
        self.calls += 1
        return next(self.actions)


def _controller(policy_bank, routes, task_policy):
    controller = CommsMAC.__new__(CommsMAC)
    controller.env_mac = _LowLevelMAC()
    controller.comms_agent = _HighLevelAgent(routes, task_policy)
    controller.n_agents = 2
    controller.args = SimpleNamespace()
    controller._dependent_policy_bank = policy_bank
    controller._dependent_policy_seeds = (7,)
    controller._fixed_dependent_policy_seed = 7
    controller._episode_index = 0
    controller._active_dependent_policy_seed = 7
    controller._active_hl_state = None
    controller._active_hl_actions = None
    controller._active_policy_key = None
    return controller


def test_candidate_policy_switches_on_task_and_budget():
    route_0 = {"chosen_next_state": 1, "comms_budget": 2.0}
    route_1 = {"chosen_next_state": 2, "comms_budget": 1.0}
    policy_0 = {"weights": 0}
    policy_1 = {"weights": 1}
    task_policy = pd.DataFrame({"state": [0, 1], "next_state": [1, 2]})
    controller = _controller(
        {
            (0, 1, 2.0, 7): policy_0,
            (1, 2, 1.0, 7): policy_1,
        },
        [route_0, route_1],
        task_policy,
    )

    controller.select_actions(_Batch(0), 0, 0)
    controller.select_actions(_Batch(0), 1, 0)
    controller.select_actions(_Batch(1), 2, 0)

    assert controller.comms_agent.calls == 2
    assert controller.env_mac.loaded_policies == [policy_0, policy_1]
    assert controller.env_mac.hidden_state_resets == 2


def test_terminal_high_level_state_reuses_completed_route():
    route = {"chosen_next_state": 1, "comms_budget": 2.0}
    policy = {"weights": 0}
    controller = _controller(
        {(0, 1, 2.0, 7): policy},
        [route],
        pd.DataFrame({"state": [0], "next_state": [1]}),
    )

    controller.select_actions(_Batch(0), 0, 0)
    controller.select_actions(_Batch(1), 1, 0)
    controller.select_actions(_Batch(-1), 2, 0)
    controller.select_actions(_Batch(-1), 3, 0)

    assert controller.comms_agent.calls == 1
    assert controller.env_mac.loaded_policies == [policy]


def test_optimizer_receives_budget_specific_success_rates(monkeypatch):
    mdp = ProjectMDP(
        msg_budget_per_agent=[0, 1],
        states=[0, 1, 2],
        transitions=[(0, 1), (1, 2)],
        initial_state=0,
        goal_states=[2],
        add_post_goal_state=False,
    )
    expected_rates = {
        (0, 1, 0.0): 0.4,
        (0, 1, 1.0): 0.9,
        (1, 2, 0.0): 0.5,
        (1, 2, 1.0): 0.95,
    }
    solution = SimpleNamespace(
        task_policy=pd.DataFrame({"state": [0], "next_state": [1]})
    )

    class _Model:
        Status = 2

        def dispose(self):
            pass

    class _ILPModel:
        transitions = None

        def __init__(self, args):
            self.model = _Model()

        def optimize_policy(self, hlmdp, success_rate_spec):
            _ILPModel.transitions = hlmdp.transition_probs.copy()
            return solution

    monkeypatch.setattr(ilp_model, "ILPModel", _ILPModel)
    simulation = Simulation.__new__(Simulation)
    simulation.args = SimpleNamespace(success_rate_spec=0.85)
    simulation.logger = SimpleNamespace(info=lambda message: None)

    assert (
        simulation._optimize_dependent_cm_policy(mdp, [(0, 1), (1, 2)], expected_rates)
        is solution
    )

    for (state, next_state, budget), success_rate in expected_rates.items():
        rows = _ILPModel.transitions.loc[
            (_ILPModel.transitions.state == state)
            & _ILPModel.transitions.action.apply(
                lambda action, target_state=next_state, selected_budget=budget: (
                    action == (target_state, selected_budget)
                )
            )
        ].set_index("next_state")["prob"]
        assert np.isclose(rows[next_state], success_rate)
        assert np.isclose(rows[mdp.fail_state], 1.0 - success_rate)


def test_dependent_grid_scenario_keeps_all_budgets_together():
    experiment = GridSearch.__new__(GridSearch)
    scenarios, _, _ = experiment._get_scenarios(
        {
            "config": {"values": ["maic_comm_opt"]},
            "msg_budget_per_agent": {"values": [[0, 1, 2]]},
            "unique_policy_per_msg_budget": {"values": [True]},
            "hl_task_sequence": {"values": [[[0, 1], [1, 2]]]},
        }
    )

    assert len(scenarios) == 1
    assert scenarios[0]["msg_budget_per_agent"] == [0, 1, 2]


def test_cm_pass_weights_converging_predecessors_by_occupancy_and_budget(
    monkeypatch,
):
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    success_rates = {
        (0, 1, 0.0): 1.0,
        (0, 1, 1.0): 0.5,
        (0, 2, 0.0): 0.5,
        (0, 2, 1.0): 1.0,
        (1, 3, 0.0): 0.8,
        (1, 3, 1.0): 0.7,
        (2, 3, 0.0): 0.2,
        (2, 3, 1.0): 0.4,
        (3, 4, 0.0): 0.9,
        (3, 4, 1.0): 1.0,
    }
    solution = SimpleNamespace(
        task_policy=pd.DataFrame(
            [
                {"state": 0, "next_state": 1, "occupancy": 0.8},
                {"state": 0, "next_state": 2, "occupancy": 0.2},
                {"state": 1, "next_state": 3, "occupancy": 0.6},
                {"state": 2, "next_state": 3, "occupancy": 0.2},
                {"state": 3, "next_state": 4, "occupancy": 0.5},
            ]
        ),
        comms_budget_policy=pd.DataFrame(
            [
                {
                    "state": state,
                    "next_state": next_state,
                    "msg_budget_per_agent": budget,
                    "action_probability": float(budget == selected_budget),
                }
                for (state, next_state), selected_budget in {
                    (0, 1): 0.0,
                    (0, 2): 1.0,
                    (1, 3): 0.0,
                    (2, 3): 1.0,
                    (3, 4): 1.0,
                }.items()
                for budget in (0.0, 1.0)
            ]
        ),
        goal_reach_probability=0.9,
    )

    class _Runner:
        def __init__(self, mac=None, env=None):
            self.mac = mac
            self.env = env
            self.test_returns = []
            self.test_stats = {}
            self.closed = False

        def close_env(self):
            self.closed = True

    final_mac = SimpleNamespace(
        comms_agent=SimpleNamespace(policy=None),
        set_dependent_policy_bank=lambda bank, seeds: setattr(
            final_mac, "policy_bank", bank
        ),
        set_dependent_policy_seed=lambda seed: None,
    )
    final_runner = _Runner(
        mac=final_mac,
        env=SimpleNamespace(terminate_on_task_completed=True),
    )
    initial_runner = _Runner(
        env=SimpleNamespace(hlmdp=object()),
    )
    simulation = Simulation.__new__(Simulation)
    simulation.args = SimpleNamespace(
        factored_hierarchical_policy=True,
        msg_budget_per_agent=[0, 1],
        dependent_policy_seeds=[11, 12],
        n_eval_eps_dependent_subtask=1,
        cm_max_iterations=2,
        cm_convergence_epsilon=0.0,
        save_model=False,
        seed=3,
    )
    simulation.runner = initial_runner
    simulation.buffer = object()
    simulation.learner = object()
    simulation.n_eval_eps = 1
    simulation.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        log_stat=lambda *args, **kwargs: None,
    )
    root_dist = {"states": [[[1, 1]]], "probs": [1.0]}
    simulation._dependent_root_init_state_dist = lambda root, task_edges: root_dist
    simulation._optimize_dependent_cm_policy = (
        lambda hlmdp, task_edges, measured_rates: solution
    )
    simulation._evaluate_dependent_cm_policy = lambda *args: None
    successor_distributions = []

    def train_candidate(reset_options, budget, policy_seed, n_eval_eps):
        edge = tuple(reset_options["hl_task"])
        if edge == (1, 3):
            state = [[10, 1]]
        elif edge == (2, 3):
            state = [[20, 1]]
        else:
            state = [[edge[0], edge[1]]]
        if edge == (3, 4):
            successor_distributions.append(reset_options["navigation_init_state_dist"])
        return (
            {"states": [state], "probs": [1.0]},
            success_rates[(edge[0], edge[1], float(budget))],
            {"seed": torch.tensor([policy_seed])},
        )

    simulation._train_dependent_candidate = train_candidate
    build_calls = 0

    def build_final_sim(args, logger):
        nonlocal build_calls
        build_calls += 1
        return args, final_runner, object(), object()

    import importlib

    dependent_subtasks_module = importlib.import_module(
        "src.simulation.dependent_subtasks"
    )
    monkeypatch.setattr(dependent_subtasks_module, "build_sim", build_final_sim)
    simulation.train_dependent_subtasks(edges)

    assert build_calls == 1
    assert initial_runner.closed and final_runner.closed
    assert len(final_mac.policy_bank) == len(edges) * 2 * 2
    assert len(successor_distributions) == 2 * 2 * 2
    expected = {((10, 1),): 0.48 / 0.56, ((20, 1),): 0.08 / 0.56}
    for distribution in successor_distributions[-4:]:
        observed = {
            tuple(tuple(position) for position in state): probability
            for state, probability in zip(distribution["states"], distribution["probs"])
        }
        assert observed.keys() == expected.keys()
        for state, probability in expected.items():
            assert np.isclose(observed[state], probability)


def test_policy_selected_task_activation_uses_learned_spawn_distribution():
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
    root_dist = {"states": [[[1, 1], [1, 3], [1, 5]]], "probs": [1.0]}
    learned_dist = {"states": [[[2, 1], [2, 3], [2, 5]]], "probs": [1.0]}
    env.set_dependent_navigation_init_state_dists({0: learned_dist})

    try:
        env.reset(
            options={
                "hl_start_state": 0,
                "navigation_init_state_dist": root_dist,
                "select_navigation_task_from_policy": True,
            }
        )
        env.step(
            {
                "hl_actions": {
                    "chosen_next_state": 2,
                    "comms_budget": 0.0,
                },
                "env_actions": np.zeros((1, 3), dtype=int),
            }
        )

        navigation_env = env.unwrapped
        assert navigation_env.current_task == 2
        assert [tuple(agent.pos) for agent in navigation_env.agents] == [
            (2, 1),
            (2, 3),
            (2, 5),
        ]
    finally:
        env.close()
