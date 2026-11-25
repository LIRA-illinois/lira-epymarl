from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import pytest
import torch as th

from src.components.episode_buffer import EpisodeBatch
from src.components.action_selectors import EpsilonGreedyActionSelector
from src.controllers.maic_controller import MAICMAC
from src.utils.policy_context import build_policy_context
from src.utils.logging import MainLogger


def test_epsilon_greedy_test_mode_preserves_training_epsilon():
    args = SimpleNamespace(
        epsilon_start=1.0,
        epsilon_finish=0.05,
        epsilon_anneal_time=100,
        evaluation_epsilon=0.0,
        correlated_exploration=False,
        correlated_exploration_prob=0.5,
    )
    selector = EpsilonGreedyActionSelector(args)
    q_values = th.zeros((2, 3, 4))
    avail_actions = th.ones((2, 3, 4), dtype=th.long)

    selector.select_action(
        q_values,
        avail_actions,
        t_env=50,
        test_mode=False,
    )
    training_epsilon = selector.epsilon
    assert training_epsilon == pytest.approx(0.525)

    selector.select_action(
        q_values,
        avail_actions,
        t_env=50,
        test_mode=True,
    )
    assert selector.epsilon == pytest.approx(training_epsilon)


def test_epsilon_logging_rejects_reset_or_upward_jump():
    logger = MainLogger.__new__(MainLogger)
    logger.step_metric = "t_env"
    logger.stats = defaultdict(list)
    logger.use_wandb = True
    logger.log_suffix = ""
    logger.wandb_current_t = -1
    logger.wandb_current_data = {}
    logged = []
    logger.wandb = SimpleNamespace(
        log=lambda data: logged.append(dict(data))
    )
    logger._last_epsilon = None
    logger._last_epsilon_t_env = None

    logger.log_stat("epsilon", 0.525, 500)
    logger.log_stat("epsilon", 0.050, 1000)
    logger.log_stat("epsilon", 1.0, 1001)
    logger.log_stat("epsilon", 0.525, 192)

    assert [value for _, value in logger.stats["epsilon"]] == [0.525, 0.05]


def test_policy_context_encodes_normalized_budget_and_subtask_one_hot():
    args = SimpleNamespace(
        n_agents=6,
        condition_on_msg_budget=True,
        condition_on_subtask=True,
        msg_budget_per_agent=[0, 2, 4],
        subtask_contexts=[[0, 1], [1, 3]],
        env_args={},
    )

    context = build_policy_context(
        args, {"msg_budget_per_agent": 2, "hl_task": [1, 3]}
    )

    assert context is not None
    np.testing.assert_allclose(context, [0.4, 0.0, 1.0])


def test_policy_context_requires_known_budget_and_subtask():
    args = SimpleNamespace(
        n_agents=3,
        condition_on_msg_budget=True,
        condition_on_subtask=True,
        msg_budget_per_agent=[0, 1],
        subtask_contexts=[[0, 1], [1, 3]],
        env_args={},
    )

    with pytest.raises(ValueError, match="integer from 0 to 2"):
        build_policy_context(args, {"msg_budget_per_agent": 3, "hl_task": [0, 1]})

    with pytest.raises(ValueError, match="not listed in subtask_contexts"):
        build_policy_context(args, {"msg_budget_per_agent": 1, "hl_task": [0, 2]})


def test_maic_controller_appends_replayed_policy_context_for_each_agent():
    args = SimpleNamespace(
        n_agents=2,
        n_actions=3,
        obs_last_action=False,
        obs_agent_id=False,
        condition_on_msg_budget=True,
        condition_on_subtask=True,
        subtask_contexts=[[0, 1], [1, 3]],
        agent="maic",
        agent_output_type="q",
        action_selector="epsilon_greedy",
        epsilon_start=1.0,
        epsilon_finish=0.05,
        epsilon_anneal_time=100,
        hidden_dim=8,
        latent_dim=2,
        nn_hidden_size=8,
        attention_dim=4,
        var_floor=0.002,
        msg_filter_type="hard_topk",
    )
    scheme = {
        "obs": {"vshape": 3, "group": "agents"},
        "actions_onehot": {"vshape": (3,), "group": "agents"},
        "avail_actions": {"vshape": (3,), "group": "agents"},
        "policy_context": {"vshape": (3,)},
    }
    mac = MAICMAC(scheme, {"agents": 2}, args)
    batch = EpisodeBatch(scheme, {"agents": 2}, 2, 1, device="cpu")
    batch.update(
        {
            "obs": np.zeros((2, 2, 3), dtype=np.float32),
            "policy_context": np.asarray(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32
            ),
        },
        ts=0,
    )

    inputs = mac._build_inputs(batch, 0)

    assert inputs.shape == (4, 6)
    th.testing.assert_close(
        inputs[:, -3:],
        th.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                   [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
    )

    th.testing.assert_close(mac._get_contextual_msg_budgets(batch, 0), th.tensor([0, 1]))
