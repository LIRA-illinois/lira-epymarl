from types import SimpleNamespace

import pytest
import torch as th

from src.modules.agents.maic_agent import MAICAgent


def _agent_args(n_agents: int, **kwargs):
    return SimpleNamespace(
        n_agents=n_agents,
        latent_dim=2,
        n_actions=4,
        hidden_dim=8,
        nn_hidden_size=8,
        attention_dim=4,
        **kwargs,
    )


@pytest.mark.parametrize("n_agents,budget", [(1, 0), (3, 1), (3, 2), (5, 2)])
def test_team_message_drop_rate_scales_with_current_message_budget(n_agents, budget):
    team_rate = 0.1
    agent = MAICAgent(
        input_shape=5, args=_agent_args(n_agents, team_msg_drop_rate=team_rate)
    )
    agent.msg_budget_per_agent = budget

    if budget == 0:
        assert agent.msg_drop_rate == 0.0
    else:
        n_possible_messages = n_agents * budget
        assert 1 - (1 - agent.msg_drop_rate) ** n_possible_messages == pytest.approx(
            team_rate
        )


def test_individual_message_drop_rate_is_preserved():
    agent = MAICAgent(input_shape=5, args=_agent_args(3, msg_drop_rate=0.1))

    assert agent.msg_drop_rate == pytest.approx(0.1)


def test_message_dropout_remains_active_in_eval_mode():
    agent = MAICAgent(input_shape=5, args=_agent_args(3, msg_drop_rate=0.5))
    agent.eval()

    msg = th.ones(1, 1, 1000, 1)
    dropped = agent._unscaled_dropout(msg)

    assert (dropped == 0).any()
    assert (dropped == 1).any()


def test_message_dropout_uses_nonzero_gated_messages_as_eligibility():
    agent = MAICAgent(input_shape=5, args=_agent_args(3, msg_drop_rate=0.5))
    msg = th.zeros(1, 1, 2000, 4)
    msg[:, :, 1000:, :] = 1.0

    dropped = agent._unscaled_dropout(msg)

    assert (dropped[:, :, :1000, :] == 0).all()
    assert (dropped[:, :, 1000:, :] == dropped[:, :, 1000:, :1]).all()
    assert (dropped[:, :, 1000:, :] == 0).any()
    assert (dropped[:, :, 1000:, :] == 1).any()


def test_message_dropout_is_applied_after_attention_filter(monkeypatch):
    agent = MAICAgent(input_shape=5, args=_agent_args(3, msg_drop_rate=1.0))
    alpha = th.zeros(1, 3, 3, 1)
    alpha[0, 0, 1, 0] = 1.0
    monkeypatch.setattr(agent, "_get_attention_weights", lambda **kwargs: alpha)

    gated_msg, actual_alpha = agent._get_messages(
        hidden_state=th.ones(3, 8),
        bs=1,
        latent=th.ones(9, 2),
        test_mode=False,
    )

    assert actual_alpha is alpha
    assert (gated_msg == 0).all()


def test_individual_and_team_message_drop_rates_are_exclusive():
    with pytest.raises(ValueError, match="either msg_drop_rate or team_msg_drop_rate"):
        MAICAgent(
            input_shape=5,
            args=_agent_args(3, msg_drop_rate=0.1, team_msg_drop_rate=0.1),
        )


def test_conditioned_budget_enforces_message_filter_during_training(monkeypatch):
    agent = MAICAgent(
        input_shape=5,
        args=_agent_args(3, condition_on_msg_budget=True),
    )
    filtered = []
    monkeypatch.setattr(
        agent,
        "_msg_filter",
        lambda alpha, compute_loss: filtered.append(compute_loss) or alpha,
    )

    agent._get_attention_weights(
        hidden_state=th.ones(3, 8),
        latent=th.ones(9, 2),
        bs=1,
        test_mode=False,
    )

    assert filtered == [False]


def test_conditioned_budgets_apply_per_episode_topk_limits():
    agent = MAICAgent(input_shape=5, args=_agent_args(3, condition_on_msg_budget=True))
    alpha = th.ones(2, 3, 3, 1)
    for agent_idx in range(3):
        alpha[:, agent_idx, agent_idx] = 0

    filtered = agent._msg_filter_hard_topk(
        alpha, False, msg_budget_per_agent=th.tensor([0, 2])
    )

    th.testing.assert_close(
        filtered.ne(0).sum(dim=2).squeeze(-1),
        th.tensor([[0, 0, 0], [2, 2, 2]]),
    )
