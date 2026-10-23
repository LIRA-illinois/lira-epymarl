from types import SimpleNamespace

import torch as th

from src.learners.maic_learner import MAICLearner
from src.modules.mixers.qmix import QMixer


def test_maic_learner_appends_policy_context_to_mixer_states():
    learner = object.__new__(MAICLearner)
    learner.policy_context_dim = 2
    states = th.arange(24, dtype=th.float32).reshape(2, 3, 4)
    context = th.tensor(
        [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
         [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]
    )

    mixer_states = learner._get_mixer_states(
        {"state": states, "policy_context": context}, slice(1, None)
    )

    assert mixer_states.shape == (2, 2, 6)
    th.testing.assert_close(mixer_states[..., :4], states[:, 1:])
    th.testing.assert_close(mixer_states[..., 4:], context[:, 1:])


def test_qmix_accepts_additional_context_dimensions():
    args = SimpleNamespace(
        n_agents=3,
        state_shape=4,
        mixing_embed_dim=5,
        hypernet_layers=1,
    )
    mixer = QMixer(args, additional_state_dim=2)

    agent_qs = th.randn(2, 3, 3)
    states_with_context = th.randn(2, 3, 6)

    q_tot = mixer(agent_qs, states_with_context)

    assert mixer.state_dim == 6
    assert q_tot.shape == (2, 3, 1)


def test_maic_learner_sizes_qmix_for_policy_context():
    args = SimpleNamespace(
        n_agents=3,
        state_shape=4,
        mixing_embed_dim=5,
        hypernet_layers=1,
        mixer="qmix",
        target_update_interval=200,
        episode_limit=15,
        lr=0.0005,
        optim_alpha=0.99,
        optim_eps=0.00001,
        learner_log_interval=2000,
    )

    learner = MAICLearner(
        th.nn.Linear(1, 1),
        {"policy_context": {"vshape": (2,)}},
        logger=None,
        args=args,
    )

    assert learner.policy_context_dim == 2
    assert learner.mixer.state_dim == 6
