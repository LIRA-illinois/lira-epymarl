from src.envs import REGISTRY


def test_mock_hierarchical_env_is_registered_and_runner_compatible():
    env = REGISTRY["mock-hierarchical-v0"](
        seed=101,
        n_agents=3,
        n_final_positions=4,
        n_comms_values=2,
        n_subtasks=5,
        max_episode_steps=1,
    )

    obs, info = env.reset(
        seed=101,
        options={
            "hl_task": [0, 1],
            "msg_budget_per_agent": 1.0,
            "hl_start_state": 0,
        },
    )

    assert env.n_agents == 3
    assert obs.shape == (3, 6)
    assert env.avail_actions.shape == (3, 2)
    assert env.get_env_info()["episode_limit"] == 1
    assert info["task_completed"] is False

    actions = env.action_space.sample()
    next_obs, reward, terminated, truncated, final_info = env.step(actions)

    assert next_obs.shape == obs.shape
    assert reward in (-1.0, 1.0)
    assert terminated
    assert not truncated
    assert "task_completed" in final_info
    assert "mock_success_rate" in final_info
    assert "msg_budget_per_agent" in final_info


def test_mock_hierarchical_env_rejects_unknown_capacity():
    env = REGISTRY["mock-hierarchical-v0"](
        seed=101,
        n_agents=3,
        n_final_positions=4,
        n_comms_values=2,
        n_subtasks=5,
    )

    try:
        env.reset(
            seed=101,
            options={
                "hl_task": [0, 1],
                "msg_budget_per_agent": 0.5,
            },
        )
    except ValueError as error:
        assert "not configured" in str(error)
    else:
        raise AssertionError("Unknown mock communication capacity did not fail")


def test_two_goal_navigation_mock_success_improves_with_capacity():
    env = REGISTRY["mock-hierarchical-v0"](
        seed=101,
        n_agents=3,
        n_comms_values=3,
        navigation_config=(
            "submodules/gym-multigrid/gym_multigrid/envs/maps/"
            "team_navigation/navigation_branching_two_goals_small_hall.yaml"
        ),
        msg_budget_per_agent=[0.0, 1.0, 2.0],
    )
    init_state_dist = {
        "states": [
            [[1, 1], [1, 4], [1, 7]],
            [[1, 2], [1, 5], [1, 8]],
        ],
        "probs": [0.5, 0.5],
    }

    success_rates = []
    for capacity in (0.0, 1.0, 2.0):
        env.reset(
            seed=101,
            options={
                "hl_task": [0, 1],
                "msg_budget_per_agent": capacity,
                "navigation_init_state_dist": init_state_dist,
            },
        )
        success_rates.append(env._mock_success_rate())
        _, _, _, _, info = env.step(env.action_space.sample())
        if info["task_completed"]:
            assert len(info["final_state"]) == 6

    assert success_rates == sorted(success_rates)
    assert success_rates[0] < success_rates[1] < success_rates[2]
