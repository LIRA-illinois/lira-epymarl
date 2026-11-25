from src.scripts.mock_hierarchical_experiment import (
    MockExperimentParameters,
    generate_joint_states,
    generate_mock_calibration_rows,
    generate_mock_training_evaluations,
    run_mock_experiment,
)


def test_mock_training_evaluations_have_five_checkpoints():
    params = MockExperimentParameters(
        n_agents=3,
        n_final_positions=4,
        n_comms_values=2,
        n_subtasks=5,
        mock_evaluation_times=(20_000, 40_000, 60_000, 80_000, 100_000),
    )
    training_df = generate_mock_training_evaluations(101, params)

    assert len(training_df["t_env"].unique()) == 5
    assert set(training_df["t_env"]) == {
        20_000,
        40_000,
        60_000,
        80_000,
        100_000,
    }
    assert len(training_df) == 5 * 5 * 2
    assert training_df["test_task_completed_mean"].between(0.05, 0.98).all()


def test_parameterized_joint_state_and_l_setting_counts():
    params = MockExperimentParameters(
        n_agents=2,
        n_final_positions=3,
        n_comms_values=4,
        n_subtasks=3,
    )

    assert len(generate_joint_states(params)) == 6
    assert params.final_joint_state_count == 6
    assert params.l_setting_count == 4 * 3 * 6


def test_generate_mock_calibration_rows():
    rows = generate_mock_calibration_rows(
        seed=17,
        n_estimation_episodes=100,
    )

    assert len(rows) == 120
    assert all(row.success_count + row.failure_count == 100 for row in rows)
    assert {
        row.setting.start_state for row in rows if row.setting.subtask_edge == (1, 3)
    } == set(generate_joint_states(MockExperimentParameters()))


def test_mock_hierarchical_experiment_generates_saves_loads_and_synthesizes(
    tmp_path,
):
    summary = run_mock_experiment(
        output_dir=tmp_path,
        seed=17,
        pg=0.65,
        epsilon_est=0.05,
        beta=0.95,
        epsilon_p=0.10,
        n_estimation_episodes=100,
        max_iterations=3,
        params=MockExperimentParameters(
            n_agents=3,
            n_final_positions=4,
            n_comms_values=2,
            n_subtasks=5,
        ),
    )

    assert summary["option_artifact_count"] == 10
    assert summary["n_final_joint_states"] == 12
    assert summary["l_setting_count"] == 120
    assert summary["outcome_alphabet_size"] == 13
    assert summary["mock_evaluation_times"] == [
        20_000,
        40_000,
        60_000,
        80_000,
        100_000,
    ]
    assert summary["m_est_theoretical"] > 0
    assert summary["epsilon_p"] >= summary["delta_p"]
    assert summary["status"] == "certified"
    assert summary["selected"] is not None
    assert summary["selected"]["empirical_goal_probability"] >= (
        summary["pg"] + summary["epsilon_p"]
    )
    assert len(summary["candidates"]) == 3


def test_mock_hierarchical_experiment_writes_ten_point_frontier(tmp_path):
    pg_values = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    summary = run_mock_experiment(
        output_dir=tmp_path,
        seed=17,
        pg=0.65,
        pg_values=pg_values,
        epsilon_est=0.05,
        beta=0.95,
        epsilon_p=0.10,
        n_estimation_episodes=100,
        max_iterations=3,
    )

    import pandas as pd

    frontier_path = tmp_path / "success_vs_cost_seed_17.csv"
    plot_dir = tmp_path / "mock_training_plots"
    assert frontier_path.is_file()
    assert summary["pg_values"] == pg_values
    frontier = pd.read_csv(frontier_path)
    assert len(frontier) == 10
    assert set(frontier.columns) >= {
        "pg",
        "ours_cost",
        "baseline_cost",
        "task_greedy_cost",
        "comms_greedy_cost",
        "status",
    }
    assert (plot_dir / "success_vs_cost_seed_17.png").is_file()
    assert (plot_dir / "success_vs_cost_seed_17.pdf").is_file()


def test_parameterized_three_comms_three_subtask_chain(tmp_path):
    params = MockExperimentParameters(
        n_agents=3,
        n_final_positions=4,
        n_comms_values=3,
        n_subtasks=3,
    )

    summary = run_mock_experiment(
        output_dir=tmp_path,
        seed=23,
        pg=0.55,
        epsilon_est=0.05,
        beta=0.95,
        epsilon_p=0.10,
        n_estimation_episodes=100,
        max_iterations=2,
        params=params,
    )

    assert summary["option_artifact_count"] == 9
    assert summary["n_agents"] == 3
    assert summary["n_final_positions"] == 4
    assert summary["n_final_joint_states"] == 12
    assert summary["n_comms_values"] == 3
    assert summary["communication_capacities"] == [0.0, 0.5, 1.0]
    assert summary["n_subtasks"] == 3
    assert summary["l_setting_count"] == 108
