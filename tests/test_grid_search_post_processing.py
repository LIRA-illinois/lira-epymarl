import sys
from argparse import Namespace

import pandas as pd
import yaml

from src.experiments import grid_search_experiment
from src.experiments.grid_search_experiment import GridSearch


def test_grid_search_cli_accepts_post_processing_time_id(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grid_search_experiment.py",
            "--experiment",
            "exp_48",
            "--post-processing",
            "aggregate_runs",
            "--time-id",
            "exp_48_26-10-23_11-04-30",
        ],
    )

    args = GridSearch.__new__(GridSearch)._parse_args()

    assert args.experiment == "exp_48"
    assert args.post_processing == "aggregate_runs"
    assert args.time_id == "exp_48_26-10-23_11-04-30"


def test_unified_post_processing_aggregates_completed_scenario_seeds(
    tmp_path,
    monkeypatch,
):
    grid_search = object.__new__(GridSearch)
    grid_search.args = Namespace(experiment="exp_50")

    time_id = "exp50_postprocess_test"
    captured = {}
    monkeypatch.setattr(
        grid_search_experiment,
        "write_seed_postprocessing",
        lambda root, output: captured.update(
            root=root,
            output=output,
        ) or {
            "frontier_raw": str(tmp_path / "frontier_raw.csv"),
        },
    )
    monkeypatch.setattr(
        grid_search_experiment,
        "main",
        lambda params: (_ for _ in ()).throw(
            AssertionError("Unified post-processing must not call legacy main")
        ),
    )


    hierarchical = {
        "task_sequence": [[0, 1], [1, 2]],
        "pg_values": [0.55],
        "epsilon_est": 0.05,
        "beta": 0.95,
        "epsilon_p": 0.20,
        "n_max": 2,
    }
    grid_search._run_post_processing(
        config={
            "time_id": {"values": [time_id]},
            "seed": {"values": [101, 202]},
            "config": {"values": ["maic"]},
            "env-config": {"values": ["team-nav-v0"]},
            "hierarchical": {"values": [hierarchical]},
            "post_processing": {"values": ["aggregate_runs"]},
        },
        conditional_config=None,
        time_id=time_id,
    )

    assert captured["root"] == (
        f"results/hierarchical_artifacts/{time_id}/01"
    )
    assert captured["output"].endswith("/postprocessing")


def test_post_processing_preserves_all_message_budgets(monkeypatch):
    grid_search = object.__new__(GridSearch)
    grid_search.args = Namespace(experiment="exp_36")
    launched_args = {}
    monkeypatch.setattr(
        grid_search_experiment,
        "main",
        lambda args: launched_args.update(args=args),
    )

    grid_search._run_post_processing(
        config={
            "time_id": {"values": ["exp_36_test"]},
            "seed": {"values": [101]},
            "config": {"values": ["maic"]},
            "env-config": {"values": ["team-nav-v0"]},
            "msg_budget_per_agent": {"values": [["0 1 2"]]},
            "unique_policy_per_msg_budget": {"values": [True]},
            "post_processing": {"values": ["aggregate_runs"]},
        },
        conditional_config=None,
        time_id="exp_36_test",
    )

    budget_argument = next(
        argument
        for argument in launched_args["args"]
        if argument.startswith("msg_budget_per_agent=")
    )
    assert budget_argument == "msg_budget_per_agent=['0 1 2']"


def test_run_priority_orders_scenarios_before_assigning_names():
    grid_search = object.__new__(GridSearch)

    scenarios, scenario_names, _ = grid_search._get_scenarios(
        {
            "env_args.n_agents": {"values": [6, 3]},
            "hl_task": {"values": [["1 3"], ["0 1"]]},
        },
        run_priority={
            "env_args.n_agents": [3, 6],
            "hl_task": [["0 1"], ["1 3"]],
        },
    )

    assert [
        (scenario["env_args.n_agents"], scenario["hl_task"]) for scenario in scenarios
    ] == [(3, ["0 1"]), (3, ["1 3"]), (6, ["0 1"]), (6, ["1 3"])]
    assert scenario_names == ["01", "02", "03", "04"]


def test_experiment_summary_uses_template_without_printing_full_table(tmp_path, capsys):
    grid_search = object.__new__(GridSearch)
    grid_search.args = Namespace(experiment="exp_test", computer="delta")
    grid_search.job_dir = str(tmp_path)
    scenarios = [
        {
            "config": "maic",
            "env-config": "team-nav-v0",
            "env_args.map_name": "small_hall",
            "msg_budget_per_agent": ["0"],
        },
        {
            "config": "maic",
            "env-config": "team-nav-v0",
            "env_args.map_name": "small_hall",
            "msg_budget_per_agent": ["1"],
        },
    ]
    run_setups = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "seed": seed,
                "time_id": "exp_test_26-10-23_10-00-00",
                "cmd": "run",
            }
            for scenario in ("01", "02")
            for seed in (11, 12)
        ]
    )

    grid_search._print_exp_info(scenarios, run_setups, ["msg_budget_per_agent"])

    output = capsys.readouterr().out
    assert "2 scenarios x 2 seeds = 4 runs" in output
    assert "Scenario | Algorithm" not in output
    assert "Scenario matrix:" not in output

    assert not (tmp_path / "scenarios_exp_test_26-10-23_10-00-00.csv").exists()

    note = (tmp_path / "summary_exp_test_26-10-23_10-00-00.md").read_text()
    assert note.startswith("## 2026-10-23 - exp_test\n")
    assert "### Experiment outline" in note
    assert "#### Experiment running checklist" in note
    assert "### Results" in note
    assert "### TODOs for future experiments" in note
    assert "**Seeds:** 11, 12" in note
    assert '| `msg_budget_per_agent` | ["0"], ["1"] |' in note
    assert "Source config" not in note
    assert "Scenario matrix" not in note


def test_unified_hierarchical_grid_preserves_full_budget_list():
    grid_search = object.__new__(GridSearch)
    hierarchical = {
        "policy_scope": "shared_across_contexts",
        "task_sequence": [[0, 1], [0, 2], [1, 3], [2, 3], (3, 4)],
        "pg_values": [0.60, 0.65],
        "epsilon_est": 0.05,
        "beta": 0.95,
        "epsilon_p": 0.20,
        "n_max": 4,
        "n_calibration_episodes": 100,
        "max_iterations": 3,
        "resume": "none",
    }

    scenarios, scenario_names, _ = grid_search._get_scenarios(
        {
            "config": {"values": ["maic"]},
            "msg_budget_per_agent": {"values": [[0, 1]]},
            "unique_policy_per_msg_budget": {"values": [True]},
            "hierarchical": {"values": [hierarchical]},
        }
    )

    assert scenario_names == ["01"]
    assert len(scenarios) == 1
    assert scenarios[0]["msg_budget_per_agent"] == [0, 1]
    assert scenarios[0]["hierarchical"]["pg_values"] == [0.60, 0.65]


def test_unified_hierarchical_run_setup_has_seed_isolated_identity():
    grid_search = object.__new__(GridSearch)
    grid_search.args = Namespace(experiment="exp_49")
    hierarchical = {
        "policy_scope": "shared_across_contexts",
        "task_sequence": [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]],
        "pg_values": [0.60, 0.65],
        "epsilon_est": 0.05,
        "beta": 0.95,
        "epsilon_p": 0.20,
        "n_max": 4,
    }

    run_setups = grid_search._get_run_setups(
        scenarios=[
            {
                "config": "maic",
                "env-config": "team-nav-v0",
                "msg_budget_per_agent": [0, 1],
                "unique_policy_per_msg_budget": True,
                "hierarchical": hierarchical,
            }
        ],
        seeds=[101],
        scenario_names=["01"],
        time_id="exp_49_test",
    )

    assert len(run_setups) == 1
    row = run_setups.iloc[0]
    assert row.seed == 101
    assert row.scenario == "01"
    assert bool(row.unified_hierarchical)
    assert row.artifact_root == (
        "results/hierarchical_artifacts/exp_49_test/01/seed_101"
    )
    assert row.pg_values == [0.60, 0.65]
    assert (
        "--artifact_root=results/hierarchical_artifacts/exp_49_test/01/seed_101"
    ) in row.cmd


def test_unified_seed_scenario_cartesian_artifact_keys():
    grid_search = object.__new__(GridSearch)
    grid_search.args = Namespace(experiment="exp_50")
    hierarchical = {
        "task_sequence": [[0, 1], [1, 2]],
        "pg_values": [0.55, 0.65],
        "epsilon_est": 0.05,
        "beta": 0.95,
        "epsilon_p": 0.20,
        "n_max": 2,
    }

    run_setups = grid_search._get_run_setups(
        scenarios=[
            {
                "config": "maic",
                "env-config": "team-nav-v0",
                "msg_budget_per_agent": [0, 1],
                "unique_policy_per_msg_budget": True,
                "hierarchical": hierarchical,
            },
            {
                "config": "qmix",
                "env-config": "team-nav-v0",
                "msg_budget_per_agent": [0, 1],
                "unique_policy_per_msg_budget": True,
                "hierarchical": hierarchical,
            },
        ],
        seeds=[101, 202],
        scenario_names=["01", "02"],
        time_id="exp50_cartesian",
    )

    assert len(run_setups) == 4
    assert set(zip(run_setups["scenario"], run_setups["seed"])) == {
        ("01", 101),
        ("01", 202),
        ("02", 101),
        ("02", 202),
    }
    assert run_setups["artifact_root"].nunique() == 4


def test_exp50_exp51_sequential_parallel_replay_configs():
    with open("experiments/exp_50/exp_config.yaml", encoding="utf-8") as f:
        exp50 = yaml.safe_load(f)["parameters"]
    with open("experiments/exp_51/exp_config.yaml", encoding="utf-8") as f:
        exp51 = yaml.safe_load(f)["parameters"]

    assert exp50["batch_size_run"]["values"] == [1]
    assert exp51["batch_size_run"]["values"] == [10]
    for parameters in (exp50, exp51):
        assert parameters["save_test_replays"]["values"] == [True]
        assert parameters["n_test_replays_save"]["values"] == [2]
        assert parameters["wandb_save_test_replays"]["values"] == [True]
        assert parameters["hierarchical"]["values"][0]["task_sequence"] == [
            [0, 1],
            [1, 2],
        ]


def test_exp49_config_uses_fixed_seed_and_nested_hierarchical_block():
    with open("experiments/exp_49/exp_config.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    parameters = config["parameters"]
    assert parameters["seed"]["values"] == [101]
    assert "n_seeds" not in parameters
    assert parameters["t_max"]["values"] == [100]
    assert parameters["test_interval"]["values"] == [50]
    assert "env" not in parameters
    assert parameters["env-config"]["values"] == ["team-nav-v0"]
    assert parameters["use_wandb"]["values"] == [True]
    hierarchical = parameters["hierarchical"]["values"][0]
    assert hierarchical["task_sequence"] == [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [3, 4],
    ]


def test_exp52_team_delay_sweep_config():
    with open("experiments/exp_52/exp_config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parameters = config["parameters"]
    assert parameters["seed"]["values"] == [101]
    assert parameters["batch_size_run"]["values"] == [10]
    assert parameters["t_max"]["values"] == [5000]
    assert parameters["env_args.n_agents_delay"]["values"] == [1]
    assert parameters["env_args.team_delay_prob"]["values"] == [
        0.05,
        0.10,
        0.20,
    ]
    assert parameters["hierarchical"]["values"][0]["policy_scope"] == (
        "shared_across_contexts"
    )


def test_exp53_mixing_config_and_two_state_navigation_distributions():
    with open("experiments/exp_53/exp_config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(
        "submodules/gym-multigrid/gym_multigrid/envs/maps/team_navigation/"
        "navigation_mixing_two_task.yaml",
        encoding="utf-8",
    ) as f:
        navigation = yaml.safe_load(f)

    parameters = config["parameters"]
    hierarchical = parameters["hierarchical"]["values"][0]
    assert parameters["msg_budget_per_agent"]["values"] == [[0, 1, 2]]
    assert parameters["env_args.team_delay_prob"]["values"] == [0.05, 0.10]
    assert hierarchical["task_sequence"] == [[0, 1], [1, 2]]
    assert hierarchical["n_max"] == 2
    assert hierarchical["epsilon_est"] == 0.01
    assert hierarchical["epsilon_p"] == 0.05

    assert [task["init_state_dist"]["probs"] for task in navigation["tasks"]] == [
        [0.5, 0.5],
        [0.5, 0.5],
    ]
    assert all(
        len(task["init_state_dist"]["states"]) == 2
        for task in navigation["tasks"]
    )


def test_exp54_evaluation_only_mock_branching_config():
    with open("experiments/exp_54/exp_config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parameters = config["parameters"]
    hierarchical = parameters["hierarchical"]["values"][0]
    assert parameters["env"]["values"] == ["mock-hierarchical-v0"]
    assert parameters["t_max"]["values"] == [1]
    assert parameters["save_test_replays"]["values"] == [False]
    assert parameters["msg_budget_per_agent"]["values"] == [[0, 1, 2]]
    assert parameters["env_args.team_delay_prob"]["values"] == [0.05, 0.10]
    assert hierarchical["task_sequence"] == [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [3, 4],
    ]
    assert hierarchical["n_max"] == 5
    assert hierarchical["epsilon_est"] == 0.01
    assert hierarchical["epsilon_p"] == 0.05
    assert hierarchical["low_level_mode"] == "train"


def test_exp55_real_three_seed_five_subtask_config():
    with open("experiments/exp_55/exp_config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(
        "submodules/gym-multigrid/gym_multigrid/envs/maps/team_navigation/"
        "navigation_branching_two_goals_small_hall.yaml",
        encoding="utf-8",
    ) as f:
        navigation = yaml.safe_load(f)

    parameters = config["parameters"]
    hierarchical = parameters["hierarchical"]["values"][0]
    assert parameters["seed"]["values"] == [101, 202, 303]
    assert parameters["t_max"]["values"] == [200]
    assert parameters["batch_size_run"]["values"] == [10]
    assert parameters["test_interval"]["values"] == [100]
    assert parameters["save_model_interval"]["values"] == [500_000]
    assert parameters["save_test_replays"]["values"] == [True]
    assert parameters["n_test_replays_save"]["values"] == [5]
    assert "env" not in parameters
    assert parameters["env-config"]["values"] == ["team-nav-v0"]
    assert parameters["env_args.n_agents_delay"]["values"] == [1]
    assert parameters["env_args.team_delay_prob"]["values"] == [0.05, 0.10]
    assert parameters["msg_budget_per_agent"]["values"] == [[0, 1, 2]]
    assert parameters["seed"]["values"] == [101, 202, 303]
    assert hierarchical["task_sequence"] == [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [3, 4],
    ]
    assert hierarchical["n_max"] == 5
    assert hierarchical["epsilon_est"] == 0.01
    assert hierarchical["epsilon_p"] == 0.05
    assert len(navigation["tasks"]) == 5
    assert all(
        len(task["init_state_dist"]["states"]) == 8
        for task in navigation["tasks"]
    )
    assert all(
        abs(sum(task["init_state_dist"]["probs"]) - 1.0) < 1e-12
        for task in navigation["tasks"]
    )


def test_exp50_tiny_hall_grid_generates_one_unified_scenario():
    with open("experiments/exp_50/exp_config.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    parameters = config["parameters"]
    grid_search = object.__new__(GridSearch)
    scenarios, scenario_names, _ = grid_search._get_scenarios(parameters)

    assert scenario_names == ["01"]
    assert len(scenarios) == 1
    assert scenarios[0]["env"] == "team-nav-v0"
    assert scenarios[0]["msg_budget_per_agent"] == [0, 1]
    assert scenarios[0]["env_args.map_name"] == "3ga_1r_very_small_hall"
    assert scenarios[0]["hierarchical"]["task_sequence"] == [[0, 1], [1, 2]]
    assert parameters["t_max"]["values"] == [1000]
    assert parameters["test_interval"]["values"] == [200]


def test_exp49_grid_generates_one_unified_scenario():
    with open("experiments/exp_49/exp_config.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    grid_search = object.__new__(GridSearch)
    scenarios, scenario_names, _ = grid_search._get_scenarios(config["parameters"])

    assert scenario_names == ["01"]
    assert len(scenarios) == 1
    assert scenarios[0]["msg_budget_per_agent"] == [0, 1]
    assert "hierarchical" in scenarios[0]
    assert scenarios[0]["hierarchical"]["pg_values"] == [
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ]
