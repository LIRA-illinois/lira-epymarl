from argparse import Namespace

import pandas as pd

from src.experiments import grid_search_experiment
from src.experiments.grid_search_experiment import GridSearch


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
        (scenario["env_args.n_agents"], scenario["hl_task"])
        for scenario in scenarios
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

    grid_search._print_exp_info(
        scenarios, run_setups, ["msg_budget_per_agent"]
    )

    output = capsys.readouterr().out
    assert "2 scenarios x 2 seeds = 4 runs" in output
    assert "Scenario | Algorithm" not in output
    assert "Scenario matrix:" not in output

    assert not (tmp_path / "scenarios_exp_test_26-10-23_10-00-00.csv").exists()

    note = (
        tmp_path / "summary_exp_test_26-10-23_10-00-00.md"
    ).read_text()
    assert note.startswith("## 2026-10-23 - exp_test\n")
    assert "### Experiment outline" in note
    assert "#### Experiment running checklist" in note
    assert "### Results" in note
    assert "### TODOs for future experiments" in note
    assert "**Seeds:** 11, 12" in note
    assert "| `msg_budget_per_agent` | [\"0\"], [\"1\"] |" in note
    assert "Source config" not in note
    assert "Scenario matrix" not in note
