"""
Run a grid search experiment, only uses wandb for logging and does not initialize
a wandb sweep or wandb agents.

Compared to a wandb sweep, assigns a unique index to each scenario and automatically
runs all scenarios in parallel across all available GPUs. Runs each command using the "screen" tool
to allow greater control over running jobs compared to wandb which only allows runs to
be stopped 1 at a time.
"""

import argparse
import datetime
import shlex
import subprocess
from argparse import Namespace
from itertools import product
from math import ceil
from os import environ, getcwd, makedirs, walk
from os.path import join
from random import SystemRandom
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pandas.core.frame import DataFrame

from src.experiments.slurm_config import (
    Computers,
    get_slurm_args,
    get_slurm_config_lines,
)
from src.main import main
from src.utils.utils import is_debugger_active, string_inputs_to_list


class GridSearch(object):
    venv_activate_path = join(".venv", "bin", "activate")
    basic_config_params: list[str] = ["config", "env-config"]

    tmux_prefix = [
        "tmux",
        "new",
        "-d",
        "-s",
    ]

    script_path = join("src", "main.py")

    def __init__(self) -> None:
        self.args = self._parse_args()
        self.args.debug = is_debugger_active() or getattr(self.args, "debug")

        self.exp_dir = join("experiments", self.args.experiment)
        # self.exp_dir = join(self.project_dir, "experiments", self.args.experiment)
        self.job_dir = join(self.exp_dir, "jobs")
        makedirs(self.job_dir, exist_ok=True)

        # setup
        config_path = join(self.exp_dir, "exp_config.yaml")
        with open(config_path, "r", encoding="utf8") as f:
            full_config: dict = yaml.safe_load(f)

        config = full_config["parameters"]
        conditional_config = full_config.get("conditional_parameters", None)

        # generate a unique time id for this experiment
        if config.get("time_id", False):
            time_id = config.get("time_id")["values"][0]
            # do not create a new wandb run if time_id provided, likely doing post-processing
            # with its own specialized logging
            config["use_wandb"] = {"values": [False]}

        else:
            time_id = f"{self.args.experiment}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:]}"

        # get seeds
        if config.get("seed"):
            seeds = config.pop("seed")["values"]
        else:
            n_seeds = config.pop("n_seeds")["values"][0] if config.get("n_seeds") else 5
            rng = SystemRandom()
            seeds = [rng.randint(0, 1000000) for _ in range(n_seeds)]

        parameters_to_print = full_config.get("parameters_to_print")

        """
        # check if running bisimulation test
        if base_config.get("env_bisimulation_test", False):
            env_bisimulation_test = base_config.pop(
                "env_bisimulation_test"
            )["values"][0]
            # add save_replay_buffer to the config so you can run the post-processing
            base_config["save_replay_buffer"] = {"values": ["True"]}

        else:
            env_bisimulation_test = False
        """

        scenarios, scenario_names = self._get_scenarios(config, conditional_config)

        run_setups = self._get_run_setups(
            scenarios=scenarios,
            seeds=seeds,
            scenario_names=scenario_names,
            time_id=time_id,
        )

        if self.args.debug:
            self._run_debug(run_setups)
            return

        else:
            if self.args.computer in Computers.lab:
                self._print_hardware_info()
            self._print_exp_info(scenarios, run_setups, parameters_to_print)

            if self.args.computer in Computers.cluster:
                # get config for slurm job on cluster
                slurm_config_path = join(self.exp_dir, "slurm_config.yaml")
                with open(slurm_config_path, "r", encoding="utf8") as f:
                    slurm_config: dict = yaml.safe_load(f)

                cluster_log_dir: str = join(
                    "results",
                    "cluster_logs",
                    self.args.experiment,
                    time_id,
                )

                slurm_config["experiment"] = self.args.experiment
                slurm_config["cluster"] = self.args.computer
                slurm_config["log_dir"] = cluster_log_dir

                job_paths = self._build_sbatch_files(
                    slurm_config,
                    run_setups,
                    cluster=self.args.computer,
                    time_id=time_id,
                )

            # check if user wants to run the experiment
            user_input = input("Run experiment now? (y/n) ").lower()
            if user_input == "y":
                print("Running experiment")
                # if env_bisimulation_test:
                #     self._run_bisimulation_test(run_setups)
                #     return

                match self.args.computer:
                    case "lab":
                        self._run_experiment_lab(run_setups.cmd)
                    case "campus" | "delta":
                        makedirs(cluster_log_dir, exist_ok=True)
                        self._run_experiment_cluster(job_paths)
                return

            print("Exiting without running experiment")
            return

    def _run_debug(self, run_setups: DataFrame) -> None:
        # just run the first command in run_setups
        for cmd in run_setups.cmd:
            # strip out any "python" prefix stuff and just have the args from src/main.py onwards
            # use shlex to avoid splitting on spaces within args substrings
            debug_cmd = shlex.split(cmd[1:])
            main(debug_cmd)

            # proc = subprocess.Popen(
            #     cmd,
            # )
            # proc.wait()
            # print("done with debugging")
            return

    def _get_scenarios(
        self, base_config: dict, conditional_config: dict | None = None
    ) -> tuple[list[dict], list[str]]:
        scenarios: list[dict] = [*self._gen_dict_combinations(base_config)]

        # handle comms budgets when a unique policy per comms value is requested
        # expand scenarios so each comms value becomes its own scenario
        updated_scenarios = []
        for scenario in scenarios:
            if scenario.get("msg_budget_per_agent") and scenario.get(
                "unique_policy_per_msg_budget"
            ):
                scenario = string_inputs_to_list(
                    scenario, "msg_budget_per_agent", output_type=int
                )
                for val in scenario.pop("msg_budget_per_agent"):
                    new_s = scenario.copy()
                    # format as a string in a list to work with parsing in main.py
                    new_s["msg_budget_per_agent"] = [f"{val}"]
                    updated_scenarios.append(new_s)
            else:
                updated_scenarios.append(scenario)

        scenarios = updated_scenarios

        # get scenario configs based on conditional params
        if conditional_config is not None:
            # loop over outer vars (EX: config, env-config)
            for outer_var, conditional_vars in conditional_config.items():
                # loop over config (EX: maic, qmix) and env-config (EX: join1-v0 and join1_original)
                for inner_var, varied_params in conditional_vars.items():
                    conditional_combos = [*self._gen_dict_combinations(varied_params)]

                    updated_scenarios = []
                    indices_remove = []
                    for i, scenario in enumerate(scenarios):
                        if scenario[outer_var] == inner_var:
                            for combo in conditional_combos:
                                if combo.get("msg_budget_per_agent") and scenario.get(
                                    "unique_policy_per_msg_budget"
                                ):
                                    combo = string_inputs_to_list(
                                        combo, "msg_budget_per_agent", output_type=int
                                    )
                                    for val in combo.pop("msg_budget_per_agent"):
                                        new_c = combo.copy()
                                        # format as a string in a list to work with parsing in main.py
                                        new_c["msg_budget_per_agent"] = [f"{val}"]
                                        updated_scenarios.append(scenario | new_c)
                                else:
                                    updated_scenarios.append(scenario | combo)

                                indices_remove.append(i)

                    # remove scenarios that were updated and bring in their updated versions
                    scenarios = [
                        s for i, s in enumerate(scenarios) if i not in indices_remove
                    ]
                    scenarios += updated_scenarios

        scenario_names: list[str] = [
            f"{scenario_idx + 1}".zfill(2) for scenario_idx, _ in enumerate(scenarios)
        ]

        return scenarios, scenario_names

    def _get_run_setups(
        self,
        scenarios: list[str],
        seeds: list[int],
        scenario_names: list[str],
        time_id: str,
    ) -> pd.DataFrame:

        run_setups: list[dict] = []

        scenario_params = {
            "experiment": self.args.experiment,
            "time_id": time_id,
        }

        for scenario_idx, params in enumerate(scenarios):
            _params: dict = params.copy()
            scenario_params["scenario"] = scenario_names[scenario_idx]

            # get cartesian product of any grid_search_params (seed, ...)
            grid_search_params = {"seed": seeds}
            grid_keys = list(grid_search_params.keys())
            grid_values = list(grid_search_params.values())

            # unique wandb group name for each experimental scenario + runtime,
            # used to group runs on the wandb for post-processing
            # for param in self.save_params:
            #     if param in _params:
            #         scenario_params[param] = _params[param]

            # options is rl alg and env
            # update is the "with" params (except params grid-searched above here)
            options = []
            updates = []
            for k, v in _params.items():
                # used for the Sacred arg parser
                if k in self.basic_config_params:
                    options.append(f"--{k}={v}")
                else:
                    updates.append(f"{k}={v}")

            # define the command to be run for each combination of grid params
            # seeds have the same scenario number, but unique_policy_per_msg_budget should generate different scenarios
            for combo in product(*grid_values):
                combo_dict = dict(zip(grid_keys, combo))

                # for each (seed, message_budget) combo, build a run
                run_params = scenario_params.copy()
                run_params.update(combo_dict)
                run_updates = [f"--{k}={v}" for k, v in run_params.items()]

                # needed for debugging while using Sacred
                # --force makes Sacred ignore this error since it sucks at checking whether params are actually used in your code or not
                # sacred.utils.ConfigAddedError: Added new config entry that is not used anywhere
                sacred_debug_suffix = ""
                # sacred_debug_suffix = "-d --force"

                python_cmd = f"python {self.script_path} {' '.join(options)} with {' '.join(updates)} {' '.join(run_updates)} {sacred_debug_suffix}"
                run_params["cmd"] = python_cmd
                run_setups.append(run_params)

        run_setups_out = pd.DataFrame.from_records(run_setups)
        return run_setups_out

    def _run_experiment_lab(self, python_cmds: pd.Series) -> None:
        # assign each runner its commands
        n_runners = self.args.n_runners
        runners = {i: "source .venv/bin/activate;" for i in range(n_runners)}
        for i, cmd in enumerate(python_cmds):
            runners[i % n_runners] += cmd + "; "

        # assign each runner its GPU
        runner_gpus = []
        for runner_idx in runners:
            runner_gpus.append(self.args.gpus[runner_idx % len(self.args.gpus)])

        # run all runners in parallel
        processes = []
        env = dict(environ.items())

        for i, runner_cmds in runners.items():
            tmux_process_name = f"{self.args.experiment}_runner_{i + 1}"
            runner_cmds = f"export CUDA_VISIBLE_DEVICES={runner_gpus[i]};" + runner_cmds

            run_cmd = [
                *self.tmux_prefix,
                tmux_process_name,
                runner_cmds,
            ]

            proc = subprocess.Popen(
                run_cmd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )

            processes.append(proc)

            for p in processes:
                p.wait()

    def _build_sbatch_files(
        self,
        slurm_config: dict,
        run_setups: pd.DataFrame,
        cluster: Literal["campus", "delta"],
        time_id: str,
    ) -> list[str]:
        # assign commands to each job and run them all in parallel within that job
        max_runs_per_job = self.args.max_runs_per_job
        python_cmds = run_setups.cmd
        n_runs = len(python_cmds)
        n_jobs = ceil(n_runs / max_runs_per_job)
        jobs = {i: [] for i in range(n_jobs)}

        # assign commands to jobs
        for i, python_cmd in enumerate(python_cmds):
            # run each python commmand in its own tmux session so you can check its progress if needed
            tmux_process_name = f"{self.args.experiment}_runner_{i + 1}"
            tmux_str = " ".join(self.tmux_prefix)

            cd_dir_cmd = f"cd {getcwd()}"
            venv_activate_cmd = "source .venv/bin/activate"
            module_load_cmd = (
                f"{'module load cuda/12.4; ' if self.args.computer == 'campus' else ''}"
                + "module load python"
            )

            cmd = f"{cd_dir_cmd}; {venv_activate_cmd}; {module_load_cmd}; {python_cmd}"
            run_cmd = f'{tmux_str} {tmux_process_name} "{cmd}"'
            jobs[i % n_jobs].append(run_cmd + " &")

        job_paths: list[str] = []
        print("Writing job files")

        # loop thru all jobs, get the slurm config args needed to generate the slurm file and generate the slurm file
        for job_idx, cmds in jobs.items():
            cmds.insert(0, "# job commands")

            # slurm config
            slurm_config["job_idx"] = job_idx + 1
            slurm_config_lines = get_slurm_config_lines(get_slurm_args(**slurm_config))

            # commands to run this project on the cluster
            project_setup_lines: list[str] = [
                "# project setup",
                f"cd {getcwd()}",
                f"source {self.venv_activate_path}",
            ]

            module_load_lines: list[str] = (
                ["module load cuda/12.4"] if self.args.computer == "campus" else []
            )
            module_load_lines += [
                "echo 'Running on node with hostname:'",
                "hostname -s",
                "nvidia-smi",
                "python src/experiments/node_test.py",
            ]

            # save the slurm files to disk
            setups: list[list[str]] = [
                slurm_config_lines,
                project_setup_lines,
                module_load_lines,
                cmds,
            ]
            job_dir = join(self.job_dir, time_id)
            makedirs(job_dir, exist_ok=True)
            job_path = join(job_dir, f"job_{cluster}_{job_idx + 1}.slurm")
            job_paths.append(job_path)
            print(f"{len(cmds) - 1} runs to {job_path}")
            self._write_sbatch(output_path=job_path, setups=setups)

        return job_paths

    def _run_experiment_cluster(self, job_paths: list[str]) -> None:
        # submit jobs to cluster
        for job_path in job_paths:
            print(f"Submitting job {job_path}")
            subprocess.run(["sbatch", job_path], check=False)

    def _parse_args(self) -> Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-e",
            "--experiment",
            type=str,
            required=True,
            help="Experiment name",
        )
        parser.add_argument(
            "-g",
            "--gpus",
            type=int,
            nargs="+",
            required=False,
            default=[0],
            help="Available GPUs to run this experimet on (lab runs only).",
        )
        parser.add_argument(
            "-r",
            "--n_runners",
            type=int,
            required=False,
            default=2,
            help="Number of parallel runners to have going at the same time (lab runs only).",
        )
        parser.add_argument(
            "-m",
            "--max_runs_per_job",
            type=int,
            default=40,
            help="Max parallel runs per slurm job (cluster runs only).",
        )
        parser.add_argument(
            "-c",
            "--computer",
            type=str,
            required=False,
            choices=["lab", "campus", "delta"],
            default="lab",
            help="Computer to run the experiment on.",
        )
        parser.add_argument(
            "-d",
            "--debug",
            action="store_true",
        )

        return parser.parse_args()

    def _print_exp_info(
        self,
        scenarios: list[dict],
        run_setups: pd.DataFrame,
        parameters_to_print: list[str] | None = None,
    ) -> None:
        """print experiment summary in a markdown-formatted table"""
        spaces = " " * 4
        n_scenarios = run_setups.scenario.nunique()
        seeds = run_setups.seed.unique()
        n_seeds = len(seeds)
        time_id = run_setups.time_id[0]

        n_runs = n_scenarios * n_seeds
        print(f"\n- Experiment summary ({self.args.computer} computer)")
        print(
            f"{spaces}- {n_scenarios} scenarios, {n_seeds} seeds per scenario, {n_runs} total runs"
        )

        print(f"{spaces}- Seeds: {seeds}\n{spaces}- time_id: {time_id}\n")

        table_header = (
            "| Scenario Name | Alg | Env | Params|" + "\n|----| ---- | ---- | ---- |"
        )
        print(table_header)

        for scenario_idx, params in enumerate(scenarios):
            # print params to markdown table
            other_params = ""

            for k, v in params.items():
                if parameters_to_print is not None:
                    if k in parameters_to_print["values"]:
                        other_params += f"{k}={v} "

                else:
                    no_print_params: list[str] = [
                        "cmd",
                        "wandb_project",
                        "wandb_mode",
                        "wandb_save_model",
                        "wandb_save_test_replays",
                        "use_wandb",
                        "save_model",
                        "save_model_interval",
                        "save_test_replays",
                        "use_sacred",
                        "save_replay_buffer",
                        "delete_local_models",
                        "live_render",
                        "save_model_interval",
                        "runner_log_interval",
                        "n_test_replays_save",
                    ]

                    if k not in self.basic_config_params + no_print_params:
                        other_params += f"{k}={v} "

            table_line = f"| {run_setups.scenario[scenario_idx * n_seeds]} | {params['config']} | {params['env-config']} | {other_params}|"
            print(table_line)

        print("")

        if self.args.computer == "lab":
            print(
                f"Using {self.args.n_runners} parallel runners with a max of {ceil(n_runs / self.args.n_runners)} sequential runs per runner"
            )

            save_path = join(self.job_dir, f"job_lab_{time_id}.txt")
            print(f"Saving python commands to {save_path}")
            with open(save_path, "w", encoding="utf8") as f:
                for cmd in run_setups.cmd:
                    f.write(f"{cmd}\n")

    def _print_hardware_info(self) -> None:
        import psutil
        from torch.cuda import get_device_properties
        from torch.cuda.memory import mem_get_info

        # available computer resources
        print(
            f"Lab computer hardware summary\nUsing {len(self.args.gpus)} GPUs with indices {self.args.gpus}\nVRAM usage"
        )
        byte_to_gb = 1024**3
        for device in self.args.gpus:
            avail_vram, total_vram = mem_get_info(device)
            avail_vram = round(avail_vram / byte_to_gb, 1)
            total_vram = round(total_vram / byte_to_gb, 1)
            used_vram = round(total_vram - avail_vram, 1)

            props = get_device_properties(device)

            print(
                f"Device {device} -- {used_vram} GB / {total_vram} GB used ({avail_vram} GB available) -- {props.name}"
            )

        ram = psutil.virtual_memory()
        used_ram = round(ram.used / byte_to_gb, 1)
        total_ram = round(ram.total / byte_to_gb, 1)
        avail_ram = round(ram.available / byte_to_gb, 1)

        print(
            "\nRAM usage --",
            f"{used_ram} GB / {total_ram} GB used ({avail_ram} GB available)",
        )

    def _write_sbatch(self, output_path: str, setups: list[list[str]]) -> None:
        with open(output_path, "w", encoding="utf8") as f:
            for setup in setups:
                for line in setup:
                    f.write(line)
                    f.write("\n")
                f.write("\n")

            # need a "wait" at the end to run parallel commands with &,
            # otherwise the batch job immediately terminates
            # if on campus computer
            if self.args.computer in ["campus", "delta"]:
                # need a short sleep before starting the checking loop for the tmux runs to start up
                f.write("sleep 1;echo Runs started;")
                f.write("\n")
                f.write("while tmux ls &>/dev/null; do sleep 1; done;")
                f.write("\n")
                f.write("echo Runs finished")
            else:
                f.write("wait")

    def _gen_dict_combinations(self, d: dict):
        """Generate all combinations of nested dict values.
        See: https://stackoverflow.com/questions/50606454/cartesian-product-of-nested-dictionaries-of-lists
        """

        def gen_combinations(nested_dict):
            for combo in product(*nested_dict.values()):
                yield combo[0]

        for combo in product(*(gen_combinations(v) for v in d.values())):
            yield dict(zip(d.keys(), combo))

    def _run_bisimulation_test(self, run_setups: pd.DataFrame) -> None:
        # get the commands for the two envs you want to compare
        # for the case of 1  seed, this is very easy, just the commands in the list of python_cmds
        processes = []
        for seed in run_setups.seed:
            df_tmp = run_setups.copy()
            df_tmp = df_tmp.loc[df_tmp.seed == seed]

            for i, cmd in enumerate(df_tmp.cmd):
                # TODO replace w/ tmux prefix if you ever come back to this
                run_cmd = [*self.bash_prefix, cmd]
                proc = subprocess.Popen(run_cmd)
                processes.append(proc)

        for p in processes:
            p.wait()

        summary = []
        for seed in run_setups.seed.unique():
            envs_match = self._check_envs(run_setups, seed)
            summary.append(envs_match)

        if np.sum(summary) == run_setups.seed.nunique():
            print("Envs match over all seeds, bisimulation test passed")
        else:
            print("Not all seeds match, bisimulation test failed")

    def _check_envs(self, run_setups: pd.DataFrame, seed: int) -> bool:
        buffer_dir = join("results", "replay_buffers")
        df_tmp = run_setups.loc[run_setups.seed == seed]

        run_dirs = [
            f"{row.time_id}_{row.seed}_{row.scenario}" for _, row in df_tmp.iterrows()
        ]

        print(f"Comparing env outputs for {run_dirs}")
        run_data = {}
        for run_dir in run_dirs:
            load_dir = join(buffer_dir, run_dir)
            for _, _, files in walk(load_dir):
                for fn in files:
                    if fn.endswith(".npy"):
                        key = fn.split(".")[0]
                        if key not in ["actions_onehot", "filled"]:
                            if key not in run_data:
                                run_data[key] = []
                            run_data[key].append(np.load(join(load_dir, fn)))

        summary = {}
        keys_pass = []
        for k, data in run_data.items():
            for arr in data[1:]:
                # check if shapes are the same
                if arr.shape == data[0].shape:
                    if np.allclose(arr, data[0]):
                        summary[k] = f"Pass - {k} match"
                        keys_pass.append(k)

                    else:
                        n_match = np.sum(np.isclose(arr, data[0]))
                        n_total = np.prod(arr.shape)
                        percentage_mismatch = (1 - (n_match / n_total)) * 100
                        summary[k] = (
                            f"Fail - {k} do not match, {n_total - n_match} / {n_total} ({round(percentage_mismatch, 5)} % of entries do not match)"
                        )
                else:
                    summary[k] = (
                        f"Fail - {k} have different sizes, {arr.shape}, {data[0].shape}"
                    )

        # Summary
        if len(keys_pass) != len(run_data):
            print(
                f"Envs do not match, issues with {len(run_data) - len(keys_pass)} / {len(run_data)} outputs"
            )
            print(f"{list(set(run_data.keys()) - set(keys_pass))}")
            for k, v in summary.items():
                print(k, v)
            return False
        else:
            print(f"Envs match for seed {seed}")
            return True


if __name__ == "__main__":
    GridSearch()
