"""
Run a grid search experiment, only uses wandb for logging and does not initialize
a wandb sweep or wandb agents.

Compared to a wandb sweep, assigns a unique index to each scenario and automatically
runs all scenarios in parallel across all available GPUs. Runs each command using the "screen" tool
to allow greater control over running jobs compared to wandb which only allows runs to
be stopped 1 at a time.
"""

from typing import Literal
from os import environ, makedirs, getcwd, walk
from os.path import join
import math
from itertools import product
import argparse
import subprocess
import datetime
from random import SystemRandom
import yaml
import pandas as pd
import numpy as np

from slurm_args import SlurmArgs


class GridSearch(object):
    def __init__(self) -> None:
        self.args = self.parse_args()
        self.venv_activate_path = join(".venv", "bin", "activate")
        self.exp_dir = join("experiments", self.args.experiment)
        self.script_path = join("src", "main.py")
        self.job_dir = join(self.exp_dir, "jobs")
        makedirs(self.job_dir, exist_ok=True)

        # setup
        exp_config_path = join(self.exp_dir, "exp_config.yaml")
        with open(exp_config_path, "r", encoding="utf8") as f:
            exp_config: dict = yaml.safe_load(f)

        self.basic_config_params: list[str] = ["config", "env-config"]

        self.bash_prefix = ["/bin/bash", "-c"]

        self.save_params: list[str] = [
            "cmd",
            "wandb_project",
            "save_model",
            "save_model_interval",
            "save_test_replays",
            "wandb_save_model",
            "use_sacred",
            "wandb_save_test_replays",
            "use_wandb",
            "save_replay_buffer",
        ]

        # unique value for this experiment
        time_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")[2:]

        if self.args.computer in ["campus", "delta"]:
            slurm_config_path = join(self.exp_dir, "slurm_config.yaml")
            with open(slurm_config_path, "r", encoding="utf8") as f:
                self.slurm_config: dict = yaml.safe_load(f)

            cluster_log_dir: str = join(
                "results",
                "cluster_logs",
                self.args.experiment,
                time_id,
            )

            self.slurm_config["experiment"] = self.args.experiment
            self.slurm_config["cluster"] = self.args.computer
            self.slurm_config["log_dir"] = cluster_log_dir

        # get seeds
        if exp_config["parameters"].get("seed", False):
            seeds = exp_config["parameters"].pop("seed")["values"]
        else:
            if exp_config["parameters"].get("n_seeds", False):
                # generate n_seeds random seeds to use in this experiment
                n_seeds = exp_config["parameters"].pop("n_seeds")["values"][0]
            else:
                # default value
                n_seeds = 5

            # true randomness from the OS
            rng = SystemRandom()
            seeds = [rng.randint(0, 1000000) for _ in range(n_seeds)]

        # check if running bisimulation test
        if exp_config["parameters"].get("env_bisimulation_test", False):
            env_bisimulation_test = exp_config["parameters"].pop(
                "env_bisimulation_test"
            )["values"][0]
            # add save_replay_buffer to the config so you can run the post-processing
            exp_config["parameters"]["save_replay_buffer"] = {"values": ["True"]}

        else:
            env_bisimulation_test = False

        scenarios, scenario_names = self.get_scenarios(exp_config)

        run_setups = self.get_run_setups(
            scenarios=scenarios,
            seeds=seeds,
            scenario_names=scenario_names,
            script_path=self.script_path,
            time_id=time_id,
        )

        if self.args.debug:
            self.run_debug(run_setups)

        else:
            self.print_info(scenarios, run_setups)

            if self.args.computer in ["campus", "delta"]:
                job_paths = self.build_sbatch_files(run_setups, cluster=self.args.computer, time_id=time_id)

            # check if user wants to run the experiment
            user_input = input("Run experiment now? (y/n) ").lower()
            if user_input == "y":
                print("Running experiment")
                if env_bisimulation_test:
                    self.run_bisimulation_test(run_setups)

                else:
                    match self.args.computer:
                        case "lab":
                            self.run_experiment_lab(run_setups.cmd)
                        case "campus" | "delta":
                            makedirs(cluster_log_dir, exist_ok=True)
                            self.run_experiment_cluster(job_paths)

            else:
                print("Exiting without running experiment")
                exit()

    def run_debug(self, run_setups):
        for cmd in run_setups.cmd:
            run_cmd = [
                *self.bash_prefix,
                cmd,
            ]
            proc = subprocess.Popen(
                run_cmd,
            )
            proc.wait()
            print("done")
            exit()


    def get_scenarios(self, exp_config: dict) -> tuple[list[dict], list[str]]:
        scenarios: list[dict] = [*self.gen_dict_combinations(exp_config["parameters"])]

        # get scenario configs based on conditional params
        if "conditional_parameters" in exp_config:
            # loop over outer vars (EX: config, env-config)
            for outer_var, conditional_vars in exp_config[
                "conditional_parameters"
            ].items():

                # loop over config (EX: maic, qmix) and env-config (EX: join1-v0 and join1_original)
                for inner_var, varied_params in conditional_vars.items():
                    conditional_combos = [*self.gen_dict_combinations(varied_params)]

                    updated_scenarios = []
                    indices_remove = []
                    for i, scenario in enumerate(scenarios):
                        if scenario[outer_var] == inner_var:
                            for combo in conditional_combos:
                                updated_scenarios.append(scenario | combo)
                                indices_remove.append(i)

                    # remove scenarios that were updated and bring in their updated versions
                    scenarios = [
                        scenario
                        for i, scenario in enumerate(scenarios)
                        if i not in indices_remove
                    ]
                    scenarios += updated_scenarios

        scenario_names: list[str] = [
            f"{scenario_idx+1}".zfill(2) for scenario_idx, _ in enumerate(scenarios)
        ]

        return scenarios, scenario_names

    def get_run_setups(
        self,
        scenarios: list[str],
        seeds: list[int],
        scenario_names: list[str],
        script_path: str,
        time_id: str,
    ) -> pd.DataFrame:

        run_setups: list[dict] = []
        for scenario_idx, params in enumerate(scenarios):
            _params = params.copy()
            # unique wandb group name for each experimental scenario + runtime, used to group runs on the wandb website for post-processing
            scenario_params = {
                "experiment": self.args.experiment,
                "scenario": scenario_names[scenario_idx],
                "time_id": f"{self.args.experiment}_{time_id}",
            }
            for param in self.save_params:
                if _params.get(param):
                    scenario_params[param] = _params.get(param)

            # options is rl alg and env
            # update is the "with" params (except seed, you add that in manually)
            options = []
            updates = []
            for k, v in _params.items():
                if k in self.basic_config_params:
                    options.append(f"--{k}={v}")
                else:
                    if k not in scenario_params:
                        updates.append(f"{k}={v}")

            # define the command to be run
            for seed in seeds:
                # unique name for each wandb run using seed
                run_params = scenario_params.copy()
                run_params["seed"] = seed

                run_updates = [f"{k}={v}" for k, v in run_params.items()]

                if self.args.debug:
                    base_cmd = "ipdb3 -c continue"
                    # needed for debugging while using Sacred
                    sacred_debug_suffix = "-d"

                else:
                    base_cmd = "python3"
                    sacred_debug_suffix = ""

                python_cmd = f"{base_cmd} {script_path} {' '.join(options)} with {' '.join(updates)} {' '.join(run_updates)} {sacred_debug_suffix}"

                run_params["cmd"] = python_cmd
                run_setups.append(run_params)

        run_setups_out = pd.DataFrame.from_records(run_setups)
        return run_setups_out

    def run_experiment_lab(self, python_cmds: pd.Series):
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
            env["CUDA_VISIBLE_DEVICES"] = f"{runner_gpus[i]}"
            screen_name = f"{self.args.experiment}_runner_{i+1}"
            screen_prefix = ["screen", "-dmS", screen_name]
            run_cmd = [
                *screen_prefix,
                *self.bash_prefix,
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

    def build_sbatch_files(
        self, run_setups: pd.DataFrame, cluster: Literal["campus", "delta"], time_id: str
    ) -> list[str]:
        # assign commands to each job and run them all in parallel within that job
        max_runs_per_job = self.args.max_runs_per_job
        python_cmds = run_setups.cmd
        n_runs = len(python_cmds)
        n_jobs = math.ceil(n_runs / max_runs_per_job)
        jobs = {i: [] for i in range(n_jobs)}

        # assign commands to jobs
        for i, cmd in enumerate(python_cmds):
            jobs[i % n_jobs].append(cmd + " &")

        job_paths: list[str] = []

        print(f"Writing job files")

        # loop thru all jobs, get the slurm config args needed to generate the slurm file and generate the slurm file
        for job_idx, cmds in jobs.items():
            cmds.insert(0, "# job commands")

            # slurm config
            self.slurm_config["job_idx"] = job_idx + 1
            slurm_args = SlurmArgs(**self.slurm_config)
            slurm_config_lines = slurm_args.get_config_lines()

            # commands to run this project on the cluster
            project_name = getcwd().split("/")[-1]
            match self.args.computer:
                case "delta":
                    cluster_project_dir = join("~", "dev", f"{project_name}")
                case "campus":
                    cluster_project_dir = join(
                        "~", "my_trg_dir", "dev", f"{project_name}"
                    )
                case _:
                    raise NotImplementedError

            project_setup_lines: list[str] = [
                "# project setup",
                f"cd {cluster_project_dir}",
                f"source {self.venv_activate_path}",
            ]

            module_load_lines: list[str] = []
            if self.args.computer == "campus":
                module_load_lines.append(
                    "module load cuda/12.4",
                )
            module_load_lines += [
                "echo 'Running on node with hostname:'",
                "hostname -s",
                "nvidia-smi",
                "python3 src/experiments/node_test.py",
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
            self.write_sbatch(output_path=job_path, setups=setups)

        return job_paths

    def run_experiment_cluster(self, job_paths: list[str]):
        # submit jobs to cluster
        for job_path in job_paths:
            print(f"Submitting job {job_path}")
            subprocess.run(["sbatch", job_path], check=False)

    def parse_args(self):
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
            type=bool,
            default=False,
        )

        return parser.parse_args()

    def print_info(
        self,
        scenarios: list[dict],
        run_setups: pd.DataFrame,
    ):
        """print useful info about the experiment"""

        n_scenarios = run_setups.scenario.nunique()
        seeds = run_setups.seed.unique()
        n_seeds = len(seeds)
        time_id = run_setups.time_id[0]

        spaces = " " * 4

        if self.args.computer == "lab":
            from torch.cuda import get_device_properties
            from torch.cuda.memory import mem_get_info
            import psutil

            # available computer resources
            print(
                f"Lab computer hardware summary\nUsing {len(self.args.gpus)} GPUs with indices {self.args.gpus}\nVRAM usage"
            )
            byte_to_gb = 1024**3
            for device in self.args.gpus:
                (avail_vram, total_vram) = mem_get_info(device)
                # convert from bytes to gigabtyes
                avail_vram, total_vram = round(avail_vram / byte_to_gb, 1), round(
                    total_vram / byte_to_gb, 1
                )
                used_vram = round(total_vram - avail_vram, 1)

                props = get_device_properties(device)

                print(
                    f"Device {device} -- {used_vram} GB / {total_vram} GB used ({avail_vram} GB available) -- {props.name}"
                )

            ram_info = psutil.virtual_memory()
            used_ram = round(ram_info.used / byte_to_gb, 1)
            total_ram = round(ram_info.total / byte_to_gb, 1)
            avail_ram = round(ram_info.available / byte_to_gb, 1)

            print(
                "\nRAM usage --",
                f"{used_ram} GB / {total_ram} GB used ({avail_ram} GB available)",
            )

        # print experiment summary in a markdown-formatted table
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
                if k not in self.basic_config_params + self.save_params:
                    other_params += f"{k}={v} "

            table_line = f"| {run_setups.scenario[scenario_idx * n_seeds]} | {params['config']} | {params['env-config']} | {other_params}|"
            print(table_line)

        print("")

        if self.args.computer == "lab":
            print(
                f"Using {self.args.n_runners} parallel runners with a max of {math.ceil(n_runs / self.args.n_runners)} sequential runs per runner"
            )

            save_path = join(self.job_dir, f"job_lab_{time_id}.txt")
            print(f"Saving python commands to {save_path}")
            with open(save_path, "w", encoding="utf8") as f:
                for cmd in run_setups.cmd:
                    f.write(f"{cmd}\n")

    def write_sbatch(self, output_path: str, setups: list[list[str]]) -> None:
        with open(output_path, "w", encoding="utf8") as f:
            for setup in setups:
                for line in setup:
                    f.write(line)
                    f.write("\n")
                f.write("\n")

            # need a "wait" at the end to run parallel commands with &,
            # otherwise the batch job immediately terminates
            f.write("wait")

    def gen_dict_combinations(self, d: dict):
        # https://stackoverflow.com/questions/50606454/cartesian-product-of-nested-dictionaries-of-lists

        def gen_combinations(d: dict):
            combinations = product(*d.values())
            for c in combinations:
                yield c[0]

        for c in product(*(gen_combinations(v) for v in d.values())):
            yield dict(zip(d.keys(), c))

    def run_bisimulation_test(self, run_setups: pd.DataFrame):
        # get the commands for the two envs you want to compare
        # for the case of 1  seed, this is very easy, just the commands in the list of python_cmds
        processes = []
        for seed in run_setups.seed:
            df_tmp = run_setups.copy()
            df_tmp = df_tmp.loc[df_tmp.seed == seed]

            for i, cmd in enumerate(df_tmp.cmd):
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
            f"{row.time_id}_{row.seed}_{row.scenario}"
            for _, row in df_tmp.iterrows()
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
