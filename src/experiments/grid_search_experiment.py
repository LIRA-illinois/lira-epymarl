"""
Run a grid search experiment, only uses wandb for logging and does not initialize
a wandb sweep or wandb agents.

Compared to a wandb sweep, assigns a unique index to each scenario and automatically
runs all scenarios in parallel across all available GPUs. Runs each command using the "screen" tool
to allow greater control over running jobs compared to wandb which only allows runs to
be stopped 1 at a time.
"""

from os import environ, makedirs, getcwd
from os.path import join
import math
from itertools import product
import argparse
import subprocess
import datetime
from random import SystemRandom
import yaml

from torch import cuda
import psutil

from slurm_args import SlurmArgs


class GridSearch(object):
    def __init__(self) -> None:
        self.args = self.parse_args()
        self.venv_activate_path = join(".venv", "bin", "activate")
        self.exp_dir = join("experiments", self.args.experiment)
        self.job_dir = join(self.exp_dir, "jobs")
        makedirs(self.job_dir, exist_ok=True)

        self.script_path = join("src", "main.py")

        # setup
        exp_config_path = join(self.exp_dir, "exp_config.yaml")
        with open(exp_config_path, "r", encoding="utf8") as f:
            exp_config: dict = yaml.safe_load(f)

        curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")[:-3]

        if self.args.computer in ["campus", "delta"]:
            slurm_config_path = join(self.exp_dir, "slurm_config.yaml")
            with open(slurm_config_path, "r", encoding="utf8") as f:
                self.slurm_config: dict = yaml.safe_load(f)

            cluster_log_dir: str = join(
                "results", self.args.experiment, curr_time, "cluster_logs"
            )
            makedirs(cluster_log_dir, exist_ok=True)

            self.slurm_config["experiment"] = self.args.experiment
            self.slurm_config["cluster"] = self.args.computer
            self.slurm_config["log_dir"] = cluster_log_dir

        # get seeds
        if exp_config["parameters"].get("seed", False):
            seeds = exp_config["parameters"].pop("seed")["values"]
        else:
            if exp_config["parameters"].get("n_seeds", False):
                # generate n_seeds random seeds to use in this experiment
                n_seeds = exp_config["parameters"].pop("n_seeds")["value"]
            else:
                # default value
                n_seeds = 5

            # true randomness from the OS
            rng = SystemRandom()
            seeds = [rng.randint(0, 1000000) for _ in range(n_seeds)]

        scenarios = [c for c in self.gen_dict_combinations(exp_config["parameters"])]
        scenario_names = [
            f"sc_{scenario_idx+1}" for scenario_idx, _ in enumerate(scenarios)
        ]

        python_cmds = self.get_python_commands(
            scenarios=scenarios,
            seeds=seeds,
            scenario_names=scenario_names,
            script_path=self.script_path,
            curr_time=curr_time,
        )

        self.print_info(scenarios, scenario_names, seeds, python_cmds)

        # check if user wants to run the experiment
        if input("Run experiment now? (y/n)").lower() == "y":
            print("Running experiment")
            if self.args.computer == "lab":
                self.run_experiment_lab(python_cmds)
            else:
                self.run_experiment_cluster(python_cmds)

        else:
            print("Exiting without running experiment")
            exit()

    def get_python_commands(
        self,
        scenarios,
        seeds,
        scenario_names,
        script_path: str,
        curr_time: str,
    ) -> list[str]:

        cmds: list[str] = []

        for scenario_idx, params in enumerate(scenarios):
            _params = params.copy()
            _params["run_name"] = (
                f"{self.args.experiment}_{curr_time}_{scenario_names[scenario_idx]}"
            )

            # options is rl alg and env
            # update is the "with" params (except seed, you add that in manually)
            options = []
            updates = []
            for k, v in _params.items():
                if k in ["config", "env-config"]:
                    options.append(f"--{k}={v}")
                else:
                    updates.append(f"{k}={v}")

            # define the command to be run
            for seed in seeds:
                python_cmd = f"python3 {script_path} {' '.join(options)} with {' '.join(updates)} seed={seed}"
                cmds.append(python_cmd)

        return cmds

    def run_experiment_lab(self, python_cmds: list[str]):
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
            bash_prefix = ["/bin/bash", "-c"]
            run_cmd = [
                *screen_prefix,
                *bash_prefix,
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

    def run_experiment_cluster(self, python_cmds: list[str]):
        # assign commands to each job and run them all in parallel within that job
        max_runs_per_job = self.args.max_runs_per_job
        n_runs = len(python_cmds)
        n_jobs = math.ceil(n_runs / max_runs_per_job)
        jobs = {i: [] for i in range(n_jobs)}

        # assign commands to jobs
        for i, cmd in enumerate(python_cmds):
            jobs[i % n_jobs].append(cmd + " &")

        # loop thru all jobs, get the slurm config args needed to generate the slurm file and generate the slurm file
        for job_idx, cmds in jobs.items():
            cmds.insert(0, "# job commands")

            # slurm config
            self.slurm_config["job_idx"] = job_idx
            slurm_args = SlurmArgs(**self.slurm_config)
            slurm_config_lines = slurm_args.get_config_lines()

            # commands to run this project on the cluster
            project_name = getcwd().split("/")[-1]
            match self.args.computer:
                case "delta":
                    cluster_project_dir = join("~", "dev", f"{project_name}")
                case "campus":
                    cluster_project_dir = join("~", "my_trg_dir", "dev", f"{project_name}")
                case _:
                    raise NotImplementedError

            project_setup_lines: list[str] = [
                "# project setup",
                f"cd {cluster_project_dir}",
                f"source {self.venv_activate_path}",
            ]

            # save the slurm files to disk
            setups: list[list[str]] = [slurm_config_lines, project_setup_lines, cmds]
            output_path = join(self.job_dir, f"job_{job_idx}.slurm")
            self.write_sbatch(output_path=output_path, setups=setups)

            subprocess.run(["sbatch", output_path], check=False)

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
            help="Available GPUs to run this experimet on.",
        )
        parser.add_argument(
            "-r",
            "--n_runners",
            type=int,
            required=False,
            default=20,
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

        return parser.parse_args()

    def print_info(
        self,
        scenarios: list[dict],
        scenario_names: list[str],
        seeds: list[int],
        python_cmds: list[str],
    ):
        """print useful info about the experiment"""
        n_scenarios, n_seeds = len(scenarios), len(seeds)

        if self.args.computer == "lab":
            # available computer resources
            print(
                f"Hardware summary\nUsing {len(self.args.gpus)} GPUs with indices {self.args.gpus}\nVRAM usage"
            )
            byte_to_gb = 1024**3
            for device in self.args.gpus:
                (avail_vram, total_vram) = cuda.memory.mem_get_info(device)
                # convert from bytes to gigabtyes
                avail_vram, total_vram = round(avail_vram / byte_to_gb, 1), round(
                    total_vram / byte_to_gb, 1
                )
                used_vram = round(total_vram - avail_vram, 1)

                props = cuda.get_device_properties(device)

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
        print(f"\nExperiment summary")
        print(
            f"{n_scenarios} scenarios, {n_seeds} seeds per scenario, {n_runs} total runs"
        )
        if self.args.computer == "lab":
            print(
                f"Using {self.args.n_runners} parallel runners with a max of {math.ceil(n_runs / self.args.n_runners)} runs per runner"
            )
        print(f"Seeds: {seeds}")

        table_header = (
            "| Scenario Name | Alg | Env | Params |\n|----| ---- | ---- | ---- |"
        )
        print(table_header)

        for scenario_idx, params in enumerate(scenarios):
            # print params to markdown table
            other_params = ""
            for k, v in params.items():
                if k not in ["config", "env-config"]:
                    other_params += f"{k}={v} "

            table_line = f"| {scenario_names[scenario_idx]} | {params['config']} | {params['env-config']} | {other_params} | "
            print(table_line)
        print("")

        save_path = join(self.exp_dir, "python_cmds.txt")
        with open(save_path, "w") as f:
            for cmd in python_cmds:
                f.write(f"{cmd}\n")

    def write_sbatch(self, output_path: str, setups: list[list[str]]) -> None:
        with open(output_path, "w", encoding="utf8") as f:
            for setup in setups:
                for line in setup:
                    f.write(line)
                    f.write("\n")
                f.write("\n")

    def gen_dict_combinations(self, d: dict):
        # https://stackoverflow.com/questions/50606454/cartesian-product-of-nested-dictionaries-of-lists

        def gen_combinations(d: dict):
            combinations = product(*d.values())
            for c in combinations:
                yield c[0]

        for c in product(*(gen_combinations(v) for v in d.values())):
            yield dict(zip(d.keys(), c))


if __name__ == "__main__":
    GridSearch()
