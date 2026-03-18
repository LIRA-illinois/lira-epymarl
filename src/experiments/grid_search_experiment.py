"""
Run a grid search experiment, only uses wandb for logging and does not initialize a wandb sweep or wandb agents.
Compared to a wandb sweep, assigns a unique index to each scenario and automatically runs all scenarios in parallel across all available GPUs
"""


import os
import sys
from itertools import product
import argparse
import subprocess
import datetime
import yaml

from torch import cuda
import psutil

class GridSearch(object):
    def __init__(self) -> None:
        self.args = self.parse_args()
        self.script_path = os.path.join("src", "main.py")

        config_dir = os.path.join("experiments", self.args.experiment)
        config_path = os.path.join(config_dir, "exp_config.yaml")

        with open(config_path, "r") as f:
            self.config: dict = yaml.safe_load(f)

        curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")[:-3]

        self.run_experiment(curr_time)

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
        return parser.parse_args()

    def run_experiment(self, curr_time):
        seeds = self.config["parameters"].pop("seed")["values"]
        scenarios = [c for c in self.gen_dict_combinations(self.config["parameters"])]
        scenario_names = [
            f"sc_{scenario_idx+1}" for scenario_idx, _ in enumerate(scenarios)
        ]

        self.print_info(scenarios, scenario_names, seeds)

        # check if user wants to run the experiment
        if input("Run experiment now? (y/n)").lower() == "y":
            print("Running experiment")
        else:
            print("Exiting without running experiment")
            exit()

        # do all runs in parallel
        processes = []
        for scenario_idx, params in enumerate(scenarios):
            scenario_name = scenario_names[scenario_idx]
            wandb_run_name = f"{self.args.experiment}_{curr_time}_{scenario_name}"
            screen_name = f"{self.args.experiment}_{scenario_name}"

            # assign scenarios to GPUs
            gpu_idx = scenario_idx % len(self.args.gpus)
            gpu_hardware_idx = self.args.gpus[gpu_idx]

            for seed in seeds:
                p = self.run_seed(
                    seed,
                    script_path=self.script_path,
                    params=params,
                    gpu_hardware_idx=gpu_hardware_idx,
                    run_name=wandb_run_name,
                    screen_name=screen_name,
                )
                processes.append(p)

        for p in processes:
            p.wait()

    def run_seed(
        self,
        seed,
        script_path,
        params,
        gpu_hardware_idx,
        run_name,
        screen_name,
    ):
        """Run a single training seed as a subprocess"""
        params_local = params.copy()
        params_local["run_name"] = run_name

        # options is rl alg and env
        # update is the "with" params (except seed, you add that in manually)
        options = []
        updates = []

        for k, v in params_local.items():
            if k in ["config", "env-config"]:
                options.append(f"--{k}={v}")
            else:
                updates.append(f"{k}={v}")

        env = {k: v for k, v in os.environ.items()}
        env["CUDA_VISIBLE_DEVICES"] = f"{gpu_hardware_idx}"

        screen_prefix = ["screen", "-dmS", screen_name]
        python_cmd = f"source .venv/bin/activate; {sys.executable} {script_path} {' '.join(options)} with {' '.join(updates)} seed={seed}"

        cmd = [
            *screen_prefix,
            "/bin/bash",
            "-c",
            python_cmd,
        ]

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        return proc

    def print_info(
        self, scenarios: list[dict], scenario_names: list[str], seeds: list[int]
    ):
        """print useful info about the experiment"""
        n_scenarios, n_seeds = len(scenarios), len(seeds)

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
        print(f"\nExperiment summary")
        print(
            f"Running {n_scenarios} scenarios, {n_seeds} seeds per scenario, {n_scenarios * n_seeds} total runs"
        )
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

            table_line = f"| {scenario_names[scenario_idx]} | {params["config"]} | {params["env-config"]} | {other_params} | "
            print(table_line)
        print("")

    def gen_dict_combinations(self, d: dict):
        # https://stackoverflow.com/questions/50606454/cartesian-product-of-nested-dictionaries-of-lists

        def gen_combinations(d: dict):
            combinations = product(*d.values())
            for c in combinations:
                yield c[0]

        for c in product(*(gen_combinations(v) for v in d.values())):
            yield dict(zip(d.keys(), c))


if __name__ == "__main__":
    experiment = GridSearch()
