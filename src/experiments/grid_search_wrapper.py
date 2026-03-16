# Run grid search using a wandb sweep. Assigns wandb agents to GPUs and runs them in parallel
import os
import sys
from itertools import product
import signal
import argparse
import subprocess
import datetime
import yaml
import wandb

# Track the currently running child so we can kill it with SIGTERM/SIGINT
_active_child = None


def _handle_signal(signum, frame):
    """Propagate termination signals to the active child subprocess."""
    if _active_child and _active_child.poll() is None:
        print(
            f"[sweep_wrapper] Received signal {signum}, killing child pid {_active_child.pid}"
        )
        try:
            os.killpg(os.getpgid(_active_child.pid), signal.SIGTERM)
        except OSError:
            _active_child.kill()
    wandb.finish(exit_code=1)
    sys.exit(1)


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


class GridSearch(object):
    def __init__(self) -> None:
        self.args = self.parse_args()
        self.script_path = os.path.join("src", "main.py")

        config_dir = os.path.join("experiments", self.args.experiment)
        config_path = os.path.join(config_dir, "wandb_config.yaml")

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
        n_seeds = len(seeds)

        scenarios = [c for c in self.gen_dict_combinations(self.config["parameters"])]
        n_scenarios = len(scenarios)

        # print useful info about the experiment
        print(
            f"Running {n_scenarios} scenarios, {n_seeds} seeds per scenario, {n_scenarios * n_seeds} total runs"
        )
        print(f"Using {len(self.args.gpus)} GPUs with indices {self.args.gpus}")

        # check if user wants to run the experiment
        run_now = input("Run experiment now? (y/n)")

        if run_now.lower() == "y":
            print("Running experiment")
        else:
            print("Exiting without running experiment")
            exit()

        # print experiment summary in a markdown-formatted table
        header = "| Scenario Name | Alg | Env | Params |\n|----| ---- | ---- | ---- |"
        print(f"\nExperiment summary\n\n{header}")

        # do all of this in parallel
        processes = []
        scenario_idx = 0
        for scenario_idx, params in enumerate(scenarios):
            scenario_name = f"sc_{scenario_idx+1}"
            wandb_run_name = f"{self.args.experiment}_{curr_time}_{scenario_name}"
            screen_name = f"{self.args.experiment}_{scenario_name}"

            # assign scenarios to GPUs
            gpu_idx = scenario_idx % len(self.args.gpus)
            gpu_hardware_idx = self.args.gpus[gpu_idx]
            print(gpu_hardware_idx)

            # read env and rl alg from the config file
            # scenario_param=${scenario_params[$scenario_idx]}
            # rl_alg=${rl_algs[$scenario_idx]}
            # env=${envs[$scenario_idx]}
            # print output to table
            other_params = ""
            for k, v in params.items():
                if k not in ["config", "env-config"]:
                    other_params += f"{k}={v} "

            table_line = f"| {scenario_name} | {params["config"]} | {params["env-config"]} | {other_params} | "
            print(table_line)

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

    def gen_dict_combinations(self, d: dict):
        # https://stackoverflow.com/questions/50606454/cartesian-product-of-nested-dictionaries-of-lists
        keys, values = d.keys(), d.values()

        for c in product(*(self.gen_combinations(v) for v in values)):
            yield dict(zip(keys, c))

    def gen_combinations(self, d: dict):
        # https://stackoverflow.com/questions/50606454/cartesian-product-of-nested-dictionaries-of-lists
        keys, values = d.keys(), d.values()
        combinations = product(*values)

        for c in combinations:
            yield c[0]


if __name__ == "__main__":
    experiment = GridSearch()
