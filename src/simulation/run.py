from types import SimpleNamespace, SimpleNamespace as SN
import datetime
from os import makedirs, listdir, cpu_count
from os.path import join, isdir
import shutil
import time
import threading
from typing import Any
import multiprocessing as mp

import pandas as pd
import matplotlib.pyplot as plt

# use agg backend to support multiprocessing
plt.switch_backend("agg")

import torch as th
import wandb

from .evaluate import run_eval_episodes, eval_worker
from .build import build_sim

from utils.utils import mp_kwargs_wrapper
from utils.general_reward_support import test_alg_config_supports_reward
from utils.logging import MainLogger
from utils.timehelper import time_left, time_str


class Simulation:
    def __init__(self, _run, _config, _log) -> None:
        mp.set_start_method("spawn", force=True)

        self.args: SN
        self.args = self._parse_config(_config, _log)

        self.logger: MainLogger
        self._build_logger(self.args, _run, _config, _log)

        self.runner: Any
        self.learner: Any
        self.buffer: Any
        self.args, self.runner, self.buffer, self.learner = build_sim(
            self.args, self.logger
        )

        self.n_eval_eps = max(1, self.args.test_nepisode // self.runner.batch_size)

    def run_sim(self) -> None:
        hierarchical: bool = hasattr(self.args, "factored_hierarchical_policy")

        if self.args.evaluate:
            self._load_checkpoint(hierarchical)

            # run evaluation on loaded checkpoint
            if self.args.evaluate or self.args.save_replay:
                self.evaluate_loaded()
                return

        # run training
        if hierarchical:
            self.train_hierarchical()

        else:
            self.train_flat()

    def train_flat(self, get_success_rates: bool = False) -> None:
        """
        original EPYMARL training for non-hierarchical policies
        """
        self.logger.info("Training Policy", log_header=True)

        # timing setup
        episode = 0
        last_test_t = 0
        last_log_t = 0
        model_save_time = 0

        start_time = time.time()
        last_time = start_time

        # training loop
        self.logger.info("Beginning training for {} timesteps".format(self.args.t_max))

        while self.runner.t_env <= self.args.t_max:
            # Run for a whole episode at a time
            result = self.runner.run(test_mode=False)
            episode_batch = result["batch"] if isinstance(result, dict) else result
            self.buffer.insert_episode_batch(episode_batch)

            # run a learning update step
            if self.buffer.can_sample(self.args.batch_size):
                episode_sample = self.buffer.sample(self.args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != self.args.device:
                    episode_sample.to(self.args.device)

                self.learner.train(episode_sample, self.runner.t_env, episode)

            # run evaluation episodes
            if self.runner.t_env - last_test_t >= self.args.test_interval:
                self.logger.info(f"t_env: {self.runner.t_env} / {self.args.t_max}")
                self.logger.info(
                    (
                        "Estimated time left: "
                        f"{time_left(last_time, last_test_t, self.runner.t_env, self.args.t_max)}. "
                        "Time passed: "
                        f"{time_str(time.time() - start_time)}"
                    )
                )

                last_time = time.time()
                last_test_t = self.runner.t_env
                self.evaluate(n_eval_eps=self.n_eval_eps)

            # why different logic for the timing of saving models vs evaluating?
            # save model to disk
            if (
                self.args.save_model
                and self.runner.t_env - model_save_time >= self.args.save_model_interval
            ):
                model_save_time = self.runner.t_env
                self.save()

            episode += self.args.batch_size_run

            # log key training / eval metrics
            if (self.runner.t_env - last_log_t) >= self.args.log_interval:
                self.logger.log_stat("episode", episode, self.runner.t_env)
                self.logger.print_recent_stats()
                last_log_t = self.runner.t_env

        # save replay buffer for other post-processing
        # if self.args.save_replay_buffer:
        #     run_id = f"{self.args.time_id}_{self.args.seed}_{self.args.scenario}"
        #     self.buffer.save(run_id)

        self.runner.close_env()
        self.logger.info("Finished Training")

    def train_hierarchical(self) -> None:
        """
        Two-stage training: first train low-level policy, then use its success rates
        in the high-level agent that interfaces with the HLMDP.
        """
        # Stage 1: Train low-level policy for a single subtask
        self.logger.info("Training Low-Level Policy", log_header=True)
        self.train_flat(get_success_rates=True)

        # evaluate the final low-level policy w/ more samples than you did during training (or whatever)
        # Stage 2: Evaluate low-level policy and build success rate model
        self.logger.info("Evaluating Low-Level Policy", log_header=True)

        ll_task_success_rates = self._evaluate_low_level_policy()
        self.logger.info(f"Low-level success rates: {ll_task_success_rates}")

        # Stage 3: Train high-level policy with learned success rates
        self.logger.info("Optimizing High-Level Policy", log_header=True)

        self._train_high_level_policy(ll_task_success_rates)

        self.runner.close_env()
        self.logger.info("Finished Hierarchical Training")

    def evaluate(self, n_eval_eps: int) -> None:
        """evaluation of low-level policy for hierarchical training, or normal evaluation if not using hierarchical training"""
        if hasattr(self.args, "comms_values_eval"):
            self._evaluate_multi_comms(
                self.args.comms_values_eval,
                n_eval_eps,
                parallel_eval=self.args.parallel_comms_eval,
            )
        else:
            self._evaluate_basic(n_eval_eps)

    def _evaluate_basic(self, n_eval_eps: int):
        """evaluate a single, non-hierarchical trained policy"""
        self.logger.info("Evaluating Policy", log_header=True)

        run_eval_episodes(
            args=self.args,
            runner=self.runner,
            n_eval_eps=n_eval_eps,
        )

    def _evaluate_multi_comms(
        self, comms_values: list[float], n_eval_eps: int, parallel_eval: bool = True
    ) -> None:
        """
        Evaluate a trained policy across multiple comms allocation values.

        Parameters
        ----------
        comms_values : list[float]
            List of comms values to evaluate (e.g., [0.0, 0.5, 1.0])
        """
        self.logger.info(
            f"Evaluating Policy Across Comms Values: {comms_values}", log_header=True
        )

        eval_data: list[dict] = []

        if parallel_eval:
            # move to CPU so tensors can be serialized for multiprocessing
            agent_state_dict = {
                k: v.cpu() for k, v in self.runner.mac.agent.state_dict().items()
            }

            wandb_attrs = ["entity", "project", "id", "name"]
            wandb_config = {}
            for attr in wandb_attrs:
                wandb_config[attr] = getattr(self.logger.wandb, attr)

            # prepare inputs for multiprocessing workers
            n_procs = min(len(comms_values), max(1, (cpu_count() or 1) - 1))
            inputs = []
            for comms_value in comms_values:

                input_args = {
                    "function": eval_worker,
                    "comms_value": comms_value,
                    "args": self.args,
                    "n_eval_eps": n_eval_eps,
                    "t_env": self.runner.t_env,
                    "agent_state_dict": agent_state_dict,
                    "logger_dir": self.logger.dir,
                    "wandb_config": wandb_config,
                }
                inputs.append(input_args)

            # run multiprocessing
            with mp.Pool(processes=n_procs) as pool:
                results: list[dict] = list(pool.map(mp_kwargs_wrapper, inputs))

            eval_data = [res["log_stats"] for res in results]

        else:
            # Serial evaluation using shared helper
            for comms_value in comms_values:
                self.logger.info(f"Evaluating with comms_value = {comms_value}")

                result = run_eval_episodes(
                    args=self.args,
                    runner=self.runner,
                    n_eval_eps=n_eval_eps,
                    t_env=self.runner.t_env,
                    comms_value=comms_value,
                )

                eval_data.append(result["log_stats"])

        # Convert to DataFrame and log
        df_eval = pd.DataFrame.from_records(eval_data)

        self.logger.log_stat_table(df_eval, t=self.runner.t_env)
        self._make_comms_eval_plots(self.logger.data_table, t=self.runner.t_env)

    def _make_comms_eval_plots(self, data_table: wandb.Table, t: int) -> None:
        """Make plots for comms evaluation.

        Plots each metric in `cols` vs `t_env` for every comms value present
        (or provided in `comms_values`) and logs images to wandb if enabled.
        """
        df = data_table.get_dataframe()

        # Columns to plot (exclude t_env as it's the x axis)
        cols = [
            "test_return_mean",
            "test_return_std",
            "test_task_completed_mean",
            "test_ep_length_mean",
        ]
        comms_values = sorted(df["comms_value"].unique())

        for col in cols:
            plt.figure()
            for comms_value in comms_values:
                df_plot = df[df.get("comms_value") == comms_value]
                label = f"Comms: {comms_value}"
                plt.plot(df_plot["t_env"], df_plot[col], marker="o", label=label)

            plt.xlabel("t_env")
            plt.ylabel(col)
            plt.title(f"{col}")
            plt.legend()
            plt.grid(True)

            save_dir = join(self.logger.dir, "figures", f"t_{t}")
            makedirs(save_dir, exist_ok=True)
            save_path = join(save_dir, f"comms_eval_{col}.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            self.logger.log_image(
                column_name=col, image_path=save_path, t=int(self.runner.t_env)
            )

    def evaluate_loaded(self) -> None:
        """probably doesn't work given new eval functions"""
        self.runner.log_train_stats_t = self.runner.t_env

        self.evaluate(n_eval_eps=self.n_eval_eps)

        self.runner.close_env()
        self.logger.log_stat("episode", self.runner.t_env, self.runner.t_env)
        self.logger.print_recent_stats()
        self.logger.info("Finished Evaluation")

    def save(self) -> None:
        model_dir = join(
            self.args.local_results_path,
            "models",
        )
        save_path = join(
            model_dir,
            self.args.unique_token,
            str(self.runner.t_env),
        )

        # "results/models/{}".format(unique_token)
        makedirs(save_path, exist_ok=True)
        self.logger.info("Saving models to {}".format(save_path))

        # learner should handle saving/loading -- delegate actor save/load to mac,
        # use appropriate filenames to do critics, optimizer states
        self.learner.save_models(save_path)

        if self.args.use_wandb and self.args.wandb_save_model:
            self.logger.log_agent(
                save_path=save_path,
                t=self.runner.t_env,
            )

        # models are saved locally and on the wandb server
        # as wandb artifacts and can be accessed with the wandb API
        if self.args.delete_local_models:
            shutil.rmtree(model_dir, ignore_errors=True)

    def _load_checkpoint(self, hierarchical: bool = False) -> None:
        # get load time step for both cases
        timesteps = []
        timestep_to_load = 0

        if self.args.eval_run_id is not None:
            artifacts = self.logger.wandb_inactive.logged_artifacts()

            # go thru metadata and get all time steps saved
            agent_artifacts = []
            for artifact in artifacts:
                # Check if this artifact is a model and has the correct step in its metadata
                if artifact.type == "agent":
                    agent_artifacts.append(artifact)
                    timesteps.append(artifact.metadata["t_env"])
        else:
            if not isdir(self.args.checkpoint_path):
                self.logger.info(
                    "Checkpoint directiory {} doesn't exist".format(
                        self.args.checkpoint_path
                    )
                )
                return

            # Go through all files in args.checkpoint_path
            for name in listdir(self.args.checkpoint_path):
                full_name = join(self.args.checkpoint_path, name)
                # Check if they are dirs the names of which are numbers
                if isdir(full_name) and name.isdigit():
                    timesteps.append(int(name))

        if self.args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(
                timesteps, key=lambda x: abs(x - self.args.load_step)
            )

        if self.args.eval_run_id is not None:
            for artifact in agent_artifacts:
                if artifact.metadata["t_env"] == timestep_to_load:
                    model_path = artifact.download()
        else:
            model_path = join(self.args.checkpoint_path, str(timestep_to_load))

        self.logger.info(f"Loading model from t={timestep_to_load} ({model_path})")
        self.learner.load_models(model_path)
        self.runner.t_env = timestep_to_load

        if self.args.eval_run_id is not None:
            # clean up local files that have been loaded into memory
            shutil.rmtree("artifacts", ignore_errors=True)

    def _parse_config(self, _config, _log) -> SimpleNamespace:
        _config = self._args_sanity_check(_config, _log)

        args = SN(**_config)
        args.device = "cuda" if args.use_cuda else "cpu"
        assert test_alg_config_supports_reward(
            args
        ), "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."

        return args

    def _args_sanity_check(self, config, _log):
        # set CUDA flags
        # config["use_cuda"] = True # Use cuda whenever possible!
        if config["use_cuda"] and not th.cuda.is_available():
            config["use_cuda"] = False
            _log.warning(
                "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
            )

        if config["test_nepisode"] < config["batch_size_run"]:
            config["test_nepisode"] = config["batch_size_run"]
        else:
            config["test_nepisode"] = (
                config["test_nepisode"] // config["batch_size_run"]
            ) * config["batch_size_run"]

        return config

    def _build_logger(self, args: SimpleNamespace, _run, _config, _log) -> None:
        # setup logger
        self.logger = MainLogger(_log)

        if hasattr(_config["env_args"], "map_name"):
            map_name = _config["env_args"]["map_name"]
        else:
            map_name = _config["env_args"]["key"]

        # run_name has a unique datetime in it, so only include curr_time if that is not available
        curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[2:][:-3]
        unique_token = (
            f"{args.run_name if args.run_name != '' else curr_time}_"
            f"{args.env}_{map_name + '_' if map_name != args.env else ''}"
            f"{args.name}_seed_{args.seed}"
        )

        args.unique_token = unique_token

        if args.use_wandb:
            if args.run_name != "":
                run_name = args.run_name
            else:
                if args.wandb_group != "":
                    run_name = args.wandb_group + f"_seed_{args.seed}"
                elif args.time_id != "" and args.scenario != "":
                    run_name = f"{args.time_id}_sc_{args.scenario}_seed_{args.seed}"
                else:
                    run_name = unique_token

            self.logger.setup_wandb(
                config=_config,
                team_name=args.wandb_team,
                project_name=args.wandb_project,
                group_name=args.run_name,
                run_name=run_name,
                mode=args.wandb_mode,
                eval_run_id=args.eval_run_id,
            )

        # deprecated, use wandb
        # sacred is on by default
        # if args.use_sacred:
        #     _log.info("Experiment Parameters:")
        #     experiment_params = pprint.pformat(_config, indent=4, width=1)
        #     _log.info("\n\n" + experiment_params + "\n")
        #     self.logger.setup_sacred(_run)
        # if args.use_tensorboard:
        #     tb_logs_direc = join(
        #         dirname(dirname(abspath(__file__))), "results", "tb_logs"
        #     )
        #     tb_exp_direc = join(tb_logs_direc, "{}").format(unique_token)
        #     self.logger.setup_tb(tb_exp_direc)

    def finish(self) -> None:
        # Finish logging
        self.logger.finish()

        # Clean up after finishing
        print("Exiting Main")

        print("Stopping all threads")
        for t in threading.enumerate():
            if t.name != "MainThread":
                print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
                t.join(timeout=1)
                print("Thread joined")

        print("Exiting script")
