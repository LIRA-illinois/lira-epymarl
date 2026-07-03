import datetime
import multiprocessing as mp
import threading
import time
from os import listdir, makedirs, walk
from os.path import abspath, isdir, join, splitext
from shutil import rmtree
from types import SimpleNamespace as SN
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import wandb

from src.simulation.build import build_sim
from src.simulation.evaluate import run_eval_episodes
from src.utils.general_reward_support import test_alg_config_supports_reward
from src.utils.logging import MainLogger, log_setup
from src.utils.timehelper import time_left, time_str

# use agg backend to support multiprocessing
plt.switch_backend("agg")


class Simulation:
    def __init__(self, _config, _log) -> None:
        mp.set_start_method("spawn", force=True)
        args: SN = self._parse_config(_config, _log)

        self.logger: MainLogger = self._build_logger(args, _config, _log)
        self.args, self.runner, self.buffer, self.learner = build_sim(args, self.logger)
        self.n_eval_eps = max(1, self.args.test_nepisode // self.runner.batch_size)

    def run(self) -> None:
        # hierarchical: bool = getattr(self.args, "factored_hierarchical_policy", False)

        if self.args.evaluate:
            self._load_checkpoint()

            # run evaluation on loaded checkpoint
            if self.args.evaluate or self.args.save_replay:
                self.evaluate_loaded()
                return

        elif hasattr(self.args, "post_processing"):
            assert self.args.post_processing in [
                "optimize_high_level_policy",
                "aggregate_runs",
            ]

            if self.args.post_processing == "optimize_high_level_policy":
                df_data = self._load_experiment_artifacts(
                    art_name="eval_stats",
                    art_version="latest",
                    art_type="run_table",
                )
                self.logger.info("Optimizing High-Level Policy", log_header=True)
                self._train_high_level_policy(df_data)
                print(self.runner.mac.comms_agent.policy.task_policy)
                print(self.runner.mac.comms_agent.policy.comms_policy)
                self.runner.close_env()
                return

            elif self.args.post_processing == "aggregate_runs":
                df_data, runs = self._load_experiment_artifacts(
                    art_name="eval_stats",
                    art_version="latest",
                    art_type="run_table",
                    return_runs=True,
                )

                # average results across seeds within each scenario
                df_avg = (
                    df_data.groupby(
                        ["scenario", "msg_budget_per_agent", "t_env_rounded"],
                        dropna=False,
                    )
                    .mean(numeric_only=True)
                    .reset_index()
                )

                for idx, (scenario, df_scenario) in enumerate(
                    df_avg.groupby("scenario"), start=np.min(df_avg.scenario)
                ):
                    self.logger.info(
                        f"Plotting aggregated eval stats for scenario {scenario}",
                    )
                    scenario_table = wandb.Table(dataframe=df_scenario)

                    # choose a run from this scenario to resume and log the plot
                    scenario_runs = [
                        run
                        for run in runs
                        if int(run.config.get("scenario", -1)) == int(scenario)
                    ]

                    wandb_run = wandb.init(
                        entity=getattr(scenario_runs[0], "entity", None),
                        project=getattr(scenario_runs[0], "project", None),
                        id=scenario_runs[0].id,
                        resume="allow",
                    )

                    self._make_comms_eval_plots(
                        scenario_table,
                        t=np.max(df_avg.t_env_rounded),
                        wandb_run=wandb_run,
                    )

                    wandb_run.finish()

                return

        # # run training
        # if hierarchical:
        #     self.train_hierarchical()

        # else:

        hl_task = getattr(self.args, "hl_task", None)
        reset_options = None
        if hl_task is not None:
            reset_options = {"hl_start_state": hl_task[0], "hl_task": hl_task}

        self.train_single_task(reset_options=reset_options)

    def train_single_task(self, reset_options: Optional[dict] = None) -> None:
        """
        original EPYMARL training for non-hierarchical policies for a project with a single task
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

        if getattr(self.args, "unique_policy_per_msg_budget", False):
            self.runner.mac.msg_budget_per_agent = (
                self.args.msg_budget_per_agent[0]
            )

        while self.runner.t_env <= self.args.t_max:
            # Run for a whole episode at a time
            result = self.runner.run(test_mode=False, reset_options=reset_options)

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
                self.evaluate(n_eval_eps=self.n_eval_eps, reset_options=reset_options)

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

        self.runner.close_env()
        self.logger.info("Finished Training")

    def _train_high_level_policy(self, df_data: pd.DataFrame) -> None:
        self.runner.env.hlmdp.transition_probs = df_data
        self.learner.optimize_hl_agent(
            self.runner.env.hlmdp, self.args.success_rate_spec
        )

    def evaluate(self, n_eval_eps: int, reset_options: Optional[dict] = None) -> None:
        """Evaluation entry point."""

        # always comms sweep if hierarchical or not
        if hasattr(self.args, "msg_budget_per_agent"):
            msg_budget_per_agent_list = self.args.msg_budget_per_agent
            self.logger.info(
                f"Evaluating Policy Across Message Budgets: {msg_budget_per_agent_list}",
                log_header=True,
            )

            eval_data: list[dict] = []

            # update the env's rng state and set it to the same value at the start of each comms value
            # used to generate the exact same initial env layouts so comms is the only variable that changes
            # this only works if np_random is not used in env.step() since the generator's state changes
            # every time it generates a random number
            env_rng = self.runner.env.get_wrapper_attr("np_random")
            init_rng_state = env_rng.bit_generator.state

            for budget in msg_budget_per_agent_list:
                env_rng.bit_generator.state = init_rng_state

                self.logger.info(f"Evaluating with msg_budget_per_agent = {budget}")

                if reset_options is None:
                    reset_options = {"msg_budget_per_agent": budget}
                else:
                    reset_options["msg_budget_per_agent"] = budget

                result = run_eval_episodes(
                    args=self.args,
                    runner=self.runner,
                    n_eval_eps=n_eval_eps,
                    t_env=self.runner.t_env,
                    reset_options=reset_options,
                )
                eval_data.append(result["log_stats"])

            df_eval = pd.DataFrame.from_records(eval_data)

            self.logger.log_table(key="eval_stats", value=df_eval, t=self.runner.t_env)
            self._make_comms_eval_plots(
                self.logger.data_tables["eval_stats"], t=self.runner.t_env
            )

            """
            if self.args.parallel_comms_eval:
                agent_state_dict = {k: v.cpu() for k, v in self.runner.mac.agent.state_dict().items()}
                wandb_attrs = ["entity", "project", "id", "name"]
                wandb_config = {attr: getattr(self.logger.wandb, attr) for attr in wandb_attrs}

                n_procs = getattr(self.args, "max_parallel_eval_processes", min(len(msg_budget_per_agents), max(1, (cpu_count() or 1) - 1)))

                inputs = []
                for msg_budget_per_agent in msg_budget_per_agents:
                    input_args = {
                        "function": eval_worker,
                        "args": self.args,
                        "n_eval_eps": n_eval_eps,
                        "t_env": self.runner.t_env,
                        "agent_state_dict": agent_state_dict,
                        "logger_dir": self.logger.dir,
                        "wandb_config": wandb_config,
                        "reset_options": {"msg_budget_per_agent": msg_budget_per_agent},
                    }
                    inputs.append(input_args)

                with mp.Pool(processes=n_procs, maxtasksperchild=2) as pool:
                    results: list[dict] = list(pool.map(mp_kwargs_wrapper, inputs))

                eval_data = [res["log_stats"] for res in results]

            else:
            """

        # non-hierarchical evaluation w/ no comms sweep
        else:
            self.logger.info("Evaluating Policy", log_header=True)
            run_eval_episodes(args=self.args, runner=self.runner, n_eval_eps=n_eval_eps)

        """
        # Hierarchical env handling
        only needed if you want to eval multiple tasks in one process
        if hasattr(self.runner.env, "hlmdp"):
            hlmdp = self.runner.env.hlmdp

            # goal-conditioned approach, 1 policy for multiple tasks, not quite there yet
            # # Full sweep across all HL actions
            # if reset_options is None:
            #     df_actions = hlmdp.transition_probs.copy()
            #     df_actions = df_actions.loc[df_actions.state_type == "normal"]
            #     hl_actions = df_actions.action.drop_duplicates().tolist()

            #     self.logger.info(f"Evaluating Policy Across Tasks", log_header=True)

            #     eval_data: list[dict] = []
            #     for action in hl_actions:
            #         state = df_actions.loc[df_actions.action == action, "state"].unique().item()
            #         chosen_next_state, comms_val = action
            #         ro = {"hl_start_state": int(state), "msg_budget_per_agent": comms_val}

            #         result = run_eval_episodes(
            #             args=self.args,
            #             runner=self.runner,
            #             n_eval_eps=n_eval_eps,
            #             t_env=self.runner.t_env,
            #             reset_options=ro,
            #         )

            #         eval_data.append(result["log_stats"])

            #         # update transition probs
            #         success_rate = result["log_stats"].get("test_task_completed_mean")
            #         df = hlmdp.transition_probs
            #         df.loc[(df.action == action) & (df.next_state == chosen_next_state), "prob"] = success_rate
            #         df.loc[(df.action == action) & (df.next_state != chosen_next_state), "prob"] = (1.0 - success_rate)

            #     df_eval = pd.DataFrame.from_records(eval_data)
            #     self.logger.log_table(df_eval, t=self.runner.t_env)

            # Single-task evaluation
            result = run_eval_episodes(
                args=self.args,
                runner=self.runner,
                n_eval_eps=n_eval_eps,
                t_env=self.runner.t_env,
                reset_options=reset_options,
            )

            df_data = pd.DataFrame.from_records([result["log_stats"]])
            self.logger.log_table(df_data, t=self.runner.t_env)

            # Optionally update HLMDP if caller provided the exact action tuple
            action = reset_options.get("action") if reset_options is not None else None
            if action is not None:
                success_rate = result["log_stats"].get("test_task_completed_mean")
                df = hlmdp.transition_probs
                chosen_next_state, _ = action
                df.loc[(df.action == action) & (df.next_state == chosen_next_state), "prob"] = success_rate
                df.loc[(df.action == action) & (df.next_state != chosen_next_state), "prob"] = (1.0 - success_rate)
        """

    def _evaluate_all_tasks(
        self,
        hlmdp,
        n_eval_eps: int,
    ) -> None:
        """
        Evaluate the policy starting from every non-terminating HLMDP state.

        Parameters
        ----------
        hlmdp : ProjectMDP
            High-level MDP instance used by the hierarchical environment.
        """
        # gather all normal-state outgoing actions (tuples)
        df_actions = hlmdp.transition_probs.copy()
        df_actions = df_actions.loc[df_actions.state_type == "normal"]

        # unique action tuples: (chosen_next_state, comms_val)
        hl_actions = df_actions.action.drop_duplicates().tolist()

        self.logger.info("Evaluating Policy Across Tasks", log_header=True)

        eval_data: list[dict] = []

        # For each unique HL action, run evals from the state it goes out of
        for action in hl_actions:
            # action is expected to be a tuple (chosen_next_state, comms_val)
            state = df_actions.loc[df_actions.action == action, "state"].unique().item()
            chosen_next_state, message_budget = action
            reset_options = {
                "hl_start_state": int(state),
                "msg_budget_per_agent": message_budget,
            }

            # set comms value if provided
            result = run_eval_episodes(
                args=self.args,
                runner=self.runner,
                n_eval_eps=n_eval_eps,
                t_env=self.runner.t_env,
                reset_options=reset_options,
            )

            eval_data.append(result["log_stats"])

            # update transition probs in hlmdp, 2 possible outcomes of task success or failure
            df = hlmdp.transition_probs
            success_rate = result["log_stats"].get("test_task_completed_mean")
            df.loc[
                (df.action == action) & (df.next_state == chosen_next_state), "prob"
            ] = success_rate
            df.loc[
                (df.action == action) & (df.next_state != chosen_next_state), "prob"
            ] = 1.0 - success_rate

        df_eval = pd.DataFrame.from_records(eval_data)
        self.logger.log_table(key="eval_stats", value=df_eval, t=self.runner.t_env)

        # TODO it may make sense to log each tasks's success rate to wandb too

        # if msg_budget_per_agents is not None:
        #     self._make_comms_eval_plots(self.logger.data_table, t=self.runner.t_env)

    def _evaluate_multi_comms(
        self,
        msg_budget_per_agents: list[float],
        n_eval_eps: int,
        parallel_eval: bool = True,
    ) -> None:
        """
        Evaluate a trained policy across multiple comms allocation values.

        Parameters
        ----------
        msg_budget_per_agents : list[float]
            List of comms values to evaluate (e.g., [0.0, 0.5, 1.0])
        """
        self.logger.info(
            f"Evaluating Policy Across Comms Values: {msg_budget_per_agents}",
            log_header=True,
        )

        eval_data: list[dict] = []

        # Serial evaluation
        if not parallel_eval:
            for mb in msg_budget_per_agents:
                self.logger.info(f"Evaluating with msg_budget_per_agent = {mb}")

                result = run_eval_episodes(
                    args=self.args,
                    runner=self.runner,
                    n_eval_eps=n_eval_eps,
                    t_env=self.runner.t_env,
                    reset_options={"msg_budget_per_agent": mb},
                )

                eval_data.append(result["log_stats"])

        # Convert to DataFrame and log
        df_eval = pd.DataFrame.from_records(eval_data)
        self.logger.log_table(key="eval_stats", value=df_eval, t=self.runner.t_env)
        self._make_comms_eval_plots(
            self.logger.data_tables["eval_stats"], t=self.runner.t_env
        )

    def _make_comms_eval_plots(
        self,
        data_table: wandb.Table,
        t: int,
        wandb_run=None,
    ) -> None:
        """Make plots for comms evaluation.

        Plots each metric in `cols` vs `t_env` for every comms value present
        (or provided in `msg_budget_per_agents`) and logs images to wandb if enabled.
        """
        df = data_table.get_dataframe()
        save_dir = abspath(join(self.logger.dir, "images", f"t_{t}"))
        makedirs(save_dir, exist_ok=True)

        # Columns to plot (exclude t_env as it's the x axis)
        cols = [
            "test_return_mean",
            "test_return_std",
            "test_task_completed_mean",
            "test_ep_length_mean",
        ]
        msg_budget_per_agents = sorted(df["msg_budget_per_agent"].unique())

        for col in cols:
            plt.figure()
            for idx, msg_budget_per_agent in enumerate(msg_budget_per_agents):
                df_plot = df[
                    df.get("msg_budget_per_agent") == msg_budget_per_agent
                ].copy()
                label = f"Comms: {msg_budget_per_agent}"
                n_samples = df_plot["test_n_episodes"].astype(int)

                # show N in legend title using first row's n
                n_samples = int(n_samples.iloc[0]) if len(n_samples) > 0 else 0

                plt.plot(
                    df_plot["t_env"],
                    df_plot[col],
                    marker="o",
                    alpha=1.0,
                    label=label,
                )
                legend_title = f"samples={n_samples}"

            plt.xlabel("t_env")
            plt.ylabel(col)
            plt.title(f"{col}")

            plt.legend(title=legend_title)
            plt.grid(True)

            save_path = join(save_dir, f"comms_eval_{col}.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        # log all images in the image dir
        if wandb_run is not None:
            for _, _, files in walk(save_dir):
                for file in files:
                    data = log_setup(self.logger.step_metric, t)
                    path = join(save_dir, file)
                    fn = (splitext(file)[0],)
                    data[f"comms_eval_aggregated/{fn}{self.logger.log_suffix}"] = (
                        wandb.Image(path)
                    )
                    wandb_run.log(data=data)
            return

        self.logger.log_images(save_dir, t=self.runner.t_env, group="comms_eval/")

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

        if self.args.use_wandb:
            self.logger.log_agent(
                save_path=save_path,
                t=self.runner.t_env,
            )

        # models are saved locally and on the wandb server
        # as wandb artifacts and can be accessed with the wandb API
        if self.args.delete_local_models:
            rmtree(model_dir, ignore_errors=True)

    def _load_experiment_artifacts(
        self,
        art_name: str = "eval_stats",
        art_version: str = "latest",
        art_type: str = "run_table",
        return_runs: bool = False,
    ):
        api = wandb.Api()

        # load all runs w/ the given time_id and get eval stats tables
        runs = api.runs(
            self.args.wandb_project,
            filters={"config.time_id": self.args.time_id},
        )

        if len(runs) == 0:
            self.logger.info(f"No wandb runs found for time_id={self.args.time_id}")
            return

        run_ids = [run.id for run in runs]
        self.logger.info(f"Time ID: {self.args.time_id}")
        self.logger.info(f"Loading {len(run_ids)} runs with ids: {run_ids}")

        dfs = []

        for i, wandb_run in enumerate(runs):
            print(f"Run {i} / {len(runs)}")

            data = api.artifact(
                name=join(
                    wandb_run.entity,
                    wandb_run.project,
                    f"run-{wandb_run.id}-{art_name}:{art_version}",
                ),
                type=art_type,
            ).get(art_name)

            df = data.get_dataframe()
            df["scenario"] = int(wandb_run.config["scenario"])
            dfs.append(df)

        df_data = pd.concat(dfs, ignore_index=True)
        # round to the nearest eval time since different seeds eval at slightly different times
        df_data["t_env_rounded"] = (
            df_data["t_env"] / wandb_run.config.get("test_interval")
        ).round() * wandb_run.config.get("test_interval")

        df_data.sort_values("scenario").reset_index(drop=True, inplace=True)

        if return_runs:
            return df_data, runs
        else:
            return df_data

    def _load_checkpoint(self) -> None:
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
            rmtree("artifacts", ignore_errors=True)

    def _parse_config(self, _config, _log) -> SN:
        _config = self._args_sanity_check(_config, _log)

        args = SN(**_config)
        args.device = "cuda" if args.use_cuda else "cpu"
        assert test_alg_config_supports_reward(args), (
            "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."
        )

        # update for parallel comms eval, can't be done offline
        # due to parallel to a single wandb run on a remote server
        if args.parallel_comms_eval:
            args.wandb_mode = "shared"

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

    def _build_logger(self, args: SN, _config, _log) -> MainLogger:
        # get unique token for this run
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

        # logger setup
        return MainLogger(_log, config=_config, args=args)

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

    # def train_hierarchical(self) -> None:
    #     """
    #     Two-stage training: first train low-level policy, then use its success rates
    #     in the high-level agent that interfaces with the HLMDP.
    #     """

    #     # Train low-level policy for a single task
    #     self.logger.info("Training Low-Level Policy", log_header=True)

    #     self.train_single_task()

    #     # may need to eval here for more samples than during training to get good statistical estimates of init state dists
    #     # only really needed for the dependent tasks

    #     # # Train high-level policy with learned success rates
    #     self.logger.info("Optimizing High-Level Policy", log_header=True)
    #     self._train_high_level_policy(self.runner.env.hlmdp)

    #     # evaluate Hl policy (only relevant for dependent tasks)

    #     self.runner.close_env()
    #     self.logger.info("Finished Hierarchical Training")
