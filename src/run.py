from types import SimpleNamespace
import datetime
from os import makedirs, listdir
from os.path import dirname, abspath, join, isdir
import pprint
import time
import threading
from types import SimpleNamespace as SN
from typing import Any
from collections import defaultdict
import numpy as np

import torch as th

from controllers import REGISTRY as mac_REGISTRY
from controllers.factored import REGISTRY as factored_mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from learners.factored import REGISTRY as factored_le_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from runners import REGISTRY as r_REGISTRY
from utils.general_reward_support import test_alg_config_supports_reward
from utils.logging import Logger
from utils.timehelper import time_left, time_str


class Simulation:
    def __init__(self, _run, _config, _log) -> None:
        self.args: SN
        self.args = self._parse_config(_config, _log)

        self.logger: Logger
        self._build_logger(self.args, _run, _config, _log)

        self.runner: Any
        self.learner: Any
        self.buffer: Any
        self._build_sim()

        self.run_sim()
        self.finish()

    def run_sim(self) -> None:
        self.hierarchical: bool = hasattr(self.args, "factored_hierarchical_policy")
        self.multi_comms_eval: bool = hasattr(self.args, "multi_comms_eval")

        if self.multi_comms_eval:
            # way of doing this without using the MDP class
            self.comms_values_eval = np.linspace(
                0.0, 1.0, num=self.args.num_comms_values
            ).tolist()

        if self.args.checkpoint_path != "":
            self._load_checkpoint(hierarchical)

            # run evaluation on loaded checkpoint
            if self.args.evaluate or self.args.save_replay:
                self.evaluate_loaded(hierarchical)
                return

        # run training
        if self.hierarchical:
            self.train_hierarchical()

        else:
            self.train_flat()

    def train_hierarchical(self) -> None:
        """
        Two-stage training: first train low-level policy, then use its success rates
        in the high-level agent that interfaces with the HLMDP.
        """
        # Stage 1: Train low-level policy for a single subtask
        self.logger.console_logger.info("=" * 50)
        self.logger.console_logger.info("Training Low-Level Policy")
        self.logger.console_logger.info("=" * 50)

        self.train_flat(get_success_rates=True)

        # evaluate the final low-level policy w/ more samples than you did during training (or whatever)
        # Stage 2: Evaluate low-level policy and build success rate model
        self.logger.console_logger.info("=" * 50)
        self.logger.console_logger.info("Evaluating Low-Level Policy")
        self.logger.console_logger.info("=" * 50)

        ll_task_success_rates = self._evaluate_low_level_policy()
        self.logger.console_logger.info(
            f"Low-level success rates: {ll_task_success_rates}"
        )

        # Stage 3: Train high-level policy with learned success rates
        self.logger.console_logger.info("=" * 50)
        self.logger.console_logger.info("Optimizing High-Level Policy")
        self.logger.console_logger.info("=" * 50)

        self._train_high_level_policy(ll_task_success_rates)

        self.runner.close_env()
        self.logger.console_logger.info("Finished Hierarchical Training")

    def train_flat(self, get_success_rates: bool = False) -> None:
        """
        original EPYMARL training for non-hierarchical policies
        """
        self.logger.console_logger.info("=" * 50)
        self.logger.console_logger.info("Training Policy")
        self.logger.console_logger.info("=" * 50)

        episode = 0
        last_test_t = -self.args.test_interval - 1
        last_log_t = 0
        model_save_time = 0

        start_time = time.time()
        last_time = start_time

        self.logger.console_logger.info(
            "Beginning training for {} timesteps".format(self.args.t_max)
        )

        while self.runner.t_env <= self.args.t_max:
            # Run for a whole episode at a time
            episode_batch = self.runner.run(test_mode=False)
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
            if (self.runner.t_env - last_test_t) / self.args.test_interval >= 1.0:
                self.logger.console_logger.info(
                    f"t_env: {self.runner.t_env} / {self.args.t_max}"
                )
                self.logger.console_logger.info(
                    (
                        "Estimated time left: "
                        f"{time_left(last_time, last_test_t, self.runner.t_env, self.args.t_max)}. "
                        "Time passed: "
                        f"{time_str(time.time() - start_time)}"
                    )
                )

                last_time = time.time()
                last_test_t = self.runner.t_env

                if self.multi_comms_eval:

                    self.evaluate_multi_comms(self.comms_values_eval)

                else:
                    self.evaluate(get_success_rates)

            # save models
            if self.args.save_model and (
                self.runner.t_env - model_save_time >= self.args.save_model_interval
                or model_save_time == 0
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
        if self.args.save_replay_buffer:
            run_id = f"{self.args.time_id}_{self.args.seed}_{self.args.scenario}"
            self.buffer.save(run_id)

        self.runner.close_env()
        self.logger.console_logger.info("Finished Training")

    def evaluate(self, get_success_rates: bool = False) -> None:
        # enable replay saving for some of the test episodes
        # if self.args.save_test_replays:
        #     self.runner.start_recording(self.args.n_test_replays_save)

        task_outcomes = defaultdict(list)

        n_test_eps = max(1, self.args.test_nepisode // self.runner.batch_size)
        for test_ep_idx in range(n_test_eps):
            # if test_ep_idx >= self.args.n_test_replays_save:
            #     self.runner.stop_recording()

            episode_batch = self.runner.run(test_mode=True)

            if get_success_rates:
                # TODO fix this up, actually get this working
                # get the current task and whether it resulted in a success or not
                # AI generated version, kinda wrong, but good starting point
                # hl_states = episode_batch["hl_state"].cpu().numpy()
                terminated = episode_batch["terminated"].cpu().numpy()

                # Track which task was attempted and whether it succeeded
                for ts in range(episode_batch.max_t_filled()):
                    # task_id = int(hl_states[0, ts, 0])  # Assuming first element is task state
                    episode_success = bool(
                        terminated[0, ts]
                    )  # Task success if episode terminated

                    task_outcomes[task_id].append(episode_success)

            # Calculate success rates and log
            for task_id, outcomes in task_outcomes.items():
                success_rate = np.mean(outcomes) if outcomes else 0.0
                ll_task_success_rates[task_id] = success_rate
                self.logger.log_stat(
                    f"test_success_rate_task_{task_id}", success_rate, self.runner.t_env
                )

    def evaluate_loaded(self, hierarchical: bool = False) -> None:
        self.runner.log_train_stats_t = self.runner.t_env

        for _ in range(self.args.test_nepisode):
            self.runner.run(test_mode=True)

        if self.args.save_replay:
            self.runner.save_replay()

        self.runner.close_env()
        self.logger.log_stat("episode", self.runner.t_env, self.runner.t_env)
        self.logger.print_recent_stats()
        self.logger.console_logger.info("Finished Evaluation")

    def evaluate_multi_comms(self, comms_values: list[float]) -> None:
        """
        Evaluate the trained policy across multiple comms allocation values.
        All logging is handled in EpisodeRunner.

        Parameters
        ----------
        comms_values : list[float]
            List of comms values to evaluate (e.g., [0.0, 0.5, 1.0])
        """
        self.logger.console_logger.info("=" * 50)
        self.logger.console_logger.info(
            f"Evaluating Policy Across Comms values: {comms_values}"
        )
        self.logger.console_logger.info("=" * 50)

        for comms_value in comms_values:
            # Run evaluation episodes
            # EpisodeRunner will log stats when test_nepisode threshold is reached
            self.logger.console_logger.info(
                f"Evaluating with comms_value = {comms_value}"
            )

            # Start recording videos for this comms value
            if self.args.save_test_replays:
                self.runner.start_recording(
                    self.args.n_test_replays_save, name_prefix=f"comms_{comms_value:.2f}"
                )

            # run evaluation episodes
            n_test_eps = max(1, self.args.test_nepisode // self.runner.batch_size)

            # Reset stats for this comms value
            # self.runner.reset_comms_stats(comms_value)

           # Update controller with new comms value
            self.runner.mac.update_comms_value(comms_value)

            for test_ep_idx in range(n_test_eps):
                self.runner.run(test_mode=True, comms_value=comms_value)

                # Stop recording after n_test_replays_save episodes
                if (
                    self.args.save_test_replays
                    and test_ep_idx >= self.args.n_test_replays_save - 1
                ):
                    self.runner.stop_recording()

            # Ensure recording is stopped before moving to next comms value
            if self.args.save_test_replays:
                self.runner.stop_recording()

    # helper methods
    def save(self) -> None:
        save_path = join(
            self.args.local_results_path,
            "models",
            self.args.unique_token,
            str(self.runner.t_env),
        )

        # "results/models/{}".format(unique_token)
        makedirs(save_path, exist_ok=True)
        self.logger.console_logger.info("Saving models to {}".format(save_path))

        # learner should handle saving/loading -- delegate actor save/load to mac,
        # use appropriate filenames to do critics, optimizer states
        self.learner.save_models(save_path)

        if self.args.use_wandb and self.args.wandb_save_model:
            for model_name in listdir(save_path):
                self.logger.log_model(
                    save_path=join(save_path, model_name),
                    t_env=self.runner.t_env,
                    model_name=model_name,
                )

    def _load_checkpoint(self, hierarchical: bool = False) -> None:
        timesteps = []
        timestep_to_load = 0

        if not isdir(self.args.checkpoint_path):
            self.logger.console_logger.info(
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

        model_path = join(self.args.checkpoint_path, str(timestep_to_load))

        self.logger.console_logger.info("Loading model from {}".format(model_path))
        self.learner.load_models(model_path)
        self.runner.t_env = timestep_to_load

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
        self.logger = Logger(_log)

        try:
            map_name = _config["env_args"]["map_name"]
        except:
            map_name = _config["env_args"]["key"]

        # run_name has a unique datetime in it, so only inclue curr_time if that is not available
        curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[2:][:-3]
        unique_token = (
            f"{args.run_name if args.run_name != '' else curr_time}_"
            f"{args.env}_{map_name + '_' if map_name != args.env else ''}"
            f"{args.name}_seed_{args.seed}"
        )

        args.unique_token = unique_token
        if args.use_tensorboard:
            tb_logs_direc = join(
                dirname(dirname(abspath(__file__))), "results", "tb_logs"
            )
            tb_exp_direc = join(tb_logs_direc, "{}").format(unique_token)
            self.logger.setup_tb(tb_exp_direc)

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
                save_replays=args.wandb_save_test_replays,
            )

        # sacred is on by default
        if args.use_sacred:
            _log.info("Experiment Parameters:")
            experiment_params = pprint.pformat(_config, indent=4, width=1)
            _log.info("\n\n" + experiment_params + "\n")
            self.logger.setup_sacred(_run)

    def _build_sim(self) -> None:
        # Init runner so we can get env info
        self.runner = r_REGISTRY[self.args.runner](args=self.args, logger=self.logger)

        # Set up schemes and groups here
        env_info = self.runner.get_env_info()
        self.args.n_agents = env_info["n_agents"]
        self.args.n_actions = env_info["n_actions"]
        self.args.state_shape = env_info["state_shape"]

        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }

        # For individual rewards in gymma reward is of shape (1, n_agents)
        if self.args.common_reward:
            scheme["reward"] = {"vshape": (1,)}
        else:
            scheme["reward"] = {"vshape": (self.args.n_agents,)}

        # support separate high-level env that interfaces with low-level env
        if hasattr(self.args, "factored_hierarchical_policy"):
            scheme["hl_state"] = {"vshape": env_info["hl_state_shape"]}

        groups = {"agents": self.args.n_agents}
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=self.args.n_actions)])
        }

        self.buffer = ReplayBuffer(
            scheme=scheme,
            groups=groups,
            buffer_size=self.args.buffer_size,
            max_seq_length=env_info["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu" if self.args.buffer_cpu_only else self.args.device,
        )

        # build controller and learner
        # buffer.scheme has preprocess in it, needed to init these objects
        if hasattr(self.args, "factored_hierarchical_policy"):
            mac = factored_mac_REGISTRY[self.args.factored_mac](
                self.buffer.scheme, groups, self.args
            )
            self.learner = factored_le_REGISTRY[self.args.factored_learner](
                mac, self.buffer.scheme, self.logger, self.args
            )
        else:
            mac = mac_REGISTRY[self.args.mac](self.buffer.scheme, groups, self.args)
            self.learner = le_REGISTRY[self.args.learner](
                mac, self.buffer.scheme, self.logger, self.args
            )

        if self.args.use_cuda:
            self.learner.cuda()

        # Give runner the scheme
        self.runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

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

        # Making sure framework really exits
        # os._exit(os.EX_OK)
