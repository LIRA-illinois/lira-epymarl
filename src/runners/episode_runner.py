from typing import Optional
from collections import defaultdict
from os import makedirs
from os.path import join
from functools import partial
import numpy as np
from torch import Tensor
from numpy.typing import NDArray
import matplotlib.image as mpl_img

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2

from utils.record_video import RecordVideoExtended


class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # registering both smac and smacv2 causes a pysc2 error
        # --> dynamically register the needed env
        if self.args.env == "sc2":
            register_smac()
        elif self.args.env == "sc2v2":
            register_smacv2()

        self.env = env_REGISTRY[self.args.env](
            **self.args.env_args,
            common_reward=self.args.common_reward,
            reward_scalarisation=self.args.reward_scalarisation,
        )

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Track stats per comms value during multi-comms evaluation
        self.comms_test_returns = {}  # {comms_val: [returns]}
        self.comms_test_stats = {}  # {comms_val: {stat_name: accumulated_value}}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def start_recording(self, n_test_replays_save: int, video_prefix: str = "replay"):
        # get video folder from wandb logger
        # make the video dir
        replay_dir = join(self.logger.dir, f"replays")
        makedirs(replay_dir, exist_ok=True)
        video_folder = join(replay_dir, f"t_{self.t_env}")
        self.logger.console_logger.info(
            f"Saving {n_test_replays_save} test episode replays to {video_folder}"
        )
        # outputs multiple formats for different uses, webm for browser compatibility and mp4 for Powerpoint
        self.env = RecordVideoExtended(
            env=self.env,
            video_folder=video_folder,
            episode_trigger=lambda e: True,
            name_prefix=video_prefix,
            output_formats=["mp4"],
        )

    def stop_recording(self):
        if isinstance(self.env, RecordVideoExtended):
            self.env.stop_recording()
            if self.logger.save_replays:
                self.logger.log_replays(
                    video_dir=self.env.video_folder, t_env=self.t_env
                )
            self.env = self.env.env
        else:
            pass

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def print_data(self, data: dict):
        print(f"t_ep: {self.t}, hl_state: {data.get('hl_state', None)}")
        # for k, v in data.items():
        #     val = v[0]
        #     if isinstance(val, np.ndarray):
        #         shape = val.shape
        #     elif isinstance(val, list):
        #         shape = (len(val), len(val[0]))
        #     else:
        #         shape = len(val)

        #     print(f"{k}, shape: {shape}, {type(val)}")
        #     print(val)
        #     print()

    def _select_actions(self, test_mode: bool) -> NDArray | tuple:
        """
        choose actions with option to use action space sampler

        Returns
        -------
        NDArray
            array of actions, shape (1, n_agents)
        """

        if self.args.action_selector == "action_space":
            actions = np.expand_dims(np.array(self.env.action_space.sample()), 0)
        else:
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            if hasattr(self.args, "manual_policy"):
                actions = self.manual_policy(actions)

            # following the format from the parallel episode runner
            if isinstance(actions, Tensor):
                actions = actions.cpu().numpy()

        return actions

    def manual_policy(self, actions):
        # class LBFActions(enum.IntEnum):
        #     # matches the order from original LBF action set
        #     STAY = 0
        #     UP = 1
        #     DOWN = 2
        #     LEFT = 3
        #     RIGHT = 4
        #     LOAD = 5

        print(f"t: {self.t}")

        if self.t in [2, 6]:
            # grab
            actions = np.array([[5, 5, 5]])
        else:
            # right
            actions = np.array([[4, 4, 4]])

        # actions = np.array([[3, 2, 3]])

        # right
        # actions = np.array([[4, 4, 4]])

        # if self.t < 5:
        #     actions = np.array([[4, 4, 4]])
        # else:
        #     actions = np.array([[3, 3, 3]])

        # cycle into the goals
        # if self.t == 0:
        #     actions = np.array([[4, 0, 4]])
        # elif self.t == 1:
        #     actions = np.array([[3, 4, 3]])
        # elif self.t % 2 == 0:
        #     actions = np.array([[4, 3, 4]])
        # else:
        #     actions = np.array([[3, 4, 3]])

        # complete 1st task, fail on 2nd task
        # if 0 < self.t <= 9:
        #     # go right
        #     actions["env_actions"] = np.array([[4, 4, 4]])
        # else:
        #     actions["env_actions"] = np.array([[0, 4, 4]])

        # go right
        # actions["env_actions"] = np.array([[4, 4, 4]])
        return actions

    def run(self, test_mode=False, comms_value: Optional[float] = None):
        """
        Run an episode.

        Parameters
        ----------
        test_mode : bool
            Whether to run in test mode (no learning)
        comms_value : float, optional
            Current comms value being evaluated. If provided, stats are tracked
            separately for this value and logged when threshold is reached.
        """
        self.reset()

        terminated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.args.n_agents)
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = self._get_pre_transition_data()
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self._select_actions(test_mode=test_mode)

            # render a frame of the env with the current HL actions chosen
            if isinstance(self.env, RecordVideoExtended):
                self.env.env.render_actions = actions
                self.env._capture_frame()

            if self.args.live_render:
                self._live_render(file_name="pre_step")

            _, reward, terminated, truncated, env_info = self._step(actions)

            terminated = terminated or truncated

            # if self.args.live_render:
            #     self._live_render(file_name="post_step")
            episode_return += reward

            post_transition_data = {
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if isinstance(actions, dict):
                post_transition_data["actions"] = actions["env_actions"]
            else:
                post_transition_data["actions"] = actions

            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            # self.print_data(post_transition_data)
            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1

        last_data = self._get_pre_transition_data()
        # self.print_data(last_data)

        if self.args.live_render:
            self._live_render(file_name="final_state")

        # print("done with ep")

        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self._select_actions(test_mode=test_mode)

        # render a frame of the env with the current HL actions chosen
        if isinstance(self.env, RecordVideoExtended):
            self.env.env.render_actions = actions
            self.env._capture_frame()

        last_actions: dict = {}
        if isinstance(actions, dict):
            last_actions["actions"] = actions["env_actions"]
        else:
            last_actions["actions"] = actions

        self.batch.update(last_actions, ts=self.t)

        # Determine which stats/returns to update
        if not test_mode:
            cur_stats = self.train_stats
            cur_returns = self.train_returns
        else:
            if comms_value is None:
                cur_stats = self.test_stats
                cur_returns = self.test_returns
            else:
                cur_stats = self.comms_test_stats.setdefault(comms_value, {})
                cur_returns = self.comms_test_returns.setdefault(comms_value, [])

        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        # log stats
        if test_mode and comms_value is None:
            # Standard test mode: log when global test_returns reaches threshold
            if len(self.test_returns) == self.args.test_nepisode:
                self._log(cur_returns, cur_stats, log_prefix)

        elif test_mode and comms_value is not None:
            # Multi-comms evaluation: log when this specific comms value reaches threshold
            if len(self.comms_test_returns[comms_value]) == self.args.test_nepisode:
                self._log_comms(comms_value, cur_returns, cur_stats, log_prefix)

        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            # Training mode logging
            self._log(cur_returns, cur_stats, log_prefix)

            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def _live_render(self, file_name: str, actions: Optional[dict] = None):
        render_save_dir = join("results", "live_renders", f"zzz_{self.args.env}")
        makedirs(render_save_dir, exist_ok=True)
        mpl_img.imsave(
            join(render_save_dir, f"{self.t}_{file_name}.png"), self.env.render()
        )
        # state = np.transpose(self.env.unwrapped.grid.encode()[:, :, 0])
        # print("pre transition state")
        # print(state)

    def _step(self, actions):
        if isinstance(actions, dict):
            obs, reward, terminated, truncated, env_info = self.env.step(actions)
        else:
            obs, reward, terminated, truncated, env_info = self.env.step(actions[0])

        return obs, reward, terminated, truncated, env_info

    def _get_pre_transition_data(self) -> dict:
        pre_transition_data = defaultdict(list)

        state = self._get_state()
        avail_actions = self._get_avail_actions()
        obs = self._get_obs()

        if isinstance(state, dict):
            pre_transition_data["hl_state"].append(state["hl_state"])
            pre_transition_data["state"].append(state["ll_state"])
        else:
            pre_transition_data["state"].append(state)

        # simliar for these too if applicable
        pre_transition_data["avail_actions"].append(avail_actions)
        pre_transition_data["obs"].append(obs)

        return pre_transition_data

    def _get_state(self):
        if isinstance(self.env, RecordVideoExtended):
            return self.env.env.get_state()
        else:
            return self.env.get_state()

    def _get_obs(self):
        if isinstance(self.env, RecordVideoExtended):
            return self.env.env.get_obs()
        else:
            return self.env.get_obs()

    def _get_avail_actions(self):
        if isinstance(self.env, RecordVideoExtended):
            return self.env.env.get_avail_actions()
        else:
            return self.env.get_avail_actions()

    def _log(self, returns, stats, prefix) -> None:
        if self.args.common_reward:
            self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
            self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        else:
            for i in range(self.args.n_agents):
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_mean",
                    np.array(returns)[:, i].mean(),
                    self.t_env,
                )
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_std",
                    np.array(returns)[:, i].std(),
                    self.t_env,
                )
            total_returns = np.array(returns).sum(axis=-1)
            self.logger.log_stat(
                prefix + "total_return_mean", total_returns.mean(), self.t_env
            )
            self.logger.log_stat(
                prefix + "total_return_std", total_returns.std(), self.t_env
            )
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()

    def _log_comms(self, comms_value: float, returns, stats, prefix) -> None:
        """
        Log stats for a specific comms value during multi-comms evaluation.
        """
        if self.args.common_reward:
            self.logger.log_stat(
                f"{prefix}return_mean_comms_{comms_value}", np.mean(returns), self.t_env
            )
            self.logger.log_stat(
                f"{prefix}return_std_comms_{comms_value}", np.std(returns), self.t_env
            )
        else:
            for i in range(self.args.n_agents):
                self.logger.log_stat(
                    f"{prefix}agent_{i}_return_mean_comms_{comms_value}",
                    np.array(returns)[:, i].mean(),
                    self.t_env,
                )
                self.logger.log_stat(
                    f"{prefix}agent_{i}_return_std_comms_{comms_value}",
                    np.array(returns)[:, i].std(),
                    self.t_env,
                )
            total_returns = np.array(returns).sum(axis=-1)
            self.logger.log_stat(
                f"{prefix}total_return_mean_comms_{comms_value}",
                total_returns.mean(),
                self.t_env,
            )
            self.logger.log_stat(
                f"{prefix}total_return_std_comms_{comms_value}",
                total_returns.std(),
                self.t_env,
            )

        # Log environment stats
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    f"{prefix}{k}_mean_comms_{comms_value}",
                    v / stats["n_episodes"],
                    self.t_env,
                )

        # Clear for next comms value
        returns.clear()
        stats.clear()
