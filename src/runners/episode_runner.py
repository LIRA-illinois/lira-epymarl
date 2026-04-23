from os import makedirs
from os.path import join
from functools import partial
import numpy as np
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

    def start_recording(self, n_test_replays_save: int):
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
            name_prefix="replay",
            output_formats=["webm", "mp4"],
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
        for k, v in data.items():
            val = v[0]
            if isinstance(val, np.ndarray):
                shape = val.shape
            elif isinstance(val, list):
                shape = (len(val), len(val[0]))
            else:
                shape = len(val)

            print(f"{k}, shape: {shape}, {type(val)}")
            print(val)
            print()

    def select_actions(self, test_mode: bool) -> NDArray:
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
            # following the format from the parallel episode runner
            actions = actions.cpu().numpy()
        return actions

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.args.n_agents)
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.get_state()],
                "avail_actions": [self.get_avail_actions()],
                "obs": [self.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)
            # self.print_data(pre_transition_data)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            # actions = self.select_actions(test_mode=test_mode)

            if self.args.live_render:
                render_save_dir = join("results", "live_renders", f"zzz_{self.args.env}")
                makedirs(render_save_dir, exist_ok=True)
                mpl_img.imsave(
                    join(render_save_dir, f"{self.t}_pre_step.png"), self.env.render()
                )
                # state = np.transpose(self.env.unwrapped.grid.encode()[:, :, 0])
                # print("pre transition state")
                # print(state)

            _, reward, terminated, truncated, env_info = self.env.step(actions[0])
            terminated = terminated or truncated
            print(self.t, reward)

            # if self.args.live_render:
            #     mpl_img.imsave(join(save_dir, f"{self.t}_post_transition.png"), self.env.render())
            #     # state = np.transpose(self.env.unwrapped.grid.encode()[:, :, 0])
            #     # print("post transition state")
            #     # print(state)

            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            # self.print_data(post_transition_data)
            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1

        last_data = {
            "state": [self.get_state()],
            "avail_actions": [self.get_avail_actions()],
            "obs": [self.get_obs()],
        }

        if self.args.live_render:
            makedirs(render_save_dir, exist_ok=True)
            mpl_img.imsave(
                join(render_save_dir, f"{self.t}_final_state.png"), self.env.render()
            )

        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.select_actions(test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
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

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def get_state(self):
        if isinstance(self.env, RecordVideoExtended):
            return self.env.env.get_state()
        else:
            return self.env.get_state()

    def get_obs(self):
        if isinstance(self.env, RecordVideoExtended):
            return self.env.env.get_obs()
        else:
            return self.env.get_obs()

    def get_avail_actions(self):
        if isinstance(self.env, RecordVideoExtended):
            return self.env.env.get_avail_actions()
        else:
            return self.env.get_avail_actions()

    def _log(self, returns, stats, prefix):
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
