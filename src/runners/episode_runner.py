from collections import defaultdict
from functools import partial
from os import makedirs
from os.path import join
from typing import Optional

import matplotlib.image as mpl_img
import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from src.components.episode_buffer import EpisodeBatch
from src.envs import REGISTRY as env_REGISTRY
from src.envs import register_smac, register_smacv2
from src.utils.record_video import RecordVideoExtended


class EpisodeRunner:
    def __init__(self, args, logger) -> None:
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
        self._t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac) -> None:
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

    def start_recording(
        self,
        n_test_replays_save: int,
        video_prefix: str = "replay",
        t_env: Optional[int] = None,
    ) -> None:
        # use env in parallel comms eval when the new runners don't have updated t_env

        # get video folder from wandb logger
        # make the video dir

        if t_env is None:
            t_env = self.t_env

        replay_dir = join(self.logger.dir, "replays", f"t_{t_env}")
        makedirs(replay_dir, exist_ok=True)
        self.logger.info(f"Saving {n_test_replays_save} test episode replays")
        self.env = RecordVideoExtended(
            env=self.env,
            video_folder=replay_dir,
            episode_trigger=lambda e: True,
            name_prefix=video_prefix,
            output_formats=["mp4"],
        )

    def stop_recording(
        self, t_env: Optional[int] = None, video_prefix: str = "replays"
    ) -> None:
        if t_env is None:
            t_env = self.t_env

        if isinstance(self.env, RecordVideoExtended):
            # save final episode (other eps saved when env.reset() is called after they finish)
            self.env.stop_recording()

            # log videos to wandb
            self.logger.log_videos(
                dir=self.env.video_folder, t=t_env, video_prefix=video_prefix
            )

            # remove the video recorder wrapper
            self.env = self.env.env

    def close_env(self):
        self.env.close()

    def run(
        self,
        test_mode: bool = False,
        return_log_stats: bool = True,
        reset_options: dict | None = None,
    ):
        """
        Run an episode.

        Parameters
        ----------
        test_mode : bool
            Whether to run in test mode (no learning)
        """
        self._reset(options=reset_options)

        terminated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.args.n_agents)
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            self.batch.update(self._get_pre_transition_data(), ts=self._t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self._select_actions(test_mode=test_mode)

            _, reward, terminated, truncated, env_info = self._step(actions)
            terminated = terminated or truncated
            episode_return += reward

            # self.print_data(post_transition_data)
            self.batch.update(
                self._get_post_transition_data(terminated, env_info, actions, reward),
                ts=self._t,
            )

            self._t += 1

        if self.args.live_render:
            self._live_render(file_name="final_state")

        # update batch with final step data
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(self._get_pre_transition_data(), ts=self._t)
        # self.print_data(self.pre_transition_data)

        # Select actions in the last stored state
        actions = self._select_actions(test_mode=test_mode)
        last_actions: dict = {}
        if isinstance(actions, dict):
            last_actions["actions"] = actions["env_actions"]
        else:
            last_actions["actions"] = actions

        self.batch.update(last_actions, ts=self._t)

        # Determine which stats/returns to update
        if not test_mode:
            cur_stats = self.train_stats
            cur_returns = self.train_returns
        else:
            cur_stats = self.test_stats
            cur_returns = self.test_returns

        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self._t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self._t

        cur_returns.append(episode_return)

        # log stats
        out = {}

        if test_mode:
            if len(self.test_returns) == self.args.test_nepisode:
                log_stats = self._get_log_stats(cur_returns, cur_stats, log_prefix)
                if return_log_stats:
                    # return data in cur_returns and cur_stats for processing outside of episode runner
                    out["log_stats"] = log_stats
                else:
                    self._log(log_stats)
        else:
            if self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
                # Training mode logging
                log_stats = self._get_log_stats(cur_returns, cur_stats, log_prefix)
                self._log(log_stats)

                if hasattr(self.mac.action_selector, "epsilon"):
                    self.logger.log_stat(
                        "epsilon", self.mac.action_selector.epsilon, self.t_env
                    )
                self.log_train_stats_t = self.t_env

        out["batch"] = self.batch
        return out

    def _reset(self, options: dict | None = None) -> None:
        self.batch = self.new_batch()
        self.env.reset(options=options)
        self._t = 0

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
                self.batch, t_ep=self._t, t_env=self.t_env, test_mode=test_mode
            )

            if hasattr(self.args, "manual_policy"):
                actions = self._manual_policy(actions)

            # following the format from the parallel episode runner
            if isinstance(actions, Tensor):
                actions = actions.cpu().numpy()

        return actions

    @property
    def t(self) -> int:
        return self._t

    def _step(self, actions):
        if self.args.live_render:
            self._live_render(file_name="pre_step")

        if self.env.has_wrapper_attr("t_render"):
            self.env.set_wrapper_attr("t_render", self.t)

        if isinstance(actions, dict):
            obs, reward, terminated, truncated, env_info = self.env.step(actions)
        else:
            obs, reward, terminated, truncated, env_info = self.env.step(actions[0])

        return obs, reward, terminated, truncated, env_info

    def _get_pre_transition_data(self) -> dict:
        data = defaultdict(list)

        state = self.env.state
        if isinstance(state, dict):
            data["hl_state"].append(state["hl_state"])
            data["state"].append(state["ll_state"])
        else:
            data["state"].append(state)

        data["avail_actions"].append(self.env.avail_actions)
        data["obs"].append(self.env.obs)

        return data

    def _get_post_transition_data(
        self, terminated: bool, env_info: dict, actions, reward
    ) -> dict:
        data = {
            "terminated": [(terminated != env_info.get("episode_limit", False),)],
        }

        if isinstance(actions, dict):
            data["actions"] = actions["env_actions"]
        else:
            data["actions"] = actions

        if self.args.common_reward:
            data["reward"] = [(reward,)]
        else:
            data["reward"] = [tuple(reward)]

        return data

    def _get_log_stats(self, returns, stats, prefix: str) -> dict:
        # populates a dict with all the stats you want to log with appropriate keys
        log_stats = {}
        # returns
        if self.args.common_reward:
            log_stats["return_mean"] = np.mean(returns)
            log_stats["return_std"] = np.std(returns)
        else:
            for i in range(self.args.n_agents):
                log_stats[f"agent_{i}_return_mean"] = np.array(returns)[:, i].mean()
                log_stats[f"agent_{i}_return_std"] = np.array(returns)[:, i].std()

            total_returns = np.array(returns).sum(axis=-1)
            log_stats["total_return_mean"] = total_returns.mean()
            log_stats["total_return_std"] = total_returns.std()

        # other stats
        for k, v in stats.items():
            if k != "n_episodes":
                log_stats[f"{k}_mean"] = v / stats["n_episodes"]
            else:
                log_stats[f"{k}"] = stats["n_episodes"]

        # add prefix to all the keys in log_stats
        log_stats_out = {}
        for k in log_stats:
            log_stats_out[f"{prefix}{k}"] = log_stats[k]

        self._clear_stats(returns, stats)
        return log_stats_out

    def _log(self, log_stats: dict) -> None:
        for k, v in log_stats.items():
            self.logger.log_stat(k, v, self.t_env)

    def _clear_stats(self, returns, stats) -> None:
        returns.clear()
        stats.clear()

    # helpers
    def _print_data(self, data: dict) -> None:
        print(f"t_ep: {self._t}, hl_state: {data.get('hl_state', None)}")
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

    def _live_render(self, file_name: str, actions: Optional[dict] = None) -> None:
        render_save_dir = join("results", "live_renders", f"zzz_{self.args.env}")
        makedirs(render_save_dir, exist_ok=True)
        mpl_img.imsave(
            join(render_save_dir, f"{self._t}_{file_name}.png"), self.env.render()
        )
        # state = np.transpose(self.env.unwrapped.grid.encode()[:, :, 0])
        # print("pre transition state")
        # print(state)

    def _manual_policy(self, actions):
        # class LBFActions(enum.IntEnum):
        #     # matches the order from original LBF action set
        #     STAY = 0
        #     UP = 1
        #     DOWN = 2
        #     LEFT = 3
        #     RIGHT = 4
        #     LOAD = 5

        # print(f"self.t: {self._t}")

        # if isinstance(actions, dict):
        #     if self.mac.msg_budget_per_agent == 0.0:
        #         # succeed 1st subtask, fail 2nd one
        #         if 0 < self._t <= 3:
        #             # go right
        #             actions["env_actions"] = np.array([[4, 4, 4]])
        #         else:
        #             actions["env_actions"] = np.array([[0, 0, 0]])

        #     # succeed both subtasks
        #     elif self.mac.msg_budget_per_agent == 1.0:
        #         # go right
        #         actions["env_actions"] = np.array([[4, 4, 4]])

        #     # # go right
        #     # actions["env_actions"] = np.array([[4, 4, 4]])

        # else:
        #     if self._t in [0, 1, 2]:
        #         actions = np.array([[5, 3, 5]])
        #     else:
        #         actions = np.array([[4, 0, 0]])

        # if self.t == 0:
        #     # all load, see if the fruit goes away
        #     actions = np.array([[5, 5, 5]])
        # elif self.t == 1:
        #     # 2 load while 3rd moves down, see if the fruit goes away
        #     actions = np.array([[5, 5, 2]])
        # elif self.t == 2:
        #     # all 3 load
        #     actions = np.array([[5, 5, 5]])
        #     # actions = np.array([[0, 0, 0]])
        # else:
        #     # actions = np.array([[5, 5, 5]])
        #     actions = np.array([[0, 0, 0]])

        # if self.mac.msg_budget_per_agent == 0.0:
        #     if self.t == 0:
        #         # right
        #         actions = np.array([[4, 4, 4]])
        #     else:
        #         # grab
        #         actions = np.array([[5, 5, 5]])

        # elif self.mac.msg_budget_per_agent == 1.0:
        #     # left
        #     actions = np.array([[3, 3, 3]])

        # if self.t in [2, 6]:
        #     # grab
        #     actions = np.array([[5, 5, 5]])
        # else:
        #     # right
        #     actions = np.array([[4, 4, 4]])

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

        return actions
