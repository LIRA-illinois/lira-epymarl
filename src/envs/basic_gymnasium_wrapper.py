from typing import Any

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import lbforaging as lbf
import join1


SUPPORTED_ENVS = (
    "foraging-v2",
    "join1-v0",
    "join1_original",
)

NON_GYMNASIUM_ENVS = {"join1_original": join1.Join1EnvOriginal}


class BasicGymnasiumWrapper(gym.Env):
    """
    Basic wrapper that supports Gymnasium and non-gymnasium envs to ensure they conform to the Gymnasium API standards. Designed for the join1 env from MAIC, but may be extended to support other envs too.
    """

    def __init__(self, env_args: dict):
        self.env_name: str = env_args.pop("key")
        self.seed: int = env_args.pop("seed")
        self.env_args: dict = env_args

        self.episode_limit = self.env_args.get("max_episode_steps")

        # register envs, get ID
        self._register_envs()

        # make the env
        self.env = self._build_env()

        # initialize the env's PRNG
        self._set_env_seed()

        # run basic checks to ensure the env follows the Gymnasium API
        # and does not have obvious issues
        self._check_env()

    def _build_env(self) -> gym.Env:
        if self.env_name in ["foraging-v2"]:
            # special way for envs that pre-register their envs with kwargs under specific names
            # normal way that follows the Gymnasium website's example
            env_id = self._get_env_id(self.env_name, self.env_args)
            return gym.make(env_id, max_episode_steps=self.episode_limit)

        elif self.env_name in NON_GYMNASIUM_ENVS:
            # envs that do not meet the Gymnasium API standards on their own
            return NON_GYMNASIUM_ENVS[self.env_name](**self.env_args)

        else:
            # normal way that follows Gymnasium's example
            return gym.make(self.env_name, **self.env_args)

    def _register_envs(self) -> None:
        # register envs supported by this wrapper
        lbf.register_envs(max_episode_steps=self.episode_limit)
        join1.register_envs()

    def _get_env_id(self, env_name: str, env_args: dict) -> str:
        match env_name:
            case "foraging-v2":
                # foraging pre-registers their envs with kwargs under specific names
                id_args = {
                    "s": env_args["field_size"],
                    "p": env_args["players"],
                    "f": env_args["max_num_food"],
                    "c": env_args["force_coop"],
                    "po": env_args["partially_observe"],
                    "pen": env_args["penalty"],
                    "mfl": (
                        env_args["max_food_level"]
                        if "max_food_level" in env_args.keys()
                        else None
                    ),
                }

                env_id = lbf.get_env_id(**id_args)

            case _:
                env_id = env_name

        return env_id

    def _set_env_seed(self):
        print(f"setting env seed to {self.seed}")
        self.env.reset(seed=self.seed)

    def _check_env(self):
        if self.env_name in NON_GYMNASIUM_ENVS:
            env = self
        else:
            env = self.env.unwrapped

        try:
            check_env(env, skip_render_check=True)
        except Exception as e:
            print(f"Env has issues: {e}")

    def _get_env_class(self):
        """get the lowest-level available version of the environment class inside any wrappers (if there are any)"""
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env

    def get_env_info(self) -> dict[str, Any]:
        env = self._get_env_class()

        info: dict = env.get_env_info()

        info["episode_limit"] = self.episode_limit
        return info

    def get_state(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            system state with shape (n_samples=1, n_agents, n_state_features)
        """
        env = self._get_env_class()
        state = env.get_state()

        # expand 0th dimension to be size (n_samples=1, n_agents, n_state_features)
        return np.expand_dims(state, 0)

    def get_avail_actions(self) -> list:
        env = self._get_env_class()
        return env.get_avail_actions()

    def get_obs(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            team obs with shape (n_samples=1, n_agents, n_obs_features)
        """
        env = self._get_env_class()
        obs = env.get_obs()

        # expand 0th dimension to be size (n_samples=1, n_agents, n_obs_features)
        return np.expand_dims(obs, 0)

    def step(self, actions: NDArray) -> tuple[NDArray, float, bool, bool, dict]:
        """
        Parameters
        ----------
        actions : NDArray
            team's joint action of size (n_agents,)
        """
        if self.env_name in NON_GYMNASIUM_ENVS:
            # only for the non-Gymnasium version of the env
            reward, terminated, env_info = self.env.step(actions)
            obs = None
            truncated = None

        else:
            obs, reward, terminated, truncated, env_info = self.env.step(actions)

        # convert from np.bool to python bool
        terminated = bool(terminated)

        return obs, reward, terminated, truncated, env_info
