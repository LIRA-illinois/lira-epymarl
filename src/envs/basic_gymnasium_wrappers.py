from typing import Any, Optional, Literal
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import lbforaging as lbf
import join1
from gym_multigrid.envs.mdp import ProjectMDP

SUPPORTED_ENVS = ("foraging-v2", "join1-v0", "join1_original", "multigrid-lbf-v0")

NON_GYMNASIUM_ENVS = {"join1_original": join1.Join1EnvOriginal}


class GymnasiumEnvWrapper(gym.Env):
    def __init__(self, env, env_args: dict):
        self.env = env
        self.env_args = env_args

        self.episode_limit = self.env_args.get("max_episode_steps")

    def get_env_info(self) -> dict[str, Any]:
        info: dict = self.env.get_env_info()
        info["episode_limit"] = self.episode_limit

        return info

    @property
    def state(self) -> NDArray:
        return self.env.state

    @property
    def avail_actions(self) -> NDArray:
        return self.env.avail_actions

    @property
    def obs(self) -> NDArray:
        return self.env.obs

    def step(self, actions: NDArray) -> tuple[NDArray, float, bool, bool, dict]:
        """
        Parameters
        ----------
        actions : NDArray
            team's joint action of size (n_agents,)
        """
        # only for the non-Gymnasium version of the env
        reward, terminated, env_info = self.env.step(actions)
        obs = None
        truncated = None

        # convert from np.bool to python bool
        terminated = bool(terminated)

        return obs, reward, terminated, truncated, env_info


class BasicGymnasiumWrapper(gym.Wrapper):
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
        # self._check_env()

        # init as a proper env wrapper
        super().__init__(self.env)

    def _build_env(self) -> gym.Env:
        if self.env_name in ["foraging-v2"]:
            # special way for envs that pre-register their envs with kwargs under specific names
            # normal way that follows the Gymnasium website's example
            env_id = self._get_env_id(self.env_name, self.env_args)
            return gym.make(env_id, max_episode_steps=self.episode_limit)

        elif self.env_name in NON_GYMNASIUM_ENVS:
            # for envs that do not meet the Gymnasium API standards on their own
            env = NON_GYMNASIUM_ENVS[self.env_name](**self.env_args)
            return GymnasiumEnvWrapper(env, self.env_args)

        else:
            # normal way that follows Gymnasium's example
            return gym.make(self.env_name, **self.env_args)

    def _register_envs(self) -> None:
        # register envs supported by this wrapper
        lbf.register_envs(max_episode_steps=self.episode_limit)
        join1.register_envs()
        import gym_multigrid

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
        # print(f"Setting env seed to {self.seed}")
        self.env.reset(seed=self.seed)

    def _check_env(self):
        try:
            check_env(self.env.unwrapped, skip_render_check=True)
        except Exception as e:
            print(f"Env has issues: {e}")

    def get_env_info(self) -> dict[str, Any]:
        info: dict = self.env.unwrapped.get_env_info()
        info["episode_limit"] = self.episode_limit
        return info

    @property
    def state(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            system state with shape (n_samples=1, n_state_features)
        """
        # you can't call self.get_wrapper_attr since that will cause an infinite loop, you have to call get_wrapper_attr for the next level in the wrapped env
        state = self.env.get_wrapper_attr("state")

        # expand 0th dimension to be size (n_samples=1, n_state_features)
        return np.expand_dims(state, 0)

    @property
    def avail_actions(self) -> list:
        return self.env.get_wrapper_attr("avail_actions")
        # return self.env.unwrapped.get_avail_actions()

    @property
    def obs(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            team obs with shape (n_samples=1, n_agents, n_obs_features)
        """
        obs = self.env.get_wrapper_attr("obs")
        # obs = self.env.unwrapped.get_obs()

        # expand 0th dimension to be size (n_samples=1, n_agents, n_obs_features)
        return np.expand_dims(obs, 0)

    def step(self, actions: NDArray) -> tuple[NDArray, float, bool, bool, dict]:
        """
        Parameters
        ----------
        actions : NDArray
            team's joint action of size (n_agents,)
        """
        obs, reward, terminated, truncated, env_info = self.env.step(actions)

        # convert from np.bool to python bool
        terminated = bool(terminated)

        return obs, reward, terminated, truncated, env_info


class HLMDPEnvWrapper(gym.Wrapper):
    def __init__(self, env_args: dict, hl_env_args: Optional[dict] = None):
        # if hl_env_args is not None:
        #     self.task_type: Literal["atomic", "composed"] = hl_env_args.pop("task_type")

        self.num_rooms: int = env_args.pop("num_rooms", 2)
        comms_values: float = env_args.pop("comms_values", [0.0])
        # optional behavior: end episode when low-level reports task_completed
        # (useful during evaluation). Default False to preserve training behavior.
        self.terminate_on_task_completed: bool = env_args.pop(
            "terminate_on_task_completed", False
        )
        self.env_args: dict = env_args

        # low-level environment
        self.env = BasicGymnasiumWrapper(env_args=env_args)
        super().__init__(self.env)

        # high-level MDP tracks valid goal transitions
        self.hlmdp = ProjectMDP(
            num_rooms=self.num_rooms,
            comms_values=comms_values,
            # task_type=self.task_type,
        )

        # this thing's action space should be a Cartesian product of the low-level env's and the MDP action space
        # you can use a dict to represent that since they're factored and different structure
        # similar for the obs space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if options is None:
            options = {}

        hl_options = {}
        ll_options = {}

        if "hl_start_state" in options:
            hl_options["hl_start_state"] = options["hl_start_state"]
            ll_options["start_room"] = options["hl_start_state"]

        _, hl_info = self.hlmdp.reset(seed=seed, options=hl_options)
        ll_obs, ll_info = self.env.reset(seed=seed, options=ll_options)

        # only used for rendering HLMDP actions
        self._pre_step_hl_actions = None

        ll_info.update(hl_info)
        return ll_obs, ll_info

    def step(self, actions: dict) -> tuple[NDArray, float, bool, bool, dict]:
        """
        Execute one step of the environment.

        Parameters
        ----------
        actions :
            High-level actions from agents, shape (n_agents,)
            Action indices into the MDP's action space
        """
        # All agents share the same high-level action
        hl_actions = actions["hl_actions"]
        ll_actions = actions["env_actions"]

        # Advance the team MDP to the next goal state
        # hl_reward: float
        # I think this would just be =1 if you reach the goal state, but we're not modeling reward so not necessary

        # Low-level agents now execute towards the shared goal
        ll_obs, ll_reward, ll_terminated, ll_truncated, ll_info = self.env.step(
            ll_actions
        )

        # stop showing the action since the agent just reached a new state the HLMDP
        if ll_info["task_completed"]:
            self._pre_step_hl_actions = None
            # may want to end the episode at task completion, like during evaluation
            if self.terminate_on_task_completed:
                ll_terminated = True

        # other envs may have other ways to fail the overall project,
        # but in LBF running out of time is the only way to do it
        project_failed = ll_truncated

        # the HLMDP's step depends on if the task was completed successfully or not and the chosen next state
        hl_obs, hl_reward, hl_terminated, hl_truncated, hl_info = self.hlmdp.step(
            hl_actions, ll_info["task_completed"], project_failed
        )

        # Combine rewards: HL rewards for goal transitions + LL rewards
        total_reward = hl_reward + ll_reward
        terminated = ll_terminated or hl_terminated
        truncated = ll_truncated or hl_truncated

        # Merge HL and LL info
        info = ll_info | hl_info

        return ll_obs, total_reward, terminated, truncated, info

    @property
    def state(self) -> dict:
        """
        Returns combined low-level and high-level state.

        Returns
        -------
        NDArray
            Joint state including LL and HL components
        """
        state = {"ll_state": self.env.state, "hl_state": self.hlmdp.state}

        return state

    @property
    def obs(self) -> NDArray:
        """
        Returns low-level obs since high-level obs not used in our alg.

        Returns
        -------
        NDArray
            shape (n_samples=1, n_agents, n_obs_features)
        """
        return self.env.obs

    @property
    def avail_actions(self) -> list:
        return self.env.avail_actions

    @property
    def episode_limit(self):
        return self.env.episode_limit

    def get_env_info(self) -> dict[str, Any]:
        ll_info: dict = self.env.get_env_info()

        hl_info = self.hlmdp.get_env_info()
        # add hl prefix to stuff in the hl info dict
        for k in list(hl_info.keys()):
            hl_info[f"hl_{k}"] = hl_info.pop(k)

        info = ll_info | hl_info

        return info

    def render(self):
        ll_img = self.env.render()
        _, ll_width = ll_img.shape[:2]
        hl_img = self.hlmdp.render(self._pre_step_hl_actions)

        # Get dimensions
        _, hl_width = hl_img.shape[:2]
        max_width = max(ll_width, hl_width)

        # pad the smaller image so it is centered
        if ll_width < max_width:
            pad_total = max_width - ll_width
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            ll_img = np.pad(
                ll_img,
                ((0, 0), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        if hl_width < max_width:
            pad_total = max_width - hl_width
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            hl_img = np.pad(
                hl_img,
                ((0, 0), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # Stack vertically with LL on bottom
        combined_img = np.vstack([hl_img, ll_img])

        # from PIL import Image
        # img = Image.fromarray(combined_img)
        # img.save("total_env.png")

        return combined_img
