import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Discrete, flatdim
from gymnasium.wrappers import TimeLimit
from gymnasium.spaces import Box, Tuple
import numpy as np
import gym_multigrid
import os 
import imageio
from gymnasium.wrappers import RecordVideo



class GymMultiGridWrapper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], 
                "render_fps": 5,}

    def __init__(
        self,
        key,
        time_limit,
        seed,
        common_reward,
        reward_scalarisation,
        **kwargs,
    ):
        self.save_replay_path = kwargs.pop("save_replay_path")
        self.save_replay_ = kwargs.pop("save_replay_")
        self.worker_id = kwargs.pop("worker_id", 0)

        self._env = gym.make(f"{key}", **kwargs)
        self._env = TimeLimit(self._env, max_episode_steps=time_limit)
        if self.save_replay_:
            self._env = RecordVideo(
                self._env,
                video_folder=self.save_replay_path[:-7],
                episode_trigger=lambda e: True,  # record all episodes
                name_prefix="replay_",
            )
        self._frames = []
        self.episode_id = 0

        self.episode_limit = time_limit
        self._obs = None
        self._info = None
        self.frame = None

        
        if isinstance(self._env.action_space, Tuple):
            self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        elif isinstance(self._env.action_space, Discrete):
            self.longest_action_space = self._env.action_space
        elif isinstance(self._env.action_space, Box):
            # Box with integer dtype represents multi-agent discrete actions
            # e.g., Box(0, 3, (2,), int64) means 2 agents with 4 actions each
            self._n_actions = int(self._env.action_space.high.flat[0]) + 1
            self.longest_action_space = Discrete(self._n_actions)
        else:
            raise NotImplementedError(f"Unsupported action space type: {type(self._env.action_space)}")

        # Obs size and num_agents will be computed from actual observation on first reset
        self._obs_size = None
        self.num_agents = None

        # Do initial reset to capture obs and compute obs_size/num_agents
        self._seed = seed
        if seed is not None:
            self.reset(seed=self._seed)
        else:
            self.reset()

        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation} (only support 'sum' or 'mean')"
                )

    def reset(self, **kwargs):
        """
        Reset the underlying environment.
        
        Returns:
            obs: A tuple of observations.
            info: Info dictionary from the underlying env.
        """
        obs, info = self._env.reset(**kwargs)
        if isinstance(obs, list):
            self.last_obs = tuple(obs)
        else:
            self.last_obs = obs

        # Compute num_agents from environment's actual agent list
        if self.num_agents is None:
            self.num_agents = len(self._env.unwrapped.agents)

        # Compute obs_size - observation is shared for all agents in this env
        if self._obs_size is None:
            if isinstance(self.last_obs, (tuple, list)):
                self._obs_size = int(np.prod(np.array(self.last_obs[0]).shape))
            else:
                # Single shared observation - total size is the flattened observation
                self._obs_size = int(np.prod(np.array(self.last_obs).shape))

        return self.last_obs, info

    def step(self, actions):
        """
        Execute one step in the environment.
        
        Args:
            actions: For multi-agent envs, a list/tuple of actions for each agent.
        
        Returns:
            A tuple (obs, rewards, done, truncated, info) where:
              - obs: Tuple of observations if multi-agent.
              - rewards: Scalar reward (aggregated if common_reward).
              - done: A boolean flag indicating termination.
              - truncated: A boolean flag indicating truncation.
              - info: Info dictionary.
        """
        # Ensure actions is a list if multi-agent.
        if self.num_agents > 1 and not isinstance(actions, list):
            actions = list(actions)
        obs, rewards, done, truncated, info = self._env.step(actions)
        if isinstance(obs, list):
            obs = tuple(obs)
        self.last_obs = obs
        
        # Aggregate rewards to a single scalar
        if isinstance(rewards, (list, tuple, np.ndarray)):
            if self.common_reward:
                rewards = float(self.reward_agg_fn(rewards))
            else:
                rewards = float(sum(rewards))
        
        return obs, rewards, done, truncated, info

    def render(self):
        # self.frame = self._env.render()
        # self._frames.append(self.frame)
        return self._env.render()

    def close(self):
        return self._env.close()
    
    def get_state_size(self):
        # State is the full observation (same for all agents in shared obs env)
        return self._obs_size

    def get_obs_size(self):
        return self._obs_size

    def get_total_actions(self):
        return self.longest_action_space.n
    
    def get_avail_actions(self):
        # Return all actions as available for each agent
        n_actions = self.get_total_actions()
        return [[1] * n_actions for _ in range(self.num_agents)]

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.num_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info
    
    def get_state(self):
        # Global state - for shared obs envs, this is just the flattened observation
        return np.array(self.last_obs).flatten()

    def get_obs(self):
        """Returns the list of observations for all agents"""
        if isinstance(self.last_obs, (tuple, list)):
            # Per-agent observations
            return [np.array(obs).flatten() for obs in self.last_obs]
        else:
            # Shared observation - each agent gets the same flattened obs
            obs_flat = np.array(self.last_obs).flatten()
            return [obs_flat.copy() for _ in range(self.num_agents)]
    
    def save_replay(self):
        pass

    # def save_gif_replays(self):
    #     if self.save_replay_ and len(self._frames) > 0:
    #         os.makedirs(os.path.dirname(self.save_replay_path), exist_ok=True)
    #         imageio.mimsave(f'{self.save_replay_path}{self.episode_id}.gif', self._frames, fps=5)
    #         print(f"Replay saved to {self.save_replay_path}{self.episode_id}.gif")  
    #         self._frames = []
    #         self.episode_id += 1
    #     else:
    #         print("No frames to save!")

    def get_stats(self):
        return {}
