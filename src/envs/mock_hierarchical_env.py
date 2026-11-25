"""Runner-compatible mock hierarchical environment for intermediate testing."""

from __future__ import annotations

import random
from typing import Any, ClassVar
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from src.envs.multiagentenv import MultiAgentEnv


class MockHierarchicalEnv(MultiAgentEnv, gym.Env):
    """Mock single-step low-level option environment."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        *,
        common_reward: bool = True,
        reward_scalarisation: str = "sum",
        seed: int = 101,
        n_agents: int = 3,
        n_final_positions: int = 4,
        n_comms_values: int = 2,
        n_subtasks: int = 5,
        max_episode_steps: int = 1,
        mock_success_offset: float = 0.55,
        mock_edge_bonus: float = 0.25,
        mock_capacity_bonus: float = 0.20,
        mock_start_bonus: float = 0.05,
        navigation_config: str | None = None,
        msg_budget_per_agent: list[float] | tuple[float, ...] | None = None,
        team_delay_prob: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del common_reward, reward_scalarisation
        self.n_agents = int(n_agents)
        self.n_final_positions = int(n_final_positions)
        self.n_comms_values = int(n_comms_values)
        self.n_subtasks = int(n_subtasks)
        self.episode_limit = int(max_episode_steps)
        self.max_episode_steps = self.episode_limit
        self.mock_success_offset = float(mock_success_offset)
        self.mock_edge_bonus = float(mock_edge_bonus)
        self.mock_capacity_bonus = float(mock_capacity_bonus)
        self.mock_start_bonus = float(mock_start_bonus)
        self.navigation_config_path = navigation_config
        team_delay_prob = float(team_delay_prob)
        if not 0.0 <= team_delay_prob <= 1.0:
            raise ValueError("mock team_delay_prob must be in [0, 1].")
        self.team_delay_prob = team_delay_prob
        self.navigation_config = {}
        if navigation_config:
            config_path = Path(navigation_config)
            if not config_path.is_file():
                raise ValueError(
                    f"Navigation config does not exist: {navigation_config}"
                )
            self.navigation_config = yaml.safe_load(
                config_path.read_text(encoding="utf-8")
            ) or {}
        self._rng = random.Random(int(seed))
        self._np_rng = np.random.default_rng(int(seed))

        if self.n_agents < 1:
            raise ValueError("mock n_agents must be positive.")
        if self.n_final_positions < 1:
            raise ValueError("mock n_final_positions must be positive.")
        if self.n_comms_values < 1:
            raise ValueError("mock n_comms_values must be positive.")
        if self.n_subtasks < 1:
            raise ValueError("mock n_subtasks must be positive.")
        if self.episode_limit < 1:
            raise ValueError("mock max_episode_steps must be positive.")

        navigation_tasks = self.navigation_config.get("tasks", [])
        if navigation_tasks:
            self.task_edges = tuple(
                (
                    int(task["from_state"]),
                    int(task["to_state"]),
                )
                for task in navigation_tasks
            )
            mdp_config = self.navigation_config.get(
                "mdp",
                self.navigation_config,
            )
            self.goal_states = tuple(
                int(state)
                for state in mdp_config.get(
                    "goal_states",
                    (self.task_edges[-1][1],),
                )
            )
        elif self.n_subtasks == 5:
            self.task_edges = (
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 3),
                (3, 4),
            )
            self.goal_states = (self.task_edges[-1][1],)
        else:
            self.task_edges = tuple(
                (index, index + 1) for index in range(self.n_subtasks)
            )
            self.goal_states = (self.task_edges[-1][1],)
        if msg_budget_per_agent:
            self.communication_capacities = tuple(
                float(capacity) for capacity in msg_budget_per_agent
            )
        elif self.n_comms_values == 1:
            self.communication_capacities = (0.0,)
        else:
            self.communication_capacities = tuple(
                index / (self.n_comms_values - 1)
                for index in range(self.n_comms_values)
            )

        self.n_actions = 2
        self.action_space = gym.spaces.MultiDiscrete([self.n_actions] * self.n_agents)
        self.state_shape = self.n_agents * self.n_final_positions
        self.obs_shape = self.n_agents * 2
        self._active_edge = self.task_edges[0]
        self._active_capacity = self.communication_capacities[-1]
        self._active_start_state = (self._active_edge[0], 0)
        self._active_start_state_index = 0
        self._active_goal_options_by_agent = ()
        self._state = np.zeros(self.state_shape, dtype=np.float32)
        self._obs = np.zeros((self.n_agents, self.obs_shape), dtype=np.float32)
        self._avail_actions = np.ones(
            (self.n_agents, self.n_actions),
            dtype=np.int64,
        )
        self._t = 0

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng.seed(int(seed))
            self._np_rng = np.random.default_rng(int(seed))
        options = options or {}

        hl_task = options.get(
            "hl_task",
            options.get("navigation_task_transition"),
        )
        if hl_task is not None:
            edge = (int(hl_task[0]), int(hl_task[1]))
            if edge not in self.task_edges:
                raise ValueError(
                    f"Mock hierarchical environment does not contain edge {edge}."
                )
            self._active_edge = edge
        else:
            subtask_index = options.get("subtask_idx")
            if subtask_index is not None:
                index = int(subtask_index)
                if not 0 <= index < len(self.task_edges):
                    raise ValueError(f"Mock subtask index {index} is out of range.")
                self._active_edge = self.task_edges[index]
            else:
                self._active_edge = self.task_edges[0]

        capacity = options.get(
            "msg_budget_per_agent",
            getattr(self, "_active_capacity", self.communication_capacities[-1]),
        )
        capacity = float(capacity)
        if capacity not in self.communication_capacities:
            raise ValueError(
                f"Mock communication capacity {capacity:g} is not configured."
            )
        self._active_capacity = capacity

        start_state = options.get("hl_start_state")
        init_state_dist = options.get("navigation_init_state_dist")
        if init_state_dist is not None:
            states = init_state_dist.get("states", [])
            probs = init_state_dist.get("probs", [])
            if states:
                state_index = self._rng.choices(
                    range(len(states)),
                    weights=probs,
                    k=1,
                )[0]
                selected_state = states[state_index]
                flat_state = tuple(
                    int(value)
                    for position in selected_state
                    for value in position
                )
                self._active_start_state = flat_state
                self._active_start_state_index = int(state_index)
            elif start_state is not None:
                self._active_start_state = (int(start_state), 0)
        elif start_state is not None:
            self._active_start_state = (int(start_state), 0)
        else:
            self._active_start_state = (self._active_edge[0], 0)

        self._active_goal_options_by_agent = (
            self._mock_goal_options_for_edge(self._active_edge)
        )
        self._t = 0
        self._state.fill(0.0)
        self._obs.fill(0.0)
        self._obs[:, 0] = 1.0
        self._avail_actions.fill(1)
        return self._obs.copy(), {"task_completed": False}

    def _mock_goal_options_for_edge(
        self,
        edge: tuple[int, int],
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        for task in self.navigation_config.get("tasks", []):
            if (
                int(task["from_state"]) == edge[0]
                and int(task["to_state"]) == edge[1]
            ):
                raw_options = task.get("goal_positions")
                if (
                    raw_options
                    and isinstance(raw_options[0], (list, tuple))
                    and any(
                        isinstance(value, (list, tuple))
                        for value in raw_options[0]
                    )
                ):
                    return tuple(
                        tuple(
                            (int(position[0]), int(position[1]))
                            for position in agent_options
                        )
                        for agent_options in raw_options
                    )
                if raw_options:
                    return tuple(
                        ((int(position[0]), int(position[1])),)
                        for position in raw_options
                    )
        return ()

    def _sample_mock_terminal_state(self) -> tuple[int, ...]:
        if not self._active_goal_options_by_agent:
            return (
                int(self._active_edge[1]),
                int(self._active_start_state[1])
                if len(self._active_start_state) > 1
                else 0,
            )
        selected_positions = [
            self._rng.choice(goal_options)
            for goal_options in self._active_goal_options_by_agent
        ]
        return tuple(
            int(value)
            for position in selected_positions
            for value in position
        )

    def step(
        self,
        actions,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del actions
        if self._t >= self.episode_limit:
            return (
                self._obs.copy(),
                0.0,
                True,
                False,
                {
                    "task_completed": False,
                    "final_state": None,
                    "episode_limit": self.episode_limit,
                },
            )

        success_rate = self._mock_success_rate()
        task_completed = self._rng.random() < success_rate
        if task_completed:
            final_state = self._sample_mock_terminal_state()
        else:
            final_state = None

        self._t += 1
        reward = 1.0 if task_completed else -1.0
        terminated = True
        truncated = False
        info = {
            "task_completed": bool(task_completed),
            "final_state": final_state if task_completed else None,
            "episode_limit": self.episode_limit,
            "msg_budget_per_agent": self._active_capacity,
            "mock_success_rate": success_rate,
        }
        return self._obs.copy(), reward, terminated, truncated, info

    def _mock_success_rate(self) -> float:
        edge_index = self.task_edges.index(self._active_edge)
        edge_position = edge_index / max(1, len(self.task_edges) - 1)
        capacity_index = self.communication_capacities.index(self._active_capacity)
        capacity_position = capacity_index / max(
            1,
            len(self.communication_capacities) - 1,
        )
        if self.navigation_config:
            start_position = (
                self._active_start_state_index
                / max(1, 1)
            )
            success = (
                0.40
                + 0.05 * edge_position
                + 0.35 * capacity_position
                + 0.05 * (1.0 - float(start_position))
                - 0.15 * self.team_delay_prob
            )
            return float(min(0.98, max(0.02, success)))

        start_position = (
            self._active_start_state[1]
            / max(1, self.n_final_positions - 1)
        )
        success = (
            self.mock_success_offset
            + self.mock_edge_bonus * edge_position
            + self.mock_capacity_bonus * capacity_position
            + self.mock_start_bonus * (1.0 - start_position)
        )
        return float(min(0.98, max(0.02, success)))

    def get_obs(self) -> list[np.ndarray]:
        return [self._obs[agent] for agent in range(self.n_agents)]

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        return self._obs[agent_id]

    def get_obs_size(self) -> int:
        return self.obs_shape

    def get_state(self) -> np.ndarray:
        return self._state

    def get_state_size(self) -> int:
        return self.state_shape

    def get_avail_actions(self) -> np.ndarray:
        return self._avail_actions

    def get_avail_agent_actions(self, agent_id: int) -> np.ndarray:
        return self._avail_actions[agent_id]

    def get_total_actions(self) -> int:
        return self.n_actions

    def get_env_info(self) -> dict[str, Any]:
        return {
            "state_shape": self.state_shape,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def obs(self) -> np.ndarray:
        return self._obs

    @property
    def avail_actions(self) -> np.ndarray:
        return self._avail_actions

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def seed(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(int(seed))
            self._np_rng = np.random.default_rng(int(seed))

    def save_replay(self) -> None:
        return None

    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name)

    def get_wrapper_attr(self, name: str) -> Any:
        if not hasattr(self, name):
            raise AttributeError(name)
        return getattr(self, name)

    def set_wrapper_attr(self, name: str, value: Any) -> None:
        setattr(self, name, value)
