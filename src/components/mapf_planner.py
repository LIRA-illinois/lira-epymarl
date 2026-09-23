from collections import deque
from random import Random
from typing import Optional


class PrioritizedBFSPlanner:
    """Small prioritized MAPF planner for grid environments."""

    def __init__(self, seed=0):
        self.random = Random(seed)

    def actions(self, env):
        base_env = env.unwrapped
        positions = [tuple(agent.pos) for agent in base_env.agents]
        goals = [self._goal(agent, base_env.current_task) for agent in base_env.agents]
        priorities = list(range(len(positions)))
        self.random.shuffle(priorities)

        reservations = {(position, 0) for position in positions}
        edge_reservations = set()
        paths = {}
        for agent_index in priorities:
            path = self._bfs(
                base_env,
                positions[agent_index],
                goals[agent_index],
                reservations,
                edge_reservations,
            )
            if path is None:
                path = [positions[agent_index]]
            paths[agent_index] = path
            for time, position in enumerate(path):
                reservations.add((position, time))
                if time:
                    edge_reservations.add((path[time - 1], position, time))

        return [
            self._action_for_delta(base_env, positions[i], paths[i][1] if len(paths[i]) > 1 else positions[i])
            for i in range(len(positions))
        ]

    def _goal(self, agent, task):
        goal = agent.task_goals.get(task)
        if goal is None or len(goal) == 0:
            return tuple(agent.pos)
        return tuple(goal[0])

    def _bfs(self, env, start, goal, reservations, edge_reservations):
        queue = deque([(start, 0)])
        parents: dict[tuple[tuple[int, int], int], Optional[tuple[tuple[int, int], int]]] = {
            (start, 0): None
        }
        max_time = env.width * env.height * 4 + len(env.agents) * 10
        while queue:
            position, time = queue.popleft()
            if time >= max_time:
                continue
            if position == goal:
                path = []
                node = (position, time)
                while node is not None:
                    path.append(node[0])
                    node = parents[node]
                return list(reversed(path))

            for next_position in self._neighbors(env, position):
                next_time = time + 1
                if (next_position, next_time) in reservations:
                    continue
                if (next_position, position, next_time) in edge_reservations:
                    continue
                node = (next_position, next_time)
                if node not in parents:
                    parents[node] = (position, time)
                    queue.append(node)
        return None

    @staticmethod
    def _neighbors(env, position):
        x, y = position
        candidates = [(x, y)]
        if hasattr(env.actions, "LEFT"):
            candidates.append((x - 1, y))
        if hasattr(env.actions, "RIGHT"):
            candidates.append((x + 1, y))
        if hasattr(env.actions, "UP"):
            candidates.append((x, y - 1))
        if hasattr(env.actions, "DOWN"):
            candidates.append((x, y + 1))
        return [candidate for candidate in candidates if PrioritizedBFSPlanner._walkable(env, candidate)]

    @staticmethod
    def _walkable(env, position):
        x, y = position
        if not (0 <= x < env.width and 0 <= y < env.height):
            return False
        cell = env.grid.get(x, y)
        return cell is None or cell.can_overlap()

    @staticmethod
    def _action_for_delta(env, current, next_position):
        dx = next_position[0] - current[0]
        dy = next_position[1] - current[1]
        if dx == 0 and dy == 0:
            return int(env.actions.STAY)
        if dx < 0:
            return int(env.actions.LEFT)
        if dx > 0:
            return int(env.actions.RIGHT)
        if dy < 0 and hasattr(env.actions, "UP"):
            return int(env.actions.UP)
        if dy > 0 and hasattr(env.actions, "DOWN"):
            return int(env.actions.DOWN)
        return int(env.actions.STAY)