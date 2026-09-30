from typing import Any

import yaml


def render_navigation_task_frames(
    groups, source_config: dict[str, Any], logger=None
) -> dict:
    """Render base-grid frames for the navigation transitions in ``groups``."""
    env_args = source_config.get("env_args", {}) or {}
    navigation_config_path = source_config.get(
        "env_args.navigation_config", env_args.get("navigation_config")
    )
    map_name = source_config.get("env_args.map_name", env_args.get("map_name"))
    if navigation_config_path is None or map_name is None:
        if logger is not None:
            logger.info("Skipping task frames: navigation config is unavailable")
        return {}

    try:
        with open(navigation_config_path) as config_file:
            navigation_config = yaml.safe_load(config_file) or {}
        tasks = navigation_config["tasks"]
        n_agents = source_config.get("env_args.n_agents", env_args.get("n_agents"))
        if n_agents is None:
            n_agents = len(tasks[0]["goal_positions"])

        from gym_multigrid.envs.team_navigation import TeamNavigationEnv

        env = TeamNavigationEnv(
            map_name=map_name,
            n_agents=int(n_agents),
            navigation_tasks=tasks,
        )
    except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
        if logger is not None:
            logger.info(f"Skipping task frames: {error}")
        return {}

    frames = {}
    try:
        for _, _, hl_task in groups:
            if not isinstance(hl_task, (list, tuple)):
                continue
            transition = tuple(hl_task)
            env.reset(seed=0, options={"navigation_task_transition": transition})
            frame = env.render_grid()
            if frame is not None:
                frames[transition] = frame
    finally:
        env.close()
    return frames


def load_navigation_mdp(source_config: dict[str, Any], logger=None) -> dict:
    """Load the high-level navigation graph from a run configuration."""
    env_args = source_config.get("env_args", {}) or {}
    navigation_config_path = source_config.get(
        "env_args.navigation_config", env_args.get("navigation_config")
    )
    if navigation_config_path is None:
        return {}

    try:
        with open(navigation_config_path) as config_file:
            navigation_config = yaml.safe_load(config_file) or {}
    except (OSError, TypeError, ValueError) as error:
        if logger is not None:
            logger.info(f"Skipping high-level graph: {error}")
        return {}
    return navigation_config.get("mdp", {}) or {}
