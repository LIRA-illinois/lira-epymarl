from copy import deepcopy
from types import SimpleNamespace as SN

import numpy as np

from src.utils.logging import LocalLogger

from .build import build_sim
from .hierarchical_types import (
    CalibrationOutcome,
    CalibrationRow,
    OptionSettingKey,
)


def run_eval_episodes(
    args: SN,
    runner,
    n_eval_eps: int,
    video_prefix: str = "replay",
    t_env: int | None = None,
    reset_options: dict | None = None,
    include_terminal_state_distribution: bool = False,
):
    """Run n_eval_eps evaluation episodes, optionally recording, and return last result."""
    task_state, msg_budget_per_agent = None, None
    if reset_options is not None:
        task_state = reset_options.get("hl_start_state", None)
        msg_budget_per_agent = reset_options.get("msg_budget_per_agent", None)

    set_msg_budget_per_agent_for_eval = (
        msg_budget_per_agent is not None
        and getattr(args, "unique_policy_per_msg_budget", False) is False
    )

    if set_msg_budget_per_agent_for_eval:
        print(f"Setting MAC msg_budget_per_agent to {msg_budget_per_agent}")
        runner.mac.msg_budget_per_agent = msg_budget_per_agent

    # filename prefix includes task state and comms value when available
    prefix_parts = [video_prefix]
    if task_state is not None:
        prefix_parts.append(f"task_{int(task_state)}")
    if msg_budget_per_agent is not None:
        prefix_parts.append(f"msg_budget_{msg_budget_per_agent:.2f}")
    file_name_prefix = "-".join(prefix_parts)

    # Process-backed runners expose environment controls through runner methods.
    if hasattr(runner, "set_env_attr"):
        runner.set_env_attr("terminate_on_task_completed", True)
    elif hasattr(runner.env, "terminate_on_task_completed"):
        runner.env.terminate_on_task_completed = True

    evaluation_result = None
    final_infos = []
    for i in range(n_eval_eps):
        if i % 50 == 0:
            runner.logger.info(f"Test Episode: {i} / {n_eval_eps}")

        return_stats = i == n_eval_eps - 1

        # Only the final evaluation run returns "log_stats".
        evaluation_result = runner.run(
            test_mode=True,
            return_log_stats=return_stats,
            reset_options=reset_options,
        )
        if include_terminal_state_distribution:
            if "final_infos" in evaluation_result:
                final_infos.extend(evaluation_result["final_infos"])
            elif evaluation_result.get("final_info") is not None:
                final_infos.append(evaluation_result["final_info"])

    # Replay generation separate from metric evaluation so the
    # normal evaluation loop does not render or encode video frames.
    if (
        args.save_test_replays
        and args.n_test_replays_save > 0
        and hasattr(runner, "start_recording")
    ):
        test_returns_snapshot = list(getattr(runner, "test_returns", []))
        test_stats_snapshot = dict(getattr(runner, "test_stats", {}))
        runner.start_recording(
            n_test_replays_save=args.n_test_replays_save,
            video_prefix=file_name_prefix,
            t_env=t_env,
        )
        try:
            for _ in range(args.n_test_replays_save):
                runner.run(
                    test_mode=True,
                    return_log_stats=False,
                    reset_options=reset_options,
                )
        finally:
            try:
                runner.stop_recording(t_env=t_env, video_prefix=file_name_prefix)
            finally:
                if hasattr(runner, "test_returns"):
                    runner.test_returns = test_returns_snapshot
                if hasattr(runner, "test_stats"):
                    runner.test_stats = test_stats_snapshot

    if evaluation_result is None:
        raise ValueError("n_eval_eps must be greater than zero")

    evaluation_result["log_stats"]["t_env"] = t_env
    if include_terminal_state_distribution:
        successful_final_states = []
        for info in final_infos:
            if info.get("task_completed", False):
                if "final_state" not in info:
                    raise RuntimeError(
                        "A successful evaluation episode did not include final_state."
                    )
                successful_final_states.append(
                    tuple(np.asarray(info["final_state"]).reshape(-1).tolist())
                )

        if successful_final_states:
            states, counts = np.unique(
                np.asarray(successful_final_states), axis=0, return_counts=True
            )
            terminal_state_distribution = {
                "states": states.tolist(),
                "probs": (counts / counts.sum()).tolist(),
            }
        else:
            terminal_state_distribution = {"states": [], "probs": []}
        evaluation_result["log_stats"]["successful_terminal_state_distribution"] = (
            terminal_state_distribution
        )

    # log stuff like current HL task and comms action
    if reset_options is not None:
        for k, v in reset_options.items():
            evaluation_result["log_stats"][k] = v

    # Restore terminate_on_task_completed to False after evaluation.
    if hasattr(runner, "set_env_attr"):
        runner.set_env_attr("terminate_on_task_completed", False)
    elif hasattr(runner.env, "terminate_on_task_completed"):
        runner.env.terminate_on_task_completed = False

    if set_msg_budget_per_agent_for_eval:
        print(
            f"Evaluation done, Setting MAC msg_budget_per_agent to default of n_agents - 1 = {args.n_agents - 1}"
        )
        runner.mac.msg_budget_per_agent = args.n_agents - 1

    return evaluation_result


def record_calibration_replays(
    runner,
    setting: OptionSettingKey,
    reset_options: dict | None,
    t_env: int | None,
) -> None:
    """Record a small replay batch for one frozen statewise option setting."""
    args = getattr(runner, "args", None)
    if args is None:
        return
    if not bool(getattr(args, "save_test_replays", False)):
        return
    n_replays = int(getattr(args, "n_test_replays_save", 0))
    if n_replays <= 0 or not hasattr(runner, "start_recording"):
        return

    prefix = (
        f"replay-task-{setting.subtask_edge[0]}-{setting.subtask_edge[1]}"
        f"-msg-{setting.communication_capacity:.2f}"
    )
    runner.start_recording(
        n_test_replays_save=n_replays,
        video_prefix=prefix,
        t_env=t_env,
    )
    try:
        for _ in range(n_replays):
            runner.run(
                test_mode=True,
                return_log_stats=False,
                reset_options=reset_options,
            )
    finally:
        runner.stop_recording(t_env=t_env, video_prefix=prefix)


def collect_statewise_calibration_row(
    runner,
    setting: OptionSettingKey,
    seed: int,
    n_eval_eps: int,
    reset_options: dict | None = None,
    record_replays: bool = False,
) -> CalibrationRow:
    """Collect categorical outcomes for one frozen (edge, capacity, start) setting."""
    if n_eval_eps <= 0:
        raise ValueError("n_eval_eps must be greater than zero.")
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("Calibration seed must be an integer.")

    reset_options = deepcopy(reset_options) if reset_options is not None else {}
    reset_options["hl_start_state"] = setting.subtask_edge[0]
    reset_options["hl_task"] = list(setting.subtask_edge)
    flat_start_state = tuple(
        int(value) for value in np.asarray(setting.start_state).reshape(-1).tolist()
    )
    if len(flat_start_state) % 2 != 0:
        raise ValueError(
            "A navigation joint start state must contain two coordinates per agent."
        )
    nested_start_state = [
        [flat_start_state[index], flat_start_state[index + 1]]
        for index in range(0, len(flat_start_state), 2)
    ]
    reset_options["navigation_init_state_dist"] = {
        "states": [nested_start_state],
        "probs": [1.0],
    }

    previous_termination_setting = None
    if hasattr(runner, "get_env_attr"):
        previous_termination_setting = runner.get_env_attr(
            "terminate_on_task_completed"
        )
        runner.set_env_attr("terminate_on_task_completed", True)
    elif hasattr(runner, "env") and hasattr(runner.env, "terminate_on_task_completed"):
        previous_termination_setting = runner.env.terminate_on_task_completed
        runner.env.terminate_on_task_completed = True

    previous_budget = None
    mac = getattr(runner, "mac", None)
    if mac is not None and hasattr(mac, "msg_budget_per_agent"):
        previous_budget = mac.msg_budget_per_agent
        mac.msg_budget_per_agent = setting.communication_capacity

    test_returns_snapshot = list(getattr(runner, "test_returns", []))
    test_stats_snapshot = dict(getattr(runner, "test_stats", {}))

    batch_size = max(1, int(getattr(runner, "batch_size", 1)))
    n_calls = -(-n_eval_eps // batch_size)
    outcomes: list[CalibrationOutcome] = []
    n_total = 0
    n_parallel_episodes = batch_size

    try:
        for call_idx in range(n_calls):
            result = runner.run(
                test_mode=True,
                return_log_stats=call_idx == n_calls - 1,
                reset_options=reset_options,
            )
            final_infos = result.get("final_infos", [result.get("final_info")])
            for info in final_infos[: n_eval_eps - n_total]:
                if info is None:
                    raise RuntimeError(
                        "Calibration runner did not return terminal episode info."
                    )
                n_total += 1
                if info.get("task_completed", False):
                    if "final_state" not in info:
                        raise RuntimeError(
                            "A successful calibration episode did not include final_state."
                        )
                    terminal_state = tuple(
                        np.asarray(info["final_state"]).reshape(-1).tolist()
                    )
                    outcomes.append(CalibrationOutcome.success(terminal_state))
                else:
                    outcomes.append(CalibrationOutcome.failure())
        if record_replays:
            record_calibration_replays(
                runner,
                setting=setting,
                reset_options=reset_options,
                t_env=getattr(runner, "t_env", None),
            )
    finally:
        if hasattr(runner, "test_returns"):
            runner.test_returns = test_returns_snapshot
        if hasattr(runner, "test_stats"):
            runner.test_stats = test_stats_snapshot
        if previous_budget is not None and mac is not None:
            mac.msg_budget_per_agent = previous_budget
        if previous_termination_setting is not None:
            if hasattr(runner, "set_env_attr"):
                runner.set_env_attr(
                    "terminate_on_task_completed",
                    previous_termination_setting,
                )
            elif hasattr(runner, "env") and hasattr(
                runner.env, "terminate_on_task_completed"
            ):
                runner.env.terminate_on_task_completed = previous_termination_setting

    if n_total != n_eval_eps:
        raise RuntimeError(
            f"Calibration runner returned terminal info for {n_total} "
            f"of {n_eval_eps} requested episodes."
        )

    return CalibrationRow(
        setting=setting,
        outcomes=tuple(outcomes),
        seed=int(seed),
    )


def collect_successful_final_state_dist(
    runner,
    n_eval_eps: int,
    reset_options: dict | None = None,
) -> tuple[dict, float]:
    """Evaluate a trained subtask policy and empirically estimate its successful
    terminal joint-state distribution and success rate.

    Used by dependent-subtask training to learn a successor subtask's
    `navigation_init_state_dist` from a predecessor's trained policy, instead of
    assuming a fixed hand-specified spawn distribution.

    Returns
    -------
    tuple[dict, float]
        (init_state_dist, success_rate) where init_state_dist follows the same
        `{"states": [...], "probs": [...]}` schema used by the navigation config YAML.
    """
    if n_eval_eps <= 0:
        raise ValueError("n_eval_eps must be greater than zero")

    if hasattr(runner, "set_env_attr"):
        runner.set_env_attr("terminate_on_task_completed", True)
    elif hasattr(runner.env, "terminate_on_task_completed"):
        runner.env.terminate_on_task_completed = True

    # snapshot test accounting so this collection pass doesn't pollute later evaluations
    test_returns_snapshot = list(getattr(runner, "test_returns", []))
    test_stats_snapshot = dict(getattr(runner, "test_stats", {}))

    batch_size = getattr(runner, "batch_size", 1)
    n_calls = -(-n_eval_eps // batch_size)  # ceil division

    successful_final_states: list[tuple] = []
    n_success, n_total = 0, 0

    try:
        for call_idx in range(n_calls):
            result = runner.run(
                test_mode=True,
                return_log_stats=call_idx == n_calls - 1,
                reset_options=reset_options,
            )
            final_infos = result.get("final_infos", [result.get("final_info")])
            for info in final_infos[: n_eval_eps - n_total]:
                if info is None:
                    raise RuntimeError(
                        "Evaluation runner did not return terminal info for an episode."
                    )
                n_total += 1
                if info.get("task_completed", False):
                    n_success += 1
                    successful_final_states.append(tuple(info["final_state"]))
    finally:
        if hasattr(runner, "test_returns"):
            runner.test_returns = test_returns_snapshot
        if hasattr(runner, "test_stats"):
            runner.test_stats = test_stats_snapshot

        if hasattr(runner, "set_env_attr"):
            runner.set_env_attr("terminate_on_task_completed", False)
        elif hasattr(runner.env, "terminate_on_task_completed"):
            runner.env.terminate_on_task_completed = False

    if n_total != n_eval_eps:
        raise RuntimeError(
            f"Evaluation runner returned terminal info for {n_total} of "
            f"{n_eval_eps} requested episodes."
        )

    success_rate = n_success / n_total if n_total > 0 else 0.0

    if not successful_final_states:
        return {"states": [], "probs": []}, success_rate

    unique_states, counts = np.unique(
        np.array(successful_final_states), return_counts=True, axis=0
    )
    init_state_dist = {
        "states": unique_states.tolist(),
        "probs": (counts / counts.sum()).tolist(),
    }
    return init_state_dist, success_rate


def eval_worker(
    args: SN,
    n_eval_eps: int,
    t_env: int,
    agent_state_dict: dict,
    logger_dir: str,
    wandb_config: dict,
    reset_options: dict,
) -> dict:
    """Worker function run inside a child process.

    Builds runner/mac/learner locally, loads model state dict, runs evaluation for `msg_budget_per_agent`, and returns stats dictionary.

    run_id is a wandb run id from the main process
    """
    # Minimal logger for worker
    logger = LocalLogger(
        dir=logger_dir,
        wandb_config=wandb_config,
        msg_budget_per_agent=reset_options["msg_budget_per_agent"],
    )

    # build env runner and other necessary objects
    args, runner, _, _ = build_sim(args, logger, agent_state_dict)

    result = run_eval_episodes(
        args=args,
        runner=runner,
        n_eval_eps=n_eval_eps,
        t_env=t_env,
        reset_options=reset_options,
    )

    logger.finish()

    return result
