from types import SimpleNamespace as SN
from typing import Optional

from utils.logging import LocalLogger
from .build import build_sim


def run_eval_episodes(
    args: SN,
    runner,
    n_eval_eps: int,
    video_prefix: str = "replay",
    t_env: Optional[int] = None,
    reset_options: Optional[dict] = None,
):
    """Run n_eval_eps evaluation episodes, optionally recording, and return last result."""
    # derive identifiers
    task_state, comms_value = None, None
    if reset_options is not None:
        task_state = reset_options.get("hl_start_state", None)
        comms_value = reset_options.get("comms_value", None)

    if comms_value is not None:
        print(f"Setting MAC comms value to {comms_value}")
        runner.mac.update_comms_value(comms_value)

    # filename prefix includes task state and comms value when available
    prefix_parts = [video_prefix]
    if task_state is not None:
        prefix_parts.append(f"task_{int(task_state)}")
    if comms_value is not None:
        prefix_parts.append(f"comms_{comms_value:.2f}")
    file_name_prefix = "-".join(prefix_parts)

    # If the runner's env supports terminating on task completion, enable it for evaluation
    if hasattr(runner.env, "terminate_on_task_completed"):
        runner.env.terminate_on_task_completed = True

    if args.save_test_replays:
        runner.start_recording(
            n_test_replays_save=args.n_test_replays_save,
            video_prefix=file_name_prefix,
            t_env=t_env,
        )

    last_result = None
    for i in range(n_eval_eps):
        if i % 50 == 0:
            runner.logger.info(f"Test Episode: {i} / {n_eval_eps}")

        return_stats = i == n_eval_eps - 1
        # last_result only has "log_stats" in it after all eps have run

        last_result = runner.run(
            test_mode=True,
            return_log_stats=return_stats,
            reset_options=reset_options,
        )

        # Stop recording after some episodes
        # -1 b/c i 0 indexed
        if args.save_test_replays and i == args.n_test_replays_save - 1:
            runner.stop_recording(t_env=t_env, video_prefix=file_name_prefix)

    last_result["log_stats"]["t_env"] = t_env

    # log stuff like current HL task and comms action
    for k, v in reset_options.items():
        last_result["log_stats"][k] = v

    # restore terminate_on_task_completed to False after evaluation
    if hasattr(runner.env, "terminate_on_task_completed"):
        runner.env.terminate_on_task_completed = False

    if comms_value is not None:
        print("Evaluation done, setting MAC comms value to default of 1.0")
        runner.mac.update_comms_value(1.0)

    return last_result


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

    Builds runner/mac/learner locally, loads model state dict, runs evaluation for `comms_value`, and returns stats dictionary.

    run_id is a wandb run id from the main process
    """
    # Minimal logger for worker
    logger = LocalLogger(
        dir=logger_dir,
        wandb_config=wandb_config,
        comms_value=reset_options["comms_value"],
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
