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
    comms_value: Optional[float] = None,
):
    """Run n_eval_eps evaluation episodes, optionally recording, and return last result."""
    if comms_value is not None:
        runner.mac.update_comms_value(comms_value)
        if args.save_test_replays:
            video_prefix = f"comms_{comms_value:.2f}"

    if args.save_test_replays:
        runner.start_recording(
            n_test_replays_save=args.n_test_replays_save,
            video_prefix=video_prefix,
            t_env=t_env,
        )

    last_result = None
    for i in range(n_eval_eps):
        if i % 50 == 0:
            runner.logger.info(
                f"Test Episode: {i} / {n_eval_eps}"
            )

        return_stats = i == n_eval_eps - 1
        # last_result only has "log_stats" in it after all eps have run
        last_result = runner.run(test_mode=True, return_log_stats=return_stats)

        # Stop recording after some episodes
        if args.save_test_replays and i >= args.n_test_replays_save:
            runner.stop_recording(t_env=t_env)

    if comms_value is not None:
        last_result["log_stats"]["t_env"] = t_env
        last_result["log_stats"]["comms_value"] = comms_value

    return last_result


def eval_worker(
    comms_value: float,
    args: SN,
    n_eval_eps: int,
    t_env: int,
    agent_state_dict: dict,
    logger_dir: str,
    wandb_config: dict,
) -> dict:
    """Worker function run inside a child process.

    Builds runner/mac/learner locally, loads model state dict, runs evaluation for `comms_value`, and returns stats dictionary.

    run_id is a wandb run id from the main process
    """
    # Minimal logger for worker
    logger = LocalLogger(logger_dir, wandb_config, comms_value)

    # build env runner and other necessary objects
    args, runner, _, _ = build_sim(args, logger, agent_state_dict)

    result = run_eval_episodes(
        args=args,
        runner=runner,
        n_eval_eps=n_eval_eps,
        t_env=t_env,
        comms_value=comms_value,
    )

    logger.finish()

    return result
