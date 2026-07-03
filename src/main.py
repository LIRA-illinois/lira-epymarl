from logging import Logger
import logging
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from collections.abc import Mapping
from copy import deepcopy
from os.path import abspath, dirname, join

import numpy as np
import yaml
from torch import manual_seed as th_manual_seed
from torch import set_num_threads as th_set_num_threads

from src.simulation.run import Simulation
from src.utils.logging import get_logger
from src.utils.utils import get_config_updates, string_inputs_to_list

logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
logging.getLogger("PIL").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)


def experiment_main(_config, logger: Logger) -> None:
    def config_copy(config):
        if isinstance(config, dict):
            return {k: config_copy(v) for k, v in config.items()}
        if isinstance(config, list):
            return [config_copy(v) for v in config]
        return deepcopy(config)

    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th_manual_seed(config["seed"])
    config["env_args"]["seed"] = config["seed"]

    if "comms_values" in config:
        config = string_inputs_to_list(config, "comms_values", output_type=float)
    if "hl_task" in config:
        config = string_inputs_to_list(config, "hl_task", output_type=int)

    # run the framework
    sim = Simulation(config, logger)
    sim.run()
    sim.finish()


def get_run_config(params: str | None) -> dict:
    def recursive_dict_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping):
                d[k] = recursive_dict_update(d.get(k, {}), v)
            else:
                d[k] = v

        return d

    def _get_config(params: str | None, arg_name: str, subfolder: str):
        config_name = None
        for _i, _v in enumerate(params):
            if _v.split("=")[0] == arg_name:
                config_name = _v.split("=")[1]
                del params[_i]
                break

        if config_name is not None:
            with open(
                join(
                    dirname(__file__),
                    "config",
                    subfolder,
                    "{}.yaml".format(config_name),
                ),
                "r",
            ) as f:
                try:
                    config_dict = yaml.load(f, Loader=yaml.FullLoader)
                except yaml.YAMLError as exc:
                    assert False, "{}.yaml error: {}".format(config_name, exc)
            return config_dict

    # Get the defaults from default.yaml
    with open(join(dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"default.yaml error: {exc}") from exc

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")

    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # get updates from commandline params
    tmp, _ = get_config_updates(params)
    config_updates = {}
    for k, v in tmp.items():
        if k.startswith("-"):
            config_updates[k.strip("-")] = v
            continue
        config_updates[k] = v
    config_dict = recursive_dict_update(config_dict, config_updates)

    return config_dict


def main(params: str | None = None) -> None:
    # argv can be passed as a space-delimited string of args
    # if you do that, it's the same as getting sys.argv
    if params is None:
        params = deepcopy(sys.argv)
    th_set_num_threads(1)

    config_dict = get_run_config(params)
    logger = get_logger()

    if "key" not in config_dict["env_args"]:
        config_dict["env_args"]["key"] = config_dict["env"]

    # sacred is off by default
    if not config_dict["use_sacred"]:
        experiment_main(config_dict, logger)

    else:
        from sacred import SETTINGS, Experiment
        from sacred.observers import FileStorageObserver
        from sacred.utils import apply_backspaces_and_linefeeds

        ex = Experiment("lira-epymarl")

        @ex.main
        def sacred_main(_run, _config, _log) -> None:
            experiment_main(_config, _log)

        # set to "no" if you want to see stdout/stderr in console
        sacred_capture_mode = "no"
        ex.logger = logger
        ex.captured_out_filter = apply_backspaces_and_linefeeds
        SETTINGS["CAPTURE_MODE"] = sacred_capture_mode
        ex.add_config(config_dict)

        # Save to disk by default
        # update the map name param for run storage
        map_name = ""
        for param in params:
            if param.startswith("env_args.map_name"):
                map_name = param.split("=")[1]
        logger.info("Saving to FileStorageObserver in results/sacred.")
        results_path = join(dirname(dirname(abspath(__file__))), "results")
        file_obs_path = join(
            results_path, "sacred", config_dict["name"], config_dict["env"], map_name
        )
        ex.observers.append(FileStorageObserver.create(file_obs_path))
        ex.run_commandline(params)


if __name__ == "__main__":
    main()
