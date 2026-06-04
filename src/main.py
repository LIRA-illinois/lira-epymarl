import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Callable
from collections.abc import Mapping
from copy import deepcopy
from os.path import dirname, abspath, join
import sys
import yaml
import numpy as np
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import torch as th
import logging

logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
logging.getLogger("PIL").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)


from utils.logging import get_logger
from simulation.run import Simulation

# ensure to make sure the `protobuf` package works (only used for tensorboard, may not be needed?)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"
ex = Experiment("pymarl")


def string_inputs_to_list(config: dict, key: str, output_type: Callable)-> dict:
    """Converts space-delimited string list to list of "type" for the given key in the config dictionary."""
    eval_str = config[key].split("=")[0][1:-1]
    float_values: list[float] = [output_type(x) for x in eval_str.split(" ")]
    config[key] = float_values

    return config


@ex.main
def my_main(_run, _config, _log):
    def config_copy(config):
        if isinstance(config, dict):
            return {k: config_copy(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [config_copy(v) for v in config]
        else:
            return deepcopy(config)

    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config["env_args"]["seed"] = config["seed"]

    if "comms_values_eval" in config:
        config = string_inputs_to_list(config, "comms_values_eval", output_type=float)
    if "hl_task" in config:
        config = string_inputs_to_list(config, "hl_task", output_type=int)

    # run the framework
    sim = Simulation(_run, config, _log)
    sim.run_sim()
    sim.finish()


def get_run_config(params) -> dict:
    def recursive_dict_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping):
                d[k] = recursive_dict_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def _get_config(params, arg_name, subfolder):
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
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")

    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    for param in params:
        if param.startswith("use_sacred"):
            config_dict["use_sacred"] = param.split("=")[1].lower() == "true"

    return config_dict


if __name__ == "__main__":
    params = deepcopy(sys.argv)
    th.set_num_threads(1)
    config_dict = get_run_config(params)

    if "key" not in config_dict["env_args"]:
        config_dict["env_args"]["key"] = config_dict["env"]

    # now add all the config to sacred
    if config_dict["use_sacred"]:
        sacred_capture_mode = "no"
        logger = get_logger()
        ex.logger = logger
        ex.captured_out_filter = apply_backspaces_and_linefeeds
    else:
        # set to "no" if you want to see stdout/stderr in console
        sacred_capture_mode = "no"
        # disable most Sacred logging
        ex.add_config({"debug": True})

    SETTINGS["CAPTURE_MODE"] = sacred_capture_mode
    ex.add_config(config_dict)

    map_name = ""
    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]

    # Save to disk by default for sacred
    if config_dict["use_sacred"]:
        logger.info("Saving to FileStorageObserver in results/sacred.")
        results_path = join(dirname(dirname(abspath(__file__))), "results")
        file_obs_path = join(
            results_path, "sacred", config_dict["name"], config_dict["env"], map_name
        )
        ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
