try:
    # until python 3.10
    from collections import Mapping
except:
    # from python 3.10
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
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from utils.logging import get_logger
from run import run

# ensure to make sure the `protobuf` package works
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

ex = Experiment("pymarl")


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

    # run the framework
    run(_run, config, _log)


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
    # config_dict = {**config_dict, **env_config, **alg_config}
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

    try:
        map_name = config_dict["env_args"]["map_name"]
    except:
        if "key" not in config_dict["env_args"]:
            config_dict["env_args"]["key"] = config_dict["env"]
        map_name = config_dict["env_args"]["key"]

    # now add all the config to sacred
    if config_dict["use_sacred"]:
        sacred_capture_mode = "fd"
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

    # for param in params:
    #     if param.startswith("env_args.map_name"):
    #         map_name = param.split("=")[1]
    #     elif param.startswith("env_args.key"):
    #         map_name = param.split("=")[1]

    # Save to disk by default for sacred
    if config_dict["use_sacred"]:
        logger.info("Saving to FileStorageObserver in results/sacred.")
        results_path = join(dirname(dirname(abspath(__file__))), "results")
        file_obs_path = join(results_path, f"sacred/{config_dict['name']}/{map_name}")
        ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
