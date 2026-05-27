from typing import Optional
import os
from os.path import join
from collections import defaultdict
from hashlib import sha256
import json
import logging
import pandas as pd

import wandb
import numpy as np

WANDB_DIR = join("results", "wandb")


def _log_setup(step_metric: str, t: int) -> dict:
    return {step_metric: t}


class MainLogger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_wandb = False

        # deprecated, use wandb
        # self.use_tb = False
        # self.use_sacred = False

        self.stats = defaultdict(list)
        # self.stats = defaultdict(lambda: [])
        self.dir: str
        self.header = "=" * 25

    def info(self, log_str: str, log_header: bool = False):
        if log_header:
            self.console_logger.info(self.header)
            self.console_logger.info(log_str)
            self.console_logger.info(self.header)
        else:
            self.console_logger.info(log_str)

    def setup_wandb(
        self,
        config,
        team_name,
        project_name,
        mode,
        group_name: str = "",
        run_name: str = "",
        eval_run_id: Optional[str] = None,
    ):
        assert (
            team_name is not None and project_name is not None
        ), "W&B logging requires specification of both `wandb_team` and `wandb_project`."
        assert mode in [
            "offline",
            "online",
        ], f"Invalid value for `wandb_mode`. Received {mode} but only 'online' and 'offline' are supported."

        self.use_wandb = True
        self.data_table: Optional[wandb.Table] = None

        self.log_suffix = ""
        # load wandb run from server for evaluation
        if eval_run_id is not None:
            self.log_suffix = "_load"
            api = wandb.Api()
            load_path = join(project_name, eval_run_id)
            self.wandb_inactive = api.run(load_path)

        # define standardized group name
        if group_name == "":
            alg_name = config["name"]
            env_name = config["env"]
            if "map_name" in config["env_args"]:
                env_name += "_" + config["env_args"]["map_name"]
            elif "key" in config["env_args"] and config["env_args"]["key"] != env_name:
                env_name += "_" + config["env_args"]["key"]

            non_hash_keys = ["seed"]
            self.config_hash = sha256(
                json.dumps(
                    {k: v for k, v in config.items() if k not in non_hash_keys},
                    sort_keys=True,
                ).encode("utf8")
            ).hexdigest()[-10:]
            group_name = "_".join([alg_name, env_name, self.config_hash])

        # start a wandb run
        self.wandb = wandb.init(
            id=eval_run_id,
            name=run_name,
            entity=team_name,
            project=project_name,
            config=config,
            group=group_name,
            dir=WANDB_DIR,
            settings=wandb.Settings(
                x_label="main_proc",
                mode="shared",
                x_primary=True,
            ),
        )

        # extra setup to support "shared" mode for parallel subprocesses that
        # log to the same wandb run id
        self.step_metric = "t_env"
        self.wandb.define_metric("*", step_metric=self.step_metric)

        # save run files here
        self.dir = self.wandb.dir
        self.info(f"WANDB RUN ID: {self.wandb.id}", log_header=True)

        # accumulate data at same timestep and only log in one batch once
        # all data has been gathered
        self.wandb_current_t = -1
        self.wandb_current_data = {}

    def log_stat(self, key, value, t: int):
        """
        logging is delayed by period due to how finish() works
        """
        # used for printing stats periodically
        self.stats[key].append((t, value))

        if self.use_wandb:
            if self.wandb_current_t != t:
                self.wandb_current_data[self.step_metric] = self.wandb_current_t
                self.wandb.log(self.wandb_current_data)
                self.wandb_current_data = {}

            self.wandb_current_t = t
            self.wandb_current_data[key + self.log_suffix] = value

        """
        # deprecated, use wandb instead
        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

            self._run_obj.log_scalar(key, value, t)
        """

    def log_stat_table(self, df_data: pd.DataFrame, t: int):
        """Log accumulated evaluation statistics as a wandb table."""
        data = _log_setup(self.step_metric, t)

        if self.data_table is None:
            self.data_table = wandb.Table(dataframe=df_data, log_mode="MUTABLE")
        else:
            # add rows to the table
            for _, row in df_data.iterrows():
                self.data_table.add_data(*row.tolist())

        data[f"eval_stats{self.log_suffix}"] = self.data_table
        self.wandb.log(data)

    def log_image(self, column_name: str, image_path: str, t: int):
        data = _log_setup(self.step_metric, t)
        data[f"comms_eval/{column_name}{self.log_suffix}"] = wandb.Image(image_path)

        self.wandb.log(data=data)

    def log_replays(self, video_dir: str, t: int):
        # log all replays in a directory to a wandb run
        for _, _, videos in os.walk(video_dir):
            for video in videos:
                video_path = join(video_dir, video)
                video_name, extension = (
                    os.path.splitext(video)[0],
                    os.path.splitext(video)[1][1:],
                )

                data = _log_setup(self.step_metric, t)
                data[f"{video_name}_{extension}{self.log_suffix}"] = wandb.Video(
                    video_path, format=extension
                )
                self.wandb.log(data=data)

    def log_agent(self, save_path: str, t: int):
        # include environment timestep as metadata for the logged agent files
        artifact_name = "agent"
        metadata = {"t_env": t}

        # log agent models as a wandb artifact so we can attach metadata
        artifact = wandb.Artifact(name=artifact_name, type=artifact_name, metadata=metadata)
        # add the model files
        for root, _, files in os.walk(save_path):
            for f in files:
                artifact.add_file(join(root, f), name=f)

        self.wandb.log_artifact(artifact)


    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(
            *self.stats["episode"][-1]
        )
        i = 0
        for k, v in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(
                    np.mean([x[1].item() for x in self.stats[k][-window:]])
                )
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.info(log_str)

    def finish(self):
        if self.use_wandb:
            if self.wandb_current_data:
                self.wandb_current_data[self.step_metric] = self.wandb_current_t
                self.wandb.log(self.wandb_current_data)
            self.wandb.finish()

    """
    deprecated, use wandb since we developed many more advance logging features using that
    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value

        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True
        self.info(f"Tensorboard logging dir: {directory_name}", log_header=True)

    def setup_sacred(self, sacred_run_dict):
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True
    """


class LocalLogger:
    """Minimal logger used inside worker processes to avoid sharing main logger.
    Assumes wandb handles logging to a central log file and just prints to console."""

    class BasicConsoleLogger:
        """basic logger that just prints to terminal"""

        def info(self, *a, **k):
            print(*a)

    def __init__(self, dir: str, wandb_config: dict, comms_value: float):
        """
        Initialize the local logger.

        Parameters
        ----------
        dir : str
            Directory to save logs to
        run_id : str
            wandb run ID to use for centralized logging
        """
        self.dir = dir
        self.console_logger = LocalLogger.BasicConsoleLogger()

        # delete main process's connection so this parallel process can start its own unique connection to the wandb server when you call wandb.init()
        del os.environ["WANDB_SERVICE"]

        # Suppresses all terminal logging output
        os.environ["WANDB_SILENT"] = "true"

        # pick up the main wandb run in shared mode to log to the wandb website
        self.wandb = wandb.init(
            **wandb_config,
            dir=WANDB_DIR,
            settings=wandb.Settings(
                x_label=f"subproc_eval_{comms_value}",
                x_primary=False,
                mode="shared",
            ),
        )
        self.step_metric = "t_env"
        self.wandb.define_metric("*", step_metric=self.step_metric)

    def info(self, log_str: str):
        self.console_logger.info(log_str)

    def log_stat(self, key, value, t: int):
        data = _log_setup(self.step_metric, t)
        data[key] = value
        self.wandb.log(data)

    def log_replays(self, video_dir: str, t: int):
        # log all replays in a directory to a wandb run
        for _, _, videos in os.walk(video_dir):
            for video in videos:
                video_path = join(video_dir, video)
                video_name, extension = (
                    os.path.splitext(video)[0],
                    os.path.splitext(video)[1][1:],
                )
                data = _log_setup(self.step_metric, t)
                data[f"{video_name}_{extension}"] = wandb.Video(
                    video_path, format=extension
                )
                self.wandb.log(data=data)

    def finish(self):
        self.wandb.finish()


# set up a custom logger
def get_logger(name: Optional[str] = None):
    logger = logging.getLogger(name=name)
    logger.handlers = []

    # output to terminal
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("DEBUG")

    return logger
