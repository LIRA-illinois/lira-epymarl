from logging import Logger
from typing import Optional
import os
from os.path import join
from collections import defaultdict
from hashlib import sha256
import json
import logging
import pandas as pd
from shutil import rmtree

import wandb
import numpy as np

# 10 minute timeout to try to prevent crashes due to wandb
# API upload limits
os.environ["WANDB_HTTP_TIMEOUT"] = "600"
RESULTS_DIR = "results"


def log_setup(step_metric: str, t: int) -> dict:
    return {step_metric: t}


class MainLogger:
    def __init__(self, console_logger, config, args) -> None:
        self.console_logger = console_logger

        self.use_wandb = False

        # deprecated, use wandb
        # self.use_tb = False
        # self.use_sacred = False

        self.stats = defaultdict(list)
        # self.stats = defaultdict(lambda: [])
        self.dir: str

        self.header = "=" * 25
        self.data_tables: dict = {}
        self.step_metric = "t_env"
        self.log_suffix = ""

        self._setup(config, args)

    def _setup(self, config, args) -> None:
        if args.use_wandb:
            if args.run_name != "":
                run_name = args.run_name
            else:
                if args.wandb_group != "":
                    run_name = args.wandb_group + f"_seed_{args.seed}"
                elif args.time_id != "" and args.scenario != "":
                    run_name = f"{args.time_id}_sc_{args.scenario}_seed_{args.seed}"
                else:
                    run_name = args.unique_token

            self._setup_wandb(
                config=config,
                team_name=args.wandb_team,
                project_name=args.wandb_project,
                group_name=args.run_name,
                run_name=run_name,
                mode=args.wandb_mode,
                eval_run_id=args.eval_run_id,
            )

        else:
            self.dir: str = os.path.join(RESULTS_DIR, "data")
            os.makedirs(self.dir, exist_ok=True)

    def _setup_wandb(
        self,
        config,
        team_name,
        project_name,
        mode,
        group_name: str = "",
        run_name: str = "",
        eval_run_id: Optional[str] = None,
    ) -> None:
        self.use_wandb = True

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
            dir=RESULTS_DIR,
            settings=wandb.Settings(
                x_label="main_proc",
                mode=mode,
                x_primary=True,
            ),
        )

        # extra setup to support "shared" mode for parallel subprocesses that
        # log to the same wandb run id
        self.wandb.define_metric("*", step_metric=self.step_metric)

        # save run files here
        self.dir = self.wandb.dir
        self.info(f"WANDB RUN ID: {self.wandb.id}", log_header=True)

        # accumulate data at same timestep and only log in one batch once
        # all data has been gathered
        self.wandb_current_t = -1
        self.wandb_current_data = {}

    def info(self, log_str: str, log_header: bool = False) -> None:
        if log_header:
            self.console_logger.info(self.header)
            self.console_logger.info(log_str)
            self.console_logger.info(self.header)
        else:
            self.console_logger.info(log_str)

    def log_stat(self, key: str, value: float, t: int) -> None:
        """
        logging is delayed by period due to how finish() works
        used for printing stats periodically
        """
        # used for printing stats periodically
        self.stats[key].append((t, value))

        if self.use_wandb:
            if self.wandb_current_t != t:
                self.wandb_current_data[self.step_metric] = self.wandb_current_t
                self.wandb.log(self.wandb_current_data)
                self.wandb_current_data = {}

            self.wandb_current_t = t
            self.wandb_current_data[f"{key}{self.log_suffix}"] = value

    def log_image(self, image_path: str, t: int, key: str = "") -> None:
        if self.use_wandb:
            data = log_setup(self.step_metric, t)
            data[f"{key}{self.log_suffix}"] = wandb.Image(image_path)

            self.wandb.log(data=data)

    def log_images(self, dir: str, t: int, key: str = "", group: str = "") -> None:
        """logs all images in a given directory to a wandb run, then removes the original directory to avoid replicated data on disk"""
        if self.use_wandb:
            # log each image separately
            for _, _, files in os.walk(dir):
                for file in files:
                    data = log_setup(self.step_metric, t)
                    path = join(dir, file)
                    fn = os.path.splitext(file)[0]
                    data[f"{group}{key}{fn}{self.log_suffix}"] = wandb.Image(path)
                    self.wandb.log(data=data)

            # remove image dir to avoid double-logging
            path = dir.split("/")

            if path[-1] == "images":
                path_delete = dir
            else:
                # cd up a dir since video_dir has the time in its name
                path_delete = join("/", *path[:-1])

            # needs an absolute path to work correctly
            rmtree(path_delete)

    def log_videos(
        self,
        dir: str,
        t: int,
        video_prefix: str = "replay",
    ) -> None:
        """logs all videos in a given directory to a wandb run, then removes the original directory to avoid replicated data on disk"""
        if self.use_wandb:
            # log all replays in a directory to a wandb run, concat videos
            # to a list then log the list for better visualization on the website
            data = log_setup(self.step_metric, t)
            video_list = []
            for _, _, videos in os.walk(dir):
                for video in videos:
                    video_path = join(dir, video)
                    extension = os.path.splitext(video)[1][1:]
                    video_list.append(wandb.Video(video_path, format=extension))

            data[f"{video_prefix}{self.log_suffix}"] = video_list
            self.wandb.log(data=data)

            # remove video_dir to avoid double-logging replays
            path = dir.split("/")
            if path[-1] == "replays":
                path_delete = dir
            else:
                # cd up a dir since video_dir has the time in its name
                path_delete = join("/", *path[:-1])

            # needs an absolute path to work correctly
            rmtree(path_delete)

    def log_table(self, key: str, value: pd.DataFrame, t: int) -> None:
        """Log accumulated evaluation statistics as a wandb table"""
        if isinstance(value, pd.DataFrame):
            if not self.data_tables.get(key, False):
                # make a new entry
                self.data_tables[key] = wandb.Table(dataframe=value, log_mode="MUTABLE")
            else:
                # add rows to the existing table from the dataframe
                for _, row in value.iterrows():
                    self.data_tables[key].add_data(*row.tolist())

        if self.use_wandb:
            # tables handled similar to artifacts, but need to add a t_env column to the table first
            table = self.data_tables[key]
            if self.step_metric not in table.columns:
                table.add_column(name=self.step_metric, data=[t] * len(table.data))

            # data must only have 1 key for the table to be interpreted as a table on wandb
            data = {f"{key}{self.log_suffix}": table}
            self.wandb.log(data)

    def log_agent(self, save_path: str, t: int) -> None:
        """logs agent models to a wandb run as an artifact"""
        if self.use_wandb:
            # include environment timestep as metadata for the logged agent files
            artifact_name = "agent"
            metadata = {f"{self.step_metric}": t}

            # log agent models as a wandb artifact so we can attach metadata
            artifact = wandb.Artifact(
                name=artifact_name, type=artifact_name, metadata=metadata
            )
            # add the model files
            for root, _, files in os.walk(save_path):
                for f in files:
                    artifact.add_file(join(root, f), name=f)

            self.wandb.log_artifact(artifact)

    def print_recent_stats(self) -> None:
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
            except ValueError:
                item = "{:.4f}".format(
                    np.mean([x[1].item() for x in self.stats[k][-window:]])
                )
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.info(log_str)

    def finish(self) -> None:
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

        def info(self, *a, **k) -> None:
            print(*a)

    def __init__(self, dir: str, wandb_config: dict, msg_budget_per_agent: float) -> None:
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
        # doesn't support
        self.log_suffix = ""

        # delete main process's connection so this parallel process can start its own unique connection to the wandb server when you call wandb.init()
        del os.environ["WANDB_SERVICE"]

        # Suppresses all terminal logging output
        os.environ["WANDB_SILENT"] = "true"

        # pick up the main wandb run in shared mode to log to the wandb website
        self.wandb = wandb.init(
            **wandb_config,
            dir=RESULTS_DIR,
            settings=wandb.Settings(
                x_label=f"subproc_eval_{msg_budget_per_agent}",
                x_primary=False,
                mode="shared",
            ),
        )
        self.step_metric = "t_env"
        self.wandb.define_metric("*", step_metric=self.step_metric)

    def info(self, log_str: str) -> None:
        self.console_logger.info(log_str)

    def log_stat(self, key, value, t: int) -> None:
        data = log_setup(self.step_metric, t)
        data[key] = value
        self.wandb.log(data)

    def log_videos(
        self,
        video_dir: str,
        t: int,
        video_prefix: str = "replay",
    ) -> None:
        # log all replays in a directory to a wandb run as a table for easier visualization
        data = log_setup(self.step_metric, t)
        # log all replays in a directory to a wandb run
        video_list = []
        for _, _, videos in os.walk(video_dir):
            for video in videos:
                video_path = join(video_dir, video)
                extension = os.path.splitext(video)[1][1:]
                video_list.append(wandb.Video(video_path, format=extension))

        data[f"{video_prefix}{self.log_suffix}"] = video_list
        self.wandb.log(data=data)

    def finish(self) -> None:
        self.wandb.finish()


# set up a custom logger
def get_logger(name: Optional[str] = None) -> Logger:
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
