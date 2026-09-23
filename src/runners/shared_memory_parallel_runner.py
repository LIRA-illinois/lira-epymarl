from copy import deepcopy
from functools import partial
from multiprocessing import Pipe, Process
from multiprocessing.shared_memory import SharedMemory
from os import makedirs
from os.path import join
from typing import Any

import numpy as np
import torch as th

from src.components.episode_buffer import EpisodeBatch
from src.envs import REGISTRY as env_REGISTRY
from src.envs import register_smac, register_smacv2
from src.utils.record_video import RecordVideoExtended


class SharedMemoryParallelRunner:
    """Parallel runner that transports array data through shared memory."""

    def __init__(self, args, logger) -> None:
        self.args = args
        self.logger = logger
        self.batch_size = args.batch_size_run

        if args.env == "sc2":
            register_smac()
        elif args.env == "sc2v2":
            register_smacv2()

        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)]
        )
        self.ps = []
        for worker_idx, (parent_conn, worker_conn) in enumerate(
            zip(self.parent_conns, self.worker_conns)
        ):
            env_args = args.env_args.copy()
            env_args["seed"] += worker_idx
            env_args["common_reward"] = args.common_reward
            env_args["reward_scalarisation"] = args.reward_scalarisation
            process = Process(
                target=shared_memory_env_worker,
                args=(worker_conn, args.env, env_args, worker_idx),
            )
            process.daemon = True
            process.start()
            self.ps.append(process)

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self._shms: dict[str, SharedMemory] = {}
        self._arrays: dict[str, np.ndarray] = {}
        self._allocate_shared_arrays()
        specs = self._shared_specs()
        for conn in self.parent_conns:
            conn.send(("attach_shared_memory", specs))
        for conn in self.parent_conns:
            conn.recv()

        self.t = 0
        self.t_env = 0
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.log_train_stats_t = -100000
        self._recording_folder = None
        self._recording_t_env = None

    @staticmethod
    def _shape(shape: Any) -> tuple[int, ...]:
        return (shape,) if isinstance(shape, int) else tuple(shape)

    def _allocate_shared_arrays(self) -> None:
        shapes = {
            "state": self._shape(self.env_info["state_shape"]),
            "obs": (
                self.env_info["n_agents"],
                *self._shape(self.env_info["obs_shape"]),
            ),
            "avail_actions": (
                self.env_info["n_agents"],
                self.env_info["n_actions"],
            ),
        }
        if "hl_state_shape" in self.env_info:
            shapes["hl_state"] = self._shape(self.env_info["hl_state_shape"])
        dtypes = {
            "state": np.float32,
            "obs": np.float32,
            "avail_actions": np.int32,
            "hl_state": np.float32,
        }
        for name, shape in shapes.items():
            sample = np.zeros(shape, dtype=dtypes[name])
            shm = SharedMemory(create=True, size=sample.nbytes * self.batch_size)
            self._shms[name] = shm
            self._arrays[name] = np.ndarray(
                (self.batch_size, *shape), dtype=sample.dtype, buffer=shm.buf
            )
            self._arrays[name].fill(0)

    def _shared_specs(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "name": self._shms[name].name,
                "shape": self._arrays[name].shape[1:],
                "dtype": str(self._arrays[name].dtype),
                "n_envs": self.batch_size,
            }
            for name in self._shms
        }

    def setup(self, scheme, groups, preprocess, mac) -> None:
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env_info

    def close_env(self) -> None:
        for conn in self.parent_conns:
            conn.send(("close", None))
        for process in self.ps:
            process.join(timeout=2)
        for conn in self.parent_conns:
            conn.close()
        for shm in self._shms.values():
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def save_replay(self) -> None:
        self.parent_conns[0].send(("save_replay", None))
        self.parent_conns[0].recv()

    def reset(self, options: dict | None = None) -> None:
        self.batch = self.new_batch()
        for conn in self.parent_conns:
            conn.send(("reset", options))
        for conn in self.parent_conns:
            conn.recv()
        self._write_pre_transition_data(ts=0)
        self.t = 0
        self.env_steps_this_run = 0

    def _write_pre_transition_data(self, ts: int) -> None:
        data = {
            "state": [self._arrays["state"][i].copy() for i in range(self.batch_size)],
            "avail_actions": [
                self._arrays["avail_actions"][i].copy() for i in range(self.batch_size)
            ],
            "obs": [self._arrays["obs"][i].copy() for i in range(self.batch_size)],
        }
        if "hl_state" in self._arrays:
            data["hl_state"] = [
                self._arrays["hl_state"][i].copy() for i in range(self.batch_size)
            ]
        self.batch.update(data, ts=ts)

    def run(
        self,
        test_mode: bool = False,
        return_log_stats: bool = True,
        reset_options: dict | None = None,
    ) -> EpisodeBatch | dict:
        self.reset(options=reset_options)
        episode_returns = (
            [0 for _ in range(self.batch_size)]
            if self.args.common_reward
            else [np.zeros(self.args.n_agents) for _ in range(self.batch_size)]
        )
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        final_infos = []

        while not all(terminated):
            active = [idx for idx, done in enumerate(terminated) if not done]
            if getattr(self.args, "action_selector", None) == "action_space":
                for idx in active:
                    self.parent_conns[idx].send(("sample_action", None))
                actions = np.stack(
                    [
                        self.parent_conns[idx].recv()
                        for idx in active
                    ]
                )
            else:
                actions = self.mac.select_actions(
                    self.batch,
                    t_ep=self.t,
                    t_env=self.t_env,
                    bs=active,
                    test_mode=test_mode,
                )
            if isinstance(actions, dict):
                env_actions = actions["env_actions"]
            else:
                env_actions = actions
            if isinstance(env_actions, th.Tensor):
                batch_actions = env_actions
                cpu_actions = env_actions.detach().cpu().numpy()
            else:
                cpu_actions = np.asarray(env_actions)
                batch_actions = cpu_actions
            self.batch.update(
                {"actions": batch_actions},
                bs=active,
                ts=self.t,
                mark_filled=False,
            )

            action_idx = 0
            for idx, conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1

            results = {}
            for idx, conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    results[idx] = conn.recv()

            post_transition_data = {"reward": [], "terminated": []}
            pre_transition_data = {"state": [], "avail_actions": [], "obs": []}
            for idx in active:
                reward, env_terminated, info = results[idx]
                if env_terminated:
                    final_infos.append(info)
                episode_returns[idx] += reward
                episode_lengths[idx] += 1
                self.env_steps_this_run += int(not test_mode)
                terminated[idx] = env_terminated

                post_transition_data["reward"].append((reward,))
                post_transition_data["terminated"].append(
                    (env_terminated and not info.get("episode_limit", False),)
                )
                pre_transition_data["state"].append(self._arrays["state"][idx].copy())
                pre_transition_data["avail_actions"].append(
                    self._arrays["avail_actions"][idx].copy()
                )
                pre_transition_data["obs"].append(self._arrays["obs"][idx].copy())
                if "hl_state" in self._arrays:
                    pre_transition_data.setdefault("hl_state", []).append(
                        self._arrays["hl_state"][idx].copy()
                    )

            self.batch.update(
                post_transition_data, bs=active, ts=self.t, mark_filled=False
            )
            self.t += 1
            self.batch.update(
                pre_transition_data, bs=active, ts=self.t, mark_filled=True
            )

        if not test_mode:
            self.t_env += self.env_steps_this_run
        log_stats = self._collect_stats(
            test_mode,
            return_log_stats,
            episode_returns,
            episode_lengths,
            final_infos,
        )
        if test_mode and return_log_stats:
            return {"batch": self.batch, "log_stats": log_stats}
        return self.batch

    def _collect_stats(
        self,
        test_mode,
        return_log_stats,
        episode_returns,
        episode_lengths,
        final_infos,
    ) -> dict:
        for conn in self.parent_conns:
            conn.send(("get_stats", None))
        env_stats = [conn.recv() for conn in self.parent_conns]
        returns = self.test_returns if test_mode else self.train_returns
        returns.extend(episode_returns)
        stats = self.test_stats if test_mode else self.train_stats
        final_info_keys = set().union(*(info.keys() for info in final_infos))
        env_stat_keys = set().union(*(info.keys() for info in env_stats))
        additional_env_stats = env_stat_keys - final_info_keys
        info_stats = final_infos if final_infos else env_stats
        stats.update(
            {
                key: stats.get(key, 0) + sum(info.get(key, 0) for info in info_stats)
                for key in set().union(*(info.keys() for info in info_stats))
            }
        )
        stats.update(
            {
                key: stats.get(key, 0)
                + sum(info.get(key, 0) for info in env_stats)
                for key in additional_env_stats
            }
        )
        stats["n_episodes"] = self.batch_size + stats.get("n_episodes", 0)
        stats["ep_length"] = sum(episode_lengths) + stats.get("ep_length", 0)
        prefix = "test_" if test_mode else ""
        log_stats = self._get_log_stats(returns, stats, prefix)

        if (
            not test_mode
            and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval
        ):
            for key, value in log_stats.items():
                self.logger.log_stat(key, value, self.t_env)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.train_returns.clear()
            self.train_stats.clear()
            self.log_train_stats_t = self.t_env

        if test_mode and return_log_stats:
            returns.clear()
            stats.clear()

        return log_stats

    def _get_log_stats(self, returns, stats, prefix: str) -> dict:
        log_stats = {}
        if self.args.common_reward:
            log_stats["return_mean"] = np.mean(returns)
            log_stats["return_std"] = np.std(returns)
        else:
            returns_array = np.asarray(returns)
            for agent_idx in range(self.args.n_agents):
                log_stats[f"agent_{agent_idx}_return_mean"] = returns_array[
                    :, agent_idx
                ].mean()
                log_stats[f"agent_{agent_idx}_return_std"] = returns_array[
                    :, agent_idx
                ].std()
            total_returns = returns_array.sum(axis=-1)
            log_stats["total_return_mean"] = total_returns.mean()
            log_stats["total_return_std"] = total_returns.std()

        for key, value in stats.items():
            if key != "n_episodes":
                log_stats[f"{key}_mean"] = value / stats["n_episodes"]
            else:
                log_stats[key] = value

        return {f"{prefix}{key}": value for key, value in log_stats.items()}

    def set_env_attr(self, name: str, value: Any) -> None:
        for conn in self.parent_conns:
            conn.send(("set_attr", (name, value)))
        for conn in self.parent_conns:
            conn.recv()

    def get_env_rng_states(self):
        for conn in self.parent_conns:
            conn.send(("get_rng_state", None))
        return [conn.recv() for conn in self.parent_conns]

    def set_env_rng_states(self, states) -> None:
        for conn, state in zip(self.parent_conns, states):
            conn.send(("set_rng_state", state))
        for conn in self.parent_conns:
            conn.recv()

    def start_recording(
        self,
        n_test_replays_save: int,
        video_prefix: str = "replay",
        t_env: int | None = None,
    ) -> None:
        if t_env is None:
            t_env = self.t_env
        replay_folder = join(self.logger.dir, "replays", f"t_{t_env}")
        makedirs(replay_folder, exist_ok=True)
        self.parent_conns[0].send(
            (
                "start_recording",
                {
                    "video_folder": replay_folder,
                    "video_prefix": video_prefix,
                },
            )
        )
        self.parent_conns[0].recv()
        self._recording_folder = replay_folder
        self._recording_t_env = t_env
        self.logger.info(f"Saving {n_test_replays_save} test episode replays")

    def stop_recording(
        self, t_env: int | None = None, video_prefix: str = "replays"
    ) -> None:
        self.parent_conns[0].send(("stop_recording", None))
        self.parent_conns[0].recv()
        if self._recording_folder is not None:
            self.logger.log_videos(
                dir=self._recording_folder,
                t=self._recording_t_env if t_env is None else t_env,
                video_prefix=video_prefix,
            )
        self._recording_folder = None
        self._recording_t_env = None


def shared_memory_env_worker(remote, env_name, env_args, worker_idx) -> None:
    env = env_REGISTRY[env_name](**env_args)
    shared_arrays = {}
    shared_memory = {}

    while True:
        command, data = remote.recv()

        if command == "get_env_info":
            env_obj: Any = env
            remote.send(env_obj.get_env_info())
        elif command == "attach_shared_memory":
            for name, spec in data.items():
                shm = SharedMemory(name=spec["name"])
                shared_memory[name] = shm
                shared_arrays[name] = np.ndarray(
                    (spec["n_envs"], *spec["shape"]),
                    dtype=np.dtype(spec["dtype"]),
                    buffer=shm.buf,
                )
            remote.send("attached")
        elif command == "reset":
            if data is None:
                env.reset()
            else:
                env.reset(options=data)
            _write_shared_state(env, shared_arrays, worker_idx)
            remote.send("reset")
        elif command == "step":
            _, reward, terminated, truncated, info = env.step(data)
            _write_shared_state(env, shared_arrays, worker_idx)
            remote.send((reward, terminated or truncated, info))
        elif command == "sample_action":
            remote.send(np.asarray(env.action_space.sample()))
        elif command == "save_replay":
            if hasattr(env, "save_replay"):
                env.save_replay()
            remote.send("replay_saved")
        elif command == "get_stats":
            remote.send(env.get_stats() if hasattr(env, "get_stats") else {})
        elif command == "start_recording":
            env = RecordVideoExtended(
                env=env,
                video_folder=data["video_folder"],
                episode_trigger=lambda _: True,
                name_prefix=data["video_prefix"],
                output_formats=["mp4"],
            )
            remote.send("recording")
        elif command == "stop_recording":
            if isinstance(env, RecordVideoExtended):
                if env.recording:
                    env.stop_recording()
                env = env.env
            remote.send("recording_stopped")
        elif command == "set_attr":
            name, value = data
            _set_env_attr(env, name, value)
            remote.send("set")
        elif command == "get_rng_state":
            rng = _get_env_attr(env, "np_random")
            remote.send(deepcopy(rng.bit_generator.state))
        elif command == "set_rng_state":
            rng = _get_env_attr(env, "np_random")
            rng.bit_generator.state = data
            remote.send("set")
        elif command == "close":
            env.close()
            for shm in shared_memory.values():
                shm.close()
            remote.close()
            break
        else:
            raise NotImplementedError(command)


def _write_shared_state(env, shared_arrays, worker_idx) -> None:
    state = env.state
    if isinstance(state, dict):
        np.copyto(shared_arrays["state"][worker_idx], np.asarray(state["ll_state"]))
        if "hl_state" in shared_arrays:
            np.copyto(
                shared_arrays["hl_state"][worker_idx],
                np.asarray(state["hl_state"]),
            )
    else:
        np.copyto(shared_arrays["state"][worker_idx], np.asarray(state))
    np.copyto(
        shared_arrays["avail_actions"][worker_idx],
        np.asarray(env.avail_actions),
    )
    np.copyto(shared_arrays["obs"][worker_idx], np.asarray(env.obs))


def _get_env_attr(env, name: str):
    env_obj: Any = env
    if hasattr(env_obj, "get_wrapper_attr"):
        return env_obj.get_wrapper_attr(name)
    return getattr(env_obj, name)


def _set_env_attr(env, name: str, value: Any) -> None:
    env_obj: Any = env
    if hasattr(env_obj, "set_wrapper_attr"):
        env_obj.set_wrapper_attr(name, value)
    else:
        setattr(env_obj, name, value)
