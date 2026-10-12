"""Ray orchestration for node-local shared-memory environment pools."""

from copy import deepcopy
from functools import partial
from os import makedirs
from os.path import join
from pathlib import Path
import random
from shutil import rmtree
from tempfile import mkdtemp
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch as th

from src.components.episode_buffer import EpisodeBatch
from src.runners.shared_memory_parallel_runner import (
    SharedMemoryParallelRunner,
    _get_env_attr,
    _set_env_attr,
)

_ACTIVE_RAY_RUNNERS = 0
_RAY_STARTED_BY_RUNNER = False


def _set_cpu_args(args) -> None:
    if hasattr(args, "device"):
        args.device = "cpu"
    if hasattr(args, "use_cuda"):
        args.use_cuda = False


def _nonleaf_tensor_memo(value, memo: dict, seen: set[int] | None = None) -> dict:
    if seen is None:
        seen = set()
    if isinstance(value, th.Tensor):
        if not value.is_leaf:
            memo[id(value)] = value.detach().cpu().clone()
        return memo
    if value is None or id(value) in seen:
        return memo
    seen.add(id(value))

    if isinstance(value, dict):
        items = (item for pair in value.items() for item in pair)
    elif isinstance(value, (list, tuple, set)):
        items = iter(value)
    elif isinstance(
        value, (th.nn.Module, SimpleNamespace)
    ) or value.__class__.__module__.startswith("src."):
        items = iter(vars(value).values())
    else:
        return memo

    for item in items:
        _nonleaf_tensor_memo(item, memo, seen)
    return memo


def _cpu_policy_copy(policy):
    policy_copy = deepcopy(policy, memo=_nonleaf_tensor_memo(policy, {}))
    seen = set()

    def move_to_cpu(value):
        if value is None or id(value) in seen:
            return value
        seen.add(id(value))

        if isinstance(value, th.Tensor):
            return value.detach().cpu()
        if isinstance(value, th.nn.Module):
            value.cpu()
            for module in value.modules():
                if hasattr(module, "args"):
                    _set_cpu_args(module.args)
            return value
        if isinstance(value, dict):
            for key, item in list(value.items()):
                value[key] = move_to_cpu(item)
            return value
        if isinstance(value, list):
            value[:] = [move_to_cpu(item) for item in value]
            return value
        if isinstance(value, tuple):
            return tuple(move_to_cpu(item) for item in value)
        if isinstance(value, set):
            return {move_to_cpu(item) for item in value}
        if isinstance(value, SimpleNamespace):
            _set_cpu_args(value)
            for name, item in vars(value).copy().items():
                setattr(value, name, move_to_cpu(item))
            return value
        if hasattr(value, "args"):
            _set_cpu_args(value.args)
        if value.__class__.__module__.startswith("src.") and hasattr(
            value, "__dict__"
        ):
            for name, item in vars(value).copy().items():
                if name != "args":
                    setattr(value, name, move_to_cpu(item))
        return value

    return move_to_cpu(policy_copy)


def _validate_mac_env_binding(mac, seen=None) -> None:
    if seen is None:
        seen = set()
    if id(mac) in seen:
        return
    seen.add(id(mac))

    selector = getattr(mac, "action_selector", None)
    if callable(getattr(selector, "set_env", None)):
        raise ValueError(
            "parallel_ray does not support action selectors that require a "
            "local environment object."
        )

    env_mac = getattr(mac, "env_mac", None)
    if env_mac is not None:
        _validate_mac_env_binding(env_mac, seen)


class _NodeLogger:
    def __init__(self) -> None:
        self.dir = mkdtemp(prefix="ray-node-runner-")

    def info(self, message: str) -> None:
        return None

    def log_stat(self, key: str, value: Any, t: int) -> None:
        return None

    def log_videos(self, **kwargs) -> None:
        return None


class _RayNodeWorker:
    """A Ray actor owning one node-local SharedMemoryParallelRunner pool."""

    def __init__(
        self,
        args,
        local_batch_size: int,
        seed_offset: int,
    ) -> None:
        if getattr(args, "live_render", False):
            raise ValueError("live_render is not supported by parallel_ray.")

        node_args = deepcopy(args)
        node_args.batch_size_run = local_batch_size
        node_args.device = "cpu"
        node_args.use_cuda = False
        node_args.env_args = deepcopy(args.env_args)
        node_args.env_args["seed"] += seed_offset
        node_args.runner_log_interval = 2**63 - 1

        worker_seed = int(args.env_args["seed"]) + seed_offset
        random.seed(worker_seed)
        np.random.seed(worker_seed % 2**32)
        th.manual_seed(worker_seed)

        self.local_batch_size = local_batch_size
        self.logger = _NodeLogger()
        try:
            self.runner = SharedMemoryParallelRunner(node_args, self.logger)
        except Exception:
            rmtree(self.logger.dir)
            raise
        self._is_setup = False

    def setup(self, scheme, groups, preprocess, mac) -> None:
        mac = deepcopy(mac)
        _validate_mac_env_binding(mac)
        self.runner.setup(scheme, groups, preprocess, mac)
        self._is_setup = True

    def get_env_info(self) -> dict:
        return self.runner.get_env_info()

    def run_batch(
        self,
        policy,
        test_mode: bool,
        t_env: int,
        reset_options: dict | None,
    ) -> dict:
        if not self._is_setup:
            raise RuntimeError("Ray node worker has not been set up.")

        policy = deepcopy(policy)
        _validate_mac_env_binding(policy)
        self.runner.mac = policy
        self.runner.t_env = t_env

        result = self.runner.run(
            test_mode=test_mode,
            return_log_stats=False,
            reset_options=reset_options,
        )
        if test_mode:
            batch = result["batch"]
            final_infos = result["final_infos"]
            returns = self.runner.test_returns
            stats = self.runner.test_stats
            self.runner.test_returns = []
            self.runner.test_stats = {}
        else:
            batch = result
            final_infos = []
            returns = self.runner.train_returns
            stats = self.runner.train_stats
            self.runner.train_returns = []
            self.runner.train_stats = {}

        return {
            "batch": batch,
            "final_infos": final_infos,
            "returns": deepcopy(returns),
            "stats": deepcopy(stats),
            "env_steps": self.runner.env_steps_this_run,
            "t": self.runner.t,
            "epsilon": getattr(
                getattr(policy, "action_selector", None), "epsilon", None
            ),
            "episode_index": getattr(policy, "_episode_index", None),
        }

    def set_env_attr(self, name: str, value: Any) -> None:
        self.runner.set_env_attr(name, value)

    def get_env_attr(self, name: str, default=None) -> Any:
        return self.runner.get_env_attr(name, default)

    def get_rng_states(self) -> list[dict]:
        return self.runner.get_env_rng_states()

    def set_rng_states(self, states: list[dict]) -> None:
        if len(states) != self.local_batch_size:
            raise ValueError(
                f"Expected {self.local_batch_size} environment RNG states, "
                f"got {len(states)}."
            )
        self.runner.set_env_rng_states(states)

    def save_replay(self) -> None:
        self.runner.save_replay()

    def start_recording(
        self,
        n_test_replays_save: int,
        video_prefix: str,
        t_env: int,
    ) -> None:
        self.runner.start_recording(
            n_test_replays_save=n_test_replays_save,
            video_prefix=video_prefix,
            t_env=t_env,
        )

    def stop_recording(self, t_env: int, video_prefix: str) -> list:
        self.runner.stop_recording(t_env=t_env, video_prefix=video_prefix)
        video_folder = Path(self.logger.dir) / "replays" / f"t_{t_env}"
        videos = (
            [
                (path.name, path.read_bytes())
                for path in sorted(video_folder.glob("*"))
                if path.is_file()
            ]
            if video_folder.exists()
            else []
        )
        if video_folder.exists():
            rmtree(video_folder)
        return videos

    def close(self) -> None:
        try:
            self.runner.close_env()
        finally:
            if Path(self.logger.dir).exists():
                rmtree(self.logger.dir)


class RayParallelRunner:
    """Run local shared-memory pools remotely and merge their episode batches."""

    def __init__(self, args, logger) -> None:
        self.args = args
        self.logger = logger
        self.batch_size = int(args.batch_size_run)
        if self.batch_size <= 0:
            raise ValueError("batch_size_run must be greater than zero.")

        envs_per_node = int(getattr(args, "ray_envs_per_node", 4))
        if envs_per_node <= 0:
            raise ValueError("ray_envs_per_node must be greater than zero.")
        self.node_batch_sizes = [
            min(envs_per_node, self.batch_size - start)
            for start in range(0, self.batch_size, envs_per_node)
        ]
        cpus_per_env = float(getattr(args, "ray_cpus_per_env", 1.0))
        memory_per_env_gb = getattr(args, "ray_memory_per_env_gb", None)
        cpu_overhead = float(getattr(args, "ray_node_overhead_cpus", 0.5))
        memory_overhead_gb = float(
            getattr(args, "ray_node_overhead_memory_gb", 0.5)
        )
        if (
            not np.isfinite(cpus_per_env)
            or not np.isfinite(cpu_overhead)
            or cpus_per_env <= 0
            or cpu_overhead < 0
        ):
            raise ValueError("Ray CPU reservations must be positive.")
        if memory_per_env_gb is None:
            raise ValueError(
                "Set ray_memory_per_env_gb to a measured per-environment memory "
                "reservation before using runner=parallel_ray."
            )
        memory_per_env_gb = float(memory_per_env_gb)
        if not np.isfinite(memory_per_env_gb) or memory_per_env_gb <= 0:
            raise ValueError("ray_memory_per_env_gb must be a positive finite value.")
        if not np.isfinite(memory_overhead_gb) or memory_overhead_gb < 0:
            raise ValueError("ray_node_overhead_memory_gb cannot be negative.")

        try:
            import ray
            from ray.util.placement_group import placement_group, remove_placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        except ImportError as error:
            raise RuntimeError(
                "Ray is optional; install marl/setup/ray_requirements.txt to use "
                "runner=parallel_ray."
            ) from error

        self._ray = ray
        self._remove_placement_group = remove_placement_group
        self._owns_ray = not ray.is_initialized()
        self._closed = False
        self._registered = False
        self._recording_active = False
        self._recording_folder: str | None = None
        self._recording_t_env: int | None = None
        self._recording_prefix = "replay"
        self._recorded_videos: list[tuple[str, bytes]] = []

        if self._owns_ray:
            ray.init(address=getattr(args, "ray_address", None))

        bundles = []
        for node_batch_size in self.node_batch_sizes:
            bundles.append(
                {
                    "CPU": node_batch_size * cpus_per_env + cpu_overhead,
                    "memory": int(
                        (
                            node_batch_size * float(memory_per_env_gb)
                            + memory_overhead_gb
                        )
                        * 1024**3
                    ),
                }
            )

        self.workers = []
        timeout_s = float(getattr(args, "ray_worker_start_timeout_s", 600))
        try:
            self.placement_group = placement_group(bundles, strategy="SPREAD")
            try:
                ray.get(self.placement_group.ready(), timeout=timeout_s)
            except ray.exceptions.GetTimeoutError as error:
                raise RuntimeError(
                    "Ray could not reserve all node-local environment pools within "
                    f"{timeout_s:g}s. Reduce batch_size_run, ray_envs_per_node, or "
                    "per-environment resource requests, or add cluster capacity."
                ) from error

            remote_worker = ray.remote(max_restarts=0)(_RayNodeWorker)
            start_idx = 0
            for node_idx, node_batch_size in enumerate(self.node_batch_sizes):
                cpu_request = node_batch_size * cpus_per_env + cpu_overhead
                memory_request = int(
                    (
                        node_batch_size * float(memory_per_env_gb)
                        + memory_overhead_gb
                    )
                    * 1024**3
                )
                scheduling = PlacementGroupSchedulingStrategy(
                    placement_group=self.placement_group,
                    placement_group_bundle_index=node_idx,
                )
                worker = remote_worker.options(
                    num_cpus=cpu_request,
                    memory=memory_request,
                    scheduling_strategy=scheduling,
                ).remote(args, node_batch_size, start_idx)
                self.workers.append(worker)
                start_idx += node_batch_size

            self.env_info = ray.get(
                self.workers[0].get_env_info.remote(), timeout=timeout_s
            )
        except Exception:
            self._abort_initialization()
            raise

        global _ACTIVE_RAY_RUNNERS, _RAY_STARTED_BY_RUNNER
        _ACTIVE_RAY_RUNNERS += 1
        _RAY_STARTED_BY_RUNNER |= self._owns_ray
        self._registered = True

        self.episode_limit = self.env_info["episode_limit"]
        self.t = 0
        self.t_env = 0
        self.env_steps_this_run = 0
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.log_train_stats_t = -100000

    def _abort_initialization(self) -> None:
        for worker in self.workers:
            self._ray.kill(worker, no_restart=True)
        if hasattr(self, "placement_group"):
            self._remove_placement_group(self.placement_group)
        if self._owns_ray:
            self._ray.shutdown()

    def set_mac_env(self, mac) -> None:
        _validate_mac_env_binding(mac)
        self.mac = mac

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
        policy_ref = self._ray.put(_cpu_policy_copy(mac))
        try:
            self._ray.get(
                [
                    worker.setup.remote(scheme, groups, preprocess, policy_ref)
                    for worker in self.workers
                ]
            )
        except Exception:
            self.close_env()
            raise

    def get_env_info(self) -> dict:
        return self.env_info

    def close_env(self) -> None:
        if self._closed:
            return
        self._closed = True
        global _ACTIVE_RAY_RUNNERS, _RAY_STARTED_BY_RUNNER
        try:
            self._ray.get([worker.close.remote() for worker in self.workers])
        finally:
            for worker in self.workers:
                self._ray.kill(worker, no_restart=True)
            self._remove_placement_group(self.placement_group)
            if self._registered:
                _ACTIVE_RAY_RUNNERS -= 1
                self._registered = False
            if _ACTIVE_RAY_RUNNERS == 0 and _RAY_STARTED_BY_RUNNER:
                self._ray.shutdown()
                _RAY_STARTED_BY_RUNNER = False

    def save_replay(self) -> None:
        self._ray.get(self.workers[0].save_replay.remote())

    def _merge_batches(self, node_results: list[dict]) -> EpisodeBatch:
        batch = self.new_batch()
        row_offset = 0
        for node_result, node_batch_size in zip(node_results, self.node_batch_sizes):
            node_batch = node_result["batch"]
            for node_row in range(node_batch_size):
                target_row = row_offset + node_row
                for name, destination in batch.data.transition_data.items():
                    destination[target_row].copy_(
                        node_batch.data.transition_data[name][node_row].to(
                            destination.device
                        )
                    )
                for name, destination in batch.data.episode_data.items():
                    destination[target_row].copy_(
                        node_batch.data.episode_data[name][node_row].to(
                            destination.device
                        )
                    )
            row_offset += node_batch_size
        return batch

    def _sync_policy_runtime_state(self, node_results: list[dict]) -> None:
        epsilons = [
            result["epsilon"]
            for result in node_results
            if result["epsilon"] is not None
        ]
        selector = getattr(self.mac, "action_selector", None)
        if epsilons and hasattr(selector, "epsilon"):
            selector.epsilon = epsilons[0]

        episode_indices = [
            result["episode_index"]
            for result in node_results
            if result["episode_index"] is not None
        ]
        if episode_indices and hasattr(self.mac, "_episode_index"):
            self.mac._episode_index = max(episode_indices)

    def run(
        self,
        test_mode: bool = False,
        return_log_stats: bool = True,
        reset_options: dict | None = None,
    ) -> EpisodeBatch | dict:
        if not hasattr(self, "mac"):
            raise RuntimeError("RayParallelRunner.setup must be called before run.")

        policy_ref = self._ray.put(_cpu_policy_copy(self.mac))
        try:
            node_results = self._ray.get(
                [
                    worker.run_batch.remote(
                        policy_ref,
                        test_mode,
                        self.t_env,
                        reset_options,
                    )
                    for worker in self.workers
                ]
            )
        finally:
            del policy_ref

        self._sync_policy_runtime_state(node_results)
        self.batch = self._merge_batches(node_results)
        episode_returns = [
            episode_return
            for node_result in node_results
            for episode_return in node_result["returns"]
        ]
        final_infos = [
            info
            for node_result in node_results
            for info in node_result["final_infos"]
        ]
        stats_increment = {}
        for node_result in node_results:
            for key, value in node_result["stats"].items():
                stats_increment[key] = stats_increment.get(key, 0) + value

        self.t = max(node_result["t"] for node_result in node_results)
        self.env_steps_this_run = sum(
            node_result["env_steps"] for node_result in node_results
        )
        if not test_mode:
            self.t_env += self.env_steps_this_run

        videos = node_results[0].get("videos", [])
        self._recorded_videos.extend(videos)

        log_stats = self._collect_stats(
            test_mode, return_log_stats, episode_returns, stats_increment
        )
        if test_mode:
            result = {"batch": self.batch, "final_infos": final_infos}
            if return_log_stats:
                result["log_stats"] = log_stats
            return result
        return self.batch

    def _collect_stats(
        self,
        test_mode: bool,
        return_log_stats: bool,
        episode_returns: list,
        stats_increment: dict,
    ) -> dict:
        returns = self.test_returns if test_mode else self.train_returns
        returns.extend(episode_returns)
        stats = self.test_stats if test_mode else self.train_stats
        for key, value in stats_increment.items():
            stats[key] = stats.get(key, 0) + value

        prefix = "test_" if test_mode else ""
        log_stats = self._get_log_stats(returns, stats, prefix)
        if (
            not test_mode
            and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval
        ):
            for key, value in log_stats.items():
                self.logger.log_stat(key, value, self.t_env)
            selector = getattr(self.mac, "action_selector", None)
            if hasattr(selector, "epsilon"):
                self.logger.log_stat("epsilon", selector.epsilon, self.t_env)
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
        self._ray.get(
            [worker.set_env_attr.remote(name, value) for worker in self.workers]
        )

    def get_env_attr(self, name: str, default=None) -> Any:
        return self._ray.get(self.workers[0].get_env_attr.remote(name, default))

    def get_env_rng_states(self) -> list[dict]:
        node_rng_states = self._ray.get(
            [worker.get_rng_states.remote() for worker in self.workers]
        )
        return [state for node_states in node_rng_states for state in node_states]

    def set_env_rng_states(self, states: list[dict]) -> None:
        if len(states) != self.batch_size:
            raise ValueError(
                f"Expected {self.batch_size} environment RNG states, got {len(states)}."
            )
        offset = 0
        refs = []
        for worker, node_batch_size in zip(self.workers, self.node_batch_sizes):
            node_states = states[offset : offset + node_batch_size]
            refs.append(worker.set_rng_states.remote(node_states))
            offset += node_batch_size
        self._ray.get(refs)

    def start_recording(
        self,
        n_test_replays_save: int,
        video_prefix: str = "replay",
        t_env: int | None = None,
    ) -> None:
        if self._recording_active:
            raise RuntimeError("Ray environment pools are already recording.")
        if t_env is None:
            t_env = self.t_env
        self._recording_folder = join(self.logger.dir, "replays", f"t_{t_env}")
        makedirs(self._recording_folder, exist_ok=True)
        self._recording_t_env = t_env
        self._recording_prefix = video_prefix
        self._recorded_videos.clear()
        self._ray.get(
            self.workers[0].start_recording.remote(
                n_test_replays_save, video_prefix, t_env
            )
        )
        self._recording_active = True
        self.logger.info(f"Saving {n_test_replays_save} test episode replays")

    def stop_recording(
        self, t_env: int | None = None, video_prefix: str = "replays"
    ) -> None:
        if self._recording_active:
            videos = self._ray.get(
                self.workers[0].stop_recording.remote(
                    self._recording_t_env, self._recording_prefix
                )
            )
            self._recorded_videos.extend(videos)
        if self._recording_folder is not None:
            for file_name, video_bytes in self._recorded_videos:
                with open(join(self._recording_folder, file_name), "wb") as video_file:
                    video_file.write(video_bytes)
            self.logger.log_videos(
                dir=self._recording_folder,
                t=self._recording_t_env if t_env is None else t_env,
                video_prefix=video_prefix,
            )
        self._recording_active = False
        self._recording_folder = None
        self._recording_t_env = None
        self._recorded_videos.clear()
