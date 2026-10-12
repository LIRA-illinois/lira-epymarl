REGISTRY: dict[str, type] = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .shared_memory_parallel_runner import SharedMemoryParallelRunner
REGISTRY["parallel_shared_memory"] = SharedMemoryParallelRunner

from .ray_parallel_runner import RayParallelRunner
REGISTRY["parallel_ray"] = RayParallelRunner
