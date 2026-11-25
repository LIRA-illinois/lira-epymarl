"""Public simulation entry surface.

Legacy implementation: ``run_legacy.py``.
Unified implementation: ``UnifiedHierarchicalSeedRunner``.
"""

from src.simulation.evaluate import run_eval_episodes
from src.simulation.hierarchical_synthesis import solve_mixed_project_policy
from src.simulation.run_legacy import Simulation, wandb
from src.simulation.unified_hierarchical_runner import (
    UnifiedHierarchicalSeedRunner,
)

__all__ = [
    "Simulation",
    "UnifiedHierarchicalSeedRunner",
    "run_eval_episodes",
    "solve_mixed_project_policy",
    "wandb",
]
