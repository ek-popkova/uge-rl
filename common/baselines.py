"""Scripted baseline policies as ``Agent`` subclasses (calibration baselines).

random: uniform over the 5 meta-actions, re-seeded per episode from ``(seed, episode seed)`` so
        its behaviour does not depend on episode order or worker layout.
idle / slower / faster: always the same meta-action.
They ignore ``deterministic`` (random is random by design) and have nothing to save or load.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from common.agent_base import Agent
from common.evaluate import run_chunk

# DiscreteMetaAction indices in highway-env 1.12.1 (checked by tests/test_evaluate.py).
ACTION_INDEX = {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2, "FASTER": 3, "SLOWER": 4}
N_ACTIONS = len(ACTION_INDEX)
BASELINE_NAMES = ("random", "idle", "slower", "faster")


class ConstantAgent(Agent):
    """Always plays the same meta-action."""

    def __init__(self, action: int) -> None:
        self.action = int(action)

    def act(self, obs: Any, deterministic: bool = True) -> int:
        return self.action

    def save(self, path: str) -> None:  # nothing to save
        pass

    def load(self, path: str) -> None:  # nothing to load
        pass


class RandomAgent(Agent):
    """Uniformly random meta-action, seeded per episode."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng([self.seed, 0])

    def start_episode(self, seed: int) -> None:
        self.rng = np.random.default_rng([self.seed, int(seed)])

    def act(self, obs: Any, deterministic: bool = True) -> int:
        return int(self.rng.integers(N_ACTIONS))

    def save(self, path: str) -> None:  # nothing to save
        pass

    def load(self, path: str) -> None:  # nothing to load
        pass


def make_baseline(name: str, seed: int = 0) -> Agent:
    """Build one of ``BASELINE_NAMES``."""
    if name == "random":
        return RandomAgent(seed)
    if name in ("idle", "slower", "faster"):
        return ConstantAgent(ACTION_INDEX[name.upper()])
    raise ValueError(f"Unknown baseline {name!r}; choose from {BASELINE_NAMES}")


def run_baseline_chunk(task: tuple[str, str, tuple[int, ...]]) -> dict[str, Any]:
    """Worker task ``(policy, condition, seeds)`` for ``common.parallel.run_tasks``.

    Uses the "default" reward env: the per-step components are recorded anyway, so the return
    under every reward preset is recomputed later (``common.metrics.add_return_columns``).
    """
    policy, condition, seeds = task
    return run_chunk((make_baseline(policy), condition, "default", tuple(seeds)))
