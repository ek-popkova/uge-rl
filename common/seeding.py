"""Seeding helpers and the seeding rules for training runs.

Seed roles (values in ``common.config``; the sets are pairwise disjoint):
    TRAIN_SEEDS        run seeds of the main grid (5 runs per algorithm x reward)
    SENSITIVITY_SEEDS  run seeds of the hyperparameter sensitivity runs
    VAL_SEEDS          episode seeds for periodic evaluation during training (learning curves)
    TEST_SEEDS         episode seeds of the final evaluation, used once
    PILOT_SEEDS        episode seeds of the environment pilot

How a training run with run seed ``s`` must seed itself (DDQN and PPO alike):
    1. ``set_global_seeds(s)`` at the start (python ``random``, numpy, torch, deterministic ops);
       this also fixes network initialisation.
    2. ``env.reset(seed=s)`` exactly once at the start; every later reset is called WITHOUT a
       seed, so episodes are drawn from the environment's own seeded random stream.
    3. All exploration randomness (epsilon-greedy, action sampling, replay sampling) comes from
       ``np.random.default_rng(s)`` (or a torch Generator seeded with ``s``), never from global
       state that other code can touch.
    4. Vectorized envs (PPO): env ``i`` gets the seed ``vector_env_seed(s, i)`` = ``s * 1000 + i``
       for its single initial reset.
    5. Derive any other seed from ``(s, purpose)``, never from the worker id or scheduling order.

Evaluation does not use any of this: ``common.evaluate`` resets with an explicit episode seed
for every episode and acts deterministically.
"""

from __future__ import annotations

import random

import numpy as np


def set_global_seeds(seed: int) -> None:
    """Seed python ``random``, numpy and (if installed) torch, and request deterministic torch ops."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is optional for the simulator-only code
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def vector_env_seed(run_seed: int, env_index: int) -> int:
    """Initial reset seed of environment ``env_index`` in a vectorized env of run ``run_seed``."""
    return run_seed * 1000 + env_index
