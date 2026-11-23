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
    4. Vectorized envs (PPO): env ``i`` gets the seed ``vector_env_seed(s, i)`` = ``100000 + s * 1000 + i``
       for its single initial reset (``derived_env_seeds(s, n_envs)`` lists them). The offset 100000
       keeps derived seeds disjoint from the pilot, validation and test seeds.
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


ENV_SEED_OFFSET = 100_000  # above every pilot, validation and test seed
MAX_ENVS_PER_RUN = 1000  # env indices 0..999 per run seed; more would collide with the next run seed


def vector_env_seed(run_seed: int, env_index: int) -> int:
    """Initial reset seed of environment ``env_index`` in a vectorized env of run ``run_seed``."""
    if not 0 <= env_index < MAX_ENVS_PER_RUN:
        raise ValueError(f"env_index must be in 0..{MAX_ENVS_PER_RUN - 1}, got {env_index}")
    return ENV_SEED_OFFSET + run_seed * 1000 + env_index


def derived_env_seeds(run_seed: int, n_envs: int) -> list[int]:
    """Initial reset seeds of the ``n_envs`` environments of a vectorized env of run ``run_seed``."""
    return [vector_env_seed(run_seed, i) for i in range(n_envs)]
