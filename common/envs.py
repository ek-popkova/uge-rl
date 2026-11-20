"""Environment factory shared by training, evaluation and scripts."""

from __future__ import annotations

import copy
from typing import Optional

import gymnasium as gym
import highway_env  # noqa: F401  (registers the highway environments)

from common.config import BASE_CONFIG, DENSITIES, ENV_ID, REWARDS
from common.rewards import CustomRewardWrapper


def make_env(
    density: str = "medium",
    reward: str = "default",
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """Create a HighwayEnv environment with the shared configuration.

    Args:
        density: Traffic density preset, one of ``common.config.DENSITIES``.
        reward: Reward preset, one of ``common.config.REWARDS``.
        seed: Seed used to reset the environment once after creation.
            ``None`` leaves seeding to the caller.
        render_mode: Gymnasium render mode (e.g. ``"rgb_array"``) or None.

    Returns:
        A Gymnasium environment wrapped with ``CustomRewardWrapper``.
    """
    if density not in DENSITIES:
        raise ValueError(f"Unknown density {density!r}; choose from {list(DENSITIES)}")
    if reward not in REWARDS:
        raise ValueError(f"Unknown reward {reward!r}; choose from {list(REWARDS)}")

    config = copy.deepcopy(BASE_CONFIG)
    config.update(DENSITIES[density])

    env = gym.make(ENV_ID, config=config, render_mode=render_mode)
    env = CustomRewardWrapper(env, REWARDS[reward])
    if seed is not None:
        env.reset(seed=seed)
    return env
