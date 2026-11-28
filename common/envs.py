"""Environment factory shared by training, evaluation and scripts."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional

import gymnasium as gym
import highway_env  # noqa: F401  (registers the highway environments)

from common.config import BASE_CONFIG, DENSITIES, ENV_ID, TEST_CONDITIONS, TRAIN_DENSITY, reward_weights
from common.rewards import CustomRewardWrapper


def make_env(
    density: str = TRAIN_DENSITY,
    reward: str = "default",
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    env_id: str = ENV_ID,
    overrides: Optional[Mapping[str, Any]] = None,
) -> gym.Env:
    """Create a HighwayEnv environment with the shared configuration.

    The env config is ``BASE_CONFIG`` with ``DENSITIES[density]`` merged on top (only
    ``vehicles_density`` differs between densities); every other parameter stays at the
    highway-fast-v0 default. Nothing is rendered unless ``render_mode`` is given.

    Args:
        density: Traffic density preset, one of ``common.config.DENSITIES``.
        reward: Reward preset, one of ``common.config.REWARDS`` or ``SUPPLEMENTARY_REWARDS`` (registered supplementary
            runs only, enforced by ``common.train_config.check_run_roles``).
        seed: Seed used to reset the environment once after creation.
            ``None`` leaves seeding to the caller.
        render_mode: Gymnasium render mode (e.g. ``"rgb_array"``) or None.
        env_id: Gymnasium environment id. Training always uses the default ``ENV_ID``; other ids
            exist only for evaluation conditions (``common.config.TEST_CONDITIONS``).
        overrides: Config values merged last, on top of ``BASE_CONFIG`` and the density preset.

    Returns:
        A Gymnasium environment wrapped with ``CustomRewardWrapper``.
    """
    if density not in DENSITIES:
        raise ValueError(f"Unknown density {density!r}; choose from {list(DENSITIES)}")
    weights = reward_weights(reward)  # REWARDS or SUPPLEMENTARY_REWARDS; raises on an unknown name

    config = copy.deepcopy(BASE_CONFIG)
    config.update(DENSITIES[density])
    config.update(copy.deepcopy(dict(overrides or {})))

    env = gym.make(env_id, config=config, render_mode=render_mode)
    env = CustomRewardWrapper(env, weights)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_test_env(condition: str, reward: str = "default", render_mode: Optional[str] = None) -> gym.Env:
    """Create the environment of a test condition from ``common.config.TEST_CONDITIONS``.

    Args:
        condition: Key of ``TEST_CONDITIONS`` ("main", "low", "high", "long", "precise").
        reward: Reward preset, one of ``common.config.REWARDS``.
        render_mode: None unless explicitly given (only ``scripts/record_video.py`` should).
    """
    if condition not in TEST_CONDITIONS:
        raise ValueError(f"Unknown test condition {condition!r}; choose from {list(TEST_CONDITIONS)}")
    spec = TEST_CONDITIONS[condition]
    return make_env(
        density=spec["density"],
        reward=reward,
        render_mode=render_mode,
        env_id=spec["env_id"],
        overrides=spec["overrides"],
    )
