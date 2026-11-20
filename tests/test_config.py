"""Tests for common.config and common.envs: values, defaults, seeds."""

from __future__ import annotations

import gymnasium as gym
import highway_env  # noqa: F401
import pytest

from common.config import (
    BASE_CONFIG,
    DENSITIES,
    ENV_ID,
    EVAL_SEEDS,
    PILOT_SEEDS,
    TRAIN_DENSITY,
    TRAIN_SEEDS,
)
from common.envs import make_env


def test_base_config_equals_env_defaults():
    defaults = gym.make(ENV_ID).unwrapped.config
    for key, value in BASE_CONFIG.items():
        assert defaults[key] == value, f"{key}: config {value} != highway-fast-v0 default {defaults[key]}"


@pytest.mark.parametrize("density", list(DENSITIES))
def test_env_config_differs_from_default_only_in_density(density):
    default = gym.make(ENV_ID).unwrapped.config
    env_cfg = make_env(density=density).unwrapped.config
    diff = {k for k in default if default[k] != env_cfg[k]}
    assert diff <= {"vehicles_density"}
    assert env_cfg["vehicles_density"] == DENSITIES[density]["vehicles_density"]
    assert env_cfg["observation"] == {"type": "Kinematics"}  # default observation: 5 vehicles


def test_make_env_defaults_and_no_rendering():
    env = make_env()  # TRAIN_DENSITY, default reward
    assert env.unwrapped.config["vehicles_density"] == DENSITIES[TRAIN_DENSITY]["vehicles_density"]
    assert env.unwrapped.render_mode is None
    assert env.unwrapped.viewer is None


def test_seed_sets_are_disjoint():
    assert not set(EVAL_SEEDS) & set(TRAIN_SEEDS)
    assert not set(EVAL_SEEDS) & set(PILOT_SEEDS)
    assert len(EVAL_SEEDS) == 100 and TRAIN_SEEDS == [0, 1, 2]
