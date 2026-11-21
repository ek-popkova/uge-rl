"""Tests for common.config and common.envs: values, defaults, reward weights, seeds."""

from __future__ import annotations

import gymnasium as gym
import highway_env  # noqa: F401
import pytest

from common.config import (
    BASE_CONFIG,
    DENSITIES,
    ENV_ID,
    EVAL_SEEDS,
    MIN_GAMMA,
    PILOT_SEEDS,
    REWARDS,
    TRAIN_DENSITY,
    TRAIN_SEEDS,
    indifference_point,
)
from common.envs import make_env


def test_reward_presets_exact_weights():
    assert REWARDS == {
        "default": {},
        "efficiency": {"w_speed": 0.8, "w_right": 0.0, "w_collision": 15.0, "w_lane": 0.0},
        "balanced": {"w_speed": 0.4, "w_right": 0.1, "w_collision": 20.0, "w_lane": 0.1},
        "safety": {"w_speed": 0.2, "w_right": 0.1, "w_collision": 30.0, "w_lane": 0.5},
    }
    assert MIN_GAMMA == 0.99


@pytest.mark.parametrize("name, expected", [("efficiency", 0.27), ("balanced", 0.61), ("safety", 0.83)])
def test_indifference_point_values(name, expected):
    w = REWARDS[name]
    p = indifference_point(w["w_speed"], w["w_right"], w["w_collision"], T=BASE_CONFIG["duration"])
    assert p is not None  # ordering G_crash < G_slow < G_succ holds
    assert round(p, 2) == expected


def test_indifference_point_none_when_ordering_fails():
    assert indifference_point(w_speed=0.8, w_right=0.0, w_collision=1.0) is None


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
