"""Tests for common.config and common.envs: values, defaults, reward weights, seeds."""

from __future__ import annotations

import itertools

import gymnasium as gym
import highway_env  # noqa: F401
import pytest

from common.config import (
    BASE_CONFIG,
    DENSITIES,
    ENV_ID,
    EVAL_SEEDS,
    MIN_GAMMA,
    PILOT_RUN_SEED,
    PILOT_SEEDS,
    REWARDS,
    SENSITIVITY_SEEDS,
    TEST_SEEDS,
    TRAIN_DENSITY,
    TRAIN_SEEDS,
    VAL_SEEDS,
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


@pytest.mark.slow
def test_base_config_equals_env_defaults():
    defaults = gym.make(ENV_ID).unwrapped.config
    for key, value in BASE_CONFIG.items():
        assert defaults[key] == value, f"{key}: config {value} != highway-fast-v0 default {defaults[key]}"


@pytest.mark.slow
@pytest.mark.parametrize("density", list(DENSITIES))
def test_env_config_differs_from_default_only_in_density(density):
    default = gym.make(ENV_ID).unwrapped.config
    env_cfg = make_env(density=density).unwrapped.config
    diff = {k for k in default if default[k] != env_cfg[k]}
    assert diff <= {"vehicles_density"}
    assert env_cfg["vehicles_density"] == DENSITIES[density]["vehicles_density"]
    assert env_cfg["observation"] == {"type": "Kinematics"}  # default observation: 5 vehicles


@pytest.mark.slow
def test_make_env_defaults_and_no_rendering():
    env = make_env()  # TRAIN_DENSITY, default reward
    assert env.unwrapped.config["vehicles_density"] == DENSITIES[TRAIN_DENSITY]["vehicles_density"]
    assert env.unwrapped.render_mode is None
    assert env.unwrapped.viewer is None


def test_seed_sets_are_disjoint():
    sets = {"pilot": PILOT_SEEDS, "train": TRAIN_SEEDS, "val": VAL_SEEDS, "test": TEST_SEEDS,
            "pilot_run": [PILOT_RUN_SEED]}
    for (a, sa), (b, sb) in itertools.combinations(sets.items(), 2):
        assert not set(sa) & set(sb), f"{a} and {b} overlap"
    assert not set(SENSITIVITY_SEEDS) & (set(PILOT_SEEDS) | set(VAL_SEEDS) | set(TEST_SEEDS) | {PILOT_RUN_SEED})


def test_seed_set_values():
    assert TRAIN_SEEDS == [0, 1, 2, 3, 4]
    assert SENSITIVITY_SEEDS == [0, 1, 2]
    assert VAL_SEEDS == list(range(20000, 20020))
    assert TEST_SEEDS == list(range(10000, 10100))
    assert PILOT_SEEDS == list(range(1000, 1050))
    assert EVAL_SEEDS is TEST_SEEDS  # backwards-compatible alias


def test_calibration_seeds_are_isolated():
    """Calibration run seeds [99, 98]: 98 is used nowhere else, and its derived vector-env seeds collide with nothing."""
    from common.config import CALIBRATION_ONLY_SEED, CALIBRATION_SEEDS
    from common.seeding import MAX_ENVS_PER_RUN, derived_env_seeds

    assert CALIBRATION_SEEDS == [99, 98] and CALIBRATION_ONLY_SEED == 98 and PILOT_RUN_SEED == 99
    others = {"pilot": PILOT_SEEDS, "train": TRAIN_SEEDS, "sensitivity": SENSITIVITY_SEEDS, "val": VAL_SEEDS,
              "test": TEST_SEEDS, "pilot_run": [PILOT_RUN_SEED]}
    for name, seeds in others.items():
        assert 98 not in seeds, f"98 is in {name}"
    cal = set(derived_env_seeds(98, MAX_ENVS_PER_RUN))
    assert len(cal) == MAX_ENVS_PER_RUN
    for name, seeds in others.items():
        assert not cal & set(seeds), f"derived seeds of 98 collide with {name}"
    for run in (*TRAIN_SEEDS, *SENSITIVITY_SEEDS, PILOT_RUN_SEED):  # 99 is also a calibration run seed: checked here too
        assert not cal & set(derived_env_seeds(run, MAX_ENVS_PER_RUN)), f"derived seeds of 98 and {run} collide"
    # the derived seeds of 99 (calibration run 1) collide with no episode-seed set either
    d99 = set(derived_env_seeds(99, MAX_ENVS_PER_RUN))
    for name, seeds in others.items():
        assert not d99 & set(seeds)


def test_harness_refuses_calibration_seed():
    from common.train_config import check_run_roles

    for variant in ("main", "tune-a", "lr=0.001"):
        with pytest.raises(ValueError, match="reserved for the SB3 calibration"):
            check_run_roles(variant, 98, "balanced", "medium", [], [])
