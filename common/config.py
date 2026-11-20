"""Shared environment and traffic-density configuration.

Environment and density values were chosen from the environment pilot (see
``docs/config_decisions.md`` and ``results/pilot/PILOT_REPORT.md``) so that both algorithms use
identical settings. Do not change them without updating the docs. The reward presets are still
the team-agreed placeholders.
"""

from __future__ import annotations

from typing import Any

ENV_ID = "highway-fast-v0"

# Base environment configuration passed to gym.make(..., config=...).
# Only parameters that are part of the study are listed; everything else stays at the
# highway-fast-v0 defaults on purpose (tests/test_config.py checks that these values equal
# the defaults, so the dict documents them without overriding anything).
BASE_CONFIG: dict[str, Any] = {
    "observation": {"type": "Kinematics"},  # defaults: vehicles_count=5 (ego + 4 nearest ahead)
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 3,
    "duration": 30,  # seconds == steps at policy_frequency=1
    "policy_frequency": 1,
    "simulation_frequency": 5,
    "vehicles_count": 20,
}

# Traffic density presets, merged on top of BASE_CONFIG. Only vehicles_density differs.
DENSITIES: dict[str, dict[str, Any]] = {
    "low": {"vehicles_density": 0.4},
    "medium": {"vehicles_density": 0.7},
    "high": {"vehicles_density": 1.0},
}
TRAIN_DENSITY = "medium"  # both algorithms train here; evaluation covers all densities


# Reward presets: weights consumed by common.rewards.CustomRewardWrapper.
# "default" means: keep the reward of the underlying HighwayEnv unchanged.
REWARDS: dict[str, dict[str, float]] = {
    "default": {},  # TODO: agree on values
    "efficiency": {"speed": 1.0, "collision": -1.0},  # TODO: agree on values
    "safety": {"speed": 0.2, "collision": -5.0},  # TODO: agree on values
    "balanced": {"speed": 0.5, "collision": -2.0},  # TODO: agree on values
}

# Seeds. Evaluation seeds must not overlap with training seeds or the pilot seeds (1000..1049).
PILOT_SEEDS: list[int] = list(range(1000, 1050))
TRAIN_SEEDS: list[int] = [0, 1, 2]
EVAL_SEEDS: list[int] = list(range(10000, 10100))
assert not set(EVAL_SEEDS) & (set(TRAIN_SEEDS) | set(PILOT_SEEDS)), "seed sets overlap"
