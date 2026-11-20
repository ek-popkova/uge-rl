"""Shared environment, traffic-density and reward configuration.

All values below are placeholders and must be agreed on by the whole team
before any experiment is run, so that both algorithms use identical settings.
"""

from __future__ import annotations

from typing import Any

ENV_ID = "highway-fast-v0"  # TODO: agree on env id (highway-fast-v0 vs highway-v0)

# Base environment configuration passed to gym.make(..., config=...).
BASE_CONFIG: dict[str, Any] = {
    "observation": {
        "type": "Kinematics",  # TODO: agree on values
    },
    "action": {
        "type": "DiscreteMetaAction",  # TODO: agree on values
    },
    "lanes_count": 4,  # TODO: agree on values
    "duration": 40,  # TODO: agree on values (episode length, seconds)
    "policy_frequency": 1,  # TODO: agree on values
    "simulation_frequency": 15,  # TODO: agree on values
}

# Traffic density presets, merged on top of BASE_CONFIG.
DENSITIES: dict[str, dict[str, Any]] = {
    "low": {"vehicles_count": 20, "vehicles_density": 0.8},  # TODO: agree on values
    "medium": {"vehicles_count": 50, "vehicles_density": 1.0},  # TODO: agree on values
    "high": {"vehicles_count": 80, "vehicles_density": 1.4},  # TODO: agree on values
}

# Reward presets: weights consumed by common.rewards.CustomRewardWrapper.
# "default" means: keep the reward of the underlying HighwayEnv unchanged.
REWARDS: dict[str, dict[str, float]] = {
    "default": {},  # TODO: agree on values
    "efficiency": {"speed": 1.0, "collision": -1.0},  # TODO: agree on values
    "safety": {"speed": 0.2, "collision": -5.0},  # TODO: agree on values
    "balanced": {"speed": 0.5, "collision": -2.0},  # TODO: agree on values
}

# Evaluation protocol shared by both algorithms.
EVAL_SEEDS: list[int] = list(range(100, 120))  # TODO: agree on values
