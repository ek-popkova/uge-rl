"""Custom reward as a Gymnasium wrapper.

Per-step custom reward (unnormalized), see ``common.config.REWARDS``:

    r = w_speed * high_speed + w_right * right_lane
        - w_collision * crashed - w_lane * lane_changed

``high_speed`` and ``right_lane`` are the components ``info["rewards"]["high_speed_reward"]`` and
``info["rewards"]["right_lane_reward"]`` of highway-env (both in [0, 1]); ``crashed`` is
``info["crashed"]``; ``lane_changed`` is 1 if the real lane index changed during the step.
Because the reward is computed from these components and not from the env's reward, the env's
``normalize_reward`` setting does not affect it.
"""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym

REQUIRED_WEIGHTS = ("w_speed", "w_right", "w_collision", "w_lane")


def compute_custom_reward(
    weights: Mapping[str, float],
    high_speed: float,
    right_lane: float,
    crashed: bool | float,
    lane_changed: bool | float,
) -> tuple[float, dict[str, float]]:
    """Evaluate the custom reward formula for one step.

    Returns:
        ``(reward, components)`` where ``components`` holds the signed weighted terms
        (collision and lane-change terms are negative or zero) and sums to ``reward``.
    """
    components = {
        "speed": weights["w_speed"] * float(high_speed),
        "right_lane": weights["w_right"] * float(right_lane),
        "collision": -weights["w_collision"] * float(bool(crashed)),
        "lane_change": -weights["w_lane"] * float(bool(lane_changed)),
    }
    return sum(components.values()), components


class CustomRewardWrapper(gym.Wrapper):
    """Replace the environment reward according to a reward preset.

    Args:
        env: A highway-env environment (needs ``env.unwrapped.vehicle`` and ``info["rewards"]``).
        weights: Reward weights, see ``common.config.REWARDS``. An empty dict keeps the
            environment's own (normalized) reward unchanged.

    Every step's ``info`` gets:
        ``lane_changed``: 1 if ``vehicle.lane_index[2]`` before the step differs from after the
            step (the real lane, not the chosen action), else 0 (also for the default preset).
        ``custom_reward_components``: signed weighted terms of the custom reward
            (``{}`` for the default preset).
    """

    def __init__(self, env: gym.Env, weights: Mapping[str, float]) -> None:
        super().__init__(env)
        self.weights = dict(weights)
        if self.weights:
            missing = [k for k in REQUIRED_WEIGHTS if k not in self.weights]
            if missing:
                raise ValueError(f"reward weights are missing {missing}: {self.weights}")
            negative = [k for k in REQUIRED_WEIGHTS if self.weights[k] < 0]
            if negative:
                raise ValueError(f"reward weights must be non-negative, got {negative}")

    def step(self, action: Any):
        lane_before = self.env.unwrapped.vehicle.lane_index[2]
        obs, reward, terminated, truncated, info = self.env.step(action)
        lane_changed = int(self.env.unwrapped.vehicle.lane_index[2] != lane_before)
        info["lane_changed"] = lane_changed
        if self.weights:
            reward, components = self._custom_reward(info, lane_changed)
            info["custom_reward_components"] = components
        else:
            info["custom_reward_components"] = {}
        return obs, reward, terminated, truncated, info

    def _custom_reward(self, info: dict[str, Any], lane_changed: int) -> tuple[float, dict[str, float]]:
        rewards = info.get("rewards")
        if rewards is None:
            raise KeyError('info["rewards"] is missing; the custom reward needs highway-env reward components')
        return compute_custom_reward(
            self.weights,
            high_speed=rewards["high_speed_reward"],
            right_lane=rewards["right_lane_reward"],
            crashed=info["crashed"],
            lane_changed=lane_changed,
        )
