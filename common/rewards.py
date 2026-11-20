"""Custom reward shaping as a Gymnasium wrapper."""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class CustomRewardWrapper(gym.Wrapper):
    """Replace the environment reward according to a reward preset.

    Args:
        env: The environment to wrap.
        weights: Reward weights, see ``common.config.REWARDS``. An empty dict
            keeps the original reward unchanged.
    """

    def __init__(self, env: gym.Env, weights: dict[str, float]) -> None:
        super().__init__(env)
        self.weights = weights

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.weights:
            reward = self._compute_reward(reward, info)
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, original_reward: float, info: dict[str, Any]) -> float:
        """Compute the shaped reward for one step.

        TODO: implement the reward logic (speed, collision, lane changes, ...).
        Until implemented, the original reward is returned.
        """
        return original_reward
