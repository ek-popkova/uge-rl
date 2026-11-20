"""Abstract agent interface shared by all algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """Minimal interface every agent (DDQN, PPO, baselines) must implement.

    The evaluation code in ``common.evaluate`` only relies on this interface,
    which guarantees that all algorithms are evaluated identically.
    """

    @abstractmethod
    def act(self, obs: Any, deterministic: bool = True) -> Any:
        """Select an action for a single observation.

        Args:
            obs: Observation returned by the environment.
            deterministic: If True, act greedily (evaluation mode); otherwise
                sample / explore (training mode).

        Returns:
            An action valid for the environment's action space.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the agent's parameters to ``path``."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the agent's parameters from ``path``."""
