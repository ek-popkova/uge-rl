"""PPO agent (stub)."""

from __future__ import annotations

from typing import Any

from common.agent_base import Agent


class PPOAgent(Agent):
    """PPO (actor-critic) agent implemented from scratch in PyTorch.

    TODO: implement the actor-critic network, rollout buffer, GAE and the
    clipped PPO update.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # TODO: build actor-critic network, optimizer, rollout buffer.
        raise NotImplementedError("TODO: implement PPOAgent.__init__")

    def act(self, obs: Any, deterministic: bool = True) -> Any:
        # TODO: sample from the policy (training) / argmax (deterministic).
        raise NotImplementedError("TODO: implement PPOAgent.act")

    def save(self, path: str) -> None:
        # TODO: save network weights.
        raise NotImplementedError("TODO: implement PPOAgent.save")

    def load(self, path: str) -> None:
        # TODO: load network weights.
        raise NotImplementedError("TODO: implement PPOAgent.load")
