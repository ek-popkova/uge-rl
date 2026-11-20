"""Double DQN agent (stub)."""

from __future__ import annotations

from typing import Any

from common.agent_base import Agent


class DDQNAgent(Agent):
    """Double DQN agent implemented from scratch in PyTorch.

    TODO: implement the Q-network, replay buffer, target network and the
    Double DQN update rule.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # TODO: build networks, optimizer, replay buffer.
        raise NotImplementedError("TODO: implement DDQNAgent.__init__")

    def act(self, obs: Any, deterministic: bool = True) -> Any:
        # TODO: epsilon-greedy (training) / greedy (deterministic) action.
        raise NotImplementedError("TODO: implement DDQNAgent.act")

    def save(self, path: str) -> None:
        # TODO: save network weights.
        raise NotImplementedError("TODO: implement DDQNAgent.save")

    def load(self, path: str) -> None:
        # TODO: load network weights.
        raise NotImplementedError("TODO: implement DDQNAgent.load")
