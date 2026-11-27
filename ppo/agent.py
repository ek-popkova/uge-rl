"""``PPOAgent``: the light policy snapshot used for validation, testing, checkpoints and videos.

It holds only the actor network (no critic, no rollout buffer, no optimizer), on the CPU, so it is cheap to pickle to
the validation worker processes (``common.evaluate``). The training state lives in ``ppo.train.PPOTrainer``.

Files written by ``save`` (``final.pt``, ``milestone_<p>pct.pt`` of the harness) hold the actor weights and the network
spec; ``PPOAgent().load(path)`` rebuilds the actor from that spec, as ``scripts/record_video.py`` does.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from common.agent_base import Agent
from common.networks import build_action_network, flatten_obs
from common.train_config import PPOConfig

AGENT_FILE_FORMAT = 1  # version of the dict written by PPOAgent.save


class PPOAgent(Agent):
    """Categorical policy ``pi_theta(a | s) = softmax(actor(s))`` over the five meta-actions.

    Args:
        cfg: gives the network spec (``hidden_sizes``, ``activation``); default ``PPOConfig()``.
        actor: an existing actor network (``PPOTrainer.make_eval_agent`` passes a copy); default: a new one.
        seed: seed of the numpy generator used only by ``act(..., deterministic=False)``.
    """

    def __init__(self, cfg: Optional[PPOConfig] = None, actor: Optional[nn.Module] = None, seed: int = 0) -> None:
        self.cfg = cfg if cfg is not None else PPOConfig()
        self.actor = actor if actor is not None else build_action_network(self.cfg)
        self.actor.eval()
        # A numpy Generator (not a torch.Generator) because the agent must be picklable for the validation workers.
        self.rng = np.random.default_rng(seed)

    def start_episode(self, seed: int) -> None:
        """Re-seed the sampling generator per episode, so stochastic play does not depend on episode order."""
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs: Any, deterministic: bool = True) -> Any:
        """Action for one observation ``(5, 5)`` (returns an int) or a batch ``(batch, 5, 5)`` (returns an array).

        deterministic: ``argmax_a logits`` (the evaluation protocol). Otherwise a sample from ``softmax(logits)``
        drawn with the Gumbel-max trick: ``argmax_a (logits_a + g_a)`` with independent standard Gumbel noise ``g`` is
        an exact sample of the softmax distribution, vectorized over the batch.
        """
        obs_array = np.asarray(obs, dtype=np.float32)
        is_single = obs_array.ndim == 2
        obs_tensor = torch.as_tensor(flatten_obs(obs_array))  # (25,) or (batch, 25)
        logits = self.actor(obs_tensor).numpy()  # (n_actions,) or (batch, n_actions)
        if deterministic:
            actions = np.argmax(logits, axis=-1)
        else:
            gumbel_noise = self.rng.gumbel(size=logits.shape)  # same shape as logits
            actions = np.argmax(logits + gumbel_noise, axis=-1)
        return int(actions) if is_single else actions

    def save(self, path: str) -> None:
        """Write the actor weights and the network spec needed to rebuild the actor."""
        torch.save(
            {
                "format": AGENT_FILE_FORMAT,
                "network": {"hidden_sizes": list(self.cfg.hidden_sizes), "activation": self.cfg.activation},
                "actor_state_dict": self.actor.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Inverse of ``save``: rebuild the actor from the stored spec and load its weights."""
        data = torch.load(path, weights_only=True)  # plain tensors and builtins only
        if data.get("format") != AGENT_FILE_FORMAT:
            raise ValueError(f"{path}: not a PPOAgent file of format {AGENT_FILE_FORMAT}")
        network = data["network"]
        self.cfg = dataclasses.replace(
            self.cfg, hidden_sizes=tuple(network["hidden_sizes"]), activation=network["activation"]
        )
        self.actor = build_action_network(self.cfg)
        self.actor.load_state_dict(data["actor_state_dict"])
        self.actor.eval()
