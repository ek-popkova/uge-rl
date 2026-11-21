"""The one MLP builder both algorithms must use (decision: docs/config_decisions.md, "Shared training decisions").

Input: the 5 x 5 Kinematics observation flattened to ``OBS_DIM = 25`` features, no extra normalization (the env
already normalizes). Hidden layers: ``cfg.hidden_sizes`` (decided: 256, 256) with ``cfg.activation`` (decided: ReLU).
Same network size for both algorithms means:

    DDQN Q-network  = ``build_action_network(ddqn_cfg)``   25 -> 256 -> 256 -> 5
    PPO actor       = ``build_action_network(ppo_cfg)``    25 -> 256 -> 256 -> 5   (same parameter count)
    PPO critic      = ``build_value_network(ppo_cfg)``     25 -> 256 -> 256 -> 1   (extra network, belongs to PPO)

Weight initialization is NOT done here: it is algorithm-specific and left to each algorithm.
"""

from __future__ import annotations

from typing import Any, Sequence

from torch import nn

from common.actions import N_ACTIONS
from common.train_config import SUPPORTED_ACTIVATIONS

OBS_SHAPE = (5, 5)  # Kinematics, vehicles_count = 5 (ego + 4 nearest) x 5 features (checked against the env in tests)
OBS_DIM = OBS_SHAPE[0] * OBS_SHAPE[1]  # 25 inputs

_ACTIVATIONS: dict[str, type[nn.Module]] = {"relu": nn.ReLU, "tanh": nn.Tanh}
ACTIVATIONS = tuple(_ACTIVATIONS)
assert ACTIVATIONS == SUPPORTED_ACTIVATIONS  # the config validates exactly what the builder can build


def flatten_obs(obs: Any) -> Any:
    """Flatten the last two (feature) dimensions: ``(5, 5) -> (25,)`` and ``(batch, 5, 5) -> (batch, 25)``.

    Works for numpy arrays and torch tensors; both algorithms use it so that they feed the network identically.
    """
    if tuple(obs.shape[-2:]) != OBS_SHAPE:
        raise ValueError(f"expected observations of shape (..., {OBS_SHAPE[0]}, {OBS_SHAPE[1]}), got {tuple(obs.shape)}")
    return obs.reshape(*obs.shape[:-2], OBS_DIM)


def build_mlp(in_dim: int, out_dim: int, hidden_sizes: Sequence[int], activation: str = "relu") -> nn.Sequential:
    """``Linear -> act -> ... -> Linear`` with no activation after the output layer and default torch initialization."""
    if activation not in _ACTIVATIONS:
        raise ValueError(f"unknown activation {activation!r}; choose from {ACTIVATIONS}")
    hidden = [int(h) for h in hidden_sizes]
    if in_dim <= 0 or out_dim <= 0 or not hidden or any(h <= 0 for h in hidden):
        raise ValueError(f"invalid MLP sizes: in_dim={in_dim}, out_dim={out_dim}, hidden_sizes={tuple(hidden_sizes)}")
    layers: list[nn.Module] = []
    width = in_dim
    for h in hidden:
        layers += [nn.Linear(width, h), _ACTIVATIONS[activation]()]
        width = h
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


def build_action_network(cfg: Any) -> nn.Sequential:
    """DDQN Q-network / PPO actor: ``OBS_DIM`` inputs, ``N_ACTIONS`` outputs (Q-values / action logits)."""
    return build_mlp(OBS_DIM, N_ACTIONS, cfg.hidden_sizes, cfg.activation)


def build_value_network(cfg: Any) -> nn.Sequential:
    """PPO critic: ``OBS_DIM`` inputs, one output (the state value)."""
    return build_mlp(OBS_DIM, 1, cfg.hidden_sizes, cfg.activation)


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def expected_parameters(in_dim: int, out_dim: int, hidden_sizes: Sequence[int]) -> int:
    """Parameter count of ``build_mlp`` from the sizes alone (weights + biases), for tests and the docs."""
    sizes = [in_dim, *[int(h) for h in hidden_sizes], out_dim]
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
