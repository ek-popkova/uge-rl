"""Training budget and hyperparameter configs shared by DDQN and PPO (no algorithm code here).

``TrainBudget`` is identical for both algorithms (same number of env steps, same validation and
checkpoint schedule). ``DDQNConfig`` and ``PPOConfig`` hold every hyperparameter the algorithms
will need. NO VALUE HERE IS DECIDED: every field marked ``# PLACEHOLDER: decide before training``
is a conventional starting point that only keeps the code runnable; the team must fix them (and
list the sensitivity grid) before any training run, see the "TO DECIDE BEFORE TRAINING" list in
CLAUDE.md. ``placeholders_in_use`` names the fields that still hold an undecided value; the training
harness prints it and stores it in ``run_meta.json``. When a value is decided, replace ``_ph(...)``
by a plain default and remove the marker.

Hyperparameter sensitivity runs override single fields with ``--set key=value`` (``parse_set_args``,
``apply_overrides``); the overridden keys are recorded in ``run_meta.json`` and ``variant_name``
turns them into the directory level ``runs/<algo>/<reward>/<variant>/seed<s>``.
"""

from __future__ import annotations

import dataclasses
import re
import typing
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional, Sequence

from common.config import MIN_GAMMA, VAL_SEEDS
from common.paths import MAIN_VARIANT, check_variant


def _ph(value: Any) -> Any:
    """A dataclass field with a default that is still a placeholder."""
    return field(default=value, metadata={"placeholder": True})


def _to_jsonable(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


class _ConfigBase:
    """JSON round-trip, validation helpers and placeholder bookkeeping for the config dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of the fields (tuples as lists), ready for ``json.dump``."""
        return {f.name: _to_jsonable(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        """Inverse of ``to_dict``; unknown keys raise, missing keys keep their default."""
        names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"{cls.__name__} has no field(s) {unknown}; fields: {sorted(names)}")
        hints = typing.get_type_hints(cls)
        return cls(**{k: _coerce(hints[k], v) for k, v in data.items()})

    def placeholder_fields(self) -> list[str]:
        """Names of the fields that are placeholders (whatever their current value)."""
        return [f.name for f in fields(self) if f.metadata.get("placeholder")]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainBudget(_ConfigBase):
    """What both algorithms get: identical for DDQN and PPO (docs/evaluation_protocol.md).

    All counts are in environment steps (transitions, summed over vectorized envs).
    """

    total_env_steps: int = _ph(100_000)  # PLACEHOLDER: decide before training
    val_every_env_steps: int = _ph(5_000)  # PLACEHOLDER: decide before training
    val_n_episodes: int = _ph(20)  # PLACEHOLDER: decide before training (<= len(VAL_SEEDS))
    checkpoint_every_env_steps: int = _ph(10_000)  # PLACEHOLDER: decide before training (resume checkpoint)
    # Fractions of total_env_steps at which a weights-only checkpoint is kept for "early vs late"
    # behaviour (videos); the final checkpoint at 100% is final.pt.
    milestone_fractions: tuple[float, ...] = _ph((0.1, 0.5))  # PLACEHOLDER: decide before training

    def __post_init__(self) -> None:
        object.__setattr__(self, "milestone_fractions", tuple(float(x) for x in self.milestone_fractions))
        for name in ("total_env_steps", "val_every_env_steps", "val_n_episodes", "checkpoint_every_env_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"TrainBudget.{name} must be positive, got {getattr(self, name)}")
        if self.val_n_episodes > len(VAL_SEEDS):
            raise ValueError(f"val_n_episodes must be <= len(VAL_SEEDS) = {len(VAL_SEEDS)}, got {self.val_n_episodes}")
        if self.val_every_env_steps > self.total_env_steps:
            raise ValueError("val_every_env_steps exceeds total_env_steps: no validation point would exist")
        if not all(0.0 < x < 1.0 for x in self.milestone_fractions):
            raise ValueError(f"milestone_fractions must lie in (0, 1), got {self.milestone_fractions}")

    @property
    def n_val_points(self) -> int:
        """Expected number of validation points (the last one is always at the end)."""
        return max(self.total_env_steps // self.val_every_env_steps, 1)


# ---------------------------------------------------------------------------------------------
# Algorithm configs
# ---------------------------------------------------------------------------------------------
def _check_common(cfg: Any) -> None:
    if not MIN_GAMMA <= cfg.gamma <= 1.0:
        raise ValueError(f"gamma must be in [{MIN_GAMMA}, 1] (common.config.MIN_GAMMA), got {cfg.gamma}")
    if cfg.lr <= 0:
        raise ValueError(f"lr must be positive, got {cfg.lr}")
    if cfg.max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be positive, got {cfg.max_grad_norm}")
    hs = tuple(int(h) for h in cfg.hidden_sizes)
    if not hs or any(h <= 0 for h in hs):
        raise ValueError(f"hidden_sizes must be a non-empty tuple of positive ints, got {cfg.hidden_sizes}")
    object.__setattr__(cfg, "hidden_sizes", hs)
    if cfg.device != "cpu":
        # CLAUDE.md rule 8: the GPU does not help; switch only with a measurement that shows a gain.
        raise ValueError(f"device must be 'cpu' (CLAUDE.md performance rule 8), got {cfg.device!r}")


@dataclass(frozen=True)
class DDQNConfig(_ConfigBase):
    """Double DQN hyperparameters."""

    lr: float = _ph(5e-4)  # PLACEHOLDER: decide before training
    gamma: float = _ph(0.99)  # PLACEHOLDER: decide before training (>= MIN_GAMMA)
    buffer_size: int = _ph(50_000)  # PLACEHOLDER: decide before training
    batch_size: int = _ph(64)  # PLACEHOLDER: decide before training
    learning_starts: int = _ph(1_000)  # PLACEHOLDER: decide before training (env steps before the first update)
    train_freq: int = _ph(1)  # PLACEHOLDER: decide before training (env steps between update rounds)
    gradient_steps: int = _ph(1)  # PLACEHOLDER: decide before training (gradient steps per update round)
    target_update_interval: int = _ph(500)  # PLACEHOLDER: decide before training (env steps)
    tau: float = _ph(1.0)  # PLACEHOLDER: decide before training (1.0 = hard copy, < 1 = Polyak averaging)
    eps_start: float = _ph(1.0)  # PLACEHOLDER: decide before training
    eps_end: float = _ph(0.05)  # PLACEHOLDER: decide before training
    eps_decay_steps: int = _ph(20_000)  # PLACEHOLDER: decide before training (linear decay, env steps)
    max_grad_norm: float = _ph(10.0)  # PLACEHOLDER: decide before training
    hidden_sizes: tuple[int, ...] = _ph((64, 64))  # PLACEHOLDER: decide before training (same as PPO)
    device: str = "cpu"  # decided: CLAUDE.md performance rule 8

    def __post_init__(self) -> None:
        _check_common(self)
        for name in ("buffer_size", "batch_size", "train_freq", "gradient_steps", "target_update_interval",
                     "eps_decay_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"DDQNConfig.{name} must be positive, got {getattr(self, name)}")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be >= 0")
        if self.batch_size > self.buffer_size or self.learning_starts > self.buffer_size:
            raise ValueError("batch_size and learning_starts must not exceed buffer_size")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if not 0.0 <= self.eps_end <= self.eps_start <= 1.0:
            raise ValueError(f"need 0 <= eps_end <= eps_start <= 1, got {self.eps_end}, {self.eps_start}")


@dataclass(frozen=True)
class PPOConfig(_ConfigBase):
    """PPO hyperparameters. One rollout has ``n_envs * n_steps`` env steps."""

    lr: float = _ph(3e-4)  # PLACEHOLDER: decide before training
    gamma: float = _ph(0.99)  # PLACEHOLDER: decide before training (>= MIN_GAMMA)
    gae_lambda: float = _ph(0.95)  # PLACEHOLDER: decide before training
    clip_range: float = _ph(0.2)  # PLACEHOLDER: decide before training
    n_envs: int = _ph(4)  # PLACEHOLDER: decide before training (vectorized envs, algorithmic)
    n_steps: int = _ph(250)  # PLACEHOLDER: decide before training (steps per env per rollout)
    n_epochs: int = _ph(10)  # PLACEHOLDER: decide before training
    minibatch_size: int = _ph(100)  # PLACEHOLDER: decide before training
    ent_coef: float = _ph(0.01)  # PLACEHOLDER: decide before training
    vf_coef: float = _ph(0.5)  # PLACEHOLDER: decide before training
    max_grad_norm: float = _ph(0.5)  # PLACEHOLDER: decide before training
    target_kl: Optional[float] = _ph(None)  # PLACEHOLDER: decide before training (None = no early stop)
    hidden_sizes: tuple[int, ...] = _ph((64, 64))  # PLACEHOLDER: decide before training (same as DDQN)
    device: str = "cpu"  # decided: CLAUDE.md performance rule 8

    def __post_init__(self) -> None:
        _check_common(self)
        for name in ("n_envs", "n_steps", "n_epochs", "minibatch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"PPOConfig.{name} must be positive, got {getattr(self, name)}")
        if self.minibatch_size > self.rollout_size:
            raise ValueError(f"minibatch_size {self.minibatch_size} exceeds the rollout size {self.rollout_size}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if not 0.0 < self.clip_range < 1.0:
            raise ValueError(f"clip_range must be in (0, 1), got {self.clip_range}")
        if self.ent_coef < 0 or self.vf_coef < 0:
            raise ValueError("ent_coef and vf_coef must be >= 0")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be positive or None")

    @property
    def rollout_size(self) -> int:
        return self.n_envs * self.n_steps


CONFIG_CLASSES = {"ddqn": DDQNConfig, "ppo": PPOConfig}


def check_comparable(ddqn_cfg: DDQNConfig, ppo_cfg: PPOConfig) -> None:
    """Both algorithms must use the same network size (and device); raises ``ValueError`` otherwise."""
    if tuple(ddqn_cfg.hidden_sizes) != tuple(ppo_cfg.hidden_sizes):
        raise ValueError(
            f"hidden_sizes differ: DDQN {tuple(ddqn_cfg.hidden_sizes)} vs PPO {tuple(ppo_cfg.hidden_sizes)} "
            "(the protocol requires the same network size for both algorithms)"
        )
    if ddqn_cfg.device != ppo_cfg.device:
        raise ValueError(f"device differs: {ddqn_cfg.device!r} vs {ppo_cfg.device!r}")


def placeholders_in_use(*objs: Any, overridden: Sequence[str] = ()) -> list[str]:
    """``ClassName.field`` of every placeholder field not replaced by an explicit override."""
    out = []
    for obj in objs:
        for name in obj.placeholder_fields():
            if name not in overridden:
                out.append(f"{type(obj).__name__}.{name}")
    return out


# ---------------------------------------------------------------------------------------------
# Overrides (``--set key=value``) and variant names
# ---------------------------------------------------------------------------------------------
_NONE_TOKENS = {"none", "null"}


def _coerce(tp: Any, value: Any) -> Any:
    """Convert ``value`` (a raw CLI string or a JSON value) to the field type ``tp``."""
    origin = typing.get_origin(tp)
    if origin is typing.Union:  # Optional[X]
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if value is None or (isinstance(value, str) and value.strip().lower() in _NONE_TOKENS):
            return None
        return _coerce(args[0], value)
    if origin is tuple:  # tuple[int, ...]
        if isinstance(value, str):
            parts = [p for p in re.split(r"[,x\s]+", value.strip().strip("()[]")) if p]
        else:
            parts = list(value)
        return tuple(_coerce(typing.get_args(tp)[0], p) for p in parts)
    if tp is bool:
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no"):
                return False
            raise ValueError(f"cannot read {value!r} as bool")
        return bool(value)
    if tp is int:
        number = float(value)
        if number != int(number):
            raise ValueError(f"{value!r} is not an integer")
        return int(number)
    if tp is float:
        return float(value)
    return value if tp is Any else tp(value)


def parse_set_args(items: Sequence[str], cls: type) -> dict[str, Any]:
    """Parse ``["lr=3e-4", "hidden_sizes=64,64"]`` into typed overrides of the dataclass ``cls``."""
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in names:
            raise ValueError(f"--set: {cls.__name__} has no field {key!r}; fields: {names}")
        if key in out:
            raise ValueError(f"--set: {key!r} given twice")
        out[key] = _coerce(hints[key], raw)
    return out


def apply_overrides(cfg: Any, overrides: Mapping[str, Any]) -> Any:
    """Copy of ``cfg`` with the overrides applied (validated again by ``__post_init__``)."""
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


def _format_value(value: Any) -> str:
    if isinstance(value, tuple):
        return "x".join(_format_value(v) for v in value)
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def variant_name(overrides: Mapping[str, Any]) -> str:
    """Directory name of a sensitivity variant, e.g. ``clip_range=0.1`` or ``gamma=0.995,lr=0.0003``.

    Keys are sorted, so the name does not depend on the order of ``--set``; no overrides give
    ``"main"``.
    """
    if not overrides:
        return MAIN_VARIANT
    name = ",".join(f"{k}={_format_value(v)}" for k, v in sorted(overrides.items()))
    check_variant(name)
    return name
