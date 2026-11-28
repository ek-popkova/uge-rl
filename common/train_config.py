"""Training budget and hyperparameter configs shared by DDQN and PPO (no algorithm code here).

``TrainBudget`` is identical for both algorithms (same number of env steps, same validation and
checkpoint schedule). ``DDQNConfig`` and ``PPOConfig`` hold every hyperparameter the algorithms
will need. The SHARED decisions are frozen here as plain defaults (gamma, network spec, milestone
fractions; ``docs/config_decisions.md``, "Shared training decisions") and may only change by team
agreement. Every field still marked ``# PLACEHOLDER: decide before training`` is a conventional starting
point that only keeps the code runnable, see the "TO DECIDE BEFORE TRAINING" list in CLAUDE.md.
``placeholders_in_use`` names the fields that still hold an undecided value; the training harness prints
it and stores it in ``run_meta.json``. When a value is decided, replace ``_ph(...)`` by a plain default and
remove the marker.

Hyperparameter sensitivity runs override single fields with ``--set key=value`` (``parse_set_args``,
``apply_overrides``); the overridden keys are recorded in ``run_meta.json`` and ``variant_name``
turns them into the directory level ``runs/<algo>/<reward>/<variant>/seed<s>``. The sensitivity grid is data
(``SENSITIVITY_GRID``, relative to the base values) and ``expand_sensitivity_grid`` lists its 24 runs; ``check_sensitivity_run`` verifies that a
sensitivity run differs from the main-grid run only in its varied parameter. Tuning runs (variant ``tune-<name>``,
run seed ``PILOT_RUN_SEED``) are checked by ``check_run_roles``. Supplementary runs (variant ``supp-<name>``) are pre-declared in
the registry ``SUPPLEMENTARY_RUNS`` (validated at import); ``expand_supplementary_runs`` lists them, ``check_run_roles`` refuses
a run that does not match its entry and every other unregistered variant, and ``check_supplementary_run`` compares a finished
run with its main-grid base run.
"""

from __future__ import annotations

import dataclasses
import math
import re
import typing
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional, Sequence

from common.config import (
    CALIBRATION_ONLY_SEED,
    MIN_GAMMA,
    PILOT_RUN_SEED,
    SENSITIVITY_DENSITY,
    SENSITIVITY_REWARD,
    SENSITIVITY_SEEDS,
    TRAIN_DENSITY,
    TRAIN_SEEDS,
    TUNING_DENSITY,
    TUNING_REWARD,
    VAL_SEEDS,
    all_reward_names,
    is_supplementary_reward,
)
from common.paths import (
    ALGORITHMS,
    MAIN_VARIANT,
    SUPPLEMENTARY_PREFIX,
    check_variant,
    is_supplementary_variant,
    is_tuning_variant,
    supplementary_variant,
)

SUPPORTED_ACTIVATIONS = ("relu", "tanh")  # common.networks builds exactly these
# Decided for BOTH algorithms; never varied by tuning or sensitivity runs and equal in the two configs.
SHARED_KEYS = frozenset({"gamma", "hidden_sizes", "activation", "obs_flatten", "ppo_shared_trunk", "device"})
POLICY_LOSSES = ("clip", "kl")  # PPOConfig.policy_loss: PPO-Clip or PPO-KL (report section 3.3)


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

    # DECIDED after the SB3 calibration (docs/calibration.md section 6, docs/config_decisions.md "Values frozen after the
    # calibration"). Fixed before the main grid; a change after it started is reported as a deviation.
    # 300,000: 8 of the 10 calibration runs (200k each) had not plateaued by the end of their budget and the plateau rule
    # gives 300k; the team accepted the cost (about 39 h on one machine for the 64 runs, split over two machines).
    # Must be a multiple of the PPO rollout size n_envs * n_steps (check_budget_matches_rollout).
    total_env_steps: int = 300_000
    # 12,000: 25 points per learning curve at 300k (the calibration proposed 25 points; 8,000 was computed for 200k).
    # Must also be a multiple of the PPO rollout size, otherwise PPO validates at shifted steps.
    val_every_env_steps: int = 12_000
    # 20 = all VAL_SEEDS: 50 episodes would need 30 new seeds, a re-derived `beats_idle` and about 4 h more machine time.
    val_n_episodes: int = 20  # <= len(VAL_SEEDS)
    # Resume checkpoint interval: an engineering setting that does not affect any result (at most this much work is lost
    # after a crash). Decided: about 10 min of wall time per checkpoint. Assumption: 12-17 env steps/s per process while
    # 8-12 runs train in parallel (aggregate 140 steps/s, docs/performance_analysis.md C5; network updates make it a bit
    # slower), so 10,000 env steps are about 10-14 min. Independent of total_env_steps.
    checkpoint_every_env_steps: int = 10_000
    # Fractions of total_env_steps at which a weights-only checkpoint is kept for "early / middle / final"
    # behaviour (videos); the final checkpoint at 100% is final.pt and is always kept. Decided by the team.
    milestone_fractions: tuple[float, ...] = (0.1, 0.25, 0.5)

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
    if cfg.activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(f"activation must be one of {SUPPORTED_ACTIVATIONS}, got {cfg.activation!r}")
    if cfg.obs_flatten is not True:
        raise ValueError("obs_flatten must be True: both networks are MLPs on the flattened 25-feature observation")
    if cfg.device != "cpu":
        # CLAUDE.md rule 8: the GPU does not help; switch only with a measurement that shows a gain.
        raise ValueError(f"device must be 'cpu' (CLAUDE.md performance rule 8), got {cfg.device!r}")


@dataclass(frozen=True)
class DDQNConfig(_ConfigBase):
    """Double DQN hyperparameters."""

    lr: float = _ph(5e-4)  # PLACEHOLDER: decide before training
    gamma: float = 0.99  # decided (shared, >= MIN_GAMMA)
    buffer_size: int = _ph(50_000)  # PLACEHOLDER: decide before training
    batch_size: int = _ph(64)  # PLACEHOLDER: decide before training
    learning_starts: int = _ph(1_000)  # PLACEHOLDER: decide before training (env steps before the first update)
    train_freq: int = _ph(1)  # PLACEHOLDER: decide before training (env steps between update rounds)
    gradient_steps: int = _ph(1)  # PLACEHOLDER: decide before training (gradient steps per update round)
    target_update_interval: int = _ph(500)  # PLACEHOLDER: decide before training (env steps)
    tau: float = _ph(1.0)  # PLACEHOLDER: decide before training (1.0 = hard copy, < 1 = Polyak averaging)
    eps_start: float = _ph(1.0)  # PLACEHOLDER: decide before training
    eps_end: float = _ph(0.05)  # PLACEHOLDER: decide before training
    # Linear epsilon decay from eps_start to eps_end over this FRACTION of total_env_steps (so it follows the budget);
    # resolved to absolute env steps at run start (``eps_decay_steps``) and stored in run_meta.json ("resolved").
    eps_decay_fraction: float = _ph(0.3)  # PLACEHOLDER: decide before training (in (0, 1])
    max_grad_norm: float = _ph(10.0)  # PLACEHOLDER: decide before training
    hidden_sizes: tuple[int, ...] = (256, 256)  # decided (shared): the Q-network equals the PPO actor
    activation: str = "relu"  # decided (shared)
    obs_flatten: bool = True  # decided (shared): 5x5 Kinematics flattened to 25 features, no extra normalization
    device: str = "cpu"  # decided: CLAUDE.md performance rule 8

    def __post_init__(self) -> None:
        _check_common(self)
        for name in ("buffer_size", "batch_size", "train_freq", "gradient_steps", "target_update_interval"):
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
        if not 0.0 < self.eps_decay_fraction <= 1.0:
            raise ValueError(f"eps_decay_fraction must be in (0, 1], got {self.eps_decay_fraction}")

    def eps_decay_steps(self, total_env_steps: int) -> int:
        """Env steps of the linear epsilon decay: ``round(eps_decay_fraction * total_env_steps)``, at least 1."""
        return max(1, round(self.eps_decay_fraction * total_env_steps))


@dataclass(frozen=True)
class PPOConfig(_ConfigBase):
    """PPO hyperparameters. One rollout has ``n_envs * n_steps`` env steps.

    The algorithm-specific values are the configuration chosen by tuning (``tune-nsteps500``, ``docs/tuning_log.md``):
    ``n_steps`` was varied and selected, ``ent_coef`` was varied and kept, the others are the tuning defaults.
    """

    lr: float = 3e-4  # decided: tuning default, not varied (base of the lr sensitivity grid)
    gamma: float = 0.99  # decided (shared, >= MIN_GAMMA)
    gae_lambda: float = 0.95  # decided: tuning default, not varied
    clip_range: float = 0.2  # decided: tuning default, not varied (base of the clip_range sensitivity grid)
    n_envs: int = 4  # decided: tuning default, not varied (vectorized envs, algorithmic)
    n_steps: int = 500  # decided by tuning (docs/tuning_log.md): steps per env per rollout, rollout 4 x 500 = 2,000
    n_epochs: int = 10  # decided: tuning default, not varied
    minibatch_size: int = 100  # decided: tuning default, not varied (20 minibatches per rollout)
    ent_coef: float = 0.01  # decided by tuning (docs/tuning_log.md): 0.05 was tried and not chosen
    vf_coef: float = 0.5  # decided: tuning default, not varied
    max_grad_norm: float = 0.5  # decided: tuning default, not varied
    target_kl: Optional[float] = None  # decided: tuning default, not varied (None = no KL early stop)
    # Policy loss (report section 3.3): "clip" = PPO-Clip (eq:ppo-clip), "kl" = PPO-KL (adaptive KL penalty). Decided.
    policy_loss: str = "clip"
    # PPO-KL only, decided (the PPO paper's values, not tuned): d_targ of the adaptive KL penalty (Schulman et al. 2017
    # used 0.003 / 0.01 / 0.03) and the initial penalty coefficient beta (the paper starts at 1 and reports low
    # sensitivity to it).
    kl_target: float = 0.01
    kl_beta_init: float = 1.0
    hidden_sizes: tuple[int, ...] = (256, 256)  # decided (shared): size of the actor AND of the critic
    activation: str = "relu"  # decided (shared)
    obs_flatten: bool = True  # decided (shared): 5x5 Kinematics flattened to 25 features, no extra normalization
    ppo_shared_trunk: bool = False  # decided: separate actor and critic networks (the critic is an extra network)
    device: str = "cpu"  # decided: CLAUDE.md performance rule 8

    def __post_init__(self) -> None:
        _check_common(self)
        for name in ("n_envs", "n_steps", "n_epochs", "minibatch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"PPOConfig.{name} must be positive, got {getattr(self, name)}")
        if self.minibatch_size > self.rollout_size:
            raise ValueError(f"minibatch_size {self.minibatch_size} exceeds the rollout size {self.rollout_size}")
        if self.rollout_size % self.minibatch_size:
            # A remainder would give a short last minibatch (a 1-sample minibatch has a normalized advantage of 0 and no
            # policy gradient), so every minibatch has exactly minibatch_size samples.
            raise ValueError(
                f"minibatch_size {self.minibatch_size} must divide the rollout size n_envs * n_steps = {self.rollout_size}"
            )
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if not 0.0 < self.clip_range < 1.0:
            raise ValueError(f"clip_range must be in (0, 1), got {self.clip_range}")
        if self.ent_coef < 0 or self.vf_coef < 0:
            raise ValueError("ent_coef and vf_coef must be >= 0")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be positive or None")
        if self.policy_loss not in POLICY_LOSSES:
            raise ValueError(f"policy_loss must be one of {POLICY_LOSSES}, got {self.policy_loss!r}")
        if self.kl_target <= 0 or self.kl_beta_init <= 0:
            raise ValueError(f"kl_target and kl_beta_init must be positive, got {self.kl_target}, {self.kl_beta_init}")

    @property
    def rollout_size(self) -> int:
        return self.n_envs * self.n_steps


CONFIG_CLASSES = {"ddqn": DDQNConfig, "ppo": PPOConfig}


def _nearest_multiples(value: int, unit: int) -> tuple[int, int]:
    """The multiples of ``unit`` just below and just above ``value`` (the lower one is at least ``unit``)."""
    below = max(unit, (value // unit) * unit)
    above = ((value + unit - 1) // unit) * unit
    return below, max(above, below)


def _nearest_divisors(g: int, near: int) -> tuple[int, int]:
    """The divisors of ``g`` just below and just above ``near`` (``near`` itself if it divides ``g``)."""
    divisors = [d for d in range(1, g + 1) if g % d == 0]
    below = max((d for d in divisors if d <= near), default=divisors[0])
    above = min((d for d in divisors if d >= near), default=divisors[-1])
    return below, above


def check_budget_matches_rollout(budget: "TrainBudget", rollout_size: int, what: str = "the PPO rollout size n_envs * n_steps") -> None:
    """``total_env_steps`` and ``val_every_env_steps`` must be multiples of the PPO rollout size ``n_envs * n_steps``.

    PPO collects ``rollout_size`` transitions per update and can only stop or validate between updates. If the budget is
    not a multiple, PPO overshoots it (or the harness refuses), and with a validation interval that is not a multiple PPO
    validates at shifted env steps, so the two algorithms no longer get exactly the same steps and the learning curves are
    no longer aligned. Raises ``ValueError`` that names the nearest valid budget values and the nearest valid rollout
    sizes (the divisors of ``gcd(total_env_steps, val_every_env_steps)``).
    """
    if rollout_size <= 0:
        raise ValueError(f"{what} must be positive, got {rollout_size}")
    problems = []
    for name in ("total_env_steps", "val_every_env_steps"):
        value = getattr(budget, name)
        if value % rollout_size:
            lo, hi = _nearest_multiples(value, rollout_size)
            problems.append(f"{name} = {value:,} is not a multiple of {what} = {rollout_size:,} (nearest valid values: {lo:,} or {hi:,})")
    if problems:
        g = math.gcd(budget.total_env_steps, budget.val_every_env_steps)
        lo, hi = _nearest_divisors(g, rollout_size)
        raise ValueError(
            "; ".join(problems) + f". Alternatively choose n_envs * n_steps that divides both {budget.total_env_steps:,} and "
            f"{budget.val_every_env_steps:,}, i.e. a divisor of {g:,} (nearest to {rollout_size:,}: {lo:,} or {hi:,}) so that "
            "both algorithms train for exactly the same number of env steps and validate at the same steps"
        )


def check_comparable(ddqn_cfg: DDQNConfig, ppo_cfg: PPOConfig, budget: Optional["TrainBudget"] = None) -> None:
    """Both algorithms must share gamma, network size, activation, observation handling and device.

    Raises ``ValueError`` naming the first field that differs. DDQN Q-network = PPO actor; the PPO critic is an extra
    network of the same size, so only ``hidden_sizes`` is compared. With ``budget`` the budget must also fit the PPO
    rollout size (``check_budget_matches_rollout``): both algorithms then train for exactly ``total_env_steps`` env steps
    and validate at the same env steps.
    """
    if budget is not None:
        check_budget_matches_rollout(budget, ppo_cfg.rollout_size)
    if ddqn_cfg.gamma != ppo_cfg.gamma:
        raise ValueError(f"gamma differs: DDQN {ddqn_cfg.gamma} vs PPO {ppo_cfg.gamma} (the protocol fixes one gamma)")
    if tuple(ddqn_cfg.hidden_sizes) != tuple(ppo_cfg.hidden_sizes):
        raise ValueError(
            f"hidden_sizes differ: DDQN {tuple(ddqn_cfg.hidden_sizes)} vs PPO {tuple(ppo_cfg.hidden_sizes)} "
            "(the protocol requires the same network size for both algorithms)"
        )
    if ddqn_cfg.activation != ppo_cfg.activation:
        raise ValueError(f"activation differs: DDQN {ddqn_cfg.activation!r} vs PPO {ppo_cfg.activation!r}")
    if ddqn_cfg.obs_flatten != ppo_cfg.obs_flatten:
        raise ValueError(f"obs_flatten differs: DDQN {ddqn_cfg.obs_flatten} vs PPO {ppo_cfg.obs_flatten}")
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


def resolve_hyperparameters(cfg: Any, budget: TrainBudget) -> dict[str, Any]:
    """Absolute values derived from relative hyperparameters at run start; stored as ``resolved`` in run_meta.json.

    DDQN: ``eps_decay_steps`` (from ``eps_decay_fraction`` and ``total_env_steps``). PPO: ``rollout_size``.
    """
    if isinstance(cfg, DDQNConfig):
        return {"eps_decay_steps": cfg.eps_decay_steps(budget.total_env_steps)}
    if isinstance(cfg, PPOConfig):
        return {"rollout_size": cfg.rollout_size}
    return {}


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


# ---------------------------------------------------------------------------------------------
# Tuning and sensitivity protocol (shared decisions, docs/config_decisions.md)
# ---------------------------------------------------------------------------------------------
# Budget fields whose change makes a run a debug run (never a result); the checkpoint interval does not affect results.
DEBUG_BUDGET_KEYS = frozenset({"total_env_steps", "val_every_env_steps", "val_n_episodes"})


def effective_budget_overrides(budget_kw: Mapping[str, Any]) -> dict[str, Any]:
    """The budget overrides whose value differs from the frozen ``TrainBudget`` default (equal values are dropped)."""
    frozen = TrainBudget()
    return {k: v for k, v in budget_kw.items() if v != getattr(frozen, k)}


def check_run_roles(
    variant: str,
    seed: int,
    reward: str,
    density: str,
    overrides: Mapping[str, Any],
    budget_overrides: Sequence[str],
    *,
    algo: str,
) -> None:
    """Every run has exactly one role; raises ``ValueError`` on a violation (called by ``resolve_setup`` before a run).

    The roles (``common.paths.variant_role``):

    * ``main``: no hyperparameter overrides.
    * tuning, ``tune-<name>``: the run seed ``PILOT_RUN_SEED``, the reward ``TUNING_REWARD`` and the density
      ``TUNING_DENSITY``, algorithm-specific hyperparameters only (never ``SHARED_KEYS``), the default budget and validation
      settings (no budget overrides). Conversely ``PILOT_RUN_SEED`` is reserved for tuning.
    * supplementary, ``supp-<name>``: ``name`` is a key of ``SUPPLEMENTARY_RUNS`` and the run matches its entry exactly
      (``check_supplementary_role``).
    * sensitivity, any other variant: the run must be one point of ``expand_sensitivity_grid()`` (same algorithm, the
      overrides of that point and nothing else, its generated variant name, reward ``SENSITIVITY_REWARD``, density
      ``SENSITIVITY_DENSITY``, a seed of ``SENSITIVITY_SEEDS``). An unregistered variant is refused, so a result run is always
      main, a sensitivity point, ``tune-*`` or ``supp-*``.

    A ``SUPPLEMENTARY_REWARDS`` preset is accepted in a registered supplementary run only. Debug runs, i.e. runs whose
    budget really differs from the frozen ``TrainBudget`` in a result-relevant value (``DEBUG_BUDGET_KEYS``: total steps,
    validation interval or validation episodes; tests and smoke runs, never results), are exempt from the sensitivity rule
    (their budget already makes them unusable as results). ``budget_overrides`` must hold only the keys whose value differs
    from the frozen budget (``effective_budget_overrides``, applied by ``resolve_setup``): ``--total-steps 300000`` is not an
    override, and a different checkpoint interval alone (no effect on results) does not make a debug run. Tuning and
    supplementary runs never take budget overrides.
    """
    overrides = dict(overrides)
    if seed == CALIBRATION_ONLY_SEED:  # reserved for the SB3 calibration runs (scripts/calibration/), never a result seed
        raise ValueError(f"run seed {CALIBRATION_ONLY_SEED} (CALIBRATION_ONLY_SEED) is reserved for the SB3 calibration runs")
    if is_tuning_variant(variant):
        _check_tuning_role(variant, seed, reward, density, overrides, budget_overrides)
        return
    if seed == PILOT_RUN_SEED:
        raise ValueError(f"run seed {PILOT_RUN_SEED} (PILOT_RUN_SEED) is reserved for tuning runs: use a 'tune-<name>' variant")
    if is_supplementary_variant(variant):
        check_supplementary_role(variant, algo, seed, reward, density, overrides, budget_overrides)
        return
    if is_supplementary_reward(reward):
        raise ValueError(
            f"reward {reward!r} is a SUPPLEMENTARY_REWARDS preset: only a registered supplementary run ('supp-<name>', "
            f"SUPPLEMENTARY_RUNS) may train with it, never a main, tuning or sensitivity run"
        )
    if variant == MAIN_VARIANT:
        if overrides:
            raise ValueError(f"the main variant takes no hyperparameter overrides, got {sorted(overrides)}")
        return
    if set(budget_overrides) & DEBUG_BUDGET_KEYS:
        return  # debug run (see the docstring)
    _check_sensitivity_role(variant, algo, seed, reward, density, overrides)


def _check_tuning_role(variant: str, seed: int, reward: str, density: str, overrides: Mapping[str, Any],
                       budget_overrides: Sequence[str]) -> None:
    problems = []
    if seed != PILOT_RUN_SEED:
        problems.append(f"tuning runs use the run seed {PILOT_RUN_SEED}, got {seed}")
    if reward != TUNING_REWARD:
        problems.append(f"tuning runs use the reward {TUNING_REWARD!r}, got {reward!r}")
    if density != TUNING_DENSITY:
        problems.append(f"tuning runs use the density {TUNING_DENSITY!r}, got {density!r}")
    shared = sorted(set(overrides) & SHARED_KEYS)
    if shared:
        problems.append(f"tuning may not vary the shared decisions {shared}")
    if budget_overrides:
        problems.append(f"tuning may not change the budget or validation settings {sorted(budget_overrides)}")
    if problems:
        raise ValueError(f"invalid tuning run {variant!r}: " + "; ".join(problems))


def _check_sensitivity_role(variant: str, algo: str, seed: int, reward: str, density: str,
                            overrides: Mapping[str, Any]) -> None:
    points = {r.variant: r for r in expand_sensitivity_grid(seeds=SENSITIVITY_SEEDS[:1]) if r.algo == algo}
    point = points.get(variant)
    if point is None or dict(point.overrides) != dict(overrides):
        raise ValueError(
            f"variant {variant!r} with overrides {dict(overrides)} is not a registered run: a result run is 'main' (no "
            f"overrides), a sensitivity point of expand_sensitivity_grid() ({algo}: {sorted(points)}, variant name generated "
            f"from --set), a tuning run 'tune-<name>' or a supplementary run 'supp-<name>' of SUPPLEMENTARY_RUNS "
            f"({sorted(SUPPLEMENTARY_RUNS)}); debug runs need budget overrides (--total-steps ...)"
        )
    problems = []
    if reward != SENSITIVITY_REWARD:
        problems.append(f"reward {SENSITIVITY_REWARD!r}, got {reward!r}")
    if density != SENSITIVITY_DENSITY:
        problems.append(f"density {SENSITIVITY_DENSITY!r}, got {density!r}")
    if seed not in SENSITIVITY_SEEDS:
        problems.append(f"a seed of {SENSITIVITY_SEEDS}, got {seed}")
    if problems:
        raise ValueError(f"sensitivity run {variant!r} must use " + "; ".join(problems))


# Sensitivity grid, fixed before training: one parameter varied at a time, every grid RELATIVE to the base value, so the
# base is in the grid by construction and the grid follows whatever base the owner chooses:
#   "factor"  value x base     lr: x1/3, x1, x3
#   "offset"  base + value     eps_decay_fraction: base -0.2, base, base +0.2;  clip_range: base -0.1, base, base +0.1
# The neutral value (factor 1 / offset 0) is the base and is NOT run again: it is the main-grid run (reward "balanced",
# seeds 0, 1, 2, variant "main"). "bounds" = (low exclusive, high, high inclusive): a resolved value outside is an error
# (never silently clipped), e.g. an eps_decay_fraction base of 0.2 or 0.9 gives an invalid grid.
SENSITIVITY_GRID: dict[str, dict[str, dict[str, Any]]] = {
    "ddqn": {
        "lr": {"mode": "factor", "values": (1 / 3, 1.0, 3.0), "bounds": (0.0, math.inf, True)},
        "eps_decay_fraction": {"mode": "offset", "values": (-0.2, 0.0, 0.2), "bounds": (0.0, 1.0, True)},
    },
    "ppo": {
        "lr": {"mode": "factor", "values": (1 / 3, 1.0, 3.0), "bounds": (0.0, math.inf, True)},
        "clip_range": {"mode": "offset", "values": (-0.1, 0.0, 0.1), "bounds": (0.0, 1.0, False)},
    },
}
_NEUTRAL = {"factor": 1.0, "offset": 0.0}


@dataclass(frozen=True)
class SensitivityRun:
    """One sensitivity run: ``overrides`` hold the resolved value of the varied parameter."""

    algo: str
    param: str
    grid_value: float  # the factor / offset as written in SENSITIVITY_GRID (x1/3, +0.2, -0.1, ...)
    variant: str
    overrides: Mapping[str, Any]
    seed: int

    def cli_args(self) -> list[str]:
        """Arguments of ``python -m <algo>.train`` that start this run (budget and every other value at its default)."""
        args = ["--reward", SENSITIVITY_REWARD, "--density", SENSITIVITY_DENSITY, "--seed", str(self.seed)]
        for key, value in self.overrides.items():
            args += ["--set", f"{key}={_format_value(value)}"]
        return args


def _resolve_grid_value(algo: str, base: float, param: str, spec: Mapping[str, Any], value: float) -> float:
    mode = spec["mode"]
    if mode not in _NEUTRAL:
        raise ValueError(f"unknown sensitivity mode {mode!r} for {algo}.{param}")
    raw = value * base if mode == "factor" else base + value
    resolved = float(f"{raw:.6g}")  # 6 significant digits: the stored value equals the one in the variant name
    low, high, high_inclusive = spec["bounds"]
    if not (low < resolved and (resolved <= high if high_inclusive else resolved < high)):
        raise ValueError(
            f"{algo}.{param}: base {base} {'x' if mode == 'factor' else '+'} {value:g} = {resolved} is outside the valid "
            f"range ({low}, {high}{']' if high_inclusive else ')'}; choose another base or change the grid"
        )
    return resolved


def expand_sensitivity_grid(
    configs: Optional[Mapping[str, Any]] = None,
    seeds: Sequence[int] = SENSITIVITY_SEEDS,
    grid: Mapping[str, Mapping[str, Mapping[str, Any]]] = SENSITIVITY_GRID,
) -> list[SensitivityRun]:
    """The sensitivity runs of ``grid``: every non-base grid value x ``seeds``; the base value is skipped.

    Values are resolved against the base config ``configs[algo]`` (default: the algorithm's default config); a resolved
    value outside its valid range raises ``ValueError``. The absolute values therefore depend on the frozen base values
    (``eps_decay_fraction`` values are fractions of ``total_env_steps``, resolved to env steps at run start). With the
    shared grid this is 4 settings x 3 seeds = 12 runs per algorithm, 24 in total.
    """
    runs: list[SensitivityRun] = []
    for algo in ALGORITHMS:
        cfg = CONFIG_CLASSES[algo]() if configs is None or algo not in configs else configs[algo]
        for param, spec in grid.get(algo, {}).items():
            base = getattr(cfg, param)
            for grid_value in spec["values"]:
                value = _resolve_grid_value(algo, base, param, spec, grid_value)  # validates every grid point
                if math.isclose(grid_value, _NEUTRAL[spec["mode"]], abs_tol=1e-12):
                    continue  # the base point is the main-grid run
                overrides = {param: value}
                for seed in seeds:
                    runs.append(SensitivityRun(algo, param, float(grid_value), variant_name(overrides), overrides, int(seed)))
    return runs


def check_sensitivity_run(sens_meta: Mapping[str, Any], main_meta: Mapping[str, Any]) -> None:
    """Fail if a sensitivity run differs from its main-grid base run in anything but the one varied parameter.

    Both arguments are ``run_meta.json`` contents (``common.run_meta.read_meta``) of the sensitivity run and of the
    main-grid run with the same algorithm, reward and seed. The base point of a sensitivity table is that main run
    (it is not re-run), which is only valid if the hyperparameters that are NOT varied, the budget and the env are
    identical. Also checks the role of the run: reward ``SENSITIVITY_REWARD``, density ``SENSITIVITY_DENSITY``, a seed of
    ``SENSITIVITY_SEEDS``, exactly one overridden key and that key is a parameter of ``SENSITIVITY_GRID``.
    """
    problems: list[str] = []
    s_id, m_id = sens_meta["identity"], main_meta["identity"]
    for key in ("algorithm", "train_reward", "train_density", "seed"):
        if s_id[key] != m_id[key]:
            problems.append(f"identity.{key}: sensitivity {s_id[key]!r} vs main {m_id[key]!r}")
    if m_id["variant"] != MAIN_VARIANT:
        problems.append(f"the reference run must be the main variant, got {m_id['variant']!r}")
    if s_id["train_reward"] != SENSITIVITY_REWARD or s_id["train_density"] != SENSITIVITY_DENSITY:
        problems.append(f"sensitivity runs use reward {SENSITIVITY_REWARD!r} and density {SENSITIVITY_DENSITY!r}")
    if s_id["seed"] not in SENSITIVITY_SEEDS:
        problems.append(f"sensitivity seed must be in {SENSITIVITY_SEEDS}, got {s_id['seed']}")
    varied = list(s_id["overridden_keys"])
    allowed = SENSITIVITY_GRID.get(s_id["algorithm"], {})
    if len(varied) != 1:
        problems.append(f"one parameter at a time: expected exactly one overridden key, got {varied}")
    elif varied[0] not in allowed:
        problems.append(f"{varied[0]!r} is not a sensitivity parameter of {s_id['algorithm']}: {sorted(allowed)}")
    s_hp, m_hp = sens_meta["hyperparameters"], main_meta["hyperparameters"]
    for key in sorted(set(s_hp) | set(m_hp)):
        if key in varied:
            if s_hp.get(key) == m_hp.get(key):
                problems.append(f"{key}: the varied value equals the main-grid value {m_hp.get(key)} (that is the base point)")
        elif s_hp.get(key) != m_hp.get(key):
            problems.append(f"{key}: not varied but differs, sensitivity {s_hp.get(key)} vs main {m_hp.get(key)}")
    if sens_meta["budget"] != main_meta["budget"]:
        problems.append(f"budget differs: sensitivity {sens_meta['budget']} vs main {main_meta['budget']}")
    s_env, m_env = sens_meta.get("env"), main_meta.get("env")
    if s_env and m_env:  # not yet described when a run is checked before it starts
        for key in ("env_id", "config", "reward_weights"):
            if s_env.get(key) != m_env.get(key):
                problems.append(f"env.{key} differs between the sensitivity run and the main run")
    if problems:
        raise ValueError(
            f"sensitivity run {s_id['variant']!r} (seed {s_id['seed']}) does not match its main-grid base run:\n  - "
            + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------------------------
# Supplementary runs: pre-declared, outside the DDQN vs PPO comparison
# ---------------------------------------------------------------------------------------------
# Experiments beyond the main grid (PPO-KL, an A2C-like PPO, later a stronger exploration, or a supplementary reward preset of
# common.config.SUPPLEMENTARY_REWARDS) are declared HERE, before they are trained, like SENSITIVITY_GRID. A run uses the variant
# "supp-<name>" (``--variant supp-<name>``, never generated from --set) and must match its entry exactly (check_run_roles):
# the algorithm, the overrides (exact values, nothing else), one of the rewards and seeds, the training density and the frozen
# budget. They never enter a DDQN vs PPO table (common.paths.load_eval_episodes, common.aggregate) and are reported separately.
# Their base point is the main-grid run with the same algorithm, reward and seed (SUPPLEMENTARY_BASE_REWARD for a supplementary
# reward, which has no main run), compared by check_supplementary_run. Adding an entry is one line in SUPPLEMENTARY_RUNS.
SUPPLEMENTARY_BASE_REWARD = SENSITIVITY_REWARD  # reference main run of a run trained with a SUPPLEMENTARY_REWARDS preset


@dataclass(frozen=True)
class FromBase:
    """An override value taken from the base config: ``FromBase("rollout_size")`` = the base config's ``rollout_size``.

    Keeps an entry valid when the owner changes the defaults after tuning (e.g. ``n_envs`` / ``n_steps``).
    """

    attr: str


@dataclass(frozen=True)
class SupplementarySpec:
    """One entry of ``SUPPLEMENTARY_RUNS``."""

    algo: str  # "ddqn" | "ppo"
    overrides: Mapping[str, Any]  # exact values or FromBase(...); algorithm-specific keys only; {} only for supplementary rewards
    rewards: tuple[str, ...]  # presets of REWARDS and / or SUPPLEMENTARY_REWARDS
    seeds: tuple[int, ...]  # a subset of TRAIN_SEEDS (the main-grid base run exists for the same seed)
    purpose: str  # one line for tables and docs

    def resolved_overrides(self, base: Optional[Any] = None) -> dict[str, Any]:
        """The overrides with every ``FromBase`` resolved against ``base`` (default: the default config of the algorithm)."""
        base = CONFIG_CLASSES[self.algo]() if base is None else base
        return {k: (getattr(base, v.attr) if isinstance(v, FromBase) else v) for k, v in sorted(self.overrides.items())}


# Decided by the team (2026-11-27). PPO only for now; the registry is algorithm-neutral (a DDQN entry has algo "ddqn").
# To add an entry, add one line. The stronger-exploration entry is added after tuning, once its ent_coef is known
# (all 4 REWARDS x seeds 0-4), e.g.
#   "ent-<value>": SupplementarySpec("ppo", {"ent_coef": <value>}, tuple(REWARDS), (0, 1, 2, 3, 4), "stronger exploration"),
SUPPLEMENTARY_RUNS: dict[str, SupplementarySpec] = {
    "ppo-kl": SupplementarySpec("ppo", {"policy_loss": "kl"}, ("balanced",), (0, 1, 2), "PPO-KL (adaptive KL penalty) vs PPO-Clip"),
    "a2c": SupplementarySpec("ppo", {"n_epochs": 1, "minibatch_size": FromBase("rollout_size")}, ("balanced",), (0, 1, 2),
                             "A2C-like: one full-batch gradient step per rollout"),
}


def validate_supplementary_runs(registry: Mapping[str, SupplementarySpec]) -> None:
    """Raise ``ValueError`` if an entry is not a valid, distinct supplementary run (``SUPPLEMENTARY_RUNS`` at import).

    Per entry: a valid name (``supp-<name>`` is a variant), a known algorithm, override keys that are fields of its config
    and not ``SHARED_KEYS``, values that give a valid config and differ from the base values, a rollout that fits the frozen
    budget, rewards of ``REWARDS`` / ``SUPPLEMENTARY_REWARDS``, seeds of ``TRAIN_SEEDS``, a purpose. No overrides only if every
    reward is supplementary (otherwise it is the main run). An entry that equals a sensitivity grid point on the sensitivity
    reward and seeds is refused (that is a sensitivity run under another name).
    """
    for name, spec in registry.items():
        where = f"SUPPLEMENTARY_RUNS[{name!r}]"
        supplementary_variant(name)
        if spec.algo not in CONFIG_CLASSES:
            raise ValueError(f"{where}: unknown algorithm {spec.algo!r}; choose from {list(CONFIG_CLASSES)}")
        cls = CONFIG_CLASSES[spec.algo]
        base = cls()
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(spec.overrides) - names)
        if unknown:
            raise ValueError(f"{where}: {cls.__name__} has no field(s) {unknown}")
        shared = sorted(set(spec.overrides) & SHARED_KEYS)
        if shared:
            raise ValueError(f"{where}: supplementary runs may not vary the shared decisions {shared}")
        for key, value in spec.overrides.items():
            if isinstance(value, FromBase) and not hasattr(base, value.attr):
                raise ValueError(f"{where}: FromBase({value.attr!r}) for {key!r}: {cls.__name__} has no attribute {value.attr!r}")
        resolved = spec.resolved_overrides(base)
        same = sorted(k for k, v in resolved.items() if getattr(base, k) == v)
        if same:
            raise ValueError(f"{where}: {same} equal the base values (no change)")
        try:
            cfg = apply_overrides(base, resolved)
        except ValueError as exc:
            raise ValueError(f"{where}: the overrides give an invalid config: {exc}") from None
        if hasattr(cfg, "rollout_size"):
            check_budget_matches_rollout(TrainBudget(), cfg.rollout_size, what=f"the rollout size of {where}")
        if not spec.rewards or len(set(spec.rewards)) != len(spec.rewards):
            raise ValueError(f"{where}: rewards must be non-empty and unique, got {spec.rewards}")
        bad = [r for r in spec.rewards if r not in all_reward_names()]
        if bad:
            raise ValueError(f"{where}: unknown reward(s) {bad}; choose from {all_reward_names()}")
        if not resolved and not all(is_supplementary_reward(r) for r in spec.rewards):
            raise ValueError(f"{where}: no overrides on a REWARDS preset is the main run; declare overrides or a supplementary reward")
        if not spec.seeds or len(set(spec.seeds)) != len(spec.seeds) or not set(spec.seeds) <= set(TRAIN_SEEDS):
            raise ValueError(f"{where}: seeds must be non-empty, unique and in TRAIN_SEEDS {TRAIN_SEEDS}, got {spec.seeds}")
        if not spec.purpose.strip():
            raise ValueError(f"{where}: purpose must be given")
        if SENSITIVITY_REWARD in spec.rewards and set(spec.seeds) & set(SENSITIVITY_SEEDS):
            for run in expand_sensitivity_grid(seeds=SENSITIVITY_SEEDS[:1]):
                if run.algo == spec.algo and dict(run.overrides) == resolved:
                    raise ValueError(f"{where}: equals the sensitivity point {run.variant!r}; run it as a sensitivity run")


@dataclass(frozen=True)
class SupplementaryRun:
    """One run of a ``SUPPLEMENTARY_RUNS`` entry (one reward, one seed); ``overrides`` are resolved."""

    name: str
    algo: str
    variant: str
    reward: str
    seed: int
    overrides: Mapping[str, Any]

    def cli_args(self) -> list[str]:
        """Arguments of ``python -m <algo>.train`` that start this run (budget at its default)."""
        args = ["--variant", self.variant, "--reward", self.reward, "--density", TRAIN_DENSITY, "--seed", str(self.seed)]
        for key, value in self.overrides.items():
            args += ["--set", f"{key}={_format_value(value)}"]
        return args


def expand_supplementary_runs(
    registry: Optional[Mapping[str, SupplementarySpec]] = None, configs: Optional[Mapping[str, Any]] = None
) -> list[SupplementaryRun]:
    """Every run of ``registry`` (default ``SUPPLEMENTARY_RUNS``): entry x reward x seed, overrides resolved against
    ``configs[algo]`` (default: the default config). With the current registry: 2 entries x 1 reward x 3 seeds = 6 runs."""
    registry = SUPPLEMENTARY_RUNS if registry is None else registry
    runs = []
    for name, spec in registry.items():
        base = None if configs is None or spec.algo not in configs else configs[spec.algo]
        overrides = spec.resolved_overrides(base)
        for reward in spec.rewards:
            for seed in spec.seeds:
                runs.append(SupplementaryRun(name, spec.algo, supplementary_variant(name), reward, int(seed), overrides))
    return runs


def check_supplementary_role(
    variant: str, algo: str, seed: int, reward: str, density: str, overrides: Mapping[str, Any],
    budget_overrides: Sequence[str], registry: Optional[Mapping[str, SupplementarySpec]] = None,
) -> None:
    """A ``supp-<name>`` run must match its registry entry exactly; raises ``ValueError`` listing every mismatch."""
    registry = SUPPLEMENTARY_RUNS if registry is None else registry
    name = variant.removeprefix(SUPPLEMENTARY_PREFIX)
    spec = registry.get(name)
    if spec is None:
        raise ValueError(
            f"supplementary run {variant!r} is not registered: declare it in SUPPLEMENTARY_RUNS (registered: "
            f"{[supplementary_variant(n) for n in registry]})"
        )
    problems = []
    if algo != spec.algo:
        problems.append(f"algorithm {spec.algo!r}, got {algo!r}")
    elif dict(overrides) != spec.resolved_overrides():
        problems.append(f"exactly the overrides {spec.resolved_overrides()}, got {dict(overrides)}")
    if reward not in spec.rewards:
        problems.append(f"a reward of {list(spec.rewards)}, got {reward!r}")
    if seed not in spec.seeds:
        problems.append(f"a seed of {list(spec.seeds)}, got {seed}")
    if density != TRAIN_DENSITY:
        problems.append(f"the training density {TRAIN_DENSITY!r}, got {density!r}")
    if budget_overrides:
        problems.append(f"the frozen budget (no budget overrides), got {sorted(budget_overrides)}")
    if problems:
        raise ValueError(f"supplementary run {variant!r} must use " + "; ".join(problems))


def supplementary_base_reward(reward: str) -> str:
    """Reward of the main-grid run a supplementary run is compared with: its own, or ``SUPPLEMENTARY_BASE_REWARD``."""
    return SUPPLEMENTARY_BASE_REWARD if is_supplementary_reward(reward) else reward


def check_supplementary_run(
    supp_meta: Mapping[str, Any], main_meta: Mapping[str, Any], registry: Optional[Mapping[str, SupplementarySpec]] = None
) -> None:
    """Fail if a supplementary run differs from its main-grid base run in anything but its declared overrides (and reward).

    Both arguments are ``run_meta.json`` contents. The base run is the main run of the same algorithm, density and seed with
    ``supplementary_base_reward(reward)``. Checks the registry entry, that the declared values hold (resolved against the
    MAIN run's hyperparameters, so ``FromBase`` follows the defaults the main grid used), that every other hyperparameter,
    the budget and the env config are identical, and the reward weights unless the run uses a supplementary reward.
    """
    registry = SUPPLEMENTARY_RUNS if registry is None else registry
    s_id, m_id = supp_meta["identity"], main_meta["identity"]
    variant = s_id["variant"]
    spec = registry.get(variant.removeprefix(SUPPLEMENTARY_PREFIX)) if is_supplementary_variant(variant) else None
    if spec is None:
        raise ValueError(f"{variant!r} is not a registered supplementary run")
    problems: list[str] = []
    if m_id["variant"] != MAIN_VARIANT:
        problems.append(f"the reference run must be the main variant, got {m_id['variant']!r}")
    for key in ("algorithm", "train_density", "seed"):
        if s_id[key] != m_id[key]:
            problems.append(f"identity.{key}: supplementary {s_id[key]!r} vs main {m_id[key]!r}")
    if m_id["train_reward"] != supplementary_base_reward(s_id["train_reward"]):
        problems.append(f"the reference run must have the reward {supplementary_base_reward(s_id['train_reward'])!r}, "
                        f"got {m_id['train_reward']!r}")
    if s_id["algorithm"] != spec.algo or s_id["train_reward"] not in spec.rewards or s_id["seed"] not in spec.seeds:
        problems.append(f"algorithm / reward / seed outside the registry entry {spec}")
    s_hp, m_hp = supp_meta["hyperparameters"], main_meta["hyperparameters"]
    declared = spec.resolved_overrides(CONFIG_CLASSES[spec.algo].from_dict(m_hp)) if spec.algo == m_id["algorithm"] else {}
    if sorted(s_id["overridden_keys"]) != sorted(declared):
        problems.append(f"overridden keys {sorted(s_id['overridden_keys'])}, declared {sorted(declared)}")
    for key in sorted(set(s_hp) | set(m_hp)):
        if key in declared:
            if s_hp.get(key) != _to_jsonable(declared[key]):
                problems.append(f"{key}: declared {declared[key]!r}, got {s_hp.get(key)!r}")
        elif s_hp.get(key) != m_hp.get(key):
            problems.append(f"{key}: not declared but differs, supplementary {s_hp.get(key)} vs main {m_hp.get(key)}")
    if supp_meta["budget"] != main_meta["budget"]:
        problems.append(f"budget differs: supplementary {supp_meta['budget']} vs main {main_meta['budget']}")
    s_env, m_env = supp_meta.get("env"), main_meta.get("env")
    if s_env and m_env:  # not yet described when a run is checked before it starts
        keys = ("env_id", "config") if is_supplementary_reward(s_id["train_reward"]) else ("env_id", "config", "reward_weights")
        for key in keys:
            if s_env.get(key) != m_env.get(key):
                problems.append(f"env.{key} differs between the supplementary run and the main run")
    if problems:
        raise ValueError(
            f"supplementary run {variant!r} (seed {s_id['seed']}) does not match its main-grid base run:\n  - "
            + "\n  - ".join(problems)
        )


validate_supplementary_runs(SUPPLEMENTARY_RUNS)
