"""Shared training-run harness: identical plumbing for DDQN and PPO (and the ``RandomTrainer`` test double).

``TrainingRun`` owns everything that is not the algorithm: CLI, seeding, quiet speed benchmark and
runtime estimate, envs, ``TrainLogger`` + ``Timers``, validation and checkpoint schedules,
milestone checkpoints, resume, ``run_meta.json``, final checkpoint and inference benchmark. The
algorithm implements ``Trainer`` (below) and nothing else; it steps the envs through
``ctx.step(...)`` (which counts env steps, times the simulator and logs finished episodes) and
reports updates with ``ctx.log_update(**metrics)``.

    from common.training import Trainer, TrainingRun
    class DDQNTrainer(Trainer): ...                      # collect_and_update, state_dict, ...
    TrainingRun.from_cli("ddqn", DDQNTrainer).run()      # python -m ddqn.train --reward balanced --seed 0

Rules implemented here (CLAUDE.md "Performance rules" and "Experiment tracking"): one env per run
created once (a ``SyncVectorEnv`` / ``AsyncVectorEnv`` for PPO), no rendering, thread limits, quiet
benchmark before the run with a runtime estimate, buffered logging, resume from checkpoints,
deterministic seeding from the run seed only, measured speed in the run metadata.

Schedules. Validation, milestone and resume checkpoints happen at the first chunk boundary at or after
their env step (``Trainer.step_granularity`` env steps per call: 1 for DDQN, ``n_envs * n_steps`` for
PPO); the actual env step is stored. ``total_env_steps`` must be a multiple of the granularity so
both algorithms train for exactly the same number of env steps. A last validation always happens at
the end.

Resume. ``resume.pt`` holds the trainer state, the pickled simulator state of every env (including
the running episode and its recorder), the current observations, python / numpy / torch RNG states,
the timers and the logger position. Rows logged beyond the checkpoint are dropped on resume, so the
resumed run reproduces the uninterrupted run exactly (up to wall-clock columns).
"""

from __future__ import annotations

# Thread limits must be in place before numpy / torch are imported (CLAUDE.md rule 7).
from common.parallel import limit_threads

limit_threads()

import argparse  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from abc import ABC, abstractmethod  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.agent_base import Agent  # noqa: E402
from common.benchmark import measure_env_speed, measure_inference, record_val_observations  # noqa: E402
from common.config import (  # noqa: E402
    BASE_CONFIG,
    DENSITIES,
    ENV_ID,
    REWARDS,
    SENSITIVITY_DENSITY,
    SENSITIVITY_REWARD,
    SENSITIVITY_SEEDS,
    TRAIN_DENSITY,
    TRAIN_SEEDS,
)
from common.evaluate import validate  # noqa: E402
from common.paths import (  # noqa: E402
    MAIN_VARIANT,
    RESULTS_DIR,
    check_variant,
    final_checkpoint,
    milestone_checkpoint,
    resume_checkpoint,
    run_dir,
    run_meta_path,
)
from common.run_meta import describe_env, new_meta, read_meta, reproducibility_info, utc_now, write_meta  # noqa: E402
from common.seeding import derived_env_seeds, set_global_seeds  # noqa: E402
from common.train_config import (  # noqa: E402
    CONFIG_CLASSES,
    SENSITIVITY_GRID,
    TrainBudget,
    apply_overrides,
    check_budget_matches_rollout,
    check_run_roles,
    check_sensitivity_run,
    parse_set_args,
    placeholders_in_use,
    resolve_hyperparameters,
    variant_name,
)
from common.train_env import finished_episodes, make_train_env, make_vector_env  # noqa: E402
from common.train_logging import LOG_FILES, Timers, TrainLogger  # noqa: E402

DEFAULT_UPDATE_LOG_EVERY = {"ddqn": 100, "ppo": 1}  # K gradient-update calls per update_log.csv row
ASSUMED_SPEEDUP_CAP = 4.0  # parallel scaling plateaus at about 3.5-5x here (docs/performance_analysis.md C5)
CHECKPOINT_FORMAT = 1


ONEDRIVE_WARNING = (
    "\n" + "!" * 100 + "\n"
    "WARNING: the project or the output directory is inside a OneDrive folder. During the SB3 calibration, OneDrive\n"
    "locking files while syncing crashed 6 runs with PermissionError, and a resumed run is not bit-identical to an\n"
    "uninterrupted one. The run goes on (run_meta.json: reproducibility.onedrive_path = true), but before long runs\n"
    "move the project out of OneDrive (or pause sync) and disable sleep.\n"
    + "!" * 100 + "\n"
)


class SimulatedInterrupt(Exception):
    """Raised by ``TrainingRun(interrupt_at_env_step=...)`` to test resume (acts like a crash)."""


# ---------------------------------------------------------------------------------------------
# Algorithm interface
# ---------------------------------------------------------------------------------------------
class TrainContext:
    """What the algorithm sees: envs, step counter, logger and timers (created by ``TrainingRun``).

    Attributes:
        env: ``TrainEnvWrapper`` (single env, DDQN) or a gymnasium vector env with SAME_STEP autoreset (PPO).
        obs: current observation(s); managed by ``reset_envs`` / ``step`` / ``reset_episode`` and restored on resume.
        env_step: environment transitions so far, summed over vector envs. The x-axis of every log.
    """

    def __init__(self, algo: str, cfg: Any, budget: TrainBudget, seed: int, reward: str, env: Any,
                 vector: bool, n_envs: int, logger: TrainLogger, timers: Timers) -> None:
        self.algo, self.cfg, self.budget, self.seed, self.reward = algo, cfg, budget, seed, reward
        self.env, self.vector, self.n_envs = env, vector, n_envs
        self.logger, self.timers = logger, timers
        self.obs: Any = None
        self.env_step = 0

    def reset_envs(self) -> Any:
        """First reset of the run with the seeds of ``common.seeding`` (run seed, or derived per-env seeds)."""
        if self.vector:
            self.obs, _ = self.env.reset(seed=derived_env_seeds(self.seed, self.n_envs))
        else:
            self.obs, _ = self.env.reset(seed=self.seed)
        return self.obs

    def reset_episode(self) -> Any:
        """Single env only: start the next episode (unseeded reset). Vector envs autoreset themselves."""
        if self.vector:
            raise RuntimeError("vector envs autoreset (SAME_STEP); do not call reset_episode")
        self.obs, _ = self.env.reset()
        return self.obs

    def step(self, action: Any):
        """``env.step(action)`` with env timing, env-step counting and logging of finished episodes."""
        with self.timers.env():
            obs, reward, terminated, truncated, info = self.env.step(action)
        self.env_step += self.n_envs
        self.obs = obs
        for env_index, record in finished_episodes(info, self.n_envs if self.vector else None):
            self.logger.log_episode(self.env_step, env_index, record)
        return obs, reward, terminated, truncated, info

    def log_update(self, n_updates: int = 1, **metrics: float) -> None:
        """Report gradient update(s); see ``common.train_logging.UPDATE_LOG_COLUMNS`` for the metric names."""
        self.logger.log_update(self.env_step, n_updates=n_updates, **metrics)

    # -- resume -------------------------------------------------------------------------------
    def get_env_states(self) -> list[bytes]:
        return list(self.env.call("get_state")) if self.vector else [self.env.get_state()]

    def set_env_states(self, blobs: list[bytes]) -> None:
        if self.vector:
            self.env.call("set_state", blobs)
        else:
            self.env.set_state(blobs)


class Trainer(ABC):
    """The algorithm-specific part of a training run (to be implemented by DDQN and PPO)."""

    #: env steps per ``collect_and_update`` call (DDQN: 1, PPO: ``n_envs * n_steps``)
    step_granularity: int = 1

    def __init__(self, ctx: TrainContext) -> None:
        self.ctx = ctx

    @classmethod
    def check_available(cls) -> None:
        """Raise ``NotImplementedError`` here if the algorithm is still a stub (checked before anything is written)."""

    @abstractmethod
    def collect_and_update(self, n_env_steps: int) -> None:
        """Advance the run by exactly ``n_env_steps`` env steps (always ``step_granularity``).

        Step the envs with ``self.ctx.step(...)`` inside ``with self.ctx.timers.env()``-free calls (it times
        the simulator itself), do the gradient updates inside ``with self.ctx.timers.update():`` and report
        each with ``self.ctx.log_update(...)`` after that block.
        """

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Everything needed to continue training: networks, optimizers, replay buffer / rollout buffer, RNGs, counters."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Inverse of ``state_dict``."""

    @abstractmethod
    def make_eval_agent(self) -> Agent:
        """A light, picklable ``Agent`` snapshot of the current policy (no replay buffer) for validation / saving."""

    @abstractmethod
    def n_parameters(self) -> int:
        """Number of trainable parameters (stored in run_meta.json)."""


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def build_parser(algo: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Train a {algo.upper()} agent (common.training harness).")
    p.add_argument("--reward", choices=list(REWARDS), default="default")
    p.add_argument("--density", choices=list(DENSITIES), default=TRAIN_DENSITY, help="training density (protocol: medium)")
    p.add_argument("--seed", type=int, default=0, help=f"run seed (main grid: {TRAIN_SEEDS})")
    p.add_argument("--variant", default=None, help="run directory level; default: main, or generated from --set")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override one hyperparameter (sensitivity runs); repeatable, recorded in run_meta.json")
    p.add_argument("--workers", type=int, default=1,
                   help="processes for env simulation (PPO: 1 = in-process vector env, >1 = one subprocess per env; "
                        "validation episodes). Results do not depend on it.")
    p.add_argument("--resume", action="store_true", help="continue from <run dir>/resume.pt")
    p.add_argument("--overwrite", action="store_true", help="delete an existing run directory content and start over")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="results root; the run goes to <out-dir>/runs/...")
    p.add_argument("--update-log-every", type=int, default=DEFAULT_UPDATE_LOG_EVERY[algo],
                   help="K: log_update calls per update_log.csv row")
    p.add_argument("--benchmark-steps", type=int, default=200, help="quiet speed benchmark length (0 = skip, no estimate)")
    p.add_argument("--max-minutes", type=float, default=45.0, help="refuse to start above this estimate")
    p.add_argument("--force", action="store_true", help="start even if the estimate exceeds --max-minutes")
    b = p.add_argument_group("budget overrides (debug / tests only; the main grid uses the TrainBudget defaults)")
    b.add_argument("--total-steps", type=int, default=None)
    b.add_argument("--val-every", type=int, default=None)
    b.add_argument("--val-episodes", type=int, default=None)
    b.add_argument("--checkpoint-every", type=int, default=None)
    return p


@dataclass
class RunSetup:
    """Resolved configuration of one run (everything that identifies it)."""

    algo: str
    reward: str
    density: str
    seed: int
    variant: str
    cfg: Any
    budget: TrainBudget
    overridden_keys: list[str]
    budget_overrides: list[str]
    workers: int
    resume: bool
    overwrite: bool
    root: Path
    update_log_every: int
    benchmark_steps: int
    max_minutes: float
    force: bool

    @property
    def run_dir(self) -> Path:
        return run_dir(self.algo, self.reward, self.seed, self.root, self.variant)

    def identity(self) -> dict[str, Any]:
        return {
            "algorithm": self.algo, "train_reward": self.reward, "train_density": self.density, "seed": self.seed,
            "variant": self.variant, "overridden_keys": self.overridden_keys,
        }


def check_budget_fits_trainer(algo: str, cfg: Any, budget: TrainBudget, debug_budget: bool) -> None:
    """PPO updates in rollouts of ``n_envs * n_steps`` env steps: the budget must fit (``check_budget_matches_rollout``).

    ``total_env_steps`` must always be a multiple, so that both algorithms train for exactly the same number of env steps.
    ``val_every_env_steps`` must be a multiple too for every run that uses the frozen budget; a debug / test budget
    (``--total-steps`` ... overrides, never used for results) may validate at the next rollout boundary instead.
    """
    if algo != "ppo":
        return
    if not debug_budget:
        check_budget_matches_rollout(budget, cfg.rollout_size)
    elif budget.total_env_steps % cfg.rollout_size:
        raise ValueError(
            f"total_env_steps {budget.total_env_steps} must be a multiple of the rollout size {cfg.rollout_size} "
            f"(n_envs * n_steps) so that both algorithms train for exactly the same number of env steps"
        )


def resolve_setup(algo: str, argv: Optional[Sequence[str]] = None) -> RunSetup:
    """Parse the CLI and build the validated config / budget (raises on invalid values)."""
    args = build_parser(algo).parse_args(argv)
    cls = CONFIG_CLASSES[algo]
    overrides = parse_set_args(args.overrides, cls)
    cfg = apply_overrides(cls(), overrides)
    budget_kw = {
        k: v for k, v in {
            "total_env_steps": args.total_steps, "val_every_env_steps": args.val_every,
            "val_n_episodes": args.val_episodes, "checkpoint_every_env_steps": args.checkpoint_every,
        }.items() if v is not None
    }
    budget = TrainBudget(**budget_kw)
    if args.variant is not None:
        variant = check_variant(args.variant)
        if variant == MAIN_VARIANT and overrides:
            raise ValueError("--variant main with --set overrides would overwrite the main run; pick another name")
    else:
        variant = variant_name(overrides)
    check_budget_fits_trainer(algo, cfg, budget, debug_budget=bool(budget_kw))
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    # tuning runs (tune-<name>) and the reserved run seed 99 stay apart from every result run
    check_run_roles(variant, args.seed, args.reward, args.density, sorted(overrides), sorted(budget_kw))
    return RunSetup(
        algo=algo, reward=args.reward, density=args.density, seed=args.seed, variant=variant, cfg=cfg, budget=budget,
        overridden_keys=sorted(overrides), budget_overrides=sorted(budget_kw), workers=args.workers,
        resume=args.resume, overwrite=args.overwrite, root=args.out_dir, update_log_every=args.update_log_every,
        benchmark_steps=args.benchmark_steps, max_minutes=args.max_minutes, force=args.force,
    )


def estimate_runtime(setup: RunSetup, env_steps_per_s: float) -> dict[str, float]:
    """Simulation-only runtime estimate in minutes (network updates are NOT included)."""
    n_envs = getattr(setup.cfg, "n_envs", 1)
    vector = setup.algo == "ppo"
    env_speedup = min(n_envs, ASSUMED_SPEEDUP_CAP) if vector and setup.workers > 1 else 1.0
    val_speedup = min(float(setup.workers), ASSUMED_SPEEDUP_CAP)
    train_steps = setup.budget.total_env_steps
    val_steps = setup.budget.n_val_points * setup.budget.val_n_episodes * int(BASE_CONFIG["duration"])
    train_min = train_steps / (env_steps_per_s * env_speedup) / 60
    val_min = val_steps / (env_steps_per_s * val_speedup) / 60
    return {"train_env_steps": train_steps, "val_env_steps": val_steps, "train_min": train_min,
            "val_min": val_min, "total_min": train_min + val_min}


# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------
def _get_rng_states() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def _set_rng_states(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def _atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class _ProgressPrinter:
    """Prints env steps, speed and ETA at most every ``interval`` seconds."""

    def __init__(self, tag: str, total: int, start: int, interval: float = 10.0) -> None:
        self.tag, self.total, self.start, self.interval = tag, total, start, interval
        self.t0 = self.last = time.perf_counter()

    def update(self, env_step: int) -> None:
        now = time.perf_counter()
        if now - self.last < self.interval:
            return
        self.last = now
        done = env_step - self.start
        speed = done / (now - self.t0)
        eta = (self.total - env_step) / speed if speed > 0 else math.nan
        print(f"[{self.tag}] {env_step}/{self.total} env steps ({100 * env_step / self.total:.0f}%), "
              f"{speed:.1f} steps/s, ETA {eta / 60:.1f} min", file=sys.stderr, flush=True)


class TrainingRun:
    """One training run of one algorithm; see the module docstring.

    Args:
        setup: resolved CLI (``resolve_setup``) or built directly in tests.
        trainer_cls: the ``Trainer`` subclass.
        interrupt_at_env_step: test hook; raise ``SimulatedInterrupt`` at the first chunk boundary at or
            after this env step (after any checkpoint of that boundary).
    """

    def __init__(self, setup: RunSetup, trainer_cls: type[Trainer], interrupt_at_env_step: Optional[int] = None,
                 quiet: bool = False) -> None:
        self.setup, self.trainer_cls = setup, trainer_cls
        self.interrupt_at = interrupt_at_env_step
        self.quiet = quiet

    @classmethod
    def from_cli(cls, algo: str, trainer_cls: type[Trainer], argv: Optional[Sequence[str]] = None, **kwargs: Any):
        return cls(resolve_setup(algo, argv), trainer_cls, **kwargs)

    def _say(self, msg: str) -> None:
        if not self.quiet:
            print(msg, flush=True)

    # -- main entry ----------------------------------------------------------------------------------
    def run(self) -> Path:
        """Run (or resume) the training; returns the run directory. Writes run_meta.json in every case."""
        s = self.setup
        self.trainer_cls.check_available()  # stubs raise NotImplementedError before anything is written
        check_budget_fits_trainer(s.algo, s.cfg, s.budget, debug_budget=bool(s.budget_overrides))
        limit_threads()
        torch.set_num_threads(1)
        rdir = s.run_dir
        meta_path = run_meta_path(s.algo, s.reward, s.seed, s.root, s.variant)
        ckpt_path = resume_checkpoint(s.algo, s.reward, s.seed, s.root, s.variant)
        resuming = self._prepare_run_dir(rdir, ckpt_path, meta_path)
        if resuming:
            meta = read_meta(meta_path)
            if meta["status"] == "finished":
                self._say(f"run already finished: {rdir}")
                return rdir
            meta["status"], meta["error"] = "running", None
            meta["timing"]["end_time"] = None
        else:
            meta = new_meta(s.identity())
        meta["hyperparameters"] = s.cfg.to_dict()
        meta["budget"] = s.budget.to_dict()
        meta["budget_overrides"] = s.budget_overrides
        meta["resolved"] = resolve_hyperparameters(s.cfg, s.budget)
        meta["placeholders_in_use"] = placeholders_in_use(
            s.cfg, s.budget, overridden=[*s.overridden_keys, *s.budget_overrides]
        )
        meta["logging"]["update_log_every"] = s.update_log_every
        meta["reproducibility"] = reproducibility_info(s.cfg.device, s.workers, s.root)
        if meta["reproducibility"]["onedrive_path"]:
            print(ONEDRIVE_WARNING, file=sys.stderr, flush=True)
        if meta["placeholders_in_use"]:
            print("WARNING: undecided PLACEHOLDER values in use: " + ", ".join(meta["placeholders_in_use"]),
                  file=sys.stderr, flush=True)

        self._check_against_main_run(meta)  # sensitivity runs must differ from the main run only in the varied key
        self._benchmark_and_estimate(meta, resuming)  # may exit; nothing is written to disk before this point
        timers = Timers()
        logger: Optional[TrainLogger] = None
        self._ctx = self._env = None
        try:
            env, logger, ctx, trainer, sched = self._setup_run(meta, timers, resuming, ckpt_path)
            write_meta(meta_path, meta)
            self._train(ctx, trainer, sched, meta, meta_path, ckpt_path)
            meta["status"] = "finished"
        except BaseException as exc:  # noqa: BLE001  (record the failure, then re-raise)
            meta["status"] = "failed"
            meta["error"] = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, (KeyboardInterrupt, SimulatedInterrupt)):
                meta["error"] += "\n" + traceback.format_exc()
            raise
        finally:
            if logger is not None:
                logger.close(emit_partial=meta["status"] == "finished")
                self._fill_timing(meta, timers, logger, self._ctx)
            meta["timing"]["end_time"] = utc_now()
            write_meta(meta_path, meta)
            if self._env is not None:
                self._env.close()
        self._say(f"finished {rdir}: {meta['timing']['final_env_step']} env steps, "
                  f"{meta['timing']['total_wall_s']:.0f} s wall, {meta['timing']['train_steps_per_s']:.1f} steps/s "
                  f"(without validation)")
        return rdir

    # -- setup -----------------------------------------------------------------------------------------
    def _prepare_run_dir(self, rdir: Path, ckpt_path: Path, meta_path: Path) -> bool:
        """Decide fresh start vs resume; protect existing results. Returns True when resuming."""
        s = self.setup
        existing = [rdir / f for f in (*LOG_FILES.values(), "run_meta.json", "resume.pt", "final.pt")]
        has_files = any(p.exists() for p in existing)
        if s.overwrite and has_files:
            for p in [*existing, *rdir.glob("milestone_*pct.pt")]:
                if p.exists():
                    p.unlink()
            has_files = False
        if s.resume and ckpt_path.exists():
            return True
        if has_files:
            raise FileExistsError(
                f"{rdir} already holds results. Use --resume to continue from resume.pt, or --overwrite to start over."
            )
        return False

    def _check_against_main_run(self, meta: dict) -> None:
        """Sensitivity run (one grid parameter overridden, reward / seed of the sensitivity protocol) whose main-grid
        base run already exists: fail before training if anything but the varied parameter differs. Without a main run
        yet the check is repeated when the sensitivity table is built (``common.aggregate.check_sensitivity_runs``)."""
        s = self.setup
        in_role = (s.variant != MAIN_VARIANT and s.reward == SENSITIVITY_REWARD and s.density == SENSITIVITY_DENSITY
                   and s.seed in SENSITIVITY_SEEDS and len(s.overridden_keys) == 1
                   and s.overridden_keys[0] in SENSITIVITY_GRID.get(s.algo, {}))
        main_path = run_meta_path(s.algo, s.reward, s.seed, s.root, MAIN_VARIANT)
        if in_role and main_path.exists():
            check_sensitivity_run(meta, read_meta(main_path))

    def _benchmark_and_estimate(self, meta: dict, resuming: bool) -> None:
        """Quiet speed benchmark (before any env of this run exists), then the runtime estimate (rule 1)."""
        s, budget = self.setup, self.setup.budget
        if resuming:
            speed = meta["timing"]["quiet_benchmark_steps_per_s"]
        elif s.benchmark_steps > 0:
            speed = measure_env_speed(partial(make_train_env, s.density, s.reward), s.benchmark_steps)
            meta["timing"]["quiet_benchmark_steps_per_s"] = speed
        else:
            speed = None
        if not speed:
            self._say("no speed benchmark (--benchmark-steps 0): no runtime estimate")
            return
        est = estimate_runtime(s, speed)
        self._say(
            f"quiet benchmark: {speed:.1f} env steps/s (one process). Plan: {est['train_env_steps']} training env "
            f"steps (~{est['train_min']:.1f} min) + {budget.n_val_points} validations x {budget.val_n_episodes} "
            f"episodes = {est['val_env_steps']} steps (~{est['val_min']:.1f} min); simulation-only estimate "
            f"{est['total_min']:.1f} min, network updates not included."
        )
        if not resuming and est["total_min"] > s.max_minutes and not s.force:
            raise SystemExit(
                f"Estimate {est['total_min']:.0f} min exceeds --max-minutes {s.max_minutes:.0f}; "
                f"reduce the plan or pass --force."
            )

    def _setup_run(self, meta: dict, timers: Timers, resuming: bool, ckpt_path: Path):
        s = self.setup
        cfg, budget = s.cfg, s.budget
        vector = s.algo == "ppo"
        n_envs = cfg.n_envs if vector else 1

        # seeds, envs, logger, context, trainer
        set_global_seeds(s.seed)
        env = (make_vector_env(n_envs, s.workers, s.density, s.reward) if vector
               else make_train_env(s.density, s.reward))
        self._env = env
        logger = TrainLogger(s.run_dir, s.algo, timers, update_log_every=s.update_log_every)
        ctx = TrainContext(s.algo, cfg, budget, s.seed, s.reward, env, vector, n_envs, logger, timers)
        self._ctx = ctx
        if not vector:
            meta["env"] = describe_env(env, ENV_ID, s.reward)
        elif s.workers <= 1:
            meta["env"] = describe_env(env.envs[0], ENV_ID, s.reward)
        else:  # the envs live in subprocesses: describe a throw-away instance of the same config
            probe = make_train_env(s.density, s.reward)
            meta["env"] = describe_env(probe, ENV_ID, s.reward)
            probe.close()
        trainer = self.trainer_cls(ctx)
        if trainer.step_granularity <= 0 or budget.total_env_steps % trainer.step_granularity:
            raise ValueError(f"total_env_steps must be a multiple of the trainer's step_granularity {trainer.step_granularity}")
        meta["model"]["n_parameters"] = int(trainer.n_parameters())

        sched = {"next_val": budget.val_every_env_steps, "next_ckpt": budget.checkpoint_every_env_steps,
                 "milestones_done": [], "last_val_step": -1}
        if resuming:
            self._load_checkpoint(ckpt_path, ctx, trainer, logger, timers, sched, meta)
        else:
            ctx.reset_envs()
        return env, logger, ctx, trainer, sched

    # -- loop --------------------------------------------------------------------------------------------
    def _train(self, ctx: TrainContext, trainer: Trainer, sched: dict, meta: dict, meta_path: Path, ckpt_path: Path) -> None:
        s, budget = self.setup, self.setup.budget
        total, g = budget.total_env_steps, trainer.step_granularity
        tag = f"{s.algo}/{s.reward}/{s.variant}/seed{s.seed}"
        progress = _ProgressPrinter(tag, total, ctx.env_step)
        while ctx.env_step < total:
            before = ctx.env_step
            n = min(g, total - before)
            trainer.collect_and_update(n)
            if ctx.env_step - before != n:
                raise RuntimeError(f"collect_and_update({n}) advanced the run by {ctx.env_step - before} env steps")
            self._boundary(ctx, trainer, sched, meta, meta_path, ckpt_path)
            ctx.logger.maybe_flush()
            if not self.quiet:
                progress.update(ctx.env_step)
            if self.interrupt_at is not None and ctx.env_step >= self.interrupt_at:
                raise SimulatedInterrupt(f"simulated interruption at env step {ctx.env_step}")

        # end of the budget: last validation, final checkpoint, inference benchmark
        if sched["last_val_step"] != ctx.env_step:
            self._validate(ctx, trainer, sched)
        agent = trainer.make_eval_agent()
        with ctx.timers.checkpoint():
            self._save_agent(agent, final_checkpoint(s.algo, s.reward, s.seed, s.root, s.variant))
        self._write_resume(ctx, trainer, sched, ckpt_path)
        t0 = time.perf_counter()  # post-training measurements are not training cost: excluded from steps/s
        obs_batch = record_val_observations(agent, n_episodes=min(5, budget.val_n_episodes))
        n_calls = 1000 if s.benchmark_steps > 0 else 50  # benchmark disabled (tests): keep it short
        meta["inference"] = measure_inference(agent, obs_batch, n=n_calls, warmup=min(100, n_calls))
        meta["timing"]["post_training_benchmark_s"] = time.perf_counter() - t0

    def _boundary(self, ctx: TrainContext, trainer: Trainer, sched: dict, meta: dict, meta_path: Path, ckpt_path: Path) -> None:
        """Validation, milestone and resume checkpoints due at the current env step."""
        s, budget = self.setup, self.setup.budget
        step = ctx.env_step
        if step >= sched["next_val"]:
            self._validate(ctx, trainer, sched)
            sched["next_val"] = (step // budget.val_every_env_steps + 1) * budget.val_every_env_steps
        for fraction in budget.milestone_fractions:
            target = math.ceil(fraction * budget.total_env_steps)
            if step >= target and fraction not in sched["milestones_done"]:
                path = milestone_checkpoint(s.algo, s.reward, s.seed, fraction, s.root, s.variant)
                with ctx.timers.checkpoint():
                    self._save_agent(trainer.make_eval_agent(), path)
                sched["milestones_done"].append(fraction)
                meta["milestones"].append({"fraction": fraction, "env_step": step, "path": path.name})
        if step >= sched["next_ckpt"]:
            self._write_resume(ctx, trainer, sched, ckpt_path)
            sched["next_ckpt"] = (step // budget.checkpoint_every_env_steps + 1) * budget.checkpoint_every_env_steps
            self._fill_timing(meta, ctx.timers, ctx.logger, ctx)
            write_meta(meta_path, meta)

    def _validate(self, ctx: TrainContext, trainer: Trainer, sched: dict) -> None:
        s = self.setup
        agent = trainer.make_eval_agent()
        with ctx.timers.val():
            metrics = validate(agent, n=s.budget.val_n_episodes, reward=s.reward, workers=s.workers)
        ctx.logger.log_val(ctx.env_step, metrics)
        sched["last_val_step"] = ctx.env_step

    @staticmethod
    def _save_agent(agent: Agent, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        agent.save(str(tmp))
        if not tmp.exists():
            raise RuntimeError(f"{type(agent).__name__}.save wrote nothing to {tmp}")
        os.replace(tmp, path)

    # -- checkpoint / resume ---------------------------------------------------------------------------------
    def _identity_blob(self) -> dict[str, Any]:
        s = self.setup
        return {"identity": s.identity(), "cfg": s.cfg.to_dict(), "budget": s.budget.to_dict(),
                "update_log_every": s.update_log_every}

    def _write_resume(self, ctx: TrainContext, trainer: Trainer, sched: dict, ckpt_path: Path) -> None:
        with ctx.timers.checkpoint():
            ctx.logger.flush()  # every row up to this env step is on disk before the checkpoint exists
            state = {
                "format": CHECKPOINT_FORMAT,
                "env_step": ctx.env_step,
                "run": self._identity_blob(),
                "trainer": trainer.state_dict(),
                "env_states": ctx.get_env_states(),
                "obs": ctx.obs,
                "rng": _get_rng_states(),
                "timers": ctx.timers.state_dict(),
                "logger": ctx.logger.state_dict(),
                "sched": dict(sched, milestones_done=list(sched["milestones_done"])),
            }
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_torch_save(state, ckpt_path)

    def _load_checkpoint(self, ckpt_path: Path, ctx: TrainContext, trainer: Trainer, logger: TrainLogger,
                         timers: Timers, sched: dict, meta: dict) -> None:
        state = torch.load(ckpt_path, weights_only=False)  # our own file: contains pickled env states and RNG tuples
        if state["format"] != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint format {state['format']}")
        if state["run"] != self._identity_blob():
            raise ValueError("the checkpoint was written by a run with a different identity / config / budget; "
                             "resume needs identical settings (workers may differ)")
        ctx.env_step = int(state["env_step"])
        timers.load_state_dict(state["timers"])
        logger.load_state_dict(state["logger"])
        dropped = logger.truncate_to(ctx.env_step)
        ctx.set_env_states(state["env_states"])
        ctx.obs = state["obs"]
        trainer.load_state_dict(state["trainer"])
        _set_rng_states(state["rng"])
        sched.update(state["sched"])
        # milestones reached after the checkpoint are written again by the resumed run
        meta["milestones"] = [m for m in meta["milestones"] if m["env_step"] <= ctx.env_step]
        meta["resumes"].append({"env_step": ctx.env_step, "time": utc_now(), "rows_dropped": dropped})
        self._say(f"resumed at env step {ctx.env_step} (dropped rows beyond it: {dropped})")

    # -- metadata -------------------------------------------------------------------------------------------------
    @staticmethod
    def _fill_timing(meta: dict, timers: Timers, logger: TrainLogger, ctx: Optional[TrainContext]) -> None:
        t = meta["timing"]
        t["total_wall_s"] = timers.wall()
        t["env_time_s"], t["update_time_s"] = timers.seconds["env"], timers.seconds["update"]
        t["val_time_s"], t["checkpoint_time_s"] = timers.seconds["val"], timers.seconds["checkpoint"]
        t["final_env_step"] = ctx.env_step if ctx is not None else t["final_env_step"]
        train_wall = timers.train_wall() - (t["post_training_benchmark_s"] or 0.0)
        t["train_steps_per_s"] = t["final_env_step"] / max(train_wall, 1e-9)
        t["n_episodes"], t["n_updates"], t["n_val_points"] = logger.n_episodes, logger.update_idx, logger.val_idx
