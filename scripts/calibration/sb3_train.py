"""One SB3 calibration training run (DQN or PPO) with the main-grid logging schemas, checkpoints and resume.

Imports torch and stable-baselines3; the env subprocesses never import this module (see ``sb3_cal.py``).

What a run writes (``results/calibration/sb3/<algo>/<reward>/seed<s>/``): ``train_log.csv`` (one row per training episode),
``update_log.csv`` (SB3's own loss values, K-call windows as in the main grid), ``val_log.csv`` (EXACTLY the main-grid
schema, written by ``common.train_logging.TrainLogger``), ``val_episodes.csv`` (the per-episode validation records),
``run_meta.json``, ``progress.json``, ``final.zip``, and ``ckpt/`` (resume state, replaced every ~10 min of wall time).

Seeding follows ``common/seeding.py``: ``set_global_seeds(seed)`` first (network initialisation), env ``i`` of the
training env gets the single initial reset seed ``vector_env_seed(seed, i)`` (DQN: ``seed`` itself), later resets are
unseeded. A resume cannot restore the simulator, so the in-flight episodes restart with the seeds
``vector_env_seed(seed, 100 + 10 * k + i)`` of resume number ``k`` and the RNGs are re-seeded from
``vector_env_seed(seed, 900 + k)``; a resumed run is therefore NOT bit-identical to an uninterrupted one (it is only
a calibration run; the main harness does restore everything).
"""

from __future__ import annotations

from scripts.calibration import sb3_cal  # light; sets nothing global

from common.parallel import limit_threads  # noqa: E402

limit_threads()

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import DQN, PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from common.config import BASE_CONFIG, ENV_ID, REWARDS, TRAIN_DENSITY, VAL_SEEDS  # noqa: E402
from common.run_meta import describe_env, new_meta, read_meta, reproducibility_info, utc_now  # noqa: E402
from common.run_meta import write_meta as _write_meta  # noqa: E402
from common.seeding import set_global_seeds, vector_env_seed  # noqa: E402
from common.train_logging import Timers, TrainLogger  # noqa: E402
from scripts.calibration.sb3_cal import (  # noqa: E402
    CHECKPOINT_EVERY_SECONDS,
    PPO_N_ENVS,
    PPO_N_STEPS,
    VAL_EVERY_ENV_STEPS,
    VAL_N_EPISODES,
    VAL_POOL_WORKERS,
    Validator,
    atomic_write_json,
    retry_on_permission_error,
    check_same_actions,
    env_fns,
    hyperparameters,
    light_agent_from_model,
    run_dir,
    total_steps,
)

class SimulatedInterrupt(Exception):
    """Raised by ``CalibrationRun(interrupt_at_env_step=...)`` to test resume (acts like a crash)."""


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    """``common.run_meta.write_meta`` that survives a momentary PermissionError (OneDrive / orchestrator reading the file)."""
    retry_on_permission_error(lambda: _write_meta(path, meta))


LOG_UPDATE_EVERY = {"dqn": 100, "ppo": 1}  # K update calls per update_log.csv row (as the main grid: DDQN 100, PPO 1)
PROGRESS_EVERY_SECONDS = 10.0
EARLY_WARNING_STEP = 100_000
RANDOM_REFERENCE = "random_val_main.json"  # written by run_plan.py (random baseline on VAL_SEEDS, main condition)


# ---------------------------------------------------------------------------------------------
# SB3 subclasses: time the update phase and log SB3's loss values; the hook object is not saved
# ---------------------------------------------------------------------------------------------
class _Hooks:
    """Timers + logger of the running run; attached to the model as ``_cal`` (excluded from ``model.save``)."""

    def __init__(self, timers: Timers, logger: TrainLogger) -> None:
        self.timers, self.logger = timers, logger


class _CalMixin:
    _cal: Optional[_Hooks] = None

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "_cal"]  # type: ignore[misc]

    def train(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if self._cal is None:
            return super().train(*args, **kwargs)  # type: ignore[misc]
        with self._cal.timers.update():
            super().train(*args, **kwargs)  # type: ignore[misc]
        self._report_update(self._cal)

    def _report_update(self, cal: _Hooks) -> None:
        raise NotImplementedError


class CalDQN(_CalMixin, DQN):
    def _report_update(self, cal: _Hooks) -> None:
        v = self.logger.name_to_value
        loss = float(v.get("train/loss", math.nan))
        cal.logger.log_update(
            self.num_timesteps, lr=float(self.lr_schedule(self._current_progress_remaining)), loss=loss, td_loss=loss,
            epsilon=float(self.exploration_rate), buffer_size=float(self.replay_buffer.size()),
        )


class CalPPO(_CalMixin, PPO):
    def _report_update(self, cal: _Hooks) -> None:
        v = self.logger.name_to_value
        g = lambda k: float(v.get(k, math.nan))  # noqa: E731
        cal.logger.log_update(
            self.num_timesteps, lr=g("train/learning_rate"), loss=g("train/loss"), policy_loss=g("train/policy_gradient_loss"),
            value_loss=g("train/value_loss"), entropy=-g("train/entropy_loss"), approx_kl=g("train/approx_kl"),
            clip_fraction=g("train/clip_fraction"), explained_variance=g("train/explained_variance"),
            n_epochs_done=float(self.n_epochs),
        )


# ---------------------------------------------------------------------------------------------
# Model / env construction
# ---------------------------------------------------------------------------------------------
def make_vec_env(algo: str, reward: str, seed: int, resume_k: int = 0):
    """Training env (created once), with the initial reset seeds of ``common.seeding`` (see the module docstring)."""
    n = PPO_N_ENVS if algo == "ppo" else 1
    venv = SubprocVecEnv(env_fns(reward, n), start_method="spawn") if algo == "ppo" else DummyVecEnv(env_fns(reward, 1))
    if resume_k == 0:
        venv._seeds = [vector_env_seed(seed, i) for i in range(n)] if algo == "ppo" else [seed]
    else:
        venv._seeds = [vector_env_seed(seed, 100 + 10 * resume_k + i) for i in range(n)]
    return venv


def sb3_kwargs(algo: str) -> dict[str, Any]:
    hp = {k: v for k, (v, _) in hyperparameters(algo).items()}
    hp.pop("activation_fn")
    net_arch = hp.pop("net_arch")
    hp["policy_kwargs"] = dict(net_arch=net_arch, activation_fn=torch.nn.ReLU)
    if algo == "ppo":
        hp.pop("n_envs")
    return hp


def build_model(algo: str, venv: Any):
    cls = CalDQN if algo == "dqn" else CalPPO
    return cls("MlpPolicy", venv, device="cpu", verbose=0, seed=None, **sb3_kwargs(algo))


def n_parameters(model: Any, algo: str) -> int:
    nets = [model.policy.q_net] if algo == "dqn" else [model.policy]
    return int(sum(p.numel() for n in nets for p in n.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------------------------
# Checkpoints (model, replay buffer for DQN, run state); latest.json is written last and names the consistent set
# ---------------------------------------------------------------------------------------------
class CheckpointStore:
    def __init__(self, rdir: Path, algo: str) -> None:
        self.dir, self.algo = rdir / "ckpt", algo

    def latest(self) -> Optional[dict[str, Any]]:
        path = self.dir / "latest.json"
        if not path.exists():
            return None
        info = json.loads(path.read_text(encoding="utf-8"))
        files = [self.dir / info["model"], self.dir / info["state"]] + ([self.dir / info["buffer"]] if info.get("buffer") else [])
        return info if all(f.exists() for f in files) else None

    def write(self, model: Any, step: int, state: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        info = {"step": step, "model": f"model_{step}.zip", "state": f"state_{step}.json", "buffer": None}
        model.save(self.dir / info["model"])
        if self.algo == "dqn":
            info["buffer"] = f"buffer_{step}.pkl"
            model.save_replay_buffer(self.dir / info["buffer"])
        atomic_write_json(self.dir / info["state"], state)
        atomic_write_json(self.dir / "latest.json", info)  # the consistent set exists only once this file points to it
        keep = {info["model"], info["state"], info["buffer"], "latest.json"}
        for f in self.dir.iterdir():
            if f.name not in keep:
                f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------
# Callback: logging, validation, checkpoints, progress
# ---------------------------------------------------------------------------------------------
class CalibrationCallback(BaseCallback):
    """Everything around the SB3 loop. Validation and checkpoints happen at the first step (DQN: every env step) or
    rollout boundary (PPO: after the update, like the main harness) at or after their env step."""

    def __init__(self, run: "CalibrationRun") -> None:
        super().__init__()
        self.run = run

    def _on_step(self) -> bool:
        r = self.run
        for i, info in enumerate(self.locals["infos"]):
            record = info.get("episode_metrics")
            if record is not None:
                r.logger.log_episode(self.num_timesteps, i, record)
        if r.algo == "dqn":
            r.boundary(self.num_timesteps, self.model)
        return True

    def _on_rollout_start(self) -> None:
        if self.run.algo == "ppo":
            self.run.boundary(self.num_timesteps, self.model)


class CalibrationRun:
    """Trains one (algorithm, reward, seed) run; ``run()`` returns the run directory."""

    def __init__(self, algo: str, reward: str, seed: int, resume: bool = False, overwrite: bool = False,
                 root: Path = sb3_cal.CAL_ROOT, total_env_steps: Optional[int] = None, val_every: int = VAL_EVERY_ENV_STEPS,
                 val_n: int = VAL_N_EPISODES, checkpoint_seconds: float = CHECKPOINT_EVERY_SECONDS,
                 val_workers: Optional[int] = None, quiet: bool = False,
                 interrupt_at_env_step: Optional[int] = None) -> None:
        self.algo, self.reward, self.seed = algo, reward, seed
        self.interrupt_at = interrupt_at_env_step  # test hook: raise SimulatedInterrupt at the first boundary at or after it
        self.root, self.resume, self.overwrite, self.quiet = root, resume, overwrite, quiet
        self.rdir = run_dir(algo, reward, seed, root)
        self.total = total_env_steps if total_env_steps is not None else total_steps(algo)
        self.val_every, self.val_n, self.checkpoint_seconds = val_every, val_n, checkpoint_seconds
        self.val_workers = (VAL_POOL_WORKERS if algo == "ppo" else 1) if val_workers is None else val_workers
        self.store = CheckpointStore(self.rdir, algo)
        self.timers = Timers()
        self.logger: TrainLogger = None  # type: ignore[assignment]
        self.validator: Optional[Validator] = None
        self.meta: dict[str, Any] = {}
        self.resumes_done = 0
        self.next_val = val_every
        self.last_val_step = -1
        self.warned = False
        self._last_ckpt = time.monotonic()
        self._last_progress = 0.0
        self._t_start = time.monotonic()
        self._step_at_start = 0

    # -- helpers --------------------------------------------------------------------------------------
    def _say(self, msg: str) -> None:
        if not self.quiet:
            print(msg, file=sys.stderr, flush=True)

    @property
    def meta_path(self) -> Path:
        return self.rdir / "run_meta.json"

    @property
    def val_episodes_path(self) -> Path:
        return self.rdir / "val_episodes.csv"

    def _random_reference(self) -> Optional[dict[str, float]]:
        path = self.root / "eval" / "baselines" / RANDOM_REFERENCE
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get(self.reward)

    # -- boundary: called every env step (DQN) / rollout start (PPO) -----------------------------------------
    def boundary(self, step: int, model: Any) -> None:
        if step >= self.next_val and step != self.last_val_step:
            self.validate(step, model)
            self.next_val = (step // self.val_every + 1) * self.val_every
        now = time.monotonic()
        if now - self._last_ckpt >= self.checkpoint_seconds:
            self.checkpoint(step, model)
        if now - self._last_progress >= PROGRESS_EVERY_SECONDS:
            self.write_progress(step)
        if self.algo == "dqn" and step % 100 == 0 or self.algo == "ppo":
            self.logger.maybe_flush()
        if self.interrupt_at is not None and step >= self.interrupt_at:
            raise SimulatedInterrupt(f"simulated interruption at env step {step}")

    def validate(self, step: int, model: Any) -> None:
        light = light_agent_from_model(model, self.algo)
        if self.logger.val_idx == 0:  # first validation: the light copy must act exactly like model.predict
            obs = np.random.default_rng(0).random((200, 5, 5)).astype(np.float32)
            bad = check_same_actions(model, light, obs)
            if bad:
                raise RuntimeError(f"light agent and model.predict disagree on {bad}/200 random observations")
        with self.timers.val():
            summary, df = self.validator.run(light)  # type: ignore[union-attr]
        sec = self.timers.seconds  # env column of val_log.csv: everything that is not update / validation / checkpoint
        sec["env"] = max(self.timers.wall() - sec["update"] - sec["val"] - sec["checkpoint"], 0.0)
        self.logger.log_val(step, summary)
        self.logger.flush()  # val_log.csv is current on disk (progress.json and the orchestrator read it)
        df.insert(0, "env_step", step)
        df.insert(0, "val_idx", self.logger.val_idx - 1)
        header = not self.val_episodes_path.exists()
        df.to_csv(self.val_episodes_path, mode="a", header=header, index=False)
        self.last_val_step = step
        if step >= EARLY_WARNING_STEP and not self.warned:
            self.early_warning()

    def early_warning(self) -> None:
        """At the first validation at or after 100k steps: still at the random-baseline level? Log prominently, never stop."""
        ref = self._random_reference()
        self.warned = True
        if ref is None:
            return
        self.logger.flush()
        val = pd.read_csv(self.rdir / "val_log.csv").tail(3)
        crash, ret = float(val["crash_rate"].mean()), float(val[f"return_{self.reward}"].mean())
        near_random = crash >= ref["crash_rate"] - 0.10 and ret <= ref["return"] + 2.0 * ref["return_se"]
        self.meta.setdefault("early_warning", {}).update(
            {"checked_at_env_step": int(val["env_step"].iloc[-1]), "crash_rate": crash, "return_own": ret,
             "random_crash_rate": ref["crash_rate"], "random_return_own": ref["return"], "random_return_se": ref["return_se"],
             "at_random_level": bool(near_random)}
        )
        if near_random:
            msg = (f"!!! EARLY WARNING {self.algo}/{self.reward}/seed{self.seed}: still at the RANDOM-baseline level at "
                   f"{int(val['env_step'].iloc[-1])} env steps (crash_rate {crash:.2f} vs random {ref['crash_rate']:.2f}, "
                   f"return_{self.reward} {ret:.1f} vs random {ref['return']:.1f} +- {ref['return_se']:.1f}). Run continues. !!!")
            print(msg, file=sys.stderr, flush=True)
            (self.rdir / "WARNING_random_level.txt").write_text(msg + "\n", encoding="utf-8")

    def checkpoint(self, step: int, model: Any) -> None:
        with self.timers.checkpoint():
            self.logger.flush()
            state = {"env_step": step, "logger": self.logger.state_dict(), "timers": self.timers.state_dict(),
                     "next_val": self.next_val, "last_val_step": self.last_val_step, "warned": self.warned,
                     "resumes_done": self.resumes_done}
            self.store.write(model, step, state)
        self._last_ckpt = time.monotonic()
        self._fill_timing(step)
        write_meta(self.meta_path, self.meta)

    def write_progress(self, step: int) -> None:
        now = time.monotonic()
        done = step - self._step_at_start
        speed = done / max(now - self._t_start, 1e-9)
        eta = (self.total - step) / speed if speed > 0 else math.nan
        val = None
        vl = self.rdir / "val_log.csv"
        if vl.exists() and self.logger.val_idx:
            try:
                last = pd.read_csv(vl).iloc[-1]
                val = {"env_step": int(last["env_step"]), "crash_rate": float(last["crash_rate"]),
                       "mean_speed": float(last["mean_speed"]), "return_own": float(last[f"return_{self.reward}"])}
            except Exception:  # noqa: BLE001  (file being written)
                val = None
        try:
            atomic_write_json(self.rdir / "progress.json", {
            "algo": self.algo, "reward": self.reward, "seed": self.seed, "env_step": int(step), "total": int(self.total),
            "steps_per_s": speed, "eta_s": eta, "wall_s": self.timers.wall(), "val_s": self.timers.seconds["val"],
            "last_val": val, "time": utc_now(), "status": "running",
            })
        except OSError:  # progress.json is only for the orchestrator: a locked file must never stop a run
            pass
        self._last_progress = now

    def _fill_timing(self, step: int) -> None:
        t = self.meta["timing"]
        sec = self.timers.seconds
        t["total_wall_s"] = self.timers.wall()
        t["update_time_s"], t["val_time_s"], t["checkpoint_time_s"] = sec["update"], sec["val"], sec["checkpoint"]
        # env + SB3 overhead (action selection, buffers) = everything that is not update, validation or checkpointing
        t["env_time_s"] = t["total_wall_s"] - sec["update"] - sec["val"] - sec["checkpoint"]
        t["final_env_step"] = step
        train_wall = self.timers.train_wall() - sec["checkpoint"]
        t["train_steps_per_s"] = step / max(train_wall, 1e-9)
        t["n_episodes"], t["n_updates"], t["n_val_points"] = self.logger.n_episodes, self.logger.update_idx, self.logger.val_idx

    # -- run ------------------------------------------------------------------------------------------------------
    def _new_meta(self) -> dict[str, Any]:
        meta = new_meta({"algorithm": f"sb3_{self.algo}", "train_reward": self.reward, "train_density": TRAIN_DENSITY,
                         "seed": self.seed, "variant": "calibration", "overridden_keys": []})
        import stable_baselines3

        hp = hyperparameters(self.algo)
        meta["hyperparameters"] = {k: v for k, (v, _) in hp.items()}
        meta["budget"] = {"total_env_steps": self.total, "val_every_env_steps": self.val_every, "val_n_episodes": self.val_n,
                          "checkpoint_every_seconds": self.checkpoint_seconds}
        meta["resolved"] = ({"exploration_steps": round(hp["exploration_fraction"][0] * self.total)} if self.algo == "dqn"
                            else {"rollout_size": PPO_N_ENVS * PPO_N_STEPS, "n_rollouts": self.total // (PPO_N_ENVS * PPO_N_STEPS)})
        meta["calibration"] = {
            "purpose": "calibration only; never part of the DDQN vs PPO comparison",
            "library": "stable-baselines3", "library_version": stable_baselines3.__version__,
            "algorithm_note": ("SB3 DQN = vanilla DQN with a target network (max target), NOT Double DQN" if self.algo == "dqn"
                               else "SB3 PPO (separate actor and critic MLPs, orthogonal init; SB3 defaults except as listed)"),
            "hyperparameter_sources": {k: s for k, (_, s) in hp.items()},
            "validation": {"seeds": f"VAL_SEEDS[:{self.val_n}]", "condition": "main", "deterministic": True,
                           "validation_workers": self.val_workers},
            "timing_note": "env_time_s = wall - update - validation - checkpoint (includes SB3 action selection and buffer overhead)",
            "resume_note": "a resume restarts the in-flight episodes with new derived seeds; not bit-identical (see sb3_train.py)",
        }
        meta["placeholders_in_use"] = []
        meta["logging"]["update_log_every"] = LOG_UPDATE_EVERY[self.algo]
        meta["reproducibility"] = reproducibility_info("cpu", 1 if self.algo == "dqn" else PPO_N_ENVS)
        meta["reproducibility"]["versions"]["stable_baselines3"] = stable_baselines3.__version__
        return meta

    def _prepare_dir(self) -> Optional[dict[str, Any]]:
        """Fresh start or resume; returns the checkpoint info when resuming. Protects existing results."""
        # the run's own result files; train_stdout.log belongs to the orchestrator (it holds the file open) and stays
        existing = [p for p in self.rdir.glob("*") if p.is_file() and p.name != "train_stdout.log"] if self.rdir.exists() else []
        if self.overwrite and existing:
            for p in existing:
                p.unlink()
            for p in (self.rdir / "ckpt").glob("*") if (self.rdir / "ckpt").exists() else []:
                p.unlink()
            existing = []
        if self.resume:
            info = self.store.latest()
            if info is not None:
                return info
            if existing:  # files but no usable checkpoint: nothing to resume from, restart cleanly
                self._say("no usable checkpoint: restarting this run from scratch")
                for p in existing:
                    p.unlink()
            return None
        if existing:
            raise FileExistsError(f"{self.rdir} already holds results: use --resume or --overwrite")
        return None

    def run(self) -> Path:
        torch.set_num_threads(1)
        info = self._prepare_dir()
        self.rdir.mkdir(parents=True, exist_ok=True)
        if info is not None and self.meta_path.exists():
            self.meta = read_meta(self.meta_path)
            if self.meta.get("status") == "finished" and (self.rdir / "final.zip").exists():
                self._say(f"run already finished: {self.rdir}")
                return self.rdir
            self.meta["status"], self.meta["error"] = "running", None
            self.meta["timing"]["end_time"] = None
        else:
            self.meta = self._new_meta()
            info = None
        self.logger = TrainLogger(self.rdir, "ddqn" if self.algo == "dqn" else "ppo", self.timers,
                                  update_log_every=LOG_UPDATE_EVERY[self.algo])
        venv = model = None
        try:
            set_global_seeds(self.seed)
            resume_k = 0
            if info is not None:
                state = json.loads((self.store.dir / info["state"]).read_text(encoding="utf-8"))
                resume_k = int(state["resumes_done"]) + 1
            venv = make_vec_env(self.algo, self.reward, self.seed, resume_k)
            if info is None:
                model = build_model(self.algo, venv)
                set_global_seeds(self.seed)  # network initialisation is done; rewind so exploration RNGs start from the seed
                torch.manual_seed(self.seed)
                model.action_space.seed(self.seed)
                step0 = 0
            else:
                model = (CalDQN if self.algo == "dqn" else CalPPO).load(self.store.dir / info["model"], env=venv, device="cpu")
                if self.algo == "dqn":
                    model.load_replay_buffer(self.store.dir / info["buffer"])
                step0 = int(state["env_step"])
                self.timers.load_state_dict(state["timers"])
                self.logger.load_state_dict(state["logger"])
                dropped = self.logger.truncate_to(step0)
                self._truncate_val_episodes(step0)
                self.next_val, self.last_val_step = int(state["next_val"]), int(state["last_val_step"])
                self.warned, self.resumes_done = bool(state["warned"]), resume_k
                set_global_seeds(vector_env_seed(self.seed, 900 + resume_k))
                model.action_space.seed(vector_env_seed(self.seed, 900 + resume_k))
                self.meta["resumes"].append({"env_step": step0, "time": utc_now(), "rows_dropped": dropped})
                self._say(f"resumed at env step {step0} (resume #{resume_k}, rows dropped beyond it: {dropped})")
            self._step_at_start = step0
            self.meta["model"]["n_parameters"] = n_parameters(model, self.algo)
            if self.algo == "dqn":
                self.meta["env"] = describe_env(venv.envs[0], ENV_ID, self.reward)
            else:
                probe = sb3_cal.make_cal_env(self.reward)
                self.meta["env"] = describe_env(probe, ENV_ID, self.reward)
                probe.close()
            model._cal = _Hooks(self.timers, self.logger)
            self.validator = Validator(self.reward, self.val_workers, self.val_n)
            write_meta(self.meta_path, self.meta)
            self._t_start = time.monotonic()
            self._last_ckpt = time.monotonic()
            remaining = self.total - step0
            model.learn(total_timesteps=remaining, callback=CalibrationCallback(self), reset_num_timesteps=info is None,
                        log_interval=10**9, progress_bar=False)
            end = int(model.num_timesteps)
            if self.last_val_step != end:
                self.validate(end, model)
            model.save(self.rdir / "final.zip")
            self.meta["status"] = "finished"
        except BaseException as exc:  # noqa: BLE001  (record the failure, then re-raise)
            self.meta["status"] = "failed"
            self.meta["error"] = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, KeyboardInterrupt):
                self.meta["error"] += "\n" + traceback.format_exc()
            raise
        finally:
            if self.validator is not None:
                self.validator.close()
            if self.logger is not None:
                self.logger.close(emit_partial=self.meta.get("status") == "finished")
            step = int(model.num_timesteps) if model is not None else self.meta["timing"].get("final_env_step", 0)
            if model is not None:
                self._fill_timing(step)
            self.meta["timing"]["end_time"] = utc_now()
            write_meta(self.meta_path, self.meta)
            if self.meta.get("status") == "finished":
                self.write_progress(step)
                prog = json.loads((self.rdir / "progress.json").read_text(encoding="utf-8"))
                prog["status"] = "finished"
                atomic_write_json(self.rdir / "progress.json", prog)
            if venv is not None:
                venv.close()
        t = self.meta["timing"]
        self._say(f"finished {self.rdir}: {t['final_env_step']} env steps, {t['total_wall_s']:.0f} s wall, "
                  f"{t['train_steps_per_s']:.1f} steps/s (without validation)")
        return self.rdir

    def _truncate_val_episodes(self, step: int) -> None:
        p = self.val_episodes_path
        if p.exists():
            df = pd.read_csv(p)
            df[df["env_step"] <= step].to_csv(p, index=False)
