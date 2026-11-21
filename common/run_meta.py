"""``run_meta.json``: everything needed to reproduce a run and to compare runs' cost.

Written when the run starts (status "running"), rewritten at every checkpoint and at the end
("finished"), and also if the run dies (status "failed" with the error; also for Ctrl-C) because the
harness wraps the run in try/finally. Sections:

    status, error
    identity          algorithm, train_reward, train_density, seed, variant, overridden_keys
    hyperparameters   full DDQNConfig / PPOConfig;  budget  full TrainBudget (+ budget_overrides)
    placeholders_in_use   undecided config values this run used (common.train_config)
    env               env id, the resolved env config actually used (env.unwrapped.config), reward weights,
                      action index mapping, observation shape, NEAR_MISS_TTC, REVERSAL_WINDOW
    reproducibility   git commit + dirty flag, library versions, OS, CPU model, logical cores, thread
                      settings, device, workers
    timing            timestamps, wall time, env / update / val / checkpoint seconds, steps/s without
                      validation, the quiet benchmark steps/s, final env step, episodes, updates, val points
    model             number of trainable parameters;  inference  ms per action (common.benchmark)
    logging, milestones, resumes
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from common.actions import ACTION_INDEX
from common.metrics import NEAR_MISS_TTC, REVERSAL_WINDOW
from common.parallel import THREAD_ENV_VARS
from common.paths import ROOT

STATUSES = ("running", "finished", "failed")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jsonable(value: Any) -> Any:
    """Recursively convert to JSON-safe types (tuples to lists, numpy scalars to Python, others to str)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


# ---------------------------------------------------------------------------------------------
# Environment description
# ---------------------------------------------------------------------------------------------
def check_action_mapping(env: Any) -> dict[str, int]:
    """Return the env's action index mapping; raise if it differs from ``common.actions.ACTION_INDEX``.

    The action histogram (``act_<NAME>``) and the scripted baselines assume this mapping, so a
    different highway-env version must fail loudly instead of mislabelling the histogram.
    """
    live = {k: int(v) for k, v in env.unwrapped.action_type.actions_indexes.items()}
    if live != ACTION_INDEX:
        raise RuntimeError(f"env action mapping {live} differs from common.actions.ACTION_INDEX {ACTION_INDEX}")
    return live


def describe_env(env: Any, env_id: str, reward_name: str) -> dict[str, Any]:
    """The env facts stored in run_meta: the resolved config actually used, not only BASE_CONFIG."""
    from common.config import REWARDS

    return {
        "env_id": env_id,
        "config": jsonable(dict(env.unwrapped.config)),
        "reward": reward_name,
        "reward_weights": dict(REWARDS[reward_name]),
        "action_index": check_action_mapping(env),
        "observation_shape": list(env.observation_space.shape),
        "near_miss_ttc": NEAR_MISS_TTC,
        "reversal_window": REVERSAL_WINDOW,
    }


# ---------------------------------------------------------------------------------------------
# Reproducibility info
# ---------------------------------------------------------------------------------------------
def git_info(cwd: Path = ROOT) -> dict[str, Any]:
    """Commit hash and dirty flag; ``None`` values if git is unavailable or this is not a repository."""
    try:
        run = lambda *args: subprocess.run(  # noqa: E731
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def cpu_model() -> str:
    """Best-effort CPU name without spawning slow subprocesses."""
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
    except Exception:  # noqa: BLE001  (informational only)
        pass
    return platform.processor() or "unknown"


def _version(module: str) -> Optional[str]:
    try:
        return str(__import__(module).__version__)
    except Exception:  # noqa: BLE001
        return None


def reproducibility_info(device: str, workers: int) -> dict[str, Any]:
    import torch

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gymnasium": _version("gymnasium"),
        "highway_env": _version("highway_env"),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
    }
    return {
        "git": git_info(),
        "versions": versions,
        "os": platform.platform(),
        "cpu_model": cpu_model(),
        "logical_cores": os.cpu_count(),
        "thread_env_vars": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "torch_num_threads": torch.get_num_threads(),
        "device": device,
        "workers": workers,
    }


# ---------------------------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------------------------
def new_meta(identity: dict[str, Any]) -> dict[str, Any]:
    """A meta dict with every field present (``None`` until it is known)."""
    return {
        "status": "running",
        "error": None,
        "identity": identity,
        "hyperparameters": None,
        "budget": None,
        "budget_overrides": [],
        "placeholders_in_use": [],
        "env": None,
        "reproducibility": None,
        "timing": {
            "start_time": utc_now(),
            "end_time": None,
            "total_wall_s": None,
            "env_time_s": None,
            "update_time_s": None,
            "val_time_s": None,
            "checkpoint_time_s": None,
            "train_steps_per_s": None,  # env steps / (wall time without validation)
            "post_training_benchmark_s": None,  # inference benchmark etc. after the last step (not training cost)
            "quiet_benchmark_steps_per_s": None,  # one process, before training (CLAUDE.md rule 1)
            "final_env_step": 0,
            "n_episodes": 0,
            "n_updates": 0,
            "n_val_points": 0,
        },
        "model": {"n_parameters": None},
        "inference": None,
        "logging": {"update_log_every": None},
        "milestones": [],
        "resumes": [],
    }


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    """Atomic write (temp file + replace), so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(jsonable(meta), fh, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_meta(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
