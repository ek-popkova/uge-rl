"""Shared, light-weight pieces of the SB3 calibration runs (no torch / stable-baselines3 import at module level).

Calibration only: Stable-Baselines3 reference implementations (SB3 DQN = vanilla DQN with a target network, NOT Double
DQN; SB3 PPO) are trained in OUR environment to calibrate the main grid (rewards, densities, seed spread, budget,
validation noise, targets). They are never part of the DDQN vs PPO comparison and never enter the main tables.

Module is imported by the env subprocesses of the PPO vector env (spawn), so it must stay cheap to import: heavy
imports live in ``sb3_train.py`` and inside functions.

Layout (``results/calibration/`` is deliberately NOT below ``results/runs`` or ``results/eval``, so
``common.paths.load_eval_episodes`` / ``load_val_logs`` / ``per_model_table`` can never see these files):

    results/calibration/benchmark.json                           quiet speed measurements (benchmark.py)
    results/calibration/plan.json                                the schedule actually used (run_plan.py)
    results/calibration/sb3/<algo>/<reward>/seed<s>/             one training run (<algo> = dqn | ppo)
        train_log.csv update_log.csv val_log.csv                 exactly the main-grid schemas (common.train_logging)
        val_episodes.csv                                         per-episode validation records (env_step, val_idx, ...)
        run_meta.json progress.json final.zip ckpt/              metadata, live progress, final model, resume state
    results/calibration/eval/sb3/<algo>/<reward>/seed<s>/<condition>.csv   P4: final model, VAL_SEEDS, 5 conditions
    results/calibration/eval/baselines/<condition>.csv           scripted baselines on VAL_SEEDS, 5 conditions
    results/calibration/plots/                                   figures of analyze.py
"""

from __future__ import annotations

import math
import os
from functools import partial
from pathlib import Path
from typing import Any

from common.agent_base import Agent
from common.config import CALIBRATION_SEEDS, REWARDS, TRAIN_DENSITY
from common.paths import RESULTS_DIR

# -- layout -------------------------------------------------------------------------------------------------------
CAL_ROOT = RESULTS_DIR / "calibration"
assert CAL_ROOT.parent == RESULTS_DIR and CAL_ROOT.name not in ("runs", "eval")  # never visible to the main-grid loaders
ALGOS = ("dqn", "ppo")
SB3_LIBRARY_VERSION = "2.9.2"


def run_dir(algo: str, reward: str, seed: int, root: Path = CAL_ROOT) -> Path:
    _check(algo, reward, seed)
    return Path(root) / "sb3" / algo / reward / f"seed{seed}"


def eval_dir(algo: str, reward: str, seed: int, root: Path = CAL_ROOT) -> Path:
    _check(algo, reward, seed)
    return Path(root) / "eval" / "sb3" / algo / reward / f"seed{seed}"


def baselines_dir(root: Path = CAL_ROOT) -> Path:
    return Path(root) / "eval" / "baselines"


def _check(algo: str, reward: str, seed: int) -> None:
    if algo not in ALGOS:
        raise ValueError(f"unknown calibration algorithm {algo!r}; choose from {ALGOS}")
    if reward not in REWARDS:
        raise ValueError(f"unknown reward {reward!r}; choose from {list(REWARDS)}")
    if seed not in CALIBRATION_SEEDS:
        raise ValueError(f"calibration runs use the seeds {CALIBRATION_SEEDS}, got {seed}")


def is_finished(algo: str, reward: str, seed: int, root: Path = CAL_ROOT) -> bool:
    """True if the run's ``run_meta.json`` says ``finished`` and ``final.zip`` exists."""
    import json

    meta = run_dir(algo, reward, seed, root) / "run_meta.json"
    if not meta.exists() or not (run_dir(algo, reward, seed, root) / "final.zip").exists():
        return False
    try:
        return json.loads(meta.read_text(encoding="utf-8")).get("status") == "finished"
    except (OSError, ValueError):
        return False


# -- budget and hyperparameters ---------------------------------------------------------------------------------------
TOTAL_ENV_STEPS = 200_000
VAL_EVERY_ENV_STEPS = 5_000
VAL_N_EPISODES = 20  # all VAL_SEEDS
CHECKPOINT_EVERY_SECONDS = 600.0  # about 10 min of wall time (the task asks for wall-time based checkpoints)
PPO_N_ENVS = 8
PPO_N_STEPS = 256
PPO_ROLLOUT = PPO_N_ENVS * PPO_N_STEPS  # 2048
VAL_POOL_WORKERS = 2  # PPO: persistent validation pool (20 episodes = 4 chunks of 5; 2 workers save memory); DQN validates in-process
VAL_CHUNK = 5


def total_steps(algo: str) -> int:
    """Env steps of a run. PPO collects whole rollouts: 200,000 is not a multiple of 2048, so 98 rollouts = 200,704."""
    if algo == "ppo":
        return math.ceil(TOTAL_ENV_STEPS / PPO_ROLLOUT) * PPO_ROLLOUT
    return TOTAL_ENV_STEPS


# Source of every value, recorded in run_meta.json and reported (docs/calibration.md).
SRC_EXAMPLE = "HighwayEnv scripts/sb3_highway_dqn.py (master, fetched 2026-11-21)"
SRC_SB3 = "SB3 2.9.2 default (the HighwayEnv example gives no value)"
SRC_TASK = "calibration task specification"
SRC_FROZEN = "frozen shared decision (common/train_config.py)"

DQN_HYPERPARAMETERS: dict[str, tuple[Any, str]] = {
    "learning_rate": (5e-4, SRC_EXAMPLE),
    "buffer_size": (15_000, SRC_EXAMPLE),
    "learning_starts": (200, SRC_EXAMPLE),
    "batch_size": (32, SRC_EXAMPLE),
    "gamma": (0.99, f"{SRC_FROZEN}; the example uses 0.8"),
    "train_freq": (1, SRC_EXAMPLE),
    "gradient_steps": (1, SRC_EXAMPLE),
    "target_update_interval": (50, SRC_EXAMPLE),
    "exploration_fraction": (0.3, f"{SRC_TASK}: epsilon reaches its final value at 30% of the run (example: SB3 default 0.1)"),
    "exploration_initial_eps": (1.0, SRC_SB3),
    "exploration_final_eps": (0.05, SRC_SB3),
    "tau": (1.0, SRC_SB3 + " (hard target update)"),
    "max_grad_norm": (10, SRC_SB3),
    "net_arch": ([256, 256], f"{SRC_FROZEN}, same as the example"),
    "activation_fn": ("ReLU", f"{SRC_FROZEN}; SB3 DQN default"),
}
PPO_HYPERPARAMETERS: dict[str, tuple[Any, str]] = {
    "learning_rate": (3e-4, f"{SRC_TASK} (= SB3 default)"),
    "n_envs": (PPO_N_ENVS, SRC_TASK),
    "n_steps": (PPO_N_STEPS, SRC_TASK),
    "batch_size": (64, f"{SRC_TASK} (= SB3 default)"),
    "n_epochs": (10, f"{SRC_TASK} (= SB3 default)"),
    "gamma": (0.99, SRC_FROZEN),
    "gae_lambda": (0.95, f"{SRC_TASK} (= SB3 default)"),
    "clip_range": (0.2, f"{SRC_TASK} (= SB3 default)"),
    "ent_coef": (0.0, f"{SRC_TASK} (= SB3 default)"),
    "vf_coef": (0.5, SRC_SB3),
    "max_grad_norm": (0.5, SRC_SB3),
    "normalize_advantage": (True, SRC_SB3),
    "target_kl": (None, SRC_SB3),
    "net_arch": ({"pi": [256, 256], "vf": [256, 256]}, SRC_FROZEN + " (separate actor and critic, as in the main grid)"),
    "activation_fn": ("ReLU", SRC_FROZEN),
}


def hyperparameters(algo: str) -> dict[str, tuple[Any, str]]:
    return DQN_HYPERPARAMETERS if algo == "dqn" else PPO_HYPERPARAMETERS


# -- environment ------------------------------------------------------------------------------------------------------
def make_cal_env(reward: str, env_index: int = 0):
    """The exact training env of the main grid (``common.envs.make_env(reward, "medium")`` + the shared episode recorder).

    Module-level and picklable through ``functools.partial`` so that SubprocVecEnv (spawn) can build it in the subprocess.
    """
    from common.train_env import make_train_env

    return make_train_env(TRAIN_DENSITY, reward, env_index=env_index)


def env_fns(reward: str, n_envs: int) -> list:
    return [partial(make_cal_env, reward, i) for i in range(n_envs)]


# -- agents ---------------------------------------------------------------------------------------------------------
class SB3Agent(Agent):
    """``Agent`` adapter around ``model.predict(obs, deterministic=True)`` of a loaded / trained SB3 model."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def act(self, obs: Any, deterministic: bool = True) -> int:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        raise NotImplementedError("load an SB3 model with load_sb3_model(...) and wrap it in SB3Agent")


class TorchPolicyAgent(Agent):
    """Light, picklable copy of the greedy policy network of an SB3 model (sent to validation / evaluation workers).

    DQN: argmax of the Q-values (``model.policy.q_net``). PPO: argmax of the actor logits (the mode of the categorical
    distribution, which is what ``predict(deterministic=True)`` returns). It holds only the network (no replay /
    rollout buffer, no SB3 objects), so unpickling needs torch but not stable-baselines3. Equality with ``model.predict``
    is checked on recorded observations by ``check_same_actions`` (and by tests/test_calibration.py).
    """

    def __init__(self, net: Any) -> None:
        self.net = net.eval()

    def act(self, obs: Any, deterministic: bool = True) -> int:
        import numpy as np
        import torch

        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
            return int(torch.argmax(self.net(x), dim=1).item())

    def save(self, path: str) -> None:
        import torch

        torch.save(self.net, path)

    def load(self, path: str) -> None:
        import torch

        self.net = torch.load(path, weights_only=False).eval()


def light_agent_from_model(model: Any, algo: str) -> TorchPolicyAgent:
    """Copy the greedy network of ``model`` into a ``TorchPolicyAgent`` (detached from the training model)."""
    import copy

    from torch import nn

    if algo == "dqn":
        net = nn.Sequential(nn.Flatten(), copy.deepcopy(model.policy.q_net.q_net))
    elif algo == "ppo":
        net = nn.Sequential(
            nn.Flatten(), copy.deepcopy(model.policy.mlp_extractor.policy_net), copy.deepcopy(model.policy.action_net)
        )
    else:
        raise ValueError(f"unknown algorithm {algo!r}")
    return TorchPolicyAgent(net.cpu())


def check_same_actions(model: Any, light: TorchPolicyAgent, observations: Any) -> int:
    """Number of observations on which ``model.predict`` and the light agent disagree (must be 0)."""
    wrapped = SB3Agent(model)
    return sum(wrapped.act(o) != light.act(o) for o in observations)


def load_sb3_model(algo: str, path: str | Path):
    """Load a finished SB3 model for inference (CPU)."""
    from stable_baselines3 import DQN, PPO

    return (DQN if algo == "dqn" else PPO).load(str(path), device="cpu")


# -- validation -------------------------------------------------------------------------------------------------------
class Validator:
    """Validation on all ``VAL_SEEDS`` in the ``main`` condition, deterministic policy.

    Identical to ``common.evaluate.validate`` (``evaluate`` + ``summarize_metrics(add_return_columns(...))``) but it keeps
    the per-episode records, and the worker processes are created once and reused for every validation (so each worker
    builds its validation env once, CLAUDE.md rule 4). ``workers <= 1`` validates in this process.
    """

    def __init__(self, reward: str, workers: int = 1, n_episodes: int = VAL_N_EPISODES, chunk: int = VAL_CHUNK) -> None:
        from common.config import VAL_SEEDS
        from common.evaluate import make_chunks
        from common.parallel import limit_threads

        self.reward, self.workers = reward, workers
        self.chunks = make_chunks(VAL_SEEDS[:n_episodes], chunk)
        self.pool = None
        if workers > 1:
            import multiprocessing as mp

            self.pool = mp.get_context("spawn").Pool(min(workers, len(self.chunks)), initializer=limit_threads)

    def run(self, agent: Agent) -> tuple[dict[str, float], Any]:
        """Returns ``(summary, per-episode DataFrame)``; the summary equals ``validate(...)``'s output."""
        from common.evaluate import records_to_frame, run_chunk
        from common.metrics import add_return_columns, summarize_metrics

        tasks = [(agent, "main", self.reward, chunk) for chunk in self.chunks]
        results = self.pool.map(run_chunk, tasks) if self.pool is not None else [run_chunk(t) for t in tasks]
        records = [r for res in results for r in res["records"]]
        df = add_return_columns(records_to_frame(records, "main", self.reward))
        return summarize_metrics(df), df

    def close(self) -> None:
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None


def retry_on_permission_error(fn, attempts: int = 12, delay: float = 0.25):
    """Windows: ``os.replace`` fails with PermissionError while another process (the orchestrator reading progress.json,
    OneDrive, an antivirus scan) has the target open. Such a collision lasts milliseconds, so retry before giving up."""
    import time

    for i in range(attempts):
        try:
            return fn()
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


def atomic_write_json(path: Path, obj: Any) -> None:
    import json
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)
        retry_on_permission_error(lambda: os.replace(tmp, path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
