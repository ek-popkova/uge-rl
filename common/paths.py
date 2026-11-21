"""File layout of trained models, training logs and evaluation results.

    results/runs/<algo>/<reward>/seed<s>/final.pt          final checkpoint (end of the step budget)
    results/runs/<algo>/<reward>/seed<s>/train_log.csv     training log
    results/runs/<algo>/<reward>/seed<s>/val_log.csv       step + validation metrics (VAL_SEEDS)
    results/eval/<algo>/<reward>/seed<s>/<condition>.csv   per-episode test results
    results/eval/baselines/<condition>.csv, summary.csv    scripted baselines (common.baselines)

``<algo>`` is "ddqn" or "ppo", ``<reward>`` a key of ``common.config.REWARDS``, ``<s>`` the
training seed (``TRAIN_SEEDS`` for the main grid), ``<condition>`` a key of
``common.config.TEST_CONDITIONS``.

Checkpoint rule: evaluate the FINAL checkpoint at the end of the step budget. If a checkpoint is
ever selected (e.g. best validation return), select it on ``VAL_SEEDS`` (``val_log.csv``), never
on ``TEST_SEEDS``. Test results are produced once per model and condition.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.config import REWARDS, TEST_CONDITIONS

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
ALGORITHMS = ("ddqn", "ppo")


def _check(algo: str, reward: str) -> None:
    if algo not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algo!r}; choose from {ALGORITHMS}")
    if reward not in REWARDS:
        raise ValueError(f"Unknown reward {reward!r}; choose from {list(REWARDS)}")


def run_dir(algo: str, reward: str, seed: int, root: Path = RESULTS_DIR) -> Path:
    """Directory of one training run."""
    _check(algo, reward)
    return Path(root) / "runs" / algo / reward / f"seed{seed}"


def final_checkpoint(algo: str, reward: str, seed: int, root: Path = RESULTS_DIR) -> Path:
    return run_dir(algo, reward, seed, root) / "final.pt"


def train_log(algo: str, reward: str, seed: int, root: Path = RESULTS_DIR) -> Path:
    return run_dir(algo, reward, seed, root) / "train_log.csv"


def val_log(algo: str, reward: str, seed: int, root: Path = RESULTS_DIR) -> Path:
    return run_dir(algo, reward, seed, root) / "val_log.csv"


def eval_dir(algo: str, reward: str, seed: int, root: Path = RESULTS_DIR) -> Path:
    _check(algo, reward)
    return Path(root) / "eval" / algo / reward / f"seed{seed}"


def eval_csv(algo: str, reward: str, seed: int, condition: str, root: Path = RESULTS_DIR) -> Path:
    if condition not in TEST_CONDITIONS:
        raise ValueError(f"Unknown test condition {condition!r}; choose from {list(TEST_CONDITIONS)}")
    return eval_dir(algo, reward, seed, root) / f"{condition}.csv"


def baselines_dir(root: Path = RESULTS_DIR) -> Path:
    return Path(root) / "eval" / "baselines"


def save_eval(df: pd.DataFrame, algo: str, reward: str, seed: int, condition: str, root: Path = RESULTS_DIR) -> Path:
    """Write per-episode test results of one model and condition (creates the directories)."""
    path = eval_csv(algo, reward, seed, condition, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_eval_episodes(root: Path = RESULTS_DIR) -> pd.DataFrame:
    """Load all per-episode test CSVs under ``<root>/eval/<algo>/<reward>/seed<s>/`` into one frame.

    Adds the columns ``algorithm``, ``train_reward`` (the preset the model was trained with),
    ``train_seed`` and ``condition``; ``reward`` stays the preset the evaluation env used.
    """
    frames = []
    for algo in ALGORITHMS:
        for path in sorted((Path(root) / "eval" / algo).glob("*/seed*/*.csv")):
            df = pd.read_csv(path)
            df["algorithm"] = algo
            df["train_reward"] = path.parent.parent.name
            df["train_seed"] = int(path.parent.name.removeprefix("seed"))
            df["condition"] = path.stem
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
