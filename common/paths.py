"""File layout of trained models, training logs and evaluation results.

    results/runs/<algo>/<reward>/<variant>/seed<s>/final.pt              final checkpoint (end of the budget)
    results/runs/<algo>/<reward>/<variant>/seed<s>/milestone_<p>pct.pt   weights at p% of the budget
    results/runs/<algo>/<reward>/<variant>/seed<s>/resume.pt             full state for --resume (overwritten)
    results/runs/<algo>/<reward>/<variant>/seed<s>/train_log.csv         one row per training episode
    results/runs/<algo>/<reward>/<variant>/seed<s>/update_log.csv        one row per (K) gradient update(s)
    results/runs/<algo>/<reward>/<variant>/seed<s>/val_log.csv           validation metrics (VAL_SEEDS)
    results/runs/<algo>/<reward>/<variant>/seed<s>/run_meta.json         identity, config, timing, versions
    results/eval/<algo>/<reward>/<variant>/seed<s>/<condition>.csv       per-episode test results
    results/eval/baselines/<condition>.csv, summary.csv                  scripted baselines (common.baselines)

``<algo>`` is "ddqn" or "ppo", ``<reward>`` a key of ``common.config.REWARDS``, ``<variant>`` is
``"main"`` for the main grid and a name generated from the overridden hyperparameters for sensitivity
runs (``common.train_config.variant_name``, e.g. ``lr=0.0003`` or ``clip_range=0.1``), so sensitivity
runs never overwrite main runs. Hyperparameter tuning runs are the variants ``tune-<name>`` (reward ``balanced``, seed
``PILOT_RUN_SEED``); ``load_eval_episodes`` skips them by default. Supplementary runs are the variants ``supp-<name>`` of the
pre-declared registry ``common.train_config.SUPPLEMENTARY_RUNS`` (they may also use a ``SUPPLEMENTARY_REWARDS`` preset as
``<reward>``); ``load_eval_episodes`` skips them by default too, so they never enter a DDQN vs PPO table. ``variant_role`` names
the role of a variant (main, sensitivity, tuning, supplementary). ``<s>`` is the training seed (``TRAIN_SEEDS`` for the main grid),
``<condition>`` a key of ``common.config.TEST_CONDITIONS``. Before the ``variant`` level existed the
layout was ``runs/<algo>/<reward>/seed<s>``; no trained run was ever written in it, so it is not
supported (``load_eval_episodes`` refuses to silently skip such directories).

Checkpoint rule: evaluate the FINAL checkpoint at the end of the step budget. If a checkpoint is
ever selected (e.g. best validation return), select it on ``VAL_SEEDS`` (``val_log.csv``), never
on ``TEST_SEEDS``. Test results are produced once per model and condition.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from common.config import PILOT_RUN_SEED, TEST_CONDITIONS, TUNING_REWARD, all_reward_names, is_supplementary_reward

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
ALGORITHMS = ("ddqn", "ppo")
MAIN_VARIANT = "main"
TUNING_PREFIX = "tune-"  # variant prefix of the hyperparameter tuning runs (never part of any result table)
SUPPLEMENTARY_PREFIX = "supp-"  # variant prefix of the registered supplementary runs (never part of a DDQN vs PPO table)
ROLES = ("main", "sensitivity", "tuning", "supplementary")
_VARIANT_RE = re.compile(r"^[A-Za-z0-9_+=,-][A-Za-z0-9_.+=,-]*$")  # also valid on Windows


def check_variant(variant: str) -> str:
    """Validate a variant directory name (letters, digits and ``_ . + = , -``; no path separators)."""
    if not isinstance(variant, str) or not _VARIANT_RE.match(variant) or variant.endswith("."):
        raise ValueError(f"Invalid variant name {variant!r}: use letters, digits and _ . + = , - only")
    return variant


def tuning_variant(name: str) -> str:
    """Variant directory of the tuning configuration ``name``: ``tune-<name>``."""
    if not name or name.startswith(TUNING_PREFIX):
        raise ValueError(f"tuning configuration name must be non-empty and without the {TUNING_PREFIX!r} prefix: {name!r}")
    return check_variant(f"{TUNING_PREFIX}{name}")


def is_tuning_variant(variant: str) -> bool:
    return variant.startswith(TUNING_PREFIX)


def supplementary_variant(name: str) -> str:
    """Variant directory of the supplementary run ``name`` (a key of ``SUPPLEMENTARY_RUNS``): ``supp-<name>``."""
    if not name or name.startswith(SUPPLEMENTARY_PREFIX) or name.startswith(TUNING_PREFIX):
        raise ValueError(f"supplementary run name must be non-empty and without a role prefix: {name!r}")
    return check_variant(f"{SUPPLEMENTARY_PREFIX}{name}")


def is_supplementary_variant(variant: str) -> bool:
    return variant.startswith(SUPPLEMENTARY_PREFIX)


def variant_role(variant: str) -> str:
    """Role of a run from its variant name: ``"main"``, ``"tuning"`` (``tune-*``), ``"supplementary"`` (``supp-*``) or
    ``"sensitivity"`` (every other name: the harness accepts no other result variant, ``check_run_roles``; debug runs with
    budget overrides also land here and belong in a separate ``--out-dir``)."""
    if variant == MAIN_VARIANT:
        return "main"
    if is_tuning_variant(variant):
        return "tuning"
    if is_supplementary_variant(variant):
        return "supplementary"
    return "sensitivity"


def _check(algo: str, reward: str, variant: str = MAIN_VARIANT) -> None:
    if algo not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algo!r}; choose from {ALGORITHMS}")
    if reward not in all_reward_names():
        raise ValueError(f"Unknown reward {reward!r}; choose from {all_reward_names()}")
    check_variant(variant)


def run_dir(algo: str, reward: str, seed: int, root: Path = RESULTS_DIR, variant: str = MAIN_VARIANT) -> Path:
    """Directory of one training run."""
    _check(algo, reward, variant)
    return Path(root) / "runs" / algo / reward / variant / f"seed{seed}"


def final_checkpoint(algo: str, reward: str, seed: int, root: Path = RESULTS_DIR, variant: str = MAIN_VARIANT) -> Path:
    return run_dir(algo, reward, seed, root, variant) / "final.pt"


def milestone_checkpoint(
    algo: str, reward: str, seed: int, fraction: float, root: Path = RESULTS_DIR, variant: str = MAIN_VARIANT
) -> Path:
    """Weights-only checkpoint kept at ``fraction`` of the step budget (0.1 -> ``milestone_10pct.pt``)."""
    return run_dir(algo, reward, seed, root, variant) / f"milestone_{fraction * 100:g}pct.pt"


def resume_checkpoint(algo: str, reward: str, seed: int, root: Path = RESULTS_DIR, variant: str = MAIN_VARIANT) -> Path:
    return run_dir(algo, reward, seed, root, variant) / "resume.pt"


def train_log(algo: str, reward: str, seed: int, root: Path = RESULTS_DIR, variant: str = MAIN_VARIANT) -> Path:
    return run_dir(algo, reward, seed, root, variant) / "train_log.csv"


def update_log(algo: str, reward: str, seed: int, root: Path = RESULTS_DIR, variant: str = MAIN_VARIANT) -> Path:
    return run_dir(algo, reward, seed, root, variant) / "update_log.csv"


def val_log(algo: str, reward: str, seed: int, root: Path = RESULTS_DIR, variant: str = MAIN_VARIANT) -> Path:
    return run_dir(algo, reward, seed, root, variant) / "val_log.csv"


def run_meta_path(algo: str, reward: str, seed: int, root: Path = RESULTS_DIR, variant: str = MAIN_VARIANT) -> Path:
    return run_dir(algo, reward, seed, root, variant) / "run_meta.json"


def eval_dir(algo: str, reward: str, seed: int, root: Path = RESULTS_DIR, variant: str = MAIN_VARIANT) -> Path:
    _check(algo, reward, variant)
    return Path(root) / "eval" / algo / reward / variant / f"seed{seed}"


def eval_csv(
    algo: str, reward: str, seed: int, condition: str, root: Path = RESULTS_DIR, variant: str = MAIN_VARIANT
) -> Path:
    if condition not in TEST_CONDITIONS:
        raise ValueError(f"Unknown test condition {condition!r}; choose from {list(TEST_CONDITIONS)}")
    return eval_dir(algo, reward, seed, root, variant) / f"{condition}.csv"


def baselines_dir(root: Path = RESULTS_DIR) -> Path:
    return Path(root) / "eval" / "baselines"


def save_eval(
    df: pd.DataFrame,
    algo: str,
    reward: str,
    seed: int,
    condition: str,
    root: Path = RESULTS_DIR,
    variant: str = MAIN_VARIANT,
) -> Path:
    """Write per-episode test results of one model and condition (creates the directories)."""
    path = eval_csv(algo, reward, seed, condition, root, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_eval_episodes(
    root: Path = RESULTS_DIR, include_tuning: bool = False, include_supplementary: bool = False
) -> pd.DataFrame:
    """Load all per-episode test CSVs under ``<root>/eval/<algo>/<reward>/<variant>/seed<s>/`` into one frame.

    Adds the columns ``algorithm``, ``train_reward`` (the preset the model was trained with),
    ``variant``, ``role`` (``variant_role``), ``train_seed`` and ``condition``; ``reward`` stays the preset the evaluation
    env used. Tuning runs (variant ``tune-<name>``, seed ``PILOT_RUN_SEED``) are skipped unless ``include_tuning`` is True:
    they never enter a result table. Supplementary runs (variant ``supp-<name>``, and every run trained with a
    ``SUPPLEMENTARY_REWARDS`` preset) are skipped unless ``include_supplementary`` is True: they never enter a DDQN vs PPO
    table.
    """
    frames = []
    for algo in ALGORITHMS:
        legacy = sorted((Path(root) / "eval" / algo).glob("*/seed*"))
        if legacy:
            raise ValueError(
                f"Old layout without a variant level found: {legacy[0]}; expected .../<reward>/<variant>/seed<s>"
            )
        for path in sorted((Path(root) / "eval" / algo).glob("*/*/seed*/*.csv")):
            variant, train_reward = path.parent.parent.name, path.parent.parent.parent.name
            if not include_tuning and is_tuning_variant(variant):
                continue
            if not include_supplementary and (is_supplementary_variant(variant) or is_supplementary_reward(train_reward)):
                continue
            df = pd.read_csv(path)
            df["algorithm"] = algo
            df["train_reward"] = train_reward
            df["variant"] = variant
            df["role"] = variant_role(variant)
            df["train_seed"] = int(path.parent.name.removeprefix("seed"))
            df["condition"] = path.stem
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_val_logs(
    algo: str, reward: str, root: Path = RESULTS_DIR, variant: str = MAIN_VARIANT
) -> dict[int, pd.DataFrame]:
    """``val_log.csv`` of every training seed of (algo, reward, variant): ``{seed: frame}``, seeds ascending.

    Input of ``common.aggregate.learning_curve_table`` and ``steps_to_threshold``.
    """
    base = run_dir(algo, reward, 0, root, variant).parent
    logs = {}
    for seed_dir in base.glob("seed*"):
        path = seed_dir / "val_log.csv"
        if path.exists():
            logs[int(seed_dir.name.removeprefix("seed"))] = pd.read_csv(path)
    return dict(sorted(logs.items()))


def load_tuning_val_logs(
    algo: str, root: Path = RESULTS_DIR, seed: int = PILOT_RUN_SEED
) -> dict[str, pd.DataFrame]:
    """``val_log.csv`` of every tuning configuration of ``algo``: ``{config name: frame}`` (name without ``tune-``).

    Input of ``common.aggregate.select_tuning_config``. Tuning runs use the reward ``TUNING_REWARD`` and the run seed
    ``PILOT_RUN_SEED``; configurations are returned in alphabetical order of their name.
    """
    if algo not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algo!r}; choose from {ALGORITHMS}")
    logs = {}
    for variant_dir in sorted((Path(root) / "runs" / algo / TUNING_REWARD).glob(f"{TUNING_PREFIX}*")):
        path = variant_dir / f"seed{seed}" / "val_log.csv"
        if path.exists():
            logs[variant_dir.name.removeprefix(TUNING_PREFIX)] = pd.read_csv(path)
    return logs
