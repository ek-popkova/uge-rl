"""Two-level aggregation of test results (see ``docs/evaluation_protocol.md``).

Level 1, per model: one model = (algorithm, train reward, training seed); its metrics under one
condition are the mean over its test episodes (100 on ``TEST_SEEDS``). ``per_model_table``.

Level 2, per (algorithm, train reward, condition): mean over the training seeds with a 95%
bootstrap confidence interval over seeds. The training seed is the unit of analysis: the episodes
of one model are NOT treated as independent samples, and pooled episodes never enter a confidence
interval (Patterson et al., 2024). The per-seed values stay available in the level-1 table for
plotting individual runs. ``across_seeds_table``.

Returns are only comparable within one reward preset: ``episode_return`` and ``return_<preset>``
live on different scales (a crash costs 15, 20 or 30). Compare reward presets by behavioural
metrics (crash rate, completion rate, mean speed, lane changes, near-miss rate, crashes per km).
The level-1 column ``return_own`` is the return under the preset the model was trained with.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from common.metrics import add_return_columns, summarize_metrics

MODEL_KEYS = ["algorithm", "train_reward", "train_seed", "condition"]
GROUP_KEYS = ["algorithm", "train_reward", "condition"]
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95


def per_model_table(episodes: pd.DataFrame) -> pd.DataFrame:
    """Level 1: one row per (algorithm, train_reward, train_seed, condition).

    ``episodes`` is the concatenation of the per-episode CSVs with the columns ``algorithm``,
    ``train_reward``, ``train_seed`` and ``condition`` added (``common.paths.load_eval_episodes``).
    """
    missing = [c for c in MODEL_KEYS if c not in episodes]
    if missing:
        raise ValueError(f"episodes lack the columns {missing}")
    episodes = add_return_columns(episodes)
    rows = []
    for key, df in episodes.groupby(MODEL_KEYS, sort=True):
        row = dict(zip(MODEL_KEYS, key))
        row.update(summarize_metrics(df))
        row["return_own"] = float(df[f"return_{row['train_reward']}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a proportion ``k / n`` (for the scripted baselines, whose
    unit is the episode because there is no training seed)."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return centre - half, centre + half


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    level: float = CI_LEVEL,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling the entries of ``values`` (the seeds).

    NaN values are dropped. With fewer than 2 values the interval is NaN. With only a handful of
    seeds the percentile bootstrap is narrow (it cannot go beyond the observed range), so read
    it together with the individual runs.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def across_seeds_table(per_model: pd.DataFrame, n_boot: int = N_BOOTSTRAP, seed: int = 0) -> pd.DataFrame:
    """Level 2: mean over training seeds and 95% bootstrap CI per (algorithm, train_reward, condition).

    Returns a long table with the columns ``algorithm, train_reward, condition, metric, mean,
    ci_low, ci_high, n_seeds``. The bootstrap RNG is seeded, so the table is reproducible.
    """
    metrics = [c for c in per_model.columns if c not in MODEL_KEYS and c != "n_episodes"]
    rng = np.random.default_rng(seed)
    rows = []
    for key, df in per_model.groupby(GROUP_KEYS, sort=True):
        for metric in metrics:
            vals = df[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(vals, n_boot=n_boot, rng=rng)
            rows.append(
                {
                    **dict(zip(GROUP_KEYS, key)),
                    "metric": metric,
                    "mean": float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan"),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_seeds": int(np.sum(~np.isnan(vals))),
                }
            )
    return pd.DataFrame(rows)


def aggregate(episodes: pd.DataFrame, n_boot: int = N_BOOTSTRAP) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both levels at once: ``(per_model, across_seeds)``."""
    per_model = per_model_table(episodes)
    return per_model, across_seeds_table(per_model, n_boot=n_boot)
