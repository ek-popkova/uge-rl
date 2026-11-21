"""Two-level aggregation of test results (see ``docs/evaluation_protocol.md``).

Level 1, per model: one model = (algorithm, train reward, training seed); its metrics under one
condition are the mean over its test episodes (100 on ``TEST_SEEDS``). ``per_model_table``.

Level 2, per (algorithm, train reward, condition): mean over the training seeds with a two-sided
95% Student-t confidence interval over seeds, mean +/- t(0.975, n-1) * sd / sqrt(n) with n the
number of seeds (sd with n-1 in the denominator). A bootstrap over a handful of values
underestimates the uncertainty (it cannot leave the observed range); the t-interval is the
standard choice for few runs. The training seed is the unit of analysis: the episodes
of one model are NOT treated as independent samples, and pooled episodes never enter a confidence
interval (Patterson et al., 2024). The per-seed values stay available in the level-1 table for
plotting individual runs. ``across_seeds_table``.

Returns are only comparable within one reward preset: ``episode_return`` and ``return_<preset>``
live on different scales (a crash costs 15, 20 or 30). Compare reward presets by behavioural
metrics (crash rate, completion rate, mean speed, lane changes, near-miss rate, crashes per km).
The level-1 column ``return_own`` is the return under the preset the model was trained with.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common.metrics import add_return_columns, summarize_metrics

MODEL_KEYS = ["algorithm", "train_reward", "train_seed", "condition"]
GROUP_KEYS = ["algorithm", "train_reward", "condition"]

# Two-sided 95% Student-t quantiles t(0.975, df) for df = 1..30 (checked against the t distribution).
_T_975 = (
    12.7062047, 4.3026527, 3.1824463, 2.7764451, 2.5705818, 2.4469119, 2.3646243, 2.3060041,
    2.2621572, 2.2281389, 2.2009852, 2.1788128, 2.1603687, 2.1447867, 2.1314495, 2.1199053,
    2.1098156, 2.1009220, 2.0930241, 2.0859634, 2.0796138, 2.0738731, 2.0686576, 2.0638986,
    2.0595386, 2.0555294, 2.0518305, 2.0484071, 2.0452296, 2.0422725,
)
_Z_975 = 1.959963984540054


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


def t_quantile_975(df: int) -> float:
    """Student-t quantile t(0.975, df) without scipy (table for df <= 30, Cornish-Fisher above)."""
    if df < 1:
        raise ValueError("degrees of freedom must be >= 1")
    if df <= len(_T_975):
        return _T_975[df - 1]
    z = _Z_975
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def t_interval(values: np.ndarray) -> tuple[float, float]:
    """Two-sided 95% t-interval of the mean over ``values`` (one value per training seed).

    mean +/- t(0.975, n-1) * sd / sqrt(n). NaN values are dropped; with fewer than 2 values there
    is no interval (NaN, NaN).
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    half = t_quantile_975(n - 1) * float(values.std(ddof=1)) / math.sqrt(n)
    mean = float(values.mean())
    return mean - half, mean + half


def across_seeds_table(per_model: pd.DataFrame) -> pd.DataFrame:
    """Level 2: mean over training seeds and 95% t-interval per (algorithm, train_reward, condition).

    Returns a long table with the columns ``algorithm, train_reward, condition, metric, mean,
    ci_low, ci_high, n_seeds``. The per-seed values are the rows of ``per_model`` (level 1), kept
    for plotting individual runs.
    """
    metrics = [c for c in per_model.columns if c not in MODEL_KEYS and c != "n_episodes"]
    rows = []
    for key, df in per_model.groupby(GROUP_KEYS, sort=True):
        for metric in metrics:
            vals = df[metric].to_numpy(dtype=float)
            lo, hi = t_interval(vals)
            valid = ~np.isnan(vals)
            rows.append(
                {
                    **dict(zip(GROUP_KEYS, key)),
                    "metric": metric,
                    "mean": float(vals[valid].mean()) if valid.any() else float("nan"),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_seeds": int(valid.sum()),
                }
            )
    return pd.DataFrame(rows)


def aggregate(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both levels at once: ``(per_model, across_seeds)``; ``per_model`` holds the per-seed values."""
    per_model = per_model_table(episodes)
    return per_model, across_seeds_table(per_model)
