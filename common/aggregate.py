"""Two-level aggregation of test results (see ``docs/evaluation_protocol.md``).

Level 1, per model: one model = (algorithm, train reward, variant, training seed); its metrics under
one condition are the mean over its test episodes (100 on ``TEST_SEEDS``). ``per_model_table``.
``variant`` is "main" for the main grid and the sensitivity-run name otherwise (``common.paths``).

Level 2, per (algorithm, train reward, variant, condition): mean over the training seeds with a two-sided
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

Training curves (``val_log.csv``, one file per run): ``steps_to_threshold`` (env steps and training
wall time until each pre-registered level of ``TARGET_PERFORMANCE`` holds at ``TARGET_CONSECUTIVE`` validation points in a row), ``learning_curve_table``
(mean and 95% t-interval over seeds per validation point) and ``curve_auc`` /
``auc_across_seeds`` (area under the learning curve). Points are aligned across seeds by validation
index, not by env step (PPO's validation steps differ slightly between runs).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from common.config import TARGET_CONSECUTIVE, TARGET_PERFORMANCE
from common.metrics import add_return_columns, summarize_metrics
from common.paths import MAIN_VARIANT

MODEL_KEYS = ["algorithm", "train_reward", "variant", "train_seed", "condition"]
GROUP_KEYS = ["algorithm", "train_reward", "variant", "condition"]

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
    ``train_reward``, ``variant``, ``train_seed`` and ``condition`` added
    (``common.paths.load_eval_episodes``); a missing ``variant`` column is read as "main".
    """
    if "variant" not in episodes:
        episodes = episodes.assign(variant=MAIN_VARIANT)
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
    """Level 2: mean over training seeds and 95% t-interval per (algorithm, train_reward, variant, condition).

    Returns a long table with the columns ``algorithm, train_reward, variant, condition, metric, mean,
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


# ---------------------------------------------------------------------------------------------
# Training curves (val_log.csv)
# ---------------------------------------------------------------------------------------------
_OPERATORS = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
}
_CURVE_COLUMNS = ["val_idx", "env_step_mean", "env_step_min", "env_step_max", "mean", "ci_low", "ci_high", "n_seeds"]


def _first_run_of_holds(val_log: pd.DataFrame, conditions: Sequence[Mapping[str, Any]], consecutive: int) -> dict[str, float]:
    ok = np.ones(len(val_log), dtype=bool)
    for cond in conditions:
        metric, op, value = cond["metric"], cond["operator"], float(cond["value"])
        if op not in _OPERATORS:
            raise ValueError(f"operator must be one of {list(_OPERATORS)}, got {op!r}")
        if metric not in val_log:
            raise ValueError(f"val_log has no column {metric!r}")
        ok &= _OPERATORS[op](val_log[metric].to_numpy(dtype=float), value)  # NaN compares False
    run = 0
    for i, hit in enumerate(ok):
        run = run + 1 if hit else 0
        if run >= consecutive:
            first = i - consecutive + 1
            row = val_log.iloc[first]
            return {
                "env_step": float(row["env_step"]),
                "train_wall_s": float(row["train_wall_s"]) if "train_wall_s" in val_log else math.nan,
                "val_idx": float(row["val_idx"]) if "val_idx" in val_log else float(first),
            }
    return {"env_step": math.nan, "train_wall_s": math.nan, "val_idx": math.nan}


def steps_to_threshold(
    val_log: pd.DataFrame,
    threshold: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]] = TARGET_PERFORMANCE,
    consecutive: Optional[int] = None,
) -> dict[str, Any]:
    """Env steps and training time until the target holds at ``consecutive`` validation points in a row.

    Args:
        val_log: ``val_log.csv`` of one run (columns ``env_step``, ``train_wall_s`` and the metrics).
        threshold: by default ``TARGET_PERFORMANCE``, a dict of named levels, each a list of conditions
            ``{"metric", "operator": "<=" | ">=" | "<" | ">", "value"}`` that must ALL hold at the same
            validation point; every level is evaluated. A single list of conditions (or one condition dict) is
            evaluated as one unnamed level.
        consecutive: Number of validation points IN A ROW at which all conditions must hold (default
            ``TARGET_CONSECUTIVE``); the reported point is the first of them. Validation uses few episodes, so a
            single lucky point (``consecutive=1``) must not count as "target reached".

    Returns:
        For named levels ``{level: {"env_step", "train_wall_s", "val_idx"}}``, otherwise the inner dict alone.
        The values are those of the FIRST point of the qualifying run; all NaN if the level is never reached.
        ``train_wall_s`` is the training wall time WITHOUT validation time (computational cost).
    """
    consecutive = TARGET_CONSECUTIVE if consecutive is None else consecutive
    if consecutive < 1:
        raise ValueError("consecutive must be >= 1")
    if isinstance(threshold, Mapping) and "metric" not in threshold:  # named levels
        if not threshold:
            raise ValueError("threshold has no levels")
        return {level: _first_run_of_holds(val_log, conds, consecutive) for level, conds in threshold.items()}
    conditions = [threshold] if isinstance(threshold, Mapping) else list(threshold)
    if not conditions:
        raise ValueError("threshold has no conditions")
    return _first_run_of_holds(val_log, conditions, consecutive)


def _as_log_list(val_logs: Union[Mapping[Any, pd.DataFrame], Sequence[pd.DataFrame]]) -> list[pd.DataFrame]:
    return list(val_logs.values()) if isinstance(val_logs, Mapping) else list(val_logs)


def learning_curve_table(
    val_logs: Union[Mapping[Any, pd.DataFrame], Sequence[pd.DataFrame]],
    metric: str = "episode_return",
) -> pd.DataFrame:
    """Mean and 95% t-interval over seeds of ``metric`` at every validation point.

    ``val_logs`` holds one ``val_log.csv`` frame per training seed (``common.paths.load_val_logs``).
    Points are aligned by validation index (``val_idx`` if present, else the row order): the k-th
    validation of every seed is one point, even if the actual env steps differ slightly (PPO validates
    at the first update boundary after each multiple of the interval). Returns one row per point with
    ``val_idx, env_step_mean, env_step_min, env_step_max, mean, ci_low, ci_high, n_seeds``; a seed
    with fewer points (unfinished run) contributes to fewer rows.
    """
    frames = []
    for df in _as_log_list(val_logs):
        if metric not in df:
            raise ValueError(f"val_log has no column {metric!r}")
        idx = df["val_idx"].to_numpy() if "val_idx" in df else np.arange(len(df))
        frames.append(
            pd.DataFrame(
                {"val_idx": idx, "env_step": df["env_step"].to_numpy(), "value": df[metric].to_numpy(dtype=float)}
            )
        )
    if not frames:
        return pd.DataFrame(columns=_CURVE_COLUMNS)
    rows = []
    for idx, g in pd.concat(frames, ignore_index=True).groupby("val_idx", sort=True):
        vals = g["value"].to_numpy(dtype=float)
        lo, hi = t_interval(vals)
        valid = ~np.isnan(vals)
        rows.append(
            {
                "val_idx": int(idx),
                "env_step_mean": float(g["env_step"].mean()),
                "env_step_min": float(g["env_step"].min()),
                "env_step_max": float(g["env_step"].max()),
                "mean": float(vals[valid].mean()) if valid.any() else math.nan,
                "ci_low": lo,
                "ci_high": hi,
                "n_seeds": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows, columns=_CURVE_COLUMNS)


def curve_auc(val_log: pd.DataFrame, metric: str = "episode_return", normalize: bool = True) -> float:
    """Area under the learning curve of one run: trapezoid rule over ``env_step``.

    The curve starts at the first validation point (no extrapolation back to step 0). With
    ``normalize=True`` (default) the area is divided by the step span, i.e. it is the average value
    of the metric over training, in the units of the metric. NaN with fewer than two points.
    """
    if metric not in val_log:
        raise ValueError(f"val_log has no column {metric!r}")
    x = val_log["env_step"].to_numpy(dtype=float)
    y = val_log[metric].to_numpy(dtype=float)
    keep = ~np.isnan(y)
    x, y = x[keep], y[keep]
    if len(x) < 2 or x[-1] <= x[0]:
        return math.nan
    area = float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))
    return area / float(x[-1] - x[0]) if normalize else area


def auc_across_seeds(
    val_logs: Union[Mapping[Any, pd.DataFrame], Sequence[pd.DataFrame]],
    metric: str = "episode_return",
    normalize: bool = True,
) -> dict[str, float]:
    """Per-seed area under the learning curve, then mean and 95% t-interval over seeds."""
    aucs = np.array([curve_auc(df, metric, normalize) for df in _as_log_list(val_logs)], dtype=float)
    lo, hi = t_interval(aucs)
    valid = ~np.isnan(aucs)
    return {
        "mean": float(aucs[valid].mean()) if valid.any() else math.nan,
        "ci_low": lo,
        "ci_high": hi,
        "n_seeds": int(valid.sum()),
    }
