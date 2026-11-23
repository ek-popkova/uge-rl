"""Pure statistics helpers of the calibration analysis (tested on hand-built cases in tests/test_calibration.py)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

# t(0.975, 4): half-width factor of the 95% t-interval over 5 seeds is t / sqrt(5)
T975_DF4 = 2.7764451051977987


def plateau_point(steps: Sequence[float], values: Sequence[float], window: int = 3, tol: float = 0.05,
                  tail_frac: float = 0.2) -> dict:
    """Plateau of a learning curve.

    ``target`` = mean of the values in the last ``tail_frac`` of the run (env_step >= (1 - tail_frac) * last step).
    ``roll[i]`` = mean of the ``window`` values ending at point i (trailing). The plateau point is the EARLIEST point i
    such that ``|roll[j] - target| <= tol * |target|`` for every j >= i. ``plateaued`` is False if that point lies inside the
    tail window itself (no evidence of a plateau before the last 20% of the run) or does not exist (including the case
    of a target so close to 0 that the tolerance is empty).
    """
    steps, values = np.asarray(steps, dtype=float), np.asarray(values, dtype=float)
    n = len(values)
    if n < window + 1:
        return {"plateau_step": math.nan, "plateau_idx": -1, "target": math.nan, "plateaued": False}
    tail = steps >= (1 - tail_frac) * steps[-1]
    target = float(values[tail].mean())
    roll = np.full(n, np.nan)
    for i in range(window - 1, n):
        roll[i] = values[i - window + 1 : i + 1].mean()
    ok = np.abs(roll - target) <= tol * abs(target)
    idx = -1
    for i in range(n - 1, window - 2, -1):  # walk back while the condition keeps holding
        if ok[i]:
            idx = i
        else:
            break
    if idx < 0:
        return {"plateau_step": math.nan, "plateau_idx": -1, "target": target, "plateaued": False}
    step = float(steps[idx])
    return {"plateau_step": step, "plateau_idx": idx, "target": target, "plateaued": step <= (1 - tail_frac) * steps[-1]}


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(min(k, n) + 1)))


def pass_probability(p_true: float, threshold: float, n_episodes: int = 20, consecutive: int = 1) -> float:
    """Probability that a policy with true crash probability ``p_true`` shows ``crash_rate <= threshold`` at
    ``consecutive`` independent validation points in a row (n_episodes each; crash rate only, speed ignored)."""
    single = binom_cdf(math.floor(threshold * n_episodes + 1e-9), n_episodes, p_true)
    return single**consecutive


def val_noise(episodes: pd.DataFrame, last_k: int = 5, ns: Sequence[int] = (20, 30, 50)) -> dict:
    """Standard error of the validation crash rate and mean speed at the last ``last_k`` validation points.

    ``episodes`` = ``val_episodes.csv`` (columns ``val_idx``, ``crashed``, ``mean_speed``). The crash-rate SE uses the pooled
    crash probability of the last points (binomial, sqrt(p (1 - p) / n)); the mean-speed SE is the mean over the points of
    sd(episode mean speed) / sqrt(n). Values for other episode counts follow the sqrt scaling sqrt(n0 / n).
    """
    last = sorted(episodes["val_idx"].unique())[-last_k:]
    sub = episodes[episodes["val_idx"].isin(last)]
    n0 = int(sub.groupby("val_idx").size().iloc[0])
    p = float(sub["crashed"].astype(float).mean())
    sd_speed = float(sub.groupby("val_idx")["mean_speed"].std(ddof=1).mean())
    out = {"n_points": len(last), "n_episodes": n0, "p_crash": p, "sd_speed": sd_speed}
    for n in ns:
        out[f"se_crash_{n}"] = math.sqrt(p * (1 - p) / n)
        out[f"se_speed_{n}"] = sd_speed / math.sqrt(n)
    return out


def tail_fluctuation(steps: Sequence[float], values: Sequence[float], tail_frac: float = 0.2) -> float:
    """Sample standard deviation of the validation values in the last ``tail_frac`` of the run."""
    steps, values = np.asarray(steps, dtype=float), np.asarray(values, dtype=float)
    tail = values[steps >= (1 - tail_frac) * steps[-1]]
    return float(tail.std(ddof=1)) if len(tail) > 1 else math.nan


def spearman(x: Sequence[float], y: Sequence[float], min_points: int = 3, pearson: bool = False) -> float:
    """Spearman rank correlation (average ranks for ties; ``pearson=True``: plain Pearson correlation of the values);
    NaN with fewer than ``min_points`` points or when one series is constant."""
    xs, ys = pd.Series(list(x), dtype=float), pd.Series(list(y), dtype=float)
    if len(xs) < min_points:
        return math.nan
    a, b = (xs.to_numpy(), ys.to_numpy()) if pearson else (xs.rank().to_numpy(), ys.rank().to_numpy())
    if a.std() == 0 or b.std() == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def seed_spread(a: float, b: float, n_seeds: int = 5) -> dict:
    """Rough spread from TWO seeds: sd = |a - b| / sqrt(2) (1 degree of freedom), and the implied half-width of the 95%
    t-interval over ``n_seeds`` seeds, t(0.975, n - 1) * sd / sqrt(n)."""
    d = abs(a - b)
    sd = d / math.sqrt(2)
    from common.aggregate import t_quantile_975

    return {"abs_diff": d, "sd": sd, "halfwidth_ci": t_quantile_975(n_seeds - 1) * sd / math.sqrt(n_seeds)}


def round_up(x: float, step: float) -> float:
    return math.ceil(x / step - 1e-9) * step


def round_down(x: float, step: float) -> float:
    return math.floor(x / step + 1e-9) * step
