"""Pick episodes by outcome from the per-episode test CSVs, so failure cases can be rendered as videos.

The test CSVs (``results/eval/<algo>/<reward>/<variant>/seed<s>/<condition>.csv``) have one row per
``TEST_SEEDS`` episode; ``env.reset(seed=seed)`` reproduces the episode exactly, so a seed is all
``scripts/record_video.py`` needs (``--from-eval-csv <csv> --outcome crashed --n 3``).

Outcomes (as defined in ``common.metrics``): ``crashed``, ``near_miss`` (not crashed, min TTC below
``NEAR_MISS_TTC``) and ``completed`` (time limit reached without a crash). Sorted by return: the worst
episodes first for ``crashed`` and ``near_miss`` (lowest return), the best first for ``completed``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

OUTCOMES = ("crashed", "near_miss", "completed")
_SHOWN = ["seed", "episode_return", "length", "crashed", "near_miss", "completed", "mean_speed", "min_ttc", "min_gap"]


def list_episodes_by_outcome(
    episodes: Union[pd.DataFrame, str, Path],
    outcome: str,
    n: Optional[int] = None,
    sort_by: str = "episode_return",
    ascending: Optional[bool] = None,
) -> pd.DataFrame:
    """Episodes with the given outcome, sorted by ``sort_by`` (default: the return), at most ``n``.

    Args:
        episodes: per-episode frame or the path of a test CSV (one model and condition).
        outcome: one of ``OUTCOMES``.
        n: keep the first ``n`` rows after sorting (``None`` = all).
        sort_by: column to sort on.
        ascending: sort direction; default lowest first for ``crashed`` / ``near_miss`` and highest first for ``completed``.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    df = pd.read_csv(episodes) if isinstance(episodes, (str, Path)) else episodes
    if outcome not in df:
        raise ValueError(f"episodes have no column {outcome!r}")
    mask = df[outcome].fillna(False).astype(bool)
    if ascending is None:
        ascending = outcome != "completed"
    out = df[mask].sort_values([sort_by, "seed"], ascending=[ascending, True], kind="stable")
    if n is not None:
        out = out.head(n)
    return out[[c for c in _SHOWN if c in out.columns or c == sort_by]].reset_index(drop=True)


def seeds_by_outcome(episodes: Union[pd.DataFrame, str, Path], outcome: str, n: Optional[int] = None, **kwargs) -> list[int]:
    """Episode seeds (``env.reset(seed=...)``) of ``list_episodes_by_outcome``."""
    return [int(s) for s in list_episodes_by_outcome(episodes, outcome, n, **kwargs)["seed"]]
