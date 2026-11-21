"""Pick episodes by outcome from the per-episode test CSVs, so failure cases can be rendered as videos.

The test CSVs (``results/eval/<algo>/<reward>/<variant>/seed<s>/<condition>.csv``) have one row per
``TEST_SEEDS`` episode; ``env.reset(seed=seed)`` reproduces the episode exactly, so a seed is all
``scripts/record_video.py`` needs (``--from-eval-csv <csv> --outcome crashed --n 3``).

Outcomes (as defined in ``common.metrics``): ``crashed``, ``near_miss`` (not crashed, min TTC below
``NEAR_MISS_TTC``) and ``completed`` (time limit reached without a crash). Sorted by return: the worst
episodes first for ``crashed`` and ``near_miss`` (lowest return), the best first for ``completed``.

``video_episode_seeds`` applies the shared video rule: the same crashed / completed / disagreement seeds for both
algorithms.
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


# ---------------------------------------------------------------------------------------------
# Video episodes shared by both algorithms (docs/evaluation_protocol.md section 11)
# ---------------------------------------------------------------------------------------------
VIDEO_CATEGORIES = ("both_crashed", "both_completed", "ddqn_crashed_only", "ppo_crashed_only")


def _outcomes(episodes: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
    df = pd.read_csv(episodes) if isinstance(episodes, (str, Path)) else episodes
    if "seed" not in df or "crashed" not in df:
        raise ValueError("episodes need the columns 'seed' and 'crashed'")
    out = pd.DataFrame({"seed": df["seed"].astype(int), "crashed": df["crashed"].fillna(False).astype(bool)})
    out["completed"] = df["completed"].fillna(False).astype(bool) if "completed" in df else ~out["crashed"]
    return out.drop_duplicates("seed").set_index("seed").sort_index()


def video_episode_seeds(
    ddqn_episodes: Union[pd.DataFrame, str, Path], ppo_episodes: Union[pd.DataFrame, str, Path]
) -> dict[str, Optional[int]]:
    """Candidate video seeds for one (reward, training seed) from the ``main`` test results of the two FINAL models.

    Returns the LOWEST seed (deterministic, not chosen by looking at videos) of each category, ``None`` if there is none:

    - ``both_crashed``: both models crashed on the seed;
    - ``both_completed``: both models reached the time limit;
    - ``ddqn_crashed_only``: DDQN crashed, PPO did not (disagreement);
    - ``ppo_crashed_only``: PPO crashed, DDQN did not (disagreement).

    The two inputs (per-episode frames or CSV paths) must cover the same seeds.
    """
    d, p = _outcomes(ddqn_episodes), _outcomes(ppo_episodes)
    if not d.index.equals(p.index):
        raise ValueError("the two models were evaluated on different seeds; the video rule needs identical TEST_SEEDS")
    masks = {
        "both_crashed": d["crashed"] & p["crashed"],
        "both_completed": d["completed"] & p["completed"],
        "ddqn_crashed_only": d["crashed"] & ~p["crashed"],
        "ppo_crashed_only": ~d["crashed"] & p["crashed"],
    }
    return {name: (int(d.index[mask.to_numpy()].min()) if mask.any() else None) for name, mask in masks.items()}
