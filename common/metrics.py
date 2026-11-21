"""Per-episode metrics: recording during an episode, return recomputation, summaries.

Per-episode columns (one row per episode, see ``EpisodeRecorder.finish``):
    seed             episode seed (``env.reset(seed=seed)``)
    episode_return   undiscounted return under the reward preset the env was built with
    return_env       undiscounted return of the environment's own (normalized) reward
    sum_high_speed   sum over steps of the high_speed reward component (in [0, 1] per step)
    sum_right_lane   sum over steps of the right_lane reward component (in [0, 1] per step)
    lane_changes     number of steps with a real lane change (sum of ``lane_changed``)
    crashed          ``info["crashed"]`` at the end of the episode
    completed        time limit reached without a crash (truncated and not crashed)
    length           number of steps
    mean_speed       mean ego speed over the steps [m/s]
    distance_m       ego x position at the end minus at the start [m]
    crashes_per_km   crashed / distance_km (NaN if the ego did not move forward)
    min_gap          minimum bumper-to-bumper gap to the leader in the ego lane [m] (NaN: no leader)
    min_ttc          minimum time-to-collision to that leader [s] (NaN: never closing in)
    near_miss        not crashed and min_ttc < NEAR_MISS_TTC

Gap and TTC are evaluated on the state after every step (and on the initial state). The leader is
the nearest vehicle ahead in the ego's current lane; gap = x_leader - x_ego - vehicle length;
TTC = gap / (v_ego - v_leader) while the ego is closing in (v_ego > v_leader), else undefined.
A gap of zero or less counts as TTC = 0.

The four ingredients ``sum_high_speed``, ``sum_right_lane``, ``lane_changes`` and ``crashed``
determine the return under every reward preset (``recompute_return``), so one evaluation run
serves all presets.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

from common.config import REWARDS

NEAR_MISS_TTC = 1.5  # [s] near miss = not crashed and min TTC below this


def leader_gap_ttc(
    ego_x: float, ego_vx: float, lead_x: float, lead_vx: float, length: float
) -> tuple[float, float]:
    """Bumper-to-bumper gap and time-to-collision to a leader (pure function).

    Returns ``(gap, ttc)``; ``ttc`` is NaN when the ego is not closing in on the leader and 0 when
    the gap is zero or negative.
    """
    gap = lead_x - ego_x - length
    closing = ego_vx - lead_vx
    if gap <= 0:
        return max(gap, 0.0), 0.0
    if closing > 1e-6:
        return gap, gap / closing
    return gap, math.nan


class EpisodeRecorder:
    """Accumulates all per-episode metrics in one pass over the steps (no extra simulation).

    Create it right after ``env.reset`` and call ``step`` after every ``env.step``.
    """

    def __init__(self, env: Any) -> None:
        self.u = env.unwrapped
        self.ego = self.u.vehicle
        self.x0 = float(self.ego.position[0])
        self.n = 0
        self.ret = 0.0
        self.ret_env = 0.0
        self.speed_sum = 0.0
        self.sum_hs = 0.0
        self.sum_rl = 0.0
        self.lane_changes = 0
        self.min_gap = math.inf
        self.min_ttc = math.inf
        self._observe_leader()  # state right after reset

    def _observe_leader(self) -> None:
        ego = self.ego
        front, _ = self.u.road.neighbour_vehicles(ego, ego.lane_index)
        if front is None:
            return
        length = 0.5 * (ego.LENGTH + front.LENGTH)
        gap, ttc = leader_gap_ttc(
            float(ego.position[0]),
            float(ego.velocity[0]),
            float(front.position[0]),
            float(front.velocity[0]),
            length,
        )
        self.min_gap = min(self.min_gap, gap)
        if not math.isnan(ttc):
            self.min_ttc = min(self.min_ttc, ttc)

    def step(self, reward: float, info: dict[str, Any]) -> None:
        """Record the result of one environment step (call after ``env.step``)."""
        self.n += 1
        self.ret += float(reward)
        self.ret_env += float(info["env_reward"])
        self.speed_sum += float(info["speed"])
        self.sum_hs += float(info["rewards"]["high_speed_reward"])
        self.sum_rl += float(info["rewards"]["right_lane_reward"])
        self.lane_changes += int(info["lane_changed"])
        self._observe_leader()

    def finish(self, seed: int, truncated: bool, info: dict[str, Any]) -> dict[str, Any]:
        """Return the finished episode as one flat record."""
        crashed = bool(info["crashed"])
        distance = float(self.ego.position[0]) - self.x0
        return {
            "seed": int(seed),
            "episode_return": self.ret,
            "return_env": self.ret_env,
            "sum_high_speed": self.sum_hs,
            "sum_right_lane": self.sum_rl,
            "lane_changes": self.lane_changes,
            "crashed": crashed,
            "completed": bool(truncated) and not crashed,
            "length": self.n,
            "mean_speed": self.speed_sum / max(self.n, 1),
            "distance_m": distance,
            "crashes_per_km": (crashed / (distance / 1000.0)) if distance > 0 else math.nan,
            "min_gap": math.nan if math.isinf(self.min_gap) else self.min_gap,
            "min_ttc": math.nan if math.isinf(self.min_ttc) else self.min_ttc,
            "near_miss": (not crashed) and self.min_ttc < NEAR_MISS_TTC,
        }


def recompute_return(df: pd.DataFrame, reward_name: str) -> pd.Series:
    """Per-episode return under reward preset ``reward_name``, from the stored components.

    ``r = w_speed * sum_high_speed + w_right * sum_right_lane - w_collision * crashed
    - w_lane * lane_changes`` (an episode ends at its first crash, so the sum of the per-step
    ``crashed`` flags equals ``crashed``). For ``"default"`` the env's own return ``return_env``
    is returned.
    """
    if reward_name not in REWARDS:
        raise ValueError(f"Unknown reward {reward_name!r}; choose from {list(REWARDS)}")
    if reward_name == "default":
        return df["return_env"].astype(float)
    w = REWARDS[reward_name]
    return (
        w["w_speed"] * df["sum_high_speed"]
        + w["w_right"] * df["sum_right_lane"]
        - w["w_collision"] * df["crashed"].astype(float)
        - w["w_lane"] * df["lane_changes"]
    )


def add_return_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``return_<preset>`` column for every reward preset."""
    out = df.copy()
    for name in REWARDS:
        out[f"return_{name}"] = recompute_return(df, name)
    return out


# Columns averaged over episodes -> name in the summary.
_MEAN_COLUMNS = {
    "episode_return": "episode_return",
    "return_env": "return_env",
    "crashed": "crash_rate",
    "completed": "completion_rate",
    "near_miss": "near_miss_rate",
    "length": "length",
    "mean_speed": "mean_speed",
    "distance_m": "distance_m",
    "lane_changes": "lane_changes",
    "min_gap": "min_gap",
    "min_ttc": "min_ttc",
}


def summarize_metrics(df: pd.DataFrame, extra_prefix: Optional[str] = "return_") -> dict[str, float]:
    """Mean metrics over the episodes of ``df`` (one model / one policy under one condition).

    ``crashes_per_km`` is the ratio of sums (total crashes / total km), not the mean of the
    per-episode ratios, which would be dominated by very short episodes. ``min_gap`` and
    ``min_ttc`` are means over the episodes where they are defined. Columns starting with
    ``extra_prefix`` (the recomputed ``return_<preset>`` columns) are averaged as well.
    """
    out: dict[str, float] = {"n_episodes": float(len(df))}
    for col, name in _MEAN_COLUMNS.items():
        if col in df:
            out[name] = float(df[col].astype(float).mean(skipna=True)) if len(df) else math.nan
    km = float(df["distance_m"].sum()) / 1000.0 if "distance_m" in df else 0.0
    out["crashes_per_km"] = float(df["crashed"].astype(float).sum()) / km if km > 0 else math.nan
    if extra_prefix:
        for col in df.columns:
            if col.startswith(extra_prefix) and col not in out:
                out[col] = float(df[col].astype(float).mean())
    return out
