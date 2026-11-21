"""Per-episode metrics: recording during an episode, return recomputation, summaries.

The same ``EpisodeRecorder`` serves evaluation (``common.evaluate``) and training
(``common.train_env``), so a training-episode row and a test-episode row are computed by identical
code. Per-episode columns (one row per episode, see ``EpisodeRecorder.finish``, order in
``EPISODE_COLUMNS``):

    seed             episode seed (``env.reset(seed=seed)``); -1 for training episodes (unseeded resets)
    episode_return   undiscounted return under the reward preset the env was built with
    return_env       undiscounted return of the environment's own (normalized) reward
    sum_high_speed   sum over steps of the high_speed reward component (in [0, 1] per step)
    sum_right_lane   sum over steps of the right_lane reward component (in [0, 1] per step)
    lane_changes     number of steps with a real lane change (sum of ``lane_changed``)
    crashed          ``info["crashed"]`` at the end of the episode
    truncated        the time limit ended the episode (``truncated`` of the last step)
    completed        time limit reached without a crash (truncated and not crashed)
    length           number of steps
    mean_speed       mean ego speed over the steps [m/s]
    speed_std        standard deviation (ddof = 0) of the ego speed over the steps [m/s]
    distance_m       ego x position at the end minus at the start [m]
    crashes_per_km   crashed / distance_km (NaN if the ego did not move forward)
    min_gap          minimum bumper-to-bumper gap to the leader in the ego lane [m] (NaN: no leader)
    min_ttc          minimum time-to-collision to that leader [s] (NaN: never closing in)
    near_miss        not crashed and min_ttc < NEAR_MISS_TTC
    near_miss_steps  steps whose state has TTC < NEAR_MISS_TTC (the final crash step is not counted)
    act_<NAME>       how often the agent chose each meta-action (LANE_LEFT, IDLE, LANE_RIGHT, FASTER,
                     SLOWER); the five counts sum to ``length``
    lane<i>_steps    steps spent in lane i after the step (0 = leftmost); sum to ``length``
    rightmost_lane_share  lane<N-1>_steps / length
    lane_change_reversals lane changes that undo the immediately preceding lane change within
                     REVERSAL_WINDOW steps (A -> B, then B -> A)
    n_speed_changes  steps in which the ego's target speed changed (FASTER / SLOWER that took effect)
    overtakes        vehicles that were ahead of the ego at the start and are behind it at the end
                     (any lane; net passes: a vehicle that passes the ego again is not counted). Compare
                     overtakes per km (``overtakes_per_km`` in the summaries), not per episode.

Gap and TTC are evaluated on the state after every step (and on the initial state, for ``min_gap``
and ``min_ttc``). The leader is the nearest vehicle ahead in the ego's current lane;
gap = x_leader - x_ego - vehicle length; TTC = gap / (v_ego - v_leader) while the ego is closing in
(v_ego > v_leader), else undefined. A gap of zero or less counts as TTC = 0. With
``with_leader=False`` (cheaper) the four leader columns ``min_gap``, ``min_ttc``, ``near_miss`` and
``near_miss_steps`` are NaN.

``overtakes`` counts passed vehicles, not manoeuvres: the road is never emptied of vehicles in
highway-env 1.12.1 and in all densities every other vehicle spawns ahead of the ego, so the
set "ahead at the start" is well defined and the count is cheap (two comparisons per vehicle, at the
start and at the end only). It does not distinguish an active overtake from a vehicle that merely
changed lane.

The four ingredients ``sum_high_speed``, ``sum_right_lane``, ``lane_changes`` and ``crashed``
determine the return under every reward preset (``recompute_return``), so one evaluation run
serves all presets.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

from common.actions import ACTION_NAMES
from common.config import BASE_CONFIG, REWARDS, TARGET_PERFORMANCE

NEAR_MISS_TTC = 1.5  # [s] near miss = not crashed and min TTC below this
# A lane change that undoes the previous one within this many steps (= seconds at policy_frequency
# 1) counts as a reversal. Fixed before training; not to be changed after seeing results.
REVERSAL_WINDOW = 3


def episode_columns(n_lanes: int) -> list[str]:
    """Ordered keys of ``EpisodeRecorder.finish`` without ``seed`` (depends on the lane count)."""
    return (
        [
            "episode_return", "return_env", "sum_high_speed", "sum_right_lane", "lane_changes",
            "crashed", "truncated", "completed", "length", "mean_speed", "speed_std", "distance_m",
            "crashes_per_km", "min_gap", "min_ttc", "near_miss", "near_miss_steps",
        ]
        + [f"act_{name}" for name in ACTION_NAMES]
        + [f"lane{i}_steps" for i in range(n_lanes)]
        + ["rightmost_lane_share", "lane_change_reversals", "n_speed_changes", "overtakes"]
    )


# Columns of every episode record under the shared BASE_CONFIG (3 lanes).
EPISODE_COLUMNS = episode_columns(int(BASE_CONFIG["lanes_count"]))


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

    Create it right after ``env.reset`` and call ``step`` after every ``env.step``. One instance can
    be reused: call ``start()`` after every ``env.reset`` (the training wrapper does this).

    Args:
        env: The (wrapped) env; only ``env.unwrapped`` is used.
        with_leader: Also track the leader gap / TTC (the costly part: one neighbour search per
            step). ``False`` leaves ``min_gap``, ``min_ttc``, ``near_miss``, ``near_miss_steps`` NaN.
    """

    def __init__(self, env: Any, with_leader: bool = True) -> None:
        self.u = env.unwrapped
        self.with_leader = with_leader
        self.n_lanes = int(self.u.config["lanes_count"])
        self.start()

    def start(self) -> None:
        """(Re-)arm the recorder on the state right after ``env.reset``."""
        self.ego = self.u.vehicle
        self.x0 = float(self.ego.position[0])
        self.n = 0
        self.ret = 0.0
        self.ret_env = 0.0
        self.speed_mean = 0.0  # Welford running mean / sum of squared deviations of the speed
        self.speed_m2 = 0.0
        self.sum_hs = 0.0
        self.sum_rl = 0.0
        self.lane_changes = 0
        self.min_gap = math.inf
        self.min_ttc = math.inf
        self.near_miss_steps = 0
        self.act_counts = [0] * len(ACTION_NAMES)
        self.lane_steps = [0] * self.n_lanes
        self.reversals = 0
        self.n_speed_changes = 0
        self._prev_lane = int(self.ego.lane_index[2])
        self._last_change: Optional[tuple[int, int, int]] = None  # (step, from lane, to lane)
        self._prev_target_speed = float(self.ego.target_speed)
        self._ahead = [v for v in self.u.road.vehicles if v is not self.ego and v.position[0] > self.x0]
        if self.with_leader:
            self._observe_leader()  # state right after reset

    def _observe_leader(self) -> float:
        """Update ``min_gap`` / ``min_ttc`` from the current state; return the TTC (NaN if undefined)."""
        ego = self.ego
        front, _ = self.u.road.neighbour_vehicles(ego, ego.lane_index)
        if front is None:
            return math.nan
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
        return ttc

    def step(self, reward: float, info: dict[str, Any], action: int) -> None:
        """Record the result of one environment step (call after ``env.step(action)``)."""
        self.n += 1
        self.ret += float(reward)
        self.ret_env += float(info["env_reward"])
        speed = float(info["speed"])
        delta = speed - self.speed_mean
        self.speed_mean += delta / self.n
        self.speed_m2 += delta * (speed - self.speed_mean)
        self.sum_hs += float(info["rewards"]["high_speed_reward"])
        self.sum_rl += float(info["rewards"]["right_lane_reward"])
        self.act_counts[int(action)] += 1

        lane = min(max(int(self.ego.lane_index[2]), 0), self.n_lanes - 1)
        self.lane_steps[lane] += 1
        if info["lane_changed"]:
            self.lane_changes += 1
            last = self._last_change
            if last is not None and self.n - last[0] <= REVERSAL_WINDOW and lane == last[1]:
                self.reversals += 1
            self._last_change = (self.n, self._prev_lane, lane)
        self._prev_lane = lane

        target_speed = float(self.ego.target_speed)
        if target_speed != self._prev_target_speed:
            self.n_speed_changes += 1
            self._prev_target_speed = target_speed

        if self.with_leader:
            ttc = self._observe_leader()
            if ttc < NEAR_MISS_TTC and not info["crashed"]:  # NaN compares False
                self.near_miss_steps += 1

    def finish(self, seed: int, truncated: bool, info: dict[str, Any]) -> dict[str, Any]:
        """Return the finished episode as one flat record (keys: ``seed`` + ``EPISODE_COLUMNS``)."""
        crashed = bool(info["crashed"])
        x_end = float(self.ego.position[0])
        distance = x_end - self.x0
        n = max(self.n, 1)
        leader = self.with_leader
        rec: dict[str, Any] = {
            "seed": int(seed),
            "episode_return": self.ret,
            "return_env": self.ret_env,
            "sum_high_speed": self.sum_hs,
            "sum_right_lane": self.sum_rl,
            "lane_changes": self.lane_changes,
            "crashed": crashed,
            "truncated": bool(truncated),
            "completed": bool(truncated) and not crashed,
            "length": self.n,
            "mean_speed": self.speed_mean if self.n else 0.0,
            "speed_std": math.sqrt(max(self.speed_m2, 0.0) / n),
            "distance_m": distance,
            "crashes_per_km": (crashed / (distance / 1000.0)) if distance > 0 else math.nan,
            "min_gap": math.nan if (not leader or math.isinf(self.min_gap)) else self.min_gap,
            "min_ttc": math.nan if (not leader or math.isinf(self.min_ttc)) else self.min_ttc,
            "near_miss": ((not crashed) and self.min_ttc < NEAR_MISS_TTC) if leader else math.nan,
            "near_miss_steps": self.near_miss_steps if leader else math.nan,
        }
        for name, count in zip(ACTION_NAMES, self.act_counts):
            rec[f"act_{name}"] = count
        for i, count in enumerate(self.lane_steps):
            rec[f"lane{i}_steps"] = count
        rec["rightmost_lane_share"] = self.lane_steps[-1] / n
        rec["lane_change_reversals"] = self.reversals
        rec["n_speed_changes"] = self.n_speed_changes
        rec["overtakes"] = sum(1 for v in self._ahead if v.position[0] < x_end)
        return rec


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


# Columns averaged over episodes -> name in the summary. Counts (act_*, lane*_steps, ...) are
# means of per-episode counts, so they depend on the episode length; divide by ``length`` for shares.
_MEAN_COLUMNS = {
    "episode_return": "episode_return",
    "return_env": "return_env",
    "crashed": "crash_rate",
    "completed": "completion_rate",
    "near_miss": "near_miss_rate",
    "length": "length",
    "mean_speed": "mean_speed",
    "speed_std": "speed_std",
    "distance_m": "distance_m",
    "lane_changes": "lane_changes",
    "min_gap": "min_gap",
    "min_ttc": "min_ttc",
    "near_miss_steps": "near_miss_steps",
    **{f"act_{name}": f"act_{name}" for name in ACTION_NAMES},
    **{f"lane{i}_steps": f"lane{i}_steps" for i in range(int(BASE_CONFIG["lanes_count"]))},
    "rightmost_lane_share": "rightmost_lane_share",
    "lane_change_reversals": "lane_change_reversals",
    "n_speed_changes": "n_speed_changes",
    "overtakes": "overtakes",
}

# Keys of ``summarize_metrics`` without the return_<preset> columns, in output order.
SUMMARY_COLUMNS = ["n_episodes", *_MEAN_COLUMNS.values(), "crashes_per_km", "overtakes_per_km"]

# Structure of common.config.TARGET_PERFORMANCE (checked at import, below).
TARGET_OPERATORS = ("<=", ">=", "<", ">")
TARGET_SAFETY_METRICS = ("crash_rate", "completion_rate", "near_miss_rate", "crashes_per_km")
TARGET_EFFICIENCY_METRICS = ("mean_speed", "distance_m", "overtakes_per_km")


def check_target_performance(target: Any) -> None:
    """Validate a target: a list of ``{"metric", "operator", "value"}`` conditions that must all hold.

    Raises ``ValueError`` for unknown metric names (not in ``SUMMARY_COLUMNS``), unknown operators,
    non-numeric values, and a target without at least one safety and one efficiency condition (a single
    kind could be met by a trivial policy: braking for safety, ignoring traffic for speed).
    """
    if not isinstance(target, (list, tuple)) or not target:
        raise ValueError("TARGET_PERFORMANCE must be a non-empty list of conditions")
    for cond in target:
        if set(cond) != {"metric", "operator", "value"}:
            raise ValueError(f"target condition needs the keys metric, operator, value: {cond}")
        if cond["metric"] not in SUMMARY_COLUMNS:
            raise ValueError(f"unknown target metric {cond['metric']!r}; known: {SUMMARY_COLUMNS}")
        if cond["operator"] not in TARGET_OPERATORS:
            raise ValueError(f"unknown target operator {cond['operator']!r}; known: {TARGET_OPERATORS}")
        if isinstance(cond["value"], bool) or not isinstance(cond["value"], (int, float)):
            raise ValueError(f"target value must be a number: {cond}")
    names = {c["metric"] for c in target}
    if not names & set(TARGET_SAFETY_METRICS) or not names & set(TARGET_EFFICIENCY_METRICS):
        raise ValueError(
            f"TARGET_PERFORMANCE needs a safety metric {TARGET_SAFETY_METRICS} and an efficiency metric "
            f"{TARGET_EFFICIENCY_METRICS}, got {sorted(names)}"
        )


check_target_performance(TARGET_PERFORMANCE)


def summarize_metrics(df: pd.DataFrame, extra_prefix: Optional[str] = "return_") -> dict[str, float]:
    """Mean metrics over the episodes of ``df`` (one model / one policy under one condition).

    ``crashes_per_km`` and ``overtakes_per_km`` are ratios of sums (total crashes or overtakes / total
    km), not means of per-episode ratios, which would be dominated by very short episodes. Overtakes are
    compared per km because the per-episode count grows with speed and survival time. ``min_gap`` and
    ``min_ttc`` are means over the episodes where they are defined. Columns starting with
    ``extra_prefix`` (the recomputed ``return_<preset>`` columns) are averaged as well.
    """
    out: dict[str, float] = {"n_episodes": float(len(df))}
    for col, name in _MEAN_COLUMNS.items():
        if col in df:
            out[name] = float(df[col].astype(float).mean(skipna=True)) if len(df) else math.nan
    km = float(df["distance_m"].sum()) / 1000.0 if "distance_m" in df else 0.0
    out["crashes_per_km"] = float(df["crashed"].astype(float).sum()) / km if km > 0 else math.nan
    out["overtakes_per_km"] = (
        float(df["overtakes"].astype(float).sum()) / km if (km > 0 and "overtakes" in df) else math.nan
    )
    if extra_prefix:
        for col in df.columns:
            if col.startswith(extra_prefix) and col not in out:
                out[col] = float(df[col].astype(float).mean())
    return out
