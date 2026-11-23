"""EpisodeRecorder behavioural metrics: hand-built cases (fake env) and invariants on the real env."""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.actions import ACTION_INDEX, ACTION_NAMES
from common.baselines import make_baseline
from common.config import BASE_CONFIG
from common.envs import make_env, make_test_env
from common.evaluate import evaluate
from common.metrics import (
    EPISODE_COLUMNS,
    NEAR_MISS_TTC,
    REVERSAL_WINDOW,
    SUMMARY_COLUMNS,
    EpisodeRecorder,
    summarize_metrics,
)

N_LANES = int(BASE_CONFIG["lanes_count"])


# ---- a fake env with exactly the attributes EpisodeRecorder reads ---------------------------------------
class FakeVehicle:
    LENGTH = 5.0

    def __init__(self, x, v=25.0, lane=2):
        self.position = np.array([x, 0.0])
        self.velocity = np.array([v, 0.0])
        self.lane_index = ("a", "b", lane)
        self.target_speed = 25.0


class FakeRoad:
    def __init__(self, vehicles):
        self.vehicles = vehicles
        self.leader = None

    def neighbour_vehicles(self, ego, lane_index):
        return self.leader, None


class FakeEnv:
    def __init__(self, others=(), lane=2):
        self.vehicle = FakeVehicle(0.0, lane=lane)
        self.road = FakeRoad([self.vehicle, *others])
        self.config = {"lanes_count": N_LANES}
        self.unwrapped = self


def drive(env, rec, steps):
    """steps: list of dicts with action, lane, speed, target, x, crashed (all optional)."""
    prev_lane = env.vehicle.lane_index[2]
    for s in steps:
        ego = env.vehicle
        lane = s.get("lane", prev_lane)
        ego.lane_index = ("a", "b", lane)
        ego.position = np.array([s.get("x", ego.position[0] + 25.0), 0.0])
        ego.target_speed = s.get("target", ego.target_speed)
        info = {
            "env_reward": 0.0, "speed": s.get("speed", 25.0), "crashed": s.get("crashed", False),
            "rewards": {"high_speed_reward": 0.5, "right_lane_reward": lane / (N_LANES - 1)},
            "lane_changed": int(lane != prev_lane),
        }
        rec.step(0.0, info, s.get("action", ACTION_INDEX["IDLE"]))
        prev_lane = lane
    return rec.finish(-1, truncated=True, info=info)


def test_columns_match_finish_keys():
    env = FakeEnv()
    rec = EpisodeRecorder(env)
    out = drive(env, rec, [{}])
    assert list(out) == ["seed", *EPISODE_COLUMNS]
    assert [c for c in SUMMARY_COLUMNS if c.startswith("act_")] == [f"act_{n}" for n in ACTION_NAMES]
    assert [c for c in EPISODE_COLUMNS if c.startswith("lane") and c.endswith("_steps")] == [
        f"lane{i}_steps" for i in range(N_LANES)
    ]


def test_action_counts_and_lane_occupancy_sum_to_length():
    env = FakeEnv()
    rec = EpisodeRecorder(env)
    steps = [
        {"action": 0, "lane": 1}, {"action": 1, "lane": 1}, {"action": 3, "lane": 1}, {"action": 3},
        {"action": 4}, {"action": 2, "lane": 2}, {"action": 1},
    ]
    out = drive(env, rec, steps)
    assert out["length"] == 7
    assert [out[f"act_{n}"] for n in ACTION_NAMES] == [1, 2, 1, 2, 1]
    assert sum(out[f"act_{n}"] for n in ACTION_NAMES) == out["length"]
    assert [out[f"lane{i}_steps"] for i in range(N_LANES)] == [0, 5, 2]
    assert sum(out[f"lane{i}_steps"] for i in range(N_LANES)) == out["length"]
    assert out["rightmost_lane_share"] == pytest.approx(2 / 7)


def test_lane_change_reversal_window():
    w = REVERSAL_WINDOW
    # A -> B at step 1, B -> A exactly w steps later: undone within the window (counts)
    env = FakeEnv(lane=2)
    out = drive(env, EpisodeRecorder(env), [{"lane": 1}] + [{}] * (w - 1) + [{"lane": 2}])
    assert out["lane_changes"] == 2 and out["lane_change_reversals"] == 1
    # one step later than the window: not a reversal
    env = FakeEnv(lane=2)
    out = drive(env, EpisodeRecorder(env), [{"lane": 1}] + [{}] * w + [{"lane": 2}])
    assert out["lane_changes"] == 2 and out["lane_change_reversals"] == 0
    # a second change in the same direction is not a reversal (2 -> 1 -> 0)
    env = FakeEnv(lane=2)
    out = drive(env, EpisodeRecorder(env), [{"lane": 1}, {"lane": 0}])
    assert out["lane_changes"] == 2 and out["lane_change_reversals"] == 0
    # zig-zag 2 -> 1 -> 2 -> 1: every change undoes the previous one -> 2 reversals
    env = FakeEnv(lane=2)
    out = drive(env, EpisodeRecorder(env), [{"lane": 1}, {"lane": 2}, {"lane": 1}])
    assert out["lane_changes"] == 3 and out["lane_change_reversals"] == 2
    assert REVERSAL_WINDOW == 3  # fixed before training; change only with the docs


def test_speed_std_and_target_speed_changes():
    env = FakeEnv()
    steps = [{"speed": 20.0, "target": 25.0}, {"speed": 30.0, "target": 30.0}, {"speed": 20.0, "target": 30.0},
             {"speed": 30.0, "target": 25.0}]
    out = drive(env, EpisodeRecorder(env), steps)
    assert out["mean_speed"] == pytest.approx(25.0) and out["speed_std"] == pytest.approx(5.0)
    assert out["n_speed_changes"] == 2  # 25 -> 30 and 30 -> 25
    env = FakeEnv()
    out = drive(env, EpisodeRecorder(env), [{"speed": 22.0}] * 3)
    assert out["speed_std"] == pytest.approx(0.0, abs=1e-9) and out["n_speed_changes"] == 0


def test_near_miss_steps_exclude_the_crash_step():
    env = FakeEnv()
    env.road.leader = FakeVehicle(40.0, v=20.0)  # gap 35, closing at 5 m/s -> ttc 7 s
    rec = EpisodeRecorder(env)
    assert rec.min_ttc == pytest.approx(7.0)
    step1 = {"x": 0.0}  # ego stays, leader moves
    env.road.leader.position = np.array([10.0, 0.0])  # gap 5, ttc 1 s  < NEAR_MISS_TTC
    env.road.leader.velocity = np.array([20.0, 0.0])
    out = drive(env, rec, [step1, step1])
    assert out["near_miss_steps"] == 2 and out["near_miss"] is True and out["min_ttc"] == pytest.approx(1.0)
    # a crash step with leader overlap (TTC 0) is a crash, not a near miss step
    env = FakeEnv()
    env.road.leader = FakeVehicle(2.0, v=20.0)
    out = drive(env, EpisodeRecorder(env), [{"x": 0.0, "crashed": True}])
    assert out["near_miss_steps"] == 0 and out["near_miss"] is False and out["min_ttc"] == 0.0
    assert NEAR_MISS_TTC == 1.5


def test_overtakes_count_vehicles_passed():
    others = [FakeVehicle(50.0), FakeVehicle(100.0), FakeVehicle(150.0), FakeVehicle(-30.0)]  # last one starts behind
    env = FakeEnv(others)
    rec = EpisodeRecorder(env)
    for v, x in zip(others, [60.0, 140.0, 220.0, 80.0]):  # the one that started behind drives past the ego
        v.position = np.array([x, 0.0])
    out = drive(env, rec, [{"x": 120.0}])
    assert out["overtakes"] == 1  # only the 50 m vehicle ends behind the ego at 120 m; nobody behind counts
    others[1].position = np.array([100.0, 0.0])
    assert EpisodeRecorder.finish(rec, -1, True, {"crashed": False})["overtakes"] == 2


def test_without_leader_leader_columns_are_nan():
    env = FakeEnv()
    env.road.leader = FakeVehicle(10.0, v=10.0)
    out = drive(env, EpisodeRecorder(env, with_leader=False), [{}, {}])
    for key in ("min_gap", "min_ttc", "near_miss", "near_miss_steps"):
        assert math.isnan(out[key])
    assert out["length"] == 2 and out["rightmost_lane_share"] == 1.0  # the rest is unaffected


# ---- the real env ---------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_real_env_invariants_and_column_order():
    df = evaluate(make_baseline("random", seed=1), "main", "balanced", list(range(10000, 10006)))
    assert list(df.columns) == ["condition", "reward", "seed", *EPISODE_COLUMNS]
    assert (df[[f"act_{n}" for n in ACTION_NAMES]].sum(axis=1) == df["length"]).all()
    assert (df[[f"lane{i}_steps" for i in range(N_LANES)]].sum(axis=1) == df["length"]).all()
    assert (df["rightmost_lane_share"] == df[f"lane{N_LANES - 1}_steps"] / df["length"]).all()
    assert (df["truncated"] == df["completed"] | (df["truncated"] & df["crashed"])).all()
    assert (df["overtakes"].between(0, 20)).all() and (df["near_miss_steps"] >= 0).all()
    assert (df["lane_change_reversals"] <= df["lane_changes"]).all() and (df["speed_std"] >= 0).all()
    assert not df.loc[df["crashed"], "near_miss"].any()
    summary = summarize_metrics(df)
    assert set(SUMMARY_COLUMNS) <= set(summary) and summary["act_IDLE"] == pytest.approx(df["act_IDLE"].mean())


@pytest.mark.slow
def test_constant_policies_have_the_expected_histogram():
    idle = evaluate(make_baseline("idle"), "main", "default", [10000, 10001])
    assert (idle["act_IDLE"] == idle["length"]).all() and (idle["lane_changes"] == 0).all()
    assert (idle["n_speed_changes"] == 0).all() and (idle["lane_change_reversals"] == 0).all()
    slower = evaluate(make_baseline("slower"), "main", "default", [10000])
    assert slower["act_SLOWER"].iloc[0] == 30 and slower["n_speed_changes"].iloc[0] >= 1


@pytest.mark.slow
@pytest.mark.parametrize("condition", ["main", "precise"])
def test_road_keeps_all_vehicles_so_overtakes_are_well_defined(condition):
    env = make_test_env(condition)
    env.reset(seed=10000)
    u = env.unwrapped
    n0 = len(u.road.vehicles)
    assert all(v.position[0] > u.vehicle.position[0] for v in u.road.vehicles if v is not u.vehicle)  # all ahead
    for _ in range(30):
        _, _, terminated, truncated, _ = env.step(ACTION_INDEX["FASTER"])
        assert len(u.road.vehicles) == n0
        if terminated or truncated:
            break


@pytest.mark.slow
def test_recorder_start_rearms_and_matches_a_fresh_recorder():
    env = make_env(reward="balanced")
    rec = None
    results = []
    for seed in (10000, 10001, 10000):
        env.reset(seed=seed)
        rec = EpisodeRecorder(env) if rec is None else rec
        if len(results):
            rec.start()
        done = False
        while not done:
            _, r, te, tr, info = env.step(ACTION_INDEX["FASTER"])
            rec.step(r, info, ACTION_INDEX["FASTER"])
            done = te or tr
        results.append(rec.finish(seed, tr, info))
    assert results[0] == results[2] and results[0] != results[1]  # reused recorder == first use for the same seed


def test_overtakes_per_km_is_a_ratio_of_sums():
    import pandas as pd

    # a short crashed episode (100 m, 1 overtake) and a long one (2000 m, 4 overtakes)
    df = pd.DataFrame({"crashed": [True, False], "distance_m": [100.0, 2000.0], "overtakes": [1, 4]})
    out = summarize_metrics(df, extra_prefix=None)
    assert out["overtakes_per_km"] == pytest.approx(5 / 2.1)  # total overtakes / total km
    mean_of_ratios = np.mean([1 / 0.1, 4 / 2.0])  # 6.0: dominated by the short episode
    assert out["overtakes_per_km"] != pytest.approx(mean_of_ratios) and mean_of_ratios == pytest.approx(6.0)
    assert out["overtakes"] == pytest.approx(2.5)  # the per-episode column stays a plain mean
    assert out["crashes_per_km"] == pytest.approx(1 / 2.1)  # same convention as crashes_per_km
    assert "overtakes_per_km" in SUMMARY_COLUMNS
    assert math.isnan(summarize_metrics(df.drop(columns="overtakes"), extra_prefix=None)["overtakes_per_km"])
    assert math.isnan(summarize_metrics(df.assign(distance_m=0.0), extra_prefix=None)["overtakes_per_km"])
