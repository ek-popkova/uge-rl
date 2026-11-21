"""Tests for the evaluation protocol: test conditions, determinism, return recomputation,
metrics, aggregation, baselines and file layout."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from common.agent_base import Agent
from common.aggregate import across_seeds_table, per_model_table, t_interval, t_quantile_975, wilson_interval
from common.baselines import ACTION_INDEX, BASELINE_NAMES, make_baseline
from common.config import (
    BASE_CONFIG,
    DENSITIES,
    PILOT_SEEDS,
    REWARDS,
    SENSITIVITY_SEEDS,
    TEST_CONDITIONS,
    TEST_SEEDS,
    TRAIN_SEEDS,
    VAL_SEEDS,
)
from common.envs import make_env, make_test_env
from common.evaluate import evaluate, make_chunks, validate
from common.metrics import NEAR_MISS_TTC, add_return_columns, leader_gap_ttc, recompute_return
from common.seeding import derived_env_seeds, vector_env_seed
from common.paths import eval_csv, final_checkpoint, load_eval_episodes, run_dir, save_eval, train_log, val_log

SEEDS = list(range(10000, 10006))  # short list from TEST_SEEDS for speed


class TorchGreedyAgent(Agent):
    """Deterministic agent with a fixed random MLP (exercises torch inside worker processes)."""

    def __init__(self) -> None:
        torch.manual_seed(0)
        self.net = torch.nn.Sequential(torch.nn.Linear(25, 16), torch.nn.Tanh(), torch.nn.Linear(16, 5))

    def act(self, obs, deterministic=True):
        with torch.no_grad():
            return int(self.net(torch.as_tensor(obs, dtype=torch.float32).flatten()).argmax())

    def save(self, path):
        pass

    def load(self, path):
        pass


# ---- conditions --------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(TEST_CONDITIONS))
def test_make_test_env_config(name):
    spec = TEST_CONDITIONS[name]
    env = make_test_env(name)
    cfg = env.unwrapped.config
    expected = {**BASE_CONFIG, **DENSITIES[spec["density"]], **spec["overrides"]}
    for key, value in expected.items():
        assert cfg[key] == value, f"{name}: {key} is {cfg[key]}, expected {value}"
    assert cfg["ego_spacing"] == 1.5
    assert env.spec.id == spec["env_id"]
    assert env.unwrapped.render_mode is None and env.unwrapped.viewer is None


def test_condition_values():
    c = {n: make_test_env(n).unwrapped.config for n in TEST_CONDITIONS}
    densities = (c["main"]["vehicles_density"], c["low"]["vehicles_density"], c["high"]["vehicles_density"])
    assert densities == (0.7, 0.4, 1.0)
    assert (c["long"]["duration"], c["long"]["vehicles_count"]) == (60, 30)
    assert c["precise"]["simulation_frequency"] == 15 and c["main"]["simulation_frequency"] == 5
    # highway-v0 checks collisions between other vehicles, highway-fast-v0 does not
    fast, precise = make_test_env("main").unwrapped, make_test_env("precise").unwrapped
    fast.reset(seed=0)
    precise.reset(seed=0)
    assert not any(v.check_collisions for v in fast.road.vehicles if v is not fast.vehicle)
    assert all(v.check_collisions for v in precise.road.vehicles)


def test_make_env_overrides_merge_last_and_defaults_unchanged():
    env = make_env(density="low", overrides={"vehicles_density": 0.9, "duration": 12})
    assert env.unwrapped.config["vehicles_density"] == 0.9 and env.unwrapped.config["duration"] == 12
    assert make_env().spec.id == "highway-fast-v0"
    assert BASE_CONFIG["duration"] == 30  # overrides never mutate the shared config
    with pytest.raises(ValueError):
        make_test_env("nope")


def test_baseline_action_indices_match_env():
    idx = make_env().unwrapped.action_type.actions_indexes
    assert {k: idx[k] for k in ACTION_INDEX} == ACTION_INDEX


# ---- determinism --------------------------------------------------------------------------------
def test_env_reuse_equals_fresh_env():
    agent = make_baseline("faster")
    a = evaluate(agent, "main", "balanced", SEEDS)  # one reused env, seeds in order
    b = evaluate(agent, "main", "balanced", SEEDS[::-1])  # same env, other order
    pd.testing.assert_frame_equal(a, b)


def test_same_policy_twice_identical_and_worker_independent():
    for agent in (make_baseline("random", seed=3), TorchGreedyAgent()):
        first = evaluate(agent, "main", "balanced", SEEDS, workers=1, chunk_size=6)
        second = evaluate(agent, "main", "balanced", SEEDS, workers=1, chunk_size=6)
        pd.testing.assert_frame_equal(first, second)
        parallel = evaluate(agent, "main", "balanced", SEEDS, workers=2, chunk_size=2)
        pd.testing.assert_frame_equal(first, parallel)


def test_random_baseline_is_seeded_per_episode_not_per_order():
    agent = make_baseline("random", seed=1)
    full = evaluate(agent, "main", "default", SEEDS)
    reversed_ = evaluate(agent, "main", "default", SEEDS[::-1])
    pd.testing.assert_frame_equal(full, reversed_)
    other = evaluate(make_baseline("random", seed=2), "main", "default", SEEDS)
    assert not full.equals(other)


def test_make_chunks_and_validation_args():
    assert make_chunks(range(7), 3) == [(0, 1, 2), (3, 4, 5), (6,)]
    with pytest.raises(ValueError):
        evaluate(make_baseline("idle"), "main", "default", [1, 1])
    with pytest.raises(ValueError):
        validate(make_baseline("idle"), n=len(VAL_SEEDS) + 1)


# ---- return recomputation ---------------------------------------------------------------------------
@pytest.mark.parametrize("reward", list(REWARDS))
@pytest.mark.parametrize("policy", ["random", "faster"])
def test_recompute_return_matches_wrapper(reward, policy):
    df = evaluate(make_baseline(policy), "main", reward, SEEDS[:4])
    np.testing.assert_allclose(recompute_return(df, reward), df["episode_return"], atol=1e-9)
    # the same run recomputed under any other preset equals a run built with that preset
    for other in REWARDS:
        ref = evaluate(make_baseline(policy), "main", other, SEEDS[:4])
        np.testing.assert_allclose(recompute_return(df, other), ref["episode_return"], atol=1e-9)


def test_return_env_is_independent_of_reward_preset():
    a = evaluate(make_baseline("random"), "main", "safety", SEEDS[:4])
    b = evaluate(make_baseline("random"), "main", "default", SEEDS[:4])
    np.testing.assert_allclose(a["return_env"], b["return_env"])
    np.testing.assert_allclose(b["return_env"], b["episode_return"])
    assert set(add_return_columns(a).columns) >= {f"return_{r}" for r in REWARDS}


# ---- metrics -------------------------------------------------------------------------------------------
def test_episode_metrics_consistency():
    df = evaluate(make_baseline("faster"), "main", "balanced", SEEDS)  # crashes often
    assert (df["completed"] == (~df["crashed"] & (df["length"] == 30))).all()
    assert not df.loc[df["crashed"], "near_miss"].any()
    assert (df["distance_m"] > 0).all()
    crashed = df["crashed"]
    np.testing.assert_allclose(df.loc[crashed, "crashes_per_km"], 1000.0 / df.loc[crashed, "distance_m"])
    assert (df.loc[~crashed, "crashes_per_km"] == 0).all()
    assert (df["min_gap"].dropna() >= 0).all()


def test_gap_and_ttc():
    gap, ttc = leader_gap_ttc(ego_x=0.0, ego_vx=25.0, lead_x=30.0, lead_vx=20.0, length=5.0)
    assert gap == pytest.approx(25.0) and ttc == pytest.approx(5.0)
    assert math.isnan(leader_gap_ttc(0.0, 20.0, 30.0, 25.0, 5.0)[1])  # leader faster: not closing in
    assert leader_gap_ttc(0.0, 25.0, 4.0, 20.0, 5.0) == (0.0, 0.0)  # overlap
    assert NEAR_MISS_TTC == 1.5


def test_validate_returns_mean_metrics_on_val_seeds():
    out = validate(make_baseline("slower"), n=2)
    assert out["n_episodes"] == 2 and out["crash_rate"] == 0.0


# ---- aggregation ----------------------------------------------------------------------------------------
def _fake_episodes() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for algo in ("ddqn", "ppo"):
        for seed in range(5):
            for ep in range(10):
                crashed = bool(rng.random() < (0.2 if algo == "ddqn" else 0.4))
                rows.append(
                    dict(
                        algorithm=algo, train_reward="balanced", train_seed=seed, condition="main", seed=ep,
                        episode_return=1.0, return_env=1.0, sum_high_speed=10.0, sum_right_lane=5.0,
                        lane_changes=1, crashed=crashed, completed=not crashed, near_miss=False, length=30,
                        mean_speed=25.0, distance_m=750.0, min_gap=10.0, min_ttc=3.0,
                    )
                )
    return pd.DataFrame(rows)


def test_aggregation_uses_seed_as_unit():
    per_model = per_model_table(_fake_episodes())
    assert len(per_model) == 10 and (per_model["n_episodes"] == 10).all()
    table = across_seeds_table(per_model)
    row = table[(table["algorithm"] == "ddqn") & (table["metric"] == "crash_rate")].iloc[0]
    seed_values = per_model[per_model["algorithm"] == "ddqn"]["crash_rate"]
    assert row["n_seeds"] == 5  # not 50 pooled episodes
    assert row["mean"] == pytest.approx(seed_values.mean())
    half = 2.7764451 * seed_values.std(ddof=1) / math.sqrt(5)  # t(0.975, 4) * sd / sqrt(n)
    assert row["ci_low"] == pytest.approx(seed_values.mean() - half)
    assert row["ci_high"] == pytest.approx(seed_values.mean() + half)
    # return_own = return under the preset the model was trained with (balanced)
    no_crash = 0.4 * 10 + 0.1 * 5 - 0.1 * 1
    assert (per_model["return_own"] <= no_crash + 1e-9).all()
    crash_rate = per_model["crash_rate"]
    np.testing.assert_allclose(per_model["return_own"], no_crash - 20.0 * crash_rate)
    assert (per_model["crashes_per_km"] >= 0).all()


def test_t_interval():
    lo, hi = t_interval(np.array([0.0, 0.0, 1.0, 1.0, 1.0]))
    half = 2.7764451 * np.std([0, 0, 1, 1, 1], ddof=1) / math.sqrt(5)
    assert (lo, hi) == (pytest.approx(0.6 - half), pytest.approx(0.6 + half))
    assert lo < 0 and hi > 1  # wider than the observed range, unlike a bootstrap
    assert all(math.isnan(v) for v in t_interval(np.array([1.0])))  # n < 2: no interval
    assert all(math.isnan(v) for v in t_interval(np.array([1.0, np.nan])))  # NaN seeds are dropped
    assert t_interval(np.array([2.0, 2.0, 2.0])) == (2.0, 2.0)


def test_t_quantiles():
    assert t_quantile_975(4) == pytest.approx(2.776445)
    assert t_quantile_975(2) == pytest.approx(4.302653)
    # table and Cornish-Fisher expansion join smoothly; large df approaches the normal quantile
    assert t_quantile_975(31) == pytest.approx(2.039513, abs=1e-3)
    assert t_quantile_975(120) == pytest.approx(1.979930, abs=1e-3)
    assert t_quantile_975(10_000) == pytest.approx(1.96, abs=1e-3)
    with pytest.raises(ValueError):
        t_quantile_975(0)


def test_wilson():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.21


# ---- baselines & layout ------------------------------------------------------------------------------------
def test_scripted_baselines_basic_behaviour():
    slower = evaluate(make_baseline("slower"), "main", "default", SEEDS)
    assert not slower["crashed"].any() and slower["completed"].all()
    assert set(BASELINE_NAMES) == {"random", "idle", "slower", "faster"}
    with pytest.raises(ValueError):
        make_baseline("nope")


def test_paths_layout_and_loader(tmp_path):
    assert run_dir("ddqn", "safety", 3, tmp_path) == tmp_path / "runs" / "ddqn" / "safety" / "seed3"
    assert final_checkpoint("ppo", "balanced", 0, tmp_path).name == "final.pt"
    assert train_log("ppo", "balanced", 0, tmp_path).name == "train_log.csv"
    assert val_log("ppo", "balanced", 0, tmp_path).name == "val_log.csv"
    expected = tmp_path / "eval" / "ppo" / "balanced" / "seed4" / "long.csv"
    assert eval_csv("ppo", "balanced", 4, "long", tmp_path) == expected
    with pytest.raises(ValueError):
        run_dir("a2c", "balanced", 0, tmp_path)
    df = evaluate(make_baseline("slower"), "main", "default", SEEDS[:2])
    save_eval(df, "ddqn", "safety", 2, "main", tmp_path)
    loaded = load_eval_episodes(tmp_path)
    assert list(loaded["algorithm"].unique()) == ["ddqn"] and loaded["train_seed"].eq(2).all()
    assert loaded["train_reward"].eq("safety").all() and loaded["condition"].eq("main").all()


def test_derived_env_seeds_disjoint_from_pilot_val_test():
    forbidden = set(PILOT_SEEDS) | set(VAL_SEEDS) | set(TEST_SEEDS)
    seen: set[int] = set()
    for run_seed in sorted(set(TRAIN_SEEDS) | set(SENSITIVITY_SEEDS)):  # sensitivity seeds reuse 0..2
        seeds = derived_env_seeds(run_seed, 64)
        assert len(seeds) == 64 and seeds[0] == 100000 + run_seed * 1000
        assert not set(seeds) & forbidden
        assert not set(seeds) & set(TRAIN_SEEDS)
        assert not set(seeds) & seen  # different run seeds never share an env seed
        seen |= set(seeds)
    assert vector_env_seed(2, 5) == 100000 + 2 * 1000 + 5
    with pytest.raises(ValueError):
        vector_env_seed(0, 1000)
