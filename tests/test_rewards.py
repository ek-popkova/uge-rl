"""Tests for common.rewards: formula, crash / lane-change steps, passthrough, real env run."""

from __future__ import annotations

import types

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from common.config import BASE_CONFIG, DENSITIES, ENV_ID, REWARDS
from common.envs import make_env
from common.rewards import CustomRewardWrapper, compute_custom_reward

W = dict(w_speed=0.9, w_right=0.1, w_collision=10.0, w_lane=0.3)


# ---- formula on hand-made component values -------------------------------------------------
def test_formula_plain_step():
    r, comp = compute_custom_reward(W, high_speed=0.5, right_lane=1.0, crashed=False, lane_changed=False)
    assert r == pytest.approx(0.9 * 0.5 + 0.1 * 1.0)
    assert comp["collision"] == 0 and comp["lane_change"] == 0


def test_formula_crash_step():
    r, comp = compute_custom_reward(W, high_speed=0.5, right_lane=1.0, crashed=True, lane_changed=False)
    assert r == pytest.approx(0.45 + 0.1 - 10.0)
    assert comp["collision"] == pytest.approx(-10.0)


def test_formula_lane_change_step():
    r, comp = compute_custom_reward(W, high_speed=1.0, right_lane=0.5, crashed=False, lane_changed=True)
    assert r == pytest.approx(0.9 + 0.05 - 0.3)
    assert comp["lane_change"] == pytest.approx(-0.3)


def test_formula_components_sum_to_reward():
    r, comp = compute_custom_reward(W, 0.3, 0.7, True, True)
    assert sum(comp.values()) == pytest.approx(r)
    assert r == pytest.approx(0.27 + 0.07 - 10.0 - 0.3)


# ---- wrapper logic on a scripted dummy env ---------------------------------------------------
class DummyEnv(gym.Env):
    """Returns scripted (env_reward, info, lane_after) tuples; `vehicle.lane_index` mimics highway-env."""

    observation_space = spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
    action_space = spaces.Discrete(5)

    def __init__(self, script):
        self.script, self.i = script, 0
        self.vehicle = types.SimpleNamespace(lane_index=("a", "b", 1))

    def reset(self, *, seed=None, options=None):
        self.i = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        reward, info, lane_after = self.script[self.i]
        self.i += 1
        self.vehicle.lane_index = ("a", "b", lane_after)
        return np.zeros(1, dtype=np.float32), reward, False, False, dict(info)


def _info(hs, right, crashed):
    return {"rewards": {"high_speed_reward": hs, "right_lane_reward": right}, "crashed": crashed}


def test_default_passes_env_reward_through():
    env = CustomRewardWrapper(DummyEnv([(0.77, _info(0.5, 1.0, False), 1), (0.05, _info(0.5, 1.0, True), 0)]), {})
    env.reset()
    _, r1, *_, info1 = env.step(1)
    _, r2, *_, info2 = env.step(1)
    assert (r1, r2) == (0.77, 0.05)
    assert info1["custom_reward_components"] == {}
    assert (info1["lane_changed"], info2["lane_changed"]) == (0, 1)  # info is still recorded


def test_custom_reward_and_real_lane_change_detection():
    # Step 1: lane 1 -> 1 (no change, whatever the action). Step 2: 1 -> 2 (change). Step 3: crash, 2 -> 2.
    script = [(9.9, _info(0.5, 1.0, False), 1), (9.9, _info(1.0, 0.5, False), 2), (9.9, _info(0.0, 0.5, True), 2)]
    env = CustomRewardWrapper(DummyEnv(script), W)
    env.reset()
    out = [env.step(a) for a in (2, 0, 1)]  # actions are irrelevant for lane_changed
    assert [o[-1]["lane_changed"] for o in out] == [0, 1, 0]
    assert [o[1] for o in out] == pytest.approx([0.55, 0.95 - 0.3, 0.05 - 10.0])  # env reward 9.9 ignored
    assert out[2][-1]["custom_reward_components"]["collision"] == pytest.approx(-10.0)


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        CustomRewardWrapper(DummyEnv([]), {"w_speed": 1.0})
    with pytest.raises(ValueError):
        CustomRewardWrapper(DummyEnv([]), {**W, "w_lane": -0.1})


# ---- short real env run ------------------------------------------------------------------------
def _run(env, n_episodes=3, seed=1000, rng_seed=0):
    """Random-action episodes; yields (reward, info, lane_before, lane_after) for every step."""
    rng = np.random.default_rng(rng_seed)
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        done = False
        while not done:
            lane_before = env.unwrapped.vehicle.lane_index[2]
            _, reward, term, trunc, info = env.step(int(rng.integers(5)))
            yield reward, info, lane_before, env.unwrapped.vehicle.lane_index[2]
            done = term or trunc


@pytest.mark.parametrize("name", ["efficiency", "balanced", "safety"])
@pytest.mark.parametrize("density", ["low", "high"])
def test_real_env_reward_equals_formula_at_every_step(name, density):
    w, n_steps, n_crash = REWARDS[name], 0, 0
    for reward, info, before, after in _run(make_env(density=density, reward=name)):
        expected = (
            w["w_speed"] * info["rewards"]["high_speed_reward"]
            + w["w_right"] * info["rewards"]["right_lane_reward"]
            - w["w_collision"] * float(info["crashed"])
            - w["w_lane"] * float(after != before)
        )
        assert reward == pytest.approx(expected)
        assert info["lane_changed"] == int(after != before)
        assert sum(info["custom_reward_components"].values()) == pytest.approx(reward)
        n_steps, n_crash = n_steps + 1, n_crash + int(info["crashed"])
    assert n_steps > 0
    if density == "high":  # dense traffic: the run must have exercised the crash branch
        assert n_crash > 0


def test_real_env_default_equals_plain_env_reward():
    config = {**BASE_CONFIG, **DENSITIES["medium"]}
    plain = gym.make(ENV_ID, config=config)
    wrapped = make_env(density="medium", reward="default")
    n = 0
    for (r1, *_), (r2, *_) in zip(_run(plain), _run(wrapped)):
        assert r1 == r2
        n += 1
    assert n > 0


def test_custom_reward_independent_of_normalize_reward():
    """The custom reward uses info['rewards'] components, so the env's normalization is irrelevant."""
    out = {}
    for normalize in (True, False):
        config = {**BASE_CONFIG, **DENSITIES["medium"], "normalize_reward": normalize}
        env = CustomRewardWrapper(gym.make(ENV_ID, config=config), REWARDS["balanced"])
        out[normalize] = [reward for reward, *_ in _run(env)]
    assert out[True] == pytest.approx(out[False])
