"""Shared helpers for the environment pilot experiments.

The pilot only *measures* the environment with scripted baseline policies.
No RL agent is trained and common/config.py is never modified or imported.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import highway_env  # noqa: F401  (registers the highway environments)
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "pilot"
ENV_ID = "highway-fast-v0"

# Same seeds for every config and every policy.
SEEDS = list(range(1000, 1050))

# Longitudinal range used to count "nearby" vehicles.
NEAR_RANGE = 100.0
# Default Kinematics observation: vehicles_count=5 includes the ego vehicle,
# so at most 4 *other* vehicles are visible to the agent.
OBS_OTHERS = 4

POLICIES = ("random", "idle", "slower")
# Action indices verified against env.unwrapped.action_type.actions_indexes.
EXPECTED_ACTIONS = {"IDLE": 1, "FASTER": 3, "SLOWER": 4}


def make_env(overrides: Optional[dict[str, Any]] = None, render_mode: Optional[str] = None) -> gym.Env:
    """Create highway-fast-v0 with only the given overrides on top of its defaults."""
    env = gym.make(ENV_ID, config=dict(overrides or {}), render_mode=render_mode)
    idx = env.unwrapped.action_type.actions_indexes
    for name, value in EXPECTED_ACTIONS.items():
        assert idx[name] == value, f"unexpected action index for {name}: {idx}"
    return env


def choose_action(policy: str, rng: np.random.Generator, n_actions: int) -> int:
    """Scripted baseline policies: random / idle (1) / faster (3) / slower (4)."""
    if policy == "random":
        return int(rng.integers(n_actions))
    if policy == "idle":
        return EXPECTED_ACTIONS["IDLE"]
    if policy == "slower":
        return EXPECTED_ACTIONS["SLOWER"]
    if policy == "faster":
        return EXPECTED_ACTIONS["FASTER"]
    raise ValueError(f"unknown policy {policy!r}")


def _nearby_counts(env_u, ego) -> tuple[int, int]:
    """Number of other vehicles within NEAR_RANGE metres longitudinally.

    Returns (all_around, ahead_only). The default observation has
    see_behind=False, so ``ahead_only`` is what the agent can actually see.
    """
    dx = np.array([v.position[0] - ego.position[0] for v in env_u.road.vehicles if v is not ego])
    if dx.size == 0:
        return 0, 0
    return int(np.sum(np.abs(dx) <= NEAR_RANGE)), int(np.sum((dx > 0) & (dx <= NEAR_RANGE)))


def _gap_ahead(env_u, ego) -> float:
    """Bumper-to-bumper gap to the vehicle ahead in the ego's current lane (nan if none)."""
    front, _ = env_u.road.neighbour_vehicles(ego, ego.lane_index)
    if front is None:
        return float("nan")
    return float(front.position[0] - ego.position[0] - ego.LENGTH)


def run_episode(env: gym.Env, policy: str, seed: int, record_steps: bool = False):
    """Run one episode and return (episode_metrics, step_log).

    ``step_log`` is empty unless ``record_steps`` is set.
    """
    rng = np.random.default_rng(seed)  # policy randomness, reproducible per seed
    env.reset(seed=seed)
    u = env.unwrapped
    ego = u.vehicle
    n_actions = env.action_space.n
    x0 = float(ego.position[0])

    ret, n_steps, lane_changes = 0.0, 0, 0
    speeds, near_all, near_ahead, gaps = [], [], [], []
    steps: list[dict[str, Any]] = []
    terminated = truncated = False
    info: dict[str, Any] = {}

    while not (terminated or truncated):
        lane_before = ego.lane_index[2]
        action = choose_action(policy, rng, n_actions)
        _, reward, terminated, truncated, info = env.step(action)
        lane_changed = int(ego.lane_index[2] != lane_before)  # real change, not the action
        n_steps += 1
        ret += float(reward)
        lane_changes += lane_changed
        speeds.append(float(info["speed"]))
        n_all, n_ah = _nearby_counts(u, ego)
        near_all.append(n_all)
        near_ahead.append(n_ah)
        gaps.append(_gap_ahead(u, ego))
        if record_steps:
            r = info["rewards"]
            steps.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "step": n_steps,
                    "action": action,
                    "reward": float(reward),
                    "collision_reward": float(r["collision_reward"]),
                    "right_lane_reward": float(r["right_lane_reward"]),
                    "high_speed_reward": float(r["high_speed_reward"]),
                    "on_road_reward": float(r["on_road_reward"]),
                    "crashed": int(bool(info["crashed"])),
                    "lane_changed": lane_changed,
                    "speed": float(info["speed"]),
                }
            )

    near_all_a, near_ahead_a = np.array(near_all), np.array(near_ahead)
    gaps_a = np.array(gaps)
    episode = {
        "seed": seed,
        "policy": policy,
        "return": ret,
        "length": n_steps,
        "crashed": int(bool(info["crashed"])),
        "terminated": int(terminated),
        "truncated": int(truncated),
        "mean_speed": float(np.mean(speeds)),
        "distance": float(ego.position[0]) - x0,
        "lane_changes": lane_changes,
        "near_mean": float(near_all_a.mean()),
        "near_max": int(near_all_a.max()),
        "near_gt4_share": float(np.mean(near_all_a > OBS_OTHERS)),
        "ahead_mean": float(near_ahead_a.mean()),
        "ahead_max": int(near_ahead_a.max()),
        "ahead_gt4_share": float(np.mean(near_ahead_a > OBS_OTHERS)),
        "min_gap": float(np.nanmin(gaps_a)) if not np.all(np.isnan(gaps_a)) else float("nan"),
    }
    return episode, steps


def run_lean(env: gym.Env, policy: str, seeds: list[int]) -> dict[str, Any]:
    """Time a bare episode loop (no metric bookkeeping) to measure env speed."""
    n_actions = env.action_space.n
    lengths, crashed, truncated_flags = [], [], []
    reset_wall = step_wall = 0.0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        env.reset(seed=seed)
        t1 = time.perf_counter()
        n, term, trunc, info = 0, False, False, {}
        while not (term or trunc):
            _, _, term, trunc, info = env.step(choose_action(policy, rng, n_actions))
            n += 1
        step_wall += time.perf_counter() - t1
        reset_wall += t1 - t0
        lengths.append(n)
        crashed.append(int(bool(info["crashed"])))
        truncated_flags.append(int(trunc))
    total_steps = int(sum(lengths))
    wall = reset_wall + step_wall
    return {
        "lengths": lengths,
        "crashed": crashed,
        "truncated": truncated_flags,
        "total_steps": total_steps,
        "wall_s": wall,
        "steps_per_s": total_steps / wall,  # resets included (what training pays)
        "step_only_per_s": total_steps / step_wall,  # env.step() only
        "reset_ms": 1000 * reset_wall / len(seeds),
    }


def record_timing(key: str, seconds: float, **extra: Any) -> None:
    """Merge the runtime of one pilot script into results/pilot/timing.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "timing.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = {"seconds": round(seconds, 1), **extra}
    path.write_text(json.dumps(data, indent=2))
