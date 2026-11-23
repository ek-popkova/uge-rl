"""Supplementary check: does Kinematics(vehicles_count=5) hide relevant vehicles?

Usage (from the repo root):
    .venv/Scripts/python.exe scripts/pilot/obs_coverage.py

The observation shows the ego plus the 4 nearest vehicles (by longitudinal
distance, see_behind=False, within PERCEPTION_DISTANCE). For every step we find
the nearest vehicle ahead within 100 m in the ego lane and in each adjacent
lane ("leaders") and check whether it is among the 4 vehicles the agent sees.
This replicates the selection in KinematicObservation.observe().
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from pilot_common import NEAR_RANGE, OBS_OTHERS, OUT_DIR, SEEDS, choose_action, make_env, record_timing

LANES = [3, 4]
DENSITIES = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
VEHICLES_COUNT = 30
POLICIES = ("idle", "slower", "random")


def run_group(task: tuple[int, float]) -> list[dict]:
    lanes, density = task
    env = make_env({"lanes_count": lanes, "vehicles_count": VEHICLES_COUNT, "vehicles_density": density})
    rows = []
    for policy in POLICIES:
        steps_total = same_present = same_hidden = any_present = any_hidden = 0
        n_hidden_sum = 0
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            env.reset(seed=seed)
            u, ego = env.unwrapped, env.unwrapped.vehicle
            done = False
            while not done:
                _, _, term, trunc, _ = env.step(choose_action(policy, rng, env.action_space.n))
                done = term or trunc
                seen = u.road.close_objects_to(
                    ego, u.PERCEPTION_DISTANCE, count=OBS_OTHERS, see_behind=False, sort=True, vehicles_only=True
                )
                seen_ids = {id(v) for v in seen}
                lane = ego.lane_index[2]
                leaders = {}
                for v in u.road.vehicles:
                    dx = v.position[0] - ego.position[0]
                    if v is ego or not (0 < dx <= NEAR_RANGE) or abs(v.lane_index[2] - lane) > 1:
                        continue
                    cur = leaders.get(v.lane_index[2])
                    if cur is None or dx < cur[0]:
                        leaders[v.lane_index[2]] = (dx, v)
                hidden = [l for l, (_, v) in leaders.items() if id(v) not in seen_ids]
                steps_total += 1
                if lane in leaders:
                    same_present += 1
                    same_hidden += int(lane in hidden)
                if leaders:
                    any_present += 1
                    any_hidden += int(len(hidden) > 0)
                n_hidden_sum += len(hidden)
        rows.append(
            {
                "lanes_count": lanes, "vehicles_density": density, "policy": policy,
                "n_steps": steps_total,
                "share_steps_same_lane_leader_hidden": same_hidden / steps_total,
                "share_steps_any_leader_hidden": any_hidden / steps_total,
                "share_of_present_same_lane_leaders_hidden": same_hidden / max(same_present, 1),
                "mean_hidden_leaders_per_step": n_hidden_sum / steps_total,
            }
        )
    env.close()
    return rows


def main() -> None:
    t0 = time.perf_counter()
    tasks = list(itertools.product(LANES, DENSITIES))
    with mp.Pool(min(12, (os.cpu_count() or 4) - 2)) as pool:
        rows = [r for res in pool.map(run_group, tasks) for r in res]
    df = pd.DataFrame(rows).sort_values(["policy", "lanes_count", "vehicles_density"])
    df.to_csv(OUT_DIR / "obs_coverage.csv", index=False)
    record_timing("obs_coverage", time.perf_counter() - t0)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
