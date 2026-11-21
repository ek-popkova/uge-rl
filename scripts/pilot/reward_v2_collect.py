"""Reward pilot v2, step 1: log per-step reward components for low/medium/high x 4 policies.

Usage (from the repo root):
    .venv/Scripts/python.exe scripts/pilot/reward_v2_collect.py [--workers N] [--resume]

Configs: common.config.BASE_CONFIG (3 lanes, 20 vehicles, 30 s, 5 Hz) with the vehicles_density of
common.config.DENSITIES (low 0.4 / medium 0.7 / high 1.0). All 4 policies x 3 densities x 50 seeds
(1000..1049) are simulated afresh (600 episodes); no earlier logs are reused.
Everything is deterministic per (config, policy, seed) and independent of the worker count.
"""

from __future__ import annotations

import os

# Limit threads per worker before numpy is imported anywhere (see CLAUDE.md rule 7).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import multiprocessing as mp
import sys
import time

import pandas as pd

from pilot_common import OUT_DIR, ROOT, SEEDS, make_env, record_timing, run_episode

sys.path.insert(0, str(ROOT))  # make `common` importable when run as a script
from common.config import BASE_CONFIG, DENSITIES, PILOT_SEEDS  # noqa: E402

V2_DIR = OUT_DIR / "reward_v2"
PART_DIR = V2_DIR / "partial"
POLICIES4 = ("random", "idle", "slower", "faster")
CONFIGS = tuple(DENSITIES)  # low, medium, high
CHUNK = 10  # seeds per task: fine-grained tasks keep all workers busy


def overrides(config: str) -> dict:
    """Env config of a density level: the shared BASE_CONFIG plus that level's vehicles_density."""
    return {**BASE_CONFIG, **DENSITIES[config]}


def part_paths(config: str, policy: str, lo: int):
    stem = f"{config}__{policy}__{lo}"
    return PART_DIR / f"{stem}.steps.csv", PART_DIR / f"{stem}.eps.csv"


def run_task(task: tuple[str, str, list[int]]) -> tuple[str, str, int, int]:
    """Worker: simulate one (config, policy, seed chunk); results go to partial files."""
    config, policy, seeds = task
    env = make_env(overrides(config))
    ep_rows, step_rows = [], []
    for seed in seeds:
        ep, steps = run_episode(env, policy, seed, record_steps=True)
        ep.update(config=config)
        ep_rows.append(ep)
        for s in steps:
            s["config"] = config
        step_rows.extend(steps)
    env.close()
    steps_path, eps_path = part_paths(config, policy, seeds[0])
    pd.DataFrame(step_rows).to_csv(steps_path, index=False)
    pd.DataFrame(ep_rows).to_csv(eps_path, index=False)  # written last: marks the task as done
    return config, policy, seeds[0], sum(e["length"] for e in ep_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--resume", action="store_true", help="skip tasks whose partial files exist")
    args = parser.parse_args()

    t_start = time.perf_counter()
    PART_DIR.mkdir(parents=True, exist_ok=True)

    assert SEEDS == PILOT_SEEDS, "pilot seeds differ from common.config.PILOT_SEEDS"

    # Tasks: 3 densities x 4 policies; seeds in chunks of CHUNK (heaviest = longest episodes first).
    todo = [(c, p) for c in ("high", "medium", "low") for p in POLICIES4]
    tasks = []
    for config, policy in todo:
        for i in range(0, len(SEEDS), CHUNK):
            chunk = SEEDS[i : i + CHUNK]
            if args.resume and all(p.exists() for p in part_paths(config, policy, chunk[0])):
                continue
            tasks.append((config, policy, chunk))

    # Speed measurement and runtime estimate before the long part (CLAUDE.md rule 1).
    env = make_env(overrides("medium")); env.reset(seed=1000); n, t0 = 0, time.perf_counter()
    while n < 100:
        _, _, te, tr, _ = env.step(1); n += 1
        if te or tr:
            env.reset(seed=1000 + n)
    sps = n / (time.perf_counter() - t0); env.close()
    est_steps = len(tasks) * CHUNK * 22  # ~22 steps per episode on average (pilot data)
    print(f"single-process speed {sps:.0f} steps/s; {len(tasks)} tasks, ~{est_steps} steps; "
          f"estimate ~{est_steps / (sps * min(args.workers, 4)) / 60:.1f} min with {args.workers} workers",
          flush=True)

    done_steps = 0
    t0 = time.perf_counter()
    if tasks:
        with mp.Pool(args.workers) as pool:
            for i, (c, p, lo, steps) in enumerate(pool.imap_unordered(run_task, tasks), 1):
                done_steps += steps
                el = time.perf_counter() - t0
                print(f"  [{i}/{len(tasks)}] {c}/{p} seeds {lo}+: {done_steps} steps, "
                      f"{done_steps / el:.0f} steps/s, ETA {el / i * (len(tasks) - i) / 60:.1f} min", flush=True)

    # Merge the partial files.
    steps_all, eps_all = [], []
    for config, policy in todo:
        for i in range(0, len(SEEDS), CHUNK):
            sp, ep = part_paths(config, policy, SEEDS[i])
            steps_all.append(pd.read_csv(sp))
            eps_all.append(pd.read_csv(ep))

    steps = pd.concat(steps_all, ignore_index=True)
    eps = pd.concat(eps_all, ignore_index=True)
    step_cols = ["config", "policy", "seed", "step", "action", "reward", "collision_reward", "right_lane_reward",
                 "high_speed_reward", "on_road_reward", "crashed", "speed", "lane_changed"]
    ep_cols = ["config", "policy", "seed", "length", "crashed", "terminated", "truncated", "mean_speed",
               "distance", "lane_changes", "return"]
    order = {"low": 0, "medium": 1, "high": 2}
    pol = {"random": 0, "idle": 1, "slower": 2, "faster": 3}
    for df, cols, key in ((steps, step_cols, ["seed", "step"]), (eps, ep_cols, ["seed"])):
        df["_c"], df["_p"] = df["config"].map(order), df["policy"].map(pol)
        df.sort_values(["_c", "_p"] + key, inplace=True)
    steps[step_cols].to_csv(V2_DIR / "steps.csv", index=False)
    eps[ep_cols].rename(columns={"return": "return_default"}).to_csv(V2_DIR / "episodes.csv", index=False)

    total = time.perf_counter() - t_start
    record_timing("reward_v2_collect", total, workers=args.workers, single_process_steps_per_s=round(sps))
    print(f"done in {total / 60:.1f} min; episodes: {len(eps)}, steps: {len(steps)}")


if __name__ == "__main__":
    main()
