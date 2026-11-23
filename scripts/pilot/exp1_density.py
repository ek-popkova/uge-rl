"""Experiment 1: traffic density sweep with three scripted baseline policies.

Usage (from the repo root):
    .venv/Scripts/python.exe scripts/pilot/exp1_density.py [--episodes N|auto]

1. Benchmark env speed (steps/s) sequentially in this process, nothing else running.
2. Estimate the sweep runtime; with ``--episodes auto`` fall back to 20 episodes
   per cell if the estimated wall time exceeds the budget.
3. Run the sweep in parallel (one task per config) and write the CSVs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from pilot_common import OUT_DIR, POLICIES, SEEDS, make_env, record_timing, run_episode, run_lean

VEHICLES_COUNT = [10, 15, 20, 30, 40]
VEHICLES_DENSITY = [0.7, 1.0, 1.5, 2.0]
LANES_COUNT = [3, 4]
# Supplementary (outside the requested grid): densities below the grid minimum.
EXTRA_VEHICLES_COUNT = [20, 30]
EXTRA_DENSITY = [0.3, 0.4, 0.5, 0.6]

BENCH_EPISODES = 10          # episodes per cell for the speed benchmark
INSTRUMENTATION_FACTOR = 2.0  # conservative: metric bookkeeping vs. bare loop
BUDGET_MIN = 45.0

METRICS = [
    "return", "length", "mean_speed", "distance", "lane_changes",
    "near_mean", "near_max", "near_gt4_share",
    "ahead_mean", "ahead_max", "ahead_gt4_share", "min_gap",
]


def config_overrides(lanes: int, count: int, density: float) -> dict:
    return {"lanes_count": lanes, "vehicles_count": count, "vehicles_density": density}


def run_cell_group(task: tuple[int, int, float, int]) -> list[dict]:
    """Worker: all policies x episodes for one config."""
    lanes, count, density, n_eps = task
    env = make_env(config_overrides(lanes, count, density))
    rows = []
    for policy in POLICIES:
        for seed in SEEDS[:n_eps]:
            ep, _ = run_episode(env, policy, seed)
            ep.update(lanes_count=lanes, vehicles_count=count, vehicles_density=density)
            rows.append(ep)
    env.close()
    return rows


def benchmark(configs: list[tuple[int, int, float]]) -> pd.DataFrame:
    rows = []
    for lanes, count, density in configs:
        env = make_env(config_overrides(lanes, count, density))
        for policy in POLICIES:
            res = run_lean(env, policy, SEEDS[:BENCH_EPISODES])
            rows.append(
                {
                    "lanes_count": lanes, "vehicles_count": count, "vehicles_density": density,
                    "policy": policy, "steps_per_s": res["steps_per_s"],
                    "step_only_per_s": res["step_only_per_s"], "reset_ms": res["reset_ms"],
                    "bench_wall_s": res["wall_s"], "bench_steps": res["total_steps"],
                }
            )
        env.close()
    return pd.DataFrame(rows)


def aggregate(episodes: pd.DataFrame, bench: pd.DataFrame | None) -> pd.DataFrame:
    keys = ["lanes_count", "vehicles_count", "vehicles_density", "policy"]
    out = []
    for key, g in episodes.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row["n_episodes"] = len(g)
        row["crash_rate"] = g["crashed"].mean()
        row["timelimit_rate"] = g["truncated"].mean()  # truncated == time limit reached
        for m in METRICS:
            row[f"{m}_mean"] = g[m].mean()  # NaNs (min_gap with no leader) are skipped
            row[f"{m}_std"] = g[m].std()
        row["min_gap_n_valid"] = int(g["min_gap"].notna().sum())
        out.append(row)
    agg = pd.DataFrame(out)
    if bench is None:
        return agg
    return agg.merge(bench[keys + ["steps_per_s", "step_only_per_s", "reset_ms"]], on=keys, how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="auto", help="episodes per cell, or 'auto'")
    parser.add_argument(
        "--extra-grid", action="store_true",
        help="supplementary sweep OUTSIDE the requested grid: lower densities "
             "(0.3-0.6); writes density_extra_*.csv and skips the speed benchmark",
    )
    parser.add_argument("--bench-only", action="store_true",
                        help="re-run only the speed benchmark and update density_sweep.csv / speed.csv")
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 2)))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    configs = list(itertools.product(LANES_COUNT, VEHICLES_COUNT, VEHICLES_DENSITY))
    if args.extra_grid:
        configs = list(itertools.product(LANES_COUNT, EXTRA_VEHICLES_COUNT, EXTRA_DENSITY))

    # 1. speed benchmark (sequential, nothing else running)
    if args.bench_only:
        t0 = time.perf_counter()
        bench = benchmark(configs)
        bench.to_csv(OUT_DIR / "speed.csv", index=False)
        sweep = pd.read_csv(OUT_DIR / "density_sweep.csv")
        speed_cols = ["steps_per_s", "step_only_per_s", "reset_ms"]
        keys = ["lanes_count", "vehicles_count", "vehicles_density", "policy"]
        sweep = sweep.drop(columns=[c for c in speed_cols if c in sweep]).merge(
            bench[keys + speed_cols], on=keys, how="left")
        sweep.to_csv(OUT_DIR / "density_sweep.csv", index=False)
        record_timing("exp1_speed_rebenchmark", time.perf_counter() - t0)
        print(f"benchmark rewritten in {time.perf_counter() - t0:.0f}s")
        return
    t0 = time.perf_counter()
    bench = None if args.extra_grid else benchmark(configs)
    bench_time = time.perf_counter() - t0
    print(f"benchmark done in {bench_time:.0f}s", flush=True)

    # 2. runtime estimate for the sweep
    if bench is None:  # supplementary grid: no benchmark, always the full 50 episodes
        serial_s_full = 0.0
    else:
        serial_s_full = bench["bench_wall_s"].sum() * (len(SEEDS) / BENCH_EPISODES) * INSTRUMENTATION_FACTOR
    wall_min_full = serial_s_full / (args.workers * 0.75) / 60  # 75% parallel efficiency
    if args.episodes == "auto":
        n_eps = len(SEEDS) if wall_min_full <= BUDGET_MIN else 20
    else:
        n_eps = int(args.episodes)
    print(
        f"estimate for 50 eps/cell: serial {serial_s_full / 60:.1f} min, "
        f"parallel({args.workers} workers) {wall_min_full:.1f} min -> using {n_eps} eps/cell",
        flush=True,
    )

    # 3. the sweep itself; heaviest configs first for load balance
    tasks = sorted(
        [(l, c, d, n_eps) for l, c, d in configs], key=lambda t: -(t[1] * t[2])
    )
    t0 = time.perf_counter()
    rows: list[dict] = []
    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_cell_group, tasks), 1):
            rows.extend(res)
            print(f"  [{i}/{len(tasks)}] configs done", flush=True)
    sweep_time = time.perf_counter() - t0

    episodes = pd.DataFrame(rows).sort_values(
        ["lanes_count", "vehicles_count", "vehicles_density", "policy", "seed"]
    )
    cols = ["lanes_count", "vehicles_count", "vehicles_density", "policy", "seed"]
    episodes = episodes[cols + [c for c in episodes.columns if c not in cols]]
    tag = "_extra" if args.extra_grid else ""
    episodes.to_csv(OUT_DIR / f"density{tag}_episodes.csv", index=False)
    aggregate(episodes, bench).to_csv(OUT_DIR / f"density{tag}_sweep.csv", index=False)
    if bench is not None:
        bench.to_csv(OUT_DIR / "speed.csv", index=False)

    total = time.perf_counter() - t_start
    record_timing(
        "exp1_density_extra" if args.extra_grid else "exp1_density", total, benchmark_s=round(bench_time, 1), sweep_s=round(sweep_time, 1),
        episodes_per_cell=n_eps, workers=args.workers,
        estimate_full_parallel_min=round(wall_min_full, 1),
    )
    print(f"done in {total / 60:.1f} min (sweep {sweep_time / 60:.1f} min, {n_eps} eps/cell)")


if __name__ == "__main__":
    main()
