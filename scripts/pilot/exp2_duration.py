"""Experiment 2: episode duration with the idle policy.

Usage (from the repo root):
    .venv/Scripts/python.exe scripts/pilot/exp2_duration.py [--extra name=count:density ...]

The default-density config (highway-fast-v0 defaults: 20 vehicles, density 1.0,
3 lanes) is always run. ``--extra`` adds further configs (e.g. the proposed
low/medium/high settings) as clearly named additional rows.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from pilot_common import OUT_DIR, SEEDS, make_env, record_timing, run_lean

DURATIONS = [20, 30, 40, 60]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", nargs="*", default=[], help="name=vehicles_count:density")
    parser.add_argument("--lanes", type=int, default=3)
    args = parser.parse_args()

    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = [("default", 20, 1.0)]
    for spec in args.extra:
        name, rest = spec.split("=")
        count, density = rest.split(":")
        configs.append((name, int(count), float(density)))

    rows = []
    for name, count, density in configs:
        for duration in DURATIONS:
            env = make_env(
                {"lanes_count": args.lanes, "vehicles_count": count,
                 "vehicles_density": density, "duration": duration}
            )
            res = run_lean(env, "idle", SEEDS)
            env.close()
            lengths = pd.Series(res["lengths"])
            crashed = pd.Series(res["crashed"])
            row = {
                "config": name, "lanes_count": args.lanes, "vehicles_count": count,
                "vehicles_density": density, "duration": duration, "policy": "idle",
                "n_episodes": len(SEEDS), "crash_rate": crashed.mean(),
                "timelimit_rate": pd.Series(res["truncated"]).mean(),
                "mean_length": lengths.mean(),
                "mean_crash_step": lengths[crashed == 1].mean(),
                "steps_per_s": res["steps_per_s"], "step_only_per_s": res["step_only_per_s"],
            }
            # Cumulative crash rate by step t (same seeds => trajectories agree up to t).
            for t in DURATIONS:
                row[f"crash_by_{t}"] = ((crashed == 1) & (lengths <= t)).mean() if t <= duration else float("nan")
            rows.append(row)
            print(name, duration, f"crash={row['crash_rate']:.2f} limit={row['timelimit_rate']:.2f}", flush=True)

    pd.DataFrame(rows).to_csv(OUT_DIR / "duration.csv", index=False)
    record_timing("exp2_duration", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
