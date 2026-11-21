"""Evaluate the scripted baselines (random, idle, slower, faster) on every test condition.

Same pipeline as for trained agents (``common.evaluate``): TEST_SEEDS, deterministic episodes,
identical starting situations. Outputs:

    results/eval/baselines/<condition>.csv   per episode, with return_<preset> for every preset
    results/eval/baselines/summary.csv       per (policy, condition) means, crash-rate Wilson CI
    results/eval/baselines/meta.json         steps/s and wall time per condition
    results/eval/baselines/_tasks/           one file per finished task (for --resume)

Usage (from the repository root):
    python -m scripts.eval_baselines --workers 6 [--resume]
"""

from __future__ import annotations

import os

# Thread limits must be set before numpy/torch are imported (CLAUDE.md rule 7).
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from common.aggregate import wilson_interval
from common.baselines import BASELINE_NAMES, make_baseline, run_baseline_chunk
from common.config import BASE_CONFIG, TEST_CONDITIONS, TEST_SEEDS
from common.envs import make_test_env
from common.evaluate import make_chunks, records_to_frame
from common.metrics import add_return_columns, summarize_metrics
from common.parallel import run_tasks
from common.paths import baselines_dir

ASSUMED_SPEEDUP_CAP = 4.0  # parallel scaling plateaus at about 3.5-5x here (docs/performance_analysis.md C5)


def condition_duration(condition: str) -> int:
    return int(TEST_CONDITIONS[condition]["overrides"].get("duration", BASE_CONFIG["duration"]))


def benchmark(condition: str, steps: int) -> float:
    """Steps/s of one condition in this (quiet) process, measured on at least ``steps`` steps."""
    env = make_test_env(condition)
    agent = make_baseline("random")
    done_steps, seed = 0, 0
    t0 = time.perf_counter()
    while done_steps < steps:
        obs, _ = env.reset(seed=1000 + seed)
        seed += 1
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _, terminated, truncated, _ = env.step(agent.act(obs))
            done_steps += 1
    elapsed = time.perf_counter() - t0
    env.close()
    return done_steps / elapsed


def task_stem(policy: str, condition: str, chunk_idx: int) -> str:
    return f"{condition}__{policy}__{chunk_idx:03d}"


def write_task(task_dir: Path, stem: str, task, res) -> None:
    policy, condition, seeds = task
    df = records_to_frame(res["records"], condition, "default")
    df.insert(0, "policy", policy)
    tmp = task_dir / f"{stem}.csv.tmp"
    df.to_csv(tmp, index=False)
    meta = {"steps": res["steps"], "wall_s": res["wall_s"]}
    (task_dir / f"{stem}.json").write_text(json.dumps(meta))
    os.replace(tmp, task_dir / f"{stem}.csv")  # csv last: its presence marks a finished task


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6, help="processes (scaling plateaus near 8 here)")
    ap.add_argument("--episodes", type=int, default=len(TEST_SEEDS), help="episodes per policy and condition")
    ap.add_argument("--chunk-size", type=int, default=5, help="episodes per task")
    ap.add_argument("--conditions", nargs="+", default=list(TEST_CONDITIONS), choices=list(TEST_CONDITIONS))
    ap.add_argument("--policies", nargs="+", default=list(BASELINE_NAMES), choices=list(BASELINE_NAMES))
    ap.add_argument("--benchmark-steps", type=int, default=200)
    ap.add_argument("--max-minutes", type=float, default=45.0, help="refuse to start above this estimate")
    ap.add_argument("--force", action="store_true", help="start even if the estimate exceeds --max-minutes")
    ap.add_argument("--resume", action="store_true", help="skip tasks that already have result files")
    ap.add_argument("--out", type=Path, default=baselines_dir())
    args = ap.parse_args()

    seeds = TEST_SEEDS[: args.episodes]
    task_dir = args.out / "_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)

    # 1. Quiet benchmark of every condition (before any worker starts), then the runtime estimate.
    speed = {}
    for cond in args.conditions:
        speed[cond] = benchmark(cond, args.benchmark_steps)
    planned = {c: len(args.policies) * len(seeds) * condition_duration(c) for c in args.conditions}
    serial_s = sum(planned[c] / speed[c] for c in args.conditions)
    speedup = min(float(args.workers), ASSUMED_SPEEDUP_CAP)
    print("condition   steps/s  planned steps (upper bound)")
    for c in args.conditions:
        print(f"{c:<10} {speed[c]:>8.1f}  {planned[c]:>8d}")
    print(
        f"total planned steps {sum(planned.values())}, serial estimate {serial_s / 60:.1f} min, "
        f"estimate with {args.workers} workers (speedup capped at {speedup:.1f}x) {serial_s / speedup / 60:.1f} min",
        flush=True,
    )
    if serial_s / speedup / 60 > args.max_minutes and not args.force:
        sys.exit(f"Estimate exceeds {args.max_minutes} min; reduce the plan or pass --force.")

    # 2. Tasks (policy x condition x seed chunk), heaviest first; skip finished ones on --resume.
    tasks, stems, cost = [], [], []
    for cond in args.conditions:
        for policy in args.policies:
            for idx, chunk in enumerate(make_chunks(seeds, args.chunk_size)):
                tasks.append((policy, cond, chunk))
                stems.append(task_stem(policy, cond, idx))
                cost.append(len(chunk) * condition_duration(cond) / speed[cond])
    order = sorted(range(len(tasks)), key=lambda i: -cost[i])
    todo = [i for i in order if not (args.resume and (task_dir / f"{stems[i]}.csv").exists())]
    print(f"{len(todo)} of {len(tasks)} tasks to run", flush=True)

    t0 = time.perf_counter()
    run_tasks(
        run_baseline_chunk,
        [tasks[i] for i in todo],
        workers=args.workers,
        desc="baselines",
        progress=True,
        on_result=lambda j, res: write_task(task_dir, stems[todo[j]], tasks[todo[j]], res),
    )
    wall_total = time.perf_counter() - t0

    # 3. Assemble per-condition CSVs, summary and metadata from the task files.
    summary_rows, meta = [], {"workers": args.workers, "wall_s_this_run": wall_total, "conditions": {}}
    for cond in args.conditions:
        frames, steps, wall = [], 0, 0.0
        for policy in args.policies:
            for idx in range(len(make_chunks(seeds, args.chunk_size))):
                stem = task_stem(policy, cond, idx)
                frames.append(pd.read_csv(task_dir / f"{stem}.csv"))
                m = json.loads((task_dir / f"{stem}.json").read_text())
                steps, wall = steps + m["steps"], wall + m["wall_s"]
        df = add_return_columns(pd.concat(frames, ignore_index=True))
        df["policy"] = pd.Categorical(df["policy"], categories=list(args.policies), ordered=True)
        df = df.sort_values(["policy", "seed"]).reset_index(drop=True)
        df["policy"] = df["policy"].astype(str)
        df.to_csv(args.out / f"{cond}.csv", index=False)
        meta["conditions"][cond] = {
            "steps": steps,
            "task_wall_s_sum": wall,
            "steps_per_s_per_process": steps / wall if wall else None,
            "quiet_benchmark_steps_per_s": speed[cond],
        }
        for policy, pdf in df.groupby("policy", sort=False):
            row = {"policy": policy, "condition": cond, **summarize_metrics(pdf)}
            row["crash_ci_low"], row["crash_ci_high"] = wilson_interval(int(pdf["crashed"].sum()), len(pdf))
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out / "summary.csv", index=False)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    show = ["policy", "condition", "crash_rate", "completion_rate", "mean_speed", "crashes_per_km", "near_miss_rate"]
    print(summary[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wall time of this run: {wall_total / 60:.1f} min")
    for cond, m in meta["conditions"].items():
        print(f"  {cond:<8} {m['task_wall_s_sum'] / 60:6.1f} process-min, {m['steps_per_s_per_process']:.1f} steps/s per process")


if __name__ == "__main__":
    main()
