"""P4: evaluate every finished final SB3 model and the scripted baselines on all five TEST_CONDITIONS.

IMPORTANT: calibration never touches TEST_SEEDS. Every episode uses the 20 ``VAL_SEEDS``.

    results/calibration/eval/sb3/<algo>/<reward>/seed<s>/<condition>.csv    per-episode records, return_<preset> columns
    results/calibration/eval/baselines/<condition>.csv                      the four scripted baselines (column ``policy``)
    results/calibration/eval/baselines/random_val_main.json                 random reference of the early warning

    python -m scripts.calibration.evaluate_final --workers 8 [--resume] [--baselines-only]

Models are evaluated with ``common.evaluate.evaluate(agent, condition, <training reward>, seeds=VAL_SEEDS)`` where the agent is
the light copy of the greedy network (checked against ``model.predict`` first). Also stores the inference time
(``common.benchmark.measure_inference``, only meaningful in a quiet process) in each run's ``run_meta.json``.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

from common.baselines import BASELINE_NAMES, run_baseline_chunk
from common.config import CALIBRATION_SEEDS, REWARDS, TEST_CONDITIONS, VAL_SEEDS
from common.evaluate import evaluate, make_chunks, records_to_frame
from common.metrics import add_return_columns, recompute_return
from common.parallel import run_tasks
from scripts.calibration import sb3_cal

BASELINE_CHUNK = 10


def condition_duration(condition: str) -> int:
    return int(TEST_CONDITIONS[condition]["overrides"].get("duration", 30))


def run_baselines(workers: int, root: Path = sb3_cal.CAL_ROOT, resume: bool = True) -> Path:
    """Scripted baselines on VAL_SEEDS under every condition; per-condition CSVs with a ``policy`` column."""
    out = sb3_cal.baselines_dir(root)
    task_dir = out / "_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    tasks, stems = [], []
    for cond in TEST_CONDITIONS:
        for policy in BASELINE_NAMES:
            for i, chunk in enumerate(make_chunks(VAL_SEEDS, BASELINE_CHUNK)):
                tasks.append((policy, cond, chunk))
                stems.append(f"{cond}__{policy}__{i:02d}")
    order = sorted(range(len(tasks)), key=lambda i: -condition_duration(tasks[i][1]))  # heaviest (long) first
    todo = [i for i in order if not (resume and (task_dir / f"{stems[i]}.csv").exists())]

    def write(j: int, res: dict) -> None:
        policy, cond, _ = tasks[todo[j]]
        df = records_to_frame(res["records"], cond, "default")
        df.insert(0, "policy", policy)
        tmp = task_dir / f"{stems[todo[j]]}.csv.tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, task_dir / f"{stems[todo[j]]}.csv")

    if todo:
        run_tasks(run_baseline_chunk, [tasks[i][:3] for i in todo], workers=workers, desc="baselines", progress=True, on_result=write)
    for cond in TEST_CONDITIONS:
        frames = [pd.read_csv(task_dir / f"{stems[i]}.csv") for i in range(len(tasks)) if tasks[i][1] == cond]
        df = add_return_columns(pd.concat(frames, ignore_index=True))
        df["policy"] = pd.Categorical(df["policy"], categories=list(BASELINE_NAMES), ordered=True)
        df = df.sort_values(["policy", "seed"]).reset_index(drop=True)
        df["policy"] = df["policy"].astype(str)
        df.to_csv(out / f"{cond}.csv", index=False)
    write_random_reference(out)
    return out


def write_random_reference(out: Path) -> None:
    """Random baseline on VAL_SEEDS / main: crash rate and mean return (+ standard error) under every reward preset."""
    df = pd.read_csv(out / "main.csv")
    rnd = df[df["policy"] == "random"]
    ref = {}
    for reward in REWARDS:
        r = recompute_return(rnd, reward)
        ref[reward] = {"crash_rate": float(rnd["crashed"].mean()), "return": float(r.mean()),
                       "return_se": float(r.std(ddof=1) / math.sqrt(len(r))), "n": int(len(r))}
    (out / "random_val_main.json").write_text(json.dumps(ref, indent=2), encoding="utf-8")


def evaluate_model(algo: str, reward: str, seed: int, workers: int, root: Path = sb3_cal.CAL_ROOT, resume: bool = True) -> None:
    import numpy as np

    rdir, edir = sb3_cal.run_dir(algo, reward, seed, root), sb3_cal.eval_dir(algo, reward, seed, root)
    edir.mkdir(parents=True, exist_ok=True)
    model = sb3_cal.load_sb3_model(algo, rdir / "final.zip")
    light = sb3_cal.light_agent_from_model(model, algo)
    obs = np.random.default_rng(1).random((500, 5, 5)).astype(np.float32)
    bad = sb3_cal.check_same_actions(model, light, obs)
    if bad:
        raise RuntimeError(f"{algo}/{reward}/seed{seed}: light agent and model.predict disagree on {bad}/500 observations")
    for cond in TEST_CONDITIONS:
        path = edir / f"{cond}.csv"
        if resume and path.exists():
            continue
        df = evaluate(light, cond, reward, seeds=VAL_SEEDS, workers=workers, chunk_size=3)
        tmp = path.with_suffix(".csv.tmp")
        add_return_columns(df).to_csv(tmp, index=False)
        os.replace(tmp, path)
    # inference time (quiet process), once
    from common.benchmark import measure_inference, record_val_observations

    meta_path = rdir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("inference"):
        from common.run_meta import write_meta

        obs_batch = record_val_observations(light, n_episodes=3)
        meta["inference"] = measure_inference(light, obs_batch, n=1000, warmup=100)
        write_meta(meta_path, meta)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="skip conditions whose CSV already exists")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else sb3_cal.CAL_ROOT
    t0 = time.perf_counter()
    run_baselines(args.workers, root, resume=True)
    print(f"baselines done ({time.perf_counter() - t0:.0f} s)", flush=True)
    if args.baselines_only:
        return
    n = 0
    for algo in sb3_cal.ALGOS:
        for reward in REWARDS:
            for seed in CALIBRATION_SEEDS:
                if sb3_cal.is_finished(algo, reward, seed, root):
                    t1 = time.perf_counter()
                    evaluate_model(algo, reward, seed, args.workers, root, resume=args.resume)
                    n += 1
                    print(f"evaluated {algo}/{reward}/seed{seed} on {len(TEST_CONDITIONS)} conditions ({time.perf_counter() - t1:.0f} s)", flush=True)
    print(f"P4 done: {n} models, {time.perf_counter() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
