"""Orchestrator of the SB3 calibration plan: quiet benchmark -> baselines -> schedule -> training runs -> P4 evaluation.

    python -m scripts.calibration.run_plan                # whole plan (benchmark, baselines, runs, P4)
    python -m scripts.calibration.run_plan                # run it again after a crash / sleep: finished runs are skipped,
                                                          # unfinished ones resume from their checkpoint
    python -m scripts.calibration.run_plan --dry-run      # benchmark + estimate + schedule only, start nothing

Rules (CLAUDE.md performance rules apply): the benchmark runs first, in a quiet process; the estimate and the schedule are
printed before anything starts; a PPO run counts as 8 busy processes and a DQN run as 1, DQN runs go in parallel with each
other and with at most one PPO run, heaviest first; the wall-time limit is enforced by dropping runs from the end of the
priority list BEFORE the start (never by shortening the 200k budget); progress is printed at most every 30 s with an ETA of
the whole plan. Every run is its own subprocess (``train_sb3.py``) that checkpoints about every 10 min, so a crash or a
sleeping laptop costs at most one interval; a crashed run is restarted from its checkpoint (up to 3 times).
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import collections
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from common.paths import ROOT
from scripts.calibration import plan as planlib
from scripts.calibration import sb3_cal

PROGRESS_PRINT_EVERY = 30.0
POLL_EVERY = 5.0
MAX_RESTARTS = 3
EVAL_RESERVE_S = 20 * 60  # time kept for P4 when the deadline hits


def available_memory_gb() -> float:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        return stat.ullAvailPhys / 1024**3
    except Exception:  # noqa: BLE001  (non-Windows: no limit)
        return float("inf")


def fmt_h(seconds: float) -> str:
    return f"{seconds / 3600:.2f} h ({seconds / 60:.0f} min)"


def print_plan(plan: planlib.Plan, bench: dict, limit_h: float) -> None:
    print("\n=== Plan estimate (cost model: scripts/calibration/plan.py) ===")
    print(f"quiet benchmark: env {bench['env_steps_per_s']:.1f} steps/s (1 process); DQN training {bench['dqn_train_steps_per_s']:.1f} "
          f"steps/s (update {1000 * bench['dqn_update_s_per_step']:.2f} ms/step, validation {bench['dqn_val_s']:.1f} s each); "
          f"PPO training {bench['ppo_train_steps_per_s']:.1f} steps/s with 8 env processes (update {bench['ppo_update_s_per_rollout']:.2f} s per "
          f"rollout, validation {bench['ppo_val_s']:.1f} s each); 8-process aggregate = {bench['aggregate_over_single']:.2f} x one process")
    print("DQN concurrency candidates (PPO lane: one PPO run at a time):")
    for c in plan.candidates:
        mark = "  <== chosen" if c["max_dqn"] == plan.max_dqn else ""
        print(f"  {c['max_dqn']} DQN + 1 PPO: makespan {c['makespan_h']:.2f} h, peak {c['peak_load']} busy processes, mean "
              f"{c['mean_load']:.1f}, memory ~{c['memory_gb']:.1f} GB{'' if c['fits_memory'] else ' (does not fit)'}{mark}")
    print("Schedule (start / end in hours after the start; quiet = wall time alone):")
    for r in sorted(plan.runs, key=lambda r: plan.schedule.start_s[r.key]):
        print(f"  P{r.priority} {r.key:<28} weight {r.weight}  quiet {r.quiet_s / 60:6.1f} min   start {plan.schedule.start_s[r.key] / 3600:5.2f} h   "
              f"end {plan.schedule.end_s[r.key] / 3600:5.2f} h")
    print(f"training makespan {fmt_h(plan.schedule.makespan_s)} + P4 evaluation / baselines {fmt_h(plan.eval_s)} = "
          f"{fmt_h(plan.total_s)}  (limit {limit_h:.1f} h)")
    print("dropped runs: " + (", ".join(plan.dropped) if plan.dropped else "none"), flush=True)


class Worker:
    def __init__(self, run: planlib.PlannedRun, root: Path) -> None:
        self.run, self.root = run, root
        self.proc: Optional[subprocess.Popen] = None
        self.restarts = 0
        self.t_start = time.time()
        self.log = None

    def launch(self) -> None:
        rdir = sb3_cal.run_dir(self.run.algo, self.run.reward, self.run.seed, self.root)
        rdir.mkdir(parents=True, exist_ok=True)
        self.log = open(rdir / "train_stdout.log", "a", encoding="utf-8")
        cmd = [sys.executable, "-m", "scripts.calibration.train_sb3", "--algo", self.run.algo, "--reward", self.run.reward,
               "--seed", str(self.run.seed), "--resume"]
        if self.root != sb3_cal.CAL_ROOT:
            cmd += ["--root", str(self.root)]
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT)

    def progress(self) -> Optional[dict]:
        p = sb3_cal.run_dir(self.run.algo, self.run.reward, self.run.seed, self.root) / "progress.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()  # the PPO env subprocesses are daemonic children and die with the parent
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.log:
            self.log.close()


def run_training(plan: planlib.Plan, bench: dict, root: Path, deadline_s: float, t_plan0: float) -> dict:
    """Run the lanes; returns per-run actual start / end / wall (seconds since t_plan0)."""
    curve = planlib.aggregate_curve(bench)
    queue = {"ppo": [r for r in plan.runs if r.algo == "ppo" and not sb3_cal.is_finished(r.algo, r.reward, r.seed, root)],
             "dqn": [r for r in plan.runs if r.algo == "dqn" and not sb3_cal.is_finished(r.algo, r.reward, r.seed, root)]}
    limit = {"ppo": 1, "dqn": plan.max_dqn}
    active: dict[str, Worker] = {}
    actual: dict[str, dict] = {r.key: {} for r in plan.runs}
    for r in plan.runs:
        if sb3_cal.is_finished(r.algo, r.reward, r.seed, root):
            actual[r.key]["status"] = "finished (before this invocation)"
    history: collections.deque = collections.deque(maxlen=120)  # (time, {key: fraction})
    last_print = 0.0
    warned: set[str] = set()
    stopped_for_deadline = False
    while queue["ppo"] or queue["dqn"] or active:
        now = time.time()
        if not stopped_for_deadline and now - t_plan0 > deadline_s - EVAL_RESERVE_S:
            stopped_for_deadline = True
            print(f"\n!!! wall-time limit reached ({(now - t_plan0) / 3600:.2f} h): stopping unfinished runs "
                  f"({', '.join(active) or 'none'}); their checkpoints stay resumable. Unstarted runs are dropped. !!!", flush=True)
            for w in list(active.values()):
                w.kill()
                actual[w.run.key].update(status="stopped at the wall-time limit", end_s=now - t_plan0)
            active.clear()
            queue["ppo"].clear()
            queue["dqn"].clear()
            break
        for lane in ("dqn", "ppo"):  # start what fits, in priority order
            while queue[lane] and sum(1 for w in active.values() if w.run.algo == lane) < limit[lane]:
                r = queue[lane].pop(0)
                w = Worker(r, root)
                w.launch()
                active[r.key] = w
                actual[r.key].update(start_s=now - t_plan0, status="running")
                print(f"[{(now - t_plan0) / 60:6.1f} min] started {r.key} (busy processes: "
                      f"{sum(x.run.weight for x in active.values())})", flush=True)
        for key, w in list(active.items()):
            code = w.proc.poll()
            if code is None:
                continue
            w.log.close()
            if sb3_cal.is_finished(w.run.algo, w.run.reward, w.run.seed, root):
                prog = w.progress() or {}
                actual[key].update(status="finished", end_s=time.time() - t_plan0, wall_s=time.time() - w.t_start,
                                   steps_per_s=prog.get("steps_per_s"))
                print(f"[{(time.time() - t_plan0) / 60:6.1f} min] finished {key} after {(time.time() - w.t_start) / 60:.1f} min", flush=True)
                del active[key]
            elif w.restarts < MAX_RESTARTS:
                w.restarts += 1
                print(f"[{(time.time() - t_plan0) / 60:6.1f} min] {key} exited with code {code}: restarting from its checkpoint "
                      f"({w.restarts}/{MAX_RESTARTS})", flush=True)
                w.launch()
            else:
                actual[key].update(status=f"failed (exit code {code})", end_s=time.time() - t_plan0)
                print(f"!!! {key} failed {MAX_RESTARTS + 1} times; see train_stdout.log. Continuing with the rest. !!!", flush=True)
                del active[key]
        # progress line at most every PROGRESS_PRINT_EVERY seconds, with an ETA for the whole plan
        if time.time() - last_print >= PROGRESS_PRINT_EVERY and active:
            last_print = time.time()
            frac, parts = {}, []
            for r in plan.runs:
                if sb3_cal.is_finished(r.algo, r.reward, r.seed, root):
                    frac[r.key] = 1.0
            for key, w in active.items():
                p = w.progress()
                f = (p["env_step"] / p["total"]) if p else 0.0
                frac[key] = f
                v = (p or {}).get("last_val")
                parts.append(f"{key.replace('/seed', '/s')} {100 * f:3.0f}%" + (f" crash {v['crash_rate']:.2f} v {v['mean_speed']:.1f}" if v else ""))
                if (sb3_cal.run_dir(w.run.algo, w.run.reward, w.run.seed, root) / "WARNING_random_level.txt").exists() and key not in warned:
                    warned.add(key)
                    print(f"!!! EARLY WARNING: {key} is still at the random-baseline level at 100k steps (run continues) !!!", flush=True)
            history.append((last_print, dict(frac)))
            eta = estimate_remaining(plan, queue, active, frac, history, curve, bench)
            print(f"[{(last_print - t_plan0) / 60:6.1f} min] " + " | ".join(parts) + f" | plan ETA ~{eta / 60:.0f} min (+P4 {plan.eval_s / 60:.0f} min)", flush=True)
        time.sleep(POLL_EVERY)
    return actual


def estimate_remaining(plan, queue, active, frac, history, curve, bench) -> float:
    """Model-based remaining training time, rescaled by the observed / predicted progress rate of the last minutes."""
    runs = []
    for r in plan.runs:
        c = planlib.PlannedRun(r.algo, r.reward, r.seed, r.priority, r.quiet_s, done=frac.get(r.key, 0.0))
        runs.append(c)
    sched = planlib.simulate(runs, plan.max_dqn, curve)
    pred = sched.makespan_s
    # calibration factor from the history window (>= 120 s)
    factor = 1.0
    if len(history) >= 2 and history[-1][0] - history[0][0] >= 120:
        (t0, f0), (t1, f1) = history[0], history[-1]
        dt = t1 - t0
        n = sum(r.weight for r in runs if r.key in active)
        obs = sum((f1.get(k, 0) - f0.get(k, 0)) / dt for k in active)
        prd = sum(planlib.speed_rel(n, curve) / (r.quiet_s * planlib.speed_rel(r.weight, curve)) for r in runs if r.key in active)
        if obs > 0 and prd > 0:
            factor = min(max(obs / prd, 0.2), 5.0)
    return pred / factor


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="benchmark, estimate and show the schedule; start nothing")
    ap.add_argument("--rebench", action="store_true", help="measure again even if benchmark.json exists")
    ap.add_argument("--limit-h", type=float, default=planlib.WALL_LIMIT_H, help="hard wall-time limit of the whole plan")
    ap.add_argument("--no-eval", action="store_true", help="skip P4")
    ap.add_argument("--dqn-slots", type=int, default=None, help="force the number of concurrent DQN runs (default: best that fits the memory)")
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else sb3_cal.CAL_ROOT
    root.mkdir(parents=True, exist_ok=True)
    t_plan0 = time.time()

    bench_path = root / "benchmark.json"
    if args.rebench or not bench_path.exists():
        print("measuring speeds in a quiet process (do not run anything else now) ...", flush=True)
        from scripts.calibration.benchmark import measure

        bench = measure()
        bench_path.write_text(json.dumps(bench, indent=2), encoding="utf-8")
    else:
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        print(f"using the benchmark of {bench['time']} ({bench_path})")

    plan = planlib.make_plan(bench, available_gb=available_memory_gb(), limit_h=args.limit_h, force_dqn=args.dqn_slots)
    print_plan(plan, bench, args.limit_h)
    (root / "plan.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "limit_h": args.limit_h, "benchmark": bench, "max_dqn": plan.max_dqn,
        "candidates": plan.candidates, "dropped": plan.dropped, "estimated_training_h": plan.schedule.makespan_s / 3600,
        "estimated_total_h": plan.total_s / 3600,
        "runs": [{"key": r.key, "priority": r.priority, "weight": r.weight, "quiet_s": r.quiet_s,
                  "est_start_h": plan.schedule.start_s[r.key] / 3600, "est_end_h": plan.schedule.end_s[r.key] / 3600} for r in plan.runs],
    }, indent=2), encoding="utf-8")
    if args.dry_run:
        return

    from scripts.calibration import evaluate_final

    if not (root / "eval" / "baselines" / "random_val_main.json").exists():
        print("\nbaselines on VAL_SEEDS (5 conditions) ...", flush=True)
        evaluate_final.run_baselines(workers=8, root=root)
    t_train0 = time.time()
    print(f"\nstarting the training runs (limit {args.limit_h:.1f} h for the whole plan)", flush=True)
    actual = run_training(plan, bench, root, args.limit_h * 3600 - (t_train0 - t_plan0), t_train0)
    train_wall = time.time() - t_train0
    print(f"\ntraining wall time {fmt_h(train_wall)} (estimated {fmt_h(plan.schedule.makespan_s)})", flush=True)

    if not args.no_eval:
        print("P4: evaluating the finished final models and the baselines ...", flush=True)
        t_eval0 = time.time()
        subprocess.run([sys.executable, "-m", "scripts.calibration.evaluate_final", "--workers", "8", "--resume"]
                       + (["--root", str(root)] if args.root else []), cwd=ROOT, check=False)
        print(f"P4 wall time {fmt_h(time.time() - t_eval0)}", flush=True)
    total = time.time() - t_plan0
    summary = {"training_wall_s": train_wall, "total_wall_s": total, "runs": actual}
    (root / "plan_actual.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"whole plan: {fmt_h(total)}", flush=True)


if __name__ == "__main__":
    main()
