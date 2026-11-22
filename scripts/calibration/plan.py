"""Run list, cost model and schedule simulation of the SB3 calibration plan (pure functions, no I/O, no simulation).

Priority order (the plan drops runs from the END of this list when the estimate exceeds the wall-time limit):
    P1  balanced x {DQN, PPO}, seed 99
    P2  efficiency, safety, default x {DQN, PPO}, seed 99
    P3  balanced x {DQN, PPO}, seed 98
Dropping order when too long: P3 first, then the "default" reward of P2 (both algorithms each time). The 200k budget of
the remaining runs is never shortened.

Cost model. ``bench`` = the quiet measurements of ``benchmark.py``. A run's quiet duration ``D`` is its wall time if it
ran alone (a DQN run on one process, a PPO run on its 8 env processes). When ``n`` processes are busy in total (a PPO
run counts 8, a DQN run 1), every process runs at the relative speed ``speed_rel(n) = aggregate(n) / n`` of the
measured scaling curve (docs/performance_analysis.md C5: aggregate steps/s relative to one process = 1.0 / 2.7 / 3.5 /
3.2 / 2.9 for 1 / 4 / 8 / 12 / 16 processes, rescaled so that the value at 8 equals the aggregate measured by the PPO
benchmark). A run of weight ``w`` progresses at ``speed_rel(n) / (D * speed_rel(w))`` of its work per second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from scripts.calibration import sb3_cal

# aggregate throughput relative to one process (docs/performance_analysis.md C5); the value at 8 is replaced by the measurement
SCALING_POINTS = {1: 1.0, 4: 2.7, 8: 3.5, 12: 3.2, 16: 2.9}
PPO_WEIGHT, DQN_WEIGHT = 8, 1
WALL_LIMIT_H = 7.0
EXPLICIT_REWARD_ORDER_P2 = ("efficiency", "safety", "default")


@dataclass
class PlannedRun:
    algo: str
    reward: str
    seed: int
    priority: int  # 1, 2 or 3
    quiet_s: float = 0.0  # wall time if run alone
    done: float = 0.0  # fraction already finished (progress files / finished runs)

    @property
    def weight(self) -> int:
        return PPO_WEIGHT if self.algo == "ppo" else DQN_WEIGHT

    @property
    def key(self) -> str:
        return f"{self.algo}/{self.reward}/seed{self.seed}"


def build_runs(include_p3: bool = True, include_default: bool = True) -> list[PlannedRun]:
    """The priority list. Within a priority group the heavier algorithm (DQN: one process, longest) comes first."""
    runs = [PlannedRun("dqn", "balanced", 99, 1), PlannedRun("ppo", "balanced", 99, 1)]
    for reward in EXPLICIT_REWARD_ORDER_P2:
        if reward == "default" and not include_default:
            continue
        runs += [PlannedRun("dqn", reward, 99, 2), PlannedRun("ppo", reward, 99, 2)]
    if include_p3:
        runs += [PlannedRun("dqn", "balanced", 98, 3), PlannedRun("ppo", "balanced", 98, 3)]
    return runs


def n_val_points(algo: str) -> int:
    return max(sb3_cal.total_steps(algo) // sb3_cal.VAL_EVERY_ENV_STEPS, 1)


def quiet_duration(algo: str, bench: dict) -> float:
    """Wall seconds of one run alone on a quiet machine: training (with updates) + all validations."""
    total = sb3_cal.total_steps(algo)
    if algo == "dqn":
        return total / bench["dqn_train_steps_per_s"] + n_val_points("dqn") * bench["dqn_val_s"]
    return total / bench["ppo_train_steps_per_s"] + n_val_points("ppo") * bench["ppo_val_s"]


def aggregate_curve(bench: Optional[dict]) -> dict[int, float]:
    pts = dict(SCALING_POINTS)
    if bench and "aggregate_over_single" in bench:
        pts[8] = min(max(float(bench["aggregate_over_single"]), 2.0), 5.0)
    return pts


def speed_rel(n: float, curve: dict[int, float]) -> float:
    """Relative speed of one process when ``n`` processes are busy (1.0 = alone)."""
    xs = sorted(curve)
    if n <= xs[0]:
        agg = curve[xs[0]] * n / xs[0]
    elif n >= xs[-1]:
        agg = curve[xs[-1]]
    else:
        agg = next(
            curve[a] + (curve[b] - curve[a]) * (n - a) / (b - a) for a, b in zip(xs[:-1], xs[1:]) if a <= n <= b
        )
    return min(agg / n, 1.0)


def eval_overhead_s(bench: dict, n_models: int) -> float:
    """P4 + baselines: every final model and the four baselines on 5 conditions x 20 episodes (long = 60 steps)."""
    from common.config import TEST_CONDITIONS

    steps_per_policy = sum(
        20 * int(TEST_CONDITIONS[c]["overrides"].get("duration", 30)) for c in TEST_CONDITIONS
    )
    agg = aggregate_curve(bench)[8] * bench["env_steps_per_s"]  # 8 workers
    return (n_models + 4) * steps_per_policy / agg + 60.0 * (n_models + 4) / 4  # + pool start-up per condition


@dataclass
class Schedule:
    start_s: dict[str, float] = field(default_factory=dict)
    end_s: dict[str, float] = field(default_factory=dict)
    makespan_s: float = 0.0
    busy_integral: float = 0.0  # process-seconds, for the mean load
    peak_load: int = 0

    @property
    def mean_load(self) -> float:
        return self.busy_integral / self.makespan_s if self.makespan_s else 0.0


def simulate(runs: Sequence[PlannedRun], max_dqn: int, curve: dict[int, float], max_ppo: int = 1) -> Schedule:
    """Two lanes in priority order: PPO runs (at most ``max_ppo`` at a time) and DQN runs (at most ``max_dqn`` at a time).

    Event-driven; ``run.done`` is the fraction already finished (0 for a fresh plan).
    """
    pending = {"ppo": [r for r in runs if r.algo == "ppo" and r.done < 1], "dqn": [r for r in runs if r.algo == "dqn" and r.done < 1]}
    limit = {"ppo": max_ppo, "dqn": max_dqn}
    active: list[tuple[PlannedRun, float]] = []  # (run, remaining fraction)
    sched = Schedule()
    t = 0.0
    while pending["ppo"] or pending["dqn"] or active:
        for lane in ("dqn", "ppo"):  # start what fits, in priority order within a lane
            while pending[lane] and sum(1 for r, _ in active if r.algo == lane) < limit[lane]:
                r = pending[lane].pop(0)
                active.append((r, 1.0 - r.done))
                sched.start_s[r.key] = t
        n = sum(r.weight for r, _ in active)
        sched.peak_load = max(sched.peak_load, n)
        rates = [speed_rel(n, curve) / (r.quiet_s * speed_rel(r.weight, curve)) for r, _ in active]
        dt = min(rem / rate for (_, rem), rate in zip(active, rates))
        t += dt
        sched.busy_integral += n * dt
        nxt = []
        for (r, rem), rate in zip(active, rates):
            rem -= rate * dt
            if rem <= 1e-9:
                sched.end_s[r.key] = t
            else:
                nxt.append((r, rem))
        active = nxt
    sched.makespan_s = t
    return sched


def memory_estimate_gb(max_dqn: int, ppo: bool = True) -> float:
    """Rough working set: a DQN run ~0.45 GB, a PPO run ~0.4 GB main + 8 x 0.27 GB env processes + the validation pool (4 x 0.3 GB)."""
    return max_dqn * 0.45 + (0.4 + PPO_WEIGHT * 0.27 + sb3_cal.VAL_POOL_WORKERS * 0.3 if ppo else 0.0)


@dataclass
class Plan:
    runs: list[PlannedRun]
    max_dqn: int
    schedule: Schedule
    eval_s: float
    total_s: float
    dropped: list[str]
    candidates: list[dict]


def make_plan(bench: dict, available_gb: float = math.inf, limit_h: float = WALL_LIMIT_H,
              dqn_slots: Sequence[int] = (2, 3, 4, 5), force_dqn: Optional[int] = None) -> Plan:
    """Build the run list, pick the DQN concurrency with the smallest makespan that fits the memory, trim to the limit."""
    curve = aggregate_curve(bench)
    dropped: list[str] = []
    for include_p3, include_default, label in ((True, True, None), (False, True, "P3: balanced seed 98 (DQN, PPO)"),
                                               (False, False, "P2: default reward (DQN, PPO)")):
        if label:
            dropped.append(label)
        runs = build_runs(include_p3, include_default)
        for r in runs:
            r.quiet_s = quiet_duration(r.algo, bench)
        n_dqn = sum(1 for r in runs if r.algo == "dqn")
        cands = []
        for k in dqn_slots:
            k = min(k, n_dqn)
            sched = simulate(runs, k, curve)
            mem = memory_estimate_gb(k)
            cands.append({"max_dqn": k, "makespan_h": sched.makespan_s / 3600, "peak_load": sched.peak_load,
                          "mean_load": sched.mean_load, "memory_gb": mem, "fits_memory": mem <= 0.7 * available_gb})
        ok = [c for c in cands if c["fits_memory"]] or [min(cands, key=lambda c: c["memory_gb"])]
        best = min(ok, key=lambda c: (c["makespan_h"], c["max_dqn"]))
        if force_dqn is not None:  # the user's choice overrides the memory filter (the candidates are still listed)
            best = next(c for c in cands if c["max_dqn"] == min(force_dqn, n_dqn))
        sched = simulate(runs, best["max_dqn"], curve)
        ev = eval_overhead_s(bench, len(runs))
        total = sched.makespan_s + ev
        if total / 3600 <= limit_h:
            return Plan(runs, best["max_dqn"], sched, ev, total, list(dropped), cands)
    raise RuntimeError(f"even without P3 and the default reward the estimate is {total / 3600:.1f} h > {limit_h} h")
