"""Analysis of the SB3 calibration runs -> docs/calibration.md, results/calibration/plots/ and docs/calibration_evidence/.

    python -m scripts.calibration.analyze

Reads only files below results/calibration/ (training logs, per-episode validation records, evaluation CSVs on VAL_SEEDS,
the baselines, run_meta.json, benchmark.json, plan_actual.json). Never touches TEST_SEEDS and writes nothing into
common/config.py or common/train_config.py: the "Proposals" section is text for the team.

Calibration only: SB3 DQN is vanilla DQN with a target network, NOT Double DQN; SB3 PPO is a reference PPO. Neither enters
the DDQN vs PPO comparison.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from common.actions import ACTION_NAMES
from common.aggregate import PLATEAU_TOL, PLATEAU_WINDOW, steps_to_plateau, steps_to_threshold, t_quantile_975
from common.config import CALIBRATION_SEEDS, REWARDS, TARGET_CONSECUTIVE, TARGET_PERFORMANCE, TEST_CONDITIONS
from common.metrics import add_return_columns, summarize_metrics
from common.paths import ROOT
from scripts.calibration import sb3_cal
from scripts.calibration import stats as st

# Small evidence copy that IS committed (results/ is git-ignored): the plots and, per run, val_log.csv and run_meta.json.
EVIDENCE_DIR = ROOT / "docs" / "calibration_evidence"
EVIDENCE_RUN_FILES = ("val_log.csv", "run_meta.json")

ALGO_LABEL = {"dqn": "SB3 DQN (vanilla, not Double)", "ppo": "SB3 PPO"}
ALGO_SHORT = {"dqn": "DQN", "ppo": "PPO"}
CONDITIONS = list(TEST_CONDITIONS)
BASELINES = ["random", "idle", "slower", "faster"]
REWARD_ORDER = ["default", "efficiency", "balanced", "safety"]
BEHAVIOUR_COLS = ["crash_rate", "completion_rate", "mean_speed", "lane_changes", "lane_change_reversals",
                  "rightmost_lane_share", "overtakes_per_km", "near_miss_rate", "crashes_per_km"]
RETURN_COLS = [f"return_{r}" for r in REWARD_ORDER]
STRONG_BASELINE_ANCHOR = [  # task: midway between idle (24.76) and faster (28.9) on VAL_SEEDS
    {"metric": "crash_rate", "operator": "<=", "value": 0.2},
    {"metric": "mean_speed", "operator": ">=", "value": 26.8},
]
STRONG_PLACEHOLDER = [  # the placeholder `strong` level the calibration was planned with; pinned here so the report stays
    {"metric": "crash_rate", "operator": "<=", "value": 0.2},  # reproducible after common/config.py froze another level
    {"metric": "mean_speed", "operator": ">=", "value": 25.5},
]
PRESENT_TARGETS: dict[str, list[dict]] = {
    "beats_idle": TARGET_PERFORMANCE["beats_idle"],
    "strong_placeholder": STRONG_PLACEHOLDER,
    "strong_baseline_anchor": STRONG_BASELINE_ANCHOR,
}
# plot colours (validated default palette, light mode, slots 1-2 of the categorical order) and neutrals for the baselines
C_DQN, C_PPO, C_BASE, C_INK, C_GRID = "#2a78d6", "#eb6834", "#6b6a66", "#0b0b0b", "#e4e3df"


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------
class Data:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.evals: dict[tuple[str, str, int, str], pd.DataFrame] = {}
        self.base: dict[str, pd.DataFrame] = {}
        for algo in sb3_cal.ALGOS:
            for reward in REWARDS:
                for seed in CALIBRATION_SEEDS:
                    if not sb3_cal.is_finished(algo, reward, seed, root):
                        continue
                    rd = sb3_cal.run_dir(algo, reward, seed, root)
                    self.runs[(algo, reward, seed)] = {
                        "val": pd.read_csv(rd / "val_log.csv"),
                        "val_ep": pd.read_csv(rd / "val_episodes.csv"),
                        "meta": json.loads((rd / "run_meta.json").read_text(encoding="utf-8")),
                    }
                    ed = sb3_cal.eval_dir(algo, reward, seed, root)
                    for cond in CONDITIONS:
                        if (ed / f"{cond}.csv").exists():
                            self.evals[(algo, reward, seed, cond)] = pd.read_csv(ed / f"{cond}.csv")
        for cond in CONDITIONS:
            p = sb3_cal.baselines_dir(root) / f"{cond}.csv"
            if p.exists():
                self.base[cond] = pd.read_csv(p)
        self.bench = _load_json(root / "benchmark.json")
        self.plan = _load_json(root / "plan.json")
        self.actual = _load_json(root / "plan_actual.json")

    def seeds(self, algo: str, reward: str) -> list[int]:
        return [s for s in CALIBRATION_SEEDS if (algo, reward, s) in self.runs]

    def summary(self, algo: str, reward: str, seed: int, cond: str = "main") -> Optional[dict]:
        df = self.evals.get((algo, reward, seed, cond))
        return None if df is None else summarize_metrics(add_return_columns(df))

    def base_summary(self, policy: str, cond: str) -> Optional[dict]:
        df = self.base.get(cond)
        if df is None:
            return None
        sub = df[df["policy"] == policy]
        return summarize_metrics(add_return_columns(sub)) if len(sub) else None


def _load_json(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------------------------------------
# markdown helpers
# ---------------------------------------------------------------------------------------------
def fmt(v: Any, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if abs(v) >= 1000 and float(v).is_integer():
            return f"{int(v):,}"
        return f"{v:,.{nd}f}"
    return str(v)


def md_table(rows: list[list[Any]], header: list[str], nd: int = 2) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(c, nd) for c in r) + " |")
    return "\n".join(out)


def pct(v: float) -> str:
    return "n/a" if v is None or math.isnan(v) else f"{100 * v:.0f}%"


# ---------------------------------------------------------------------------------------------
# Q1: rewards
# ---------------------------------------------------------------------------------------------
def action_share(summ: dict) -> dict[str, float]:
    total = sum(summ[f"act_{a}"] for a in ACTION_NAMES)
    return {a: summ[f"act_{a}"] / total for a in ACTION_NAMES}


def q1_tables(d: Data, seed: int = 99) -> tuple[str, dict]:
    text, info = [], {}
    for algo in sb3_cal.ALGOS:
        rows_b, rows_a, rows_r = [], [], []
        per_reward = {}
        for reward in REWARD_ORDER:
            s = d.summary(algo, reward, seed)
            if s is None:
                continue
            per_reward[reward] = s
            rows_b.append([reward, *[s[c] for c in BEHAVIOUR_COLS]])
            sh = action_share(s)
            rows_a.append([reward, *[pct(sh[a]) for a in ACTION_NAMES]])
            rows_r.append([reward, *[s[c] for c in RETURN_COLS]])
        info[algo] = per_reward
        if not rows_b:
            continue
        text.append(f"#### {ALGO_LABEL[algo]}, final models, `main` condition, seed {seed}, 20 `VAL_SEEDS` episodes\n")
        text.append(md_table(rows_b, ["trained with", *BEHAVIOUR_COLS]))
        text.append("\nAction histogram (share of the steps) and return under every preset (rows: training reward; the diagonal "
                    "is the model's own return):\n")
        text.append(md_table([ra + rr[1:] for ra, rr in zip(rows_a, rows_r)],
                             ["trained with", *[a for a in ACTION_NAMES], *RETURN_COLS], nd=1))
        text.append("")
    return "\n".join(text), info


def intended_ordering(per_reward: dict[str, dict]) -> dict[str, Any]:
    """Check the intended ordering of the three custom presets on the final main-condition metrics."""
    need = ["efficiency", "balanced", "safety"]
    if not all(r in per_reward for r in need):
        return {}
    e, b, s = (per_reward[r] for r in need)
    chk = {
        "speed: efficiency > balanced > safety": e["mean_speed"] > b["mean_speed"] > s["mean_speed"],
        "crash rate: efficiency > balanced > safety": e["crash_rate"] > b["crash_rate"] > s["crash_rate"],
        "crashes/km: efficiency > balanced > safety": e["crashes_per_km"] > b["crashes_per_km"] > s["crashes_per_km"],
        "safety has the fewest lane changes": s["lane_changes"] < min(e["lane_changes"], b["lane_changes"]),
        "efficiency fastest": e["mean_speed"] == max(x["mean_speed"] for x in per_reward.values()),
        "safety slowest": s["mean_speed"] == min(x["mean_speed"] for x in per_reward.values()),
    }
    return chk


def noise_se(d: Data, algo: str, reward: str, seed: int, metric: str) -> float:
    """Standard error of a 20-episode mean of ``metric`` of one final model (binomial for the crash rate)."""
    df = d.evals.get((algo, reward, seed, "main"))
    if df is None:
        return math.nan
    if metric == "crash_rate":
        p = float(df["crashed"].mean())
        return math.sqrt(p * (1 - p) / len(df))
    col = {"mean_speed": "mean_speed", "lane_changes": "lane_changes"}[metric]
    return float(df[col].std(ddof=1) / math.sqrt(len(df)))


def q1_spread_comparison(d: Data, per_reward_by_algo: dict) -> tuple[str, list[str]]:
    """Pairwise differences between presets vs the seed-to-seed spread (balanced seed 99 vs 98) and 20-episode noise."""
    metrics = ["crash_rate", "mean_speed", "lane_changes"]
    lines, indist = [], []
    for algo, per_reward in per_reward_by_algo.items():
        if len(per_reward) < 2:
            continue
        s99, s98 = d.summary(algo, "balanced", 99), d.summary(algo, "balanced", 98)
        rows = []
        pairs = [(a, b) for i, a in enumerate(REWARD_ORDER) for b in REWARD_ORDER[i + 1:] if a in per_reward and b in per_reward]
        for a, b in pairs:
            exceed = []
            for m in metrics:
                delta = per_reward[a][m] - per_reward[b][m]
                spread = abs(s99[m] - s98[m]) if (s99 and s98) else math.nan
                noise = 2 * math.sqrt(2) * np.nanmean([noise_se(d, algo, a, 99, m), noise_se(d, algo, b, 99, m)])
                thr = max(spread if not math.isnan(spread) else 0.0, noise)
                exceeds = abs(delta) > thr
                exceed.append(exceeds)
                rows.append([f"{a} - {b}", m, delta, spread, noise, exceeds])
            if not any(exceed):
                indist.append(f"{ALGO_SHORT[algo]}: {a} vs {b}")
        lines.append(f"#### {ALGO_LABEL[algo]}\n")
        lines.append(md_table(rows, ["pair", "metric", "difference", "seed spread |99-98| (balanced)",
                                     "2 sqrt(2) SE (20 episodes)", "exceeds both?"]))
        lines.append("")
    return "\n".join(lines), indist


def q1_return_vs_behaviour(d: Data, seed: int = 99) -> str:
    """'Does a higher return mean better driving?': rank correlation of return_<preset> with behaviour across final models."""
    rows, models = [], []
    for algo in sb3_cal.ALGOS:
        for reward in REWARD_ORDER:
            s = d.summary(algo, reward, seed)
            if s:
                models.append((algo, reward, s))
    if len(models) < 3:
        return "Not enough models.\n"
    lines = [f"Final models of seed {seed}, `main` condition ({len(models)} models of both algorithms; the model's own preset is "
             "marked with *). For every preset p: Spearman correlation across the models between `return_p` and (a) completion rate, "
             "(b) crash-free rate, (c) mean speed, (d) near-miss rate. With few models these are rough orderings, not tests.\n"]
    rows = []
    for p in REWARD_ORDER:
        ret = [m[2][f"return_{p}"] for m in models]
        rows.append([p, st.spearman(ret, [m[2]["completion_rate"] for m in models]),
                     st.spearman(ret, [1 - m[2]["crash_rate"] for m in models]),
                     st.spearman(ret, [m[2]["mean_speed"] for m in models]),
                     st.spearman(ret, [m[2]["near_miss_rate"] for m in models])])
    lines.append(md_table(rows, ["return under", "rho completion", "rho crash-free", "rho mean speed", "rho near-miss rate"]))
    lines.append("\nPer model: return under every preset next to the behaviour (rank of the model by that return in brackets):\n")
    rows = []
    for algo, reward, s in models:
        rks = []
        for p in REWARD_ORDER:
            order = sorted(models, key=lambda m: -m[2][f"return_{p}"])
            rks.append(f"{s[f'return_{p}']:.1f} (#{1 + [(m[0], m[1]) for m in order].index((algo, reward))})")
        rows.append([f"{ALGO_SHORT[algo]} {reward}*", *rks, s["crash_rate"], s["mean_speed"], s["lane_changes"]])
    lines.append(md_table(rows, ["model", *[f"return_{p}" for p in REWARD_ORDER], "crash", "speed", "lane changes"]))
    # the central question: does the ordering by return agree with the ordering by crash rate within an algorithm?
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Q2: conditions
# ---------------------------------------------------------------------------------------------
def q2_tables(d: Data, seed: int = 99) -> tuple[str, list[str], dict]:
    text, flags, info = [], [], {}
    for cond in CONDITIONS:
        rows, crashes_learned, crashes_all = [], [], []
        for algo in sb3_cal.ALGOS:
            for reward in REWARD_ORDER:
                s = d.summary(algo, reward, seed, cond)
                if s:
                    rows.append([f"{ALGO_SHORT[algo]} {reward}", s["crash_rate"], s["completion_rate"], s["mean_speed"], s["crashes_per_km"]])
                    crashes_learned.append(s["crash_rate"])
        for pol in BASELINES:
            s = d.base_summary(pol, cond)
            if s:
                rows.append([f"baseline {pol}", s["crash_rate"], s["completion_rate"], s["mean_speed"], s["crashes_per_km"]])
                crashes_all.append(s["crash_rate"])
        info[cond] = {"learned": crashes_learned, "baselines": crashes_all}
        text.append(f"**`{cond}`**\n")
        text.append(md_table(rows, ["model", "crash rate", "completion", "mean speed [m/s]", "crashes/km"]))
        text.append("")
        if crashes_learned:
            lo, hi = min(crashes_learned), max(crashes_learned)
            allv = crashes_learned + crashes_all
            floor_ceiling = max(allv) <= 0.10 or min(allv) >= 0.90 or max(crashes_learned) <= 0.10 or min(crashes_learned) >= 0.90
            if hi - lo <= 0.10 or floor_ceiling:
                why = ("learned models within 10 points of each other" if hi - lo <= 0.10 else "learned models all near 0% or all near 100%")
                flags.append(f"`{cond}`: crash rate of the learned models {pct(lo)}..{pct(hi)} ({why})")
    return "\n".join(text), flags, info


def q2_gaps(d: Data, seed: int = 99) -> str:
    rows = []
    entries = [(f"{ALGO_SHORT[a]} {r}", lambda c, a=a, r=r: d.summary(a, r, seed, c)) for a in sb3_cal.ALGOS for r in REWARD_ORDER]
    entries += [(f"baseline {p}", lambda c, p=p: d.base_summary(p, c)) for p in BASELINES]
    for name, get in entries:
        m, pr, lg, lo, hi = get("main"), get("precise"), get("long"), get("low"), get("high")
        if not m:
            continue
        ratio = (lg["crashes_per_km"] / m["crashes_per_km"]) if (lg and m["crashes_per_km"] and m["crashes_per_km"] > 0) else math.nan
        rows.append([name, m["crash_rate"], pr["crash_rate"] - m["crash_rate"] if pr else math.nan,
                     lo["crash_rate"] - m["crash_rate"] if lo else math.nan, hi["crash_rate"] - m["crash_rate"] if hi else math.nan,
                     m["crashes_per_km"], lg["crashes_per_km"] if lg else math.nan, ratio])
    return md_table(rows, ["model", "crash rate main", "precise - main", "low - main", "high - main", "crashes/km main",
                           "crashes/km long", "long / main ratio"])


# ---------------------------------------------------------------------------------------------
# Q3: seeds
# ---------------------------------------------------------------------------------------------
def last3(val: pd.DataFrame, col: str) -> float:
    return float(val[col].tail(3).mean())


def q3_section(d: Data) -> tuple[str, dict]:
    lines, info = [], {}
    metrics = ["crash_rate", "completion_rate", "mean_speed", "lane_changes", "near_miss_rate", "return_balanced"]
    for algo in sb3_cal.ALGOS:
        if not (99 in d.seeds(algo, "balanced") and 98 in d.seeds(algo, "balanced")):
            lines.append(f"{ALGO_LABEL[algo]}: seed 98 of the balanced reward is missing, no spread available.\n")
            continue
        s99, s98 = d.summary(algo, "balanced", 99), d.summary(algo, "balanced", 98)
        v99, v98 = d.runs[(algo, "balanced", 99)]["val"], d.runs[(algo, "balanced", 98)]["val"]
        rows = []
        info[algo] = {}
        for m in metrics:
            a, b = s99[m], s98[m]
            sp = st.seed_spread(a, b)
            a3, b3 = last3(v99, m), last3(v98, m)
            sp3 = st.seed_spread(a3, b3)
            info[algo][m] = {"final_eval": sp, "last3": sp3}
            rows.append([m, a, b, sp["abs_diff"], a3, b3, sp3["abs_diff"], sp3["sd"], sp3["halfwidth_ci"]])
        lines.append(f"#### {ALGO_LABEL[algo]}: balanced, seed 99 vs seed 98\n")
        lines.append(md_table(rows, ["metric", "seed 99 (final, 20 ep.)", "seed 98 (final, 20 ep.)", "|diff|",
                                     "seed 99 (mean last 3 val.)", "seed 98 (mean last 3 val.)", "|diff| last 3", "implied sd (|d|/sqrt 2)",
                                     "5-seed 95% t half-width"], nd=3))
        # curve shape: correlation of the two own-reward return curves and of crash-rate curves, and AUC difference
        n = min(len(v99), len(v98))
        rc = st.spearman(v99["return_balanced"][:n], v98["return_balanced"][:n], pearson=True)
        cc = st.spearman(v99["crash_rate"][:n], v98["crash_rate"][:n], pearson=True)
        pl99 = st.plateau_point(v99["env_step"], v99["return_balanced"])
        pl98 = st.plateau_point(v98["env_step"], v98["return_balanced"])
        info[algo]["curve_corr_return"], info[algo]["curve_corr_crash"] = rc, cc
        lines.append(f"\nCurve shapes: correlation of the two `return_balanced` curves over the {n} validation points {rc:.2f}, of the "
                     f"`crash_rate` curves {cc:.2f}; plateau point of `return_balanced` {fmt(pl99['plateau_step'], 0)} vs "
                     f"{fmt(pl98['plateau_step'], 0)} env steps (see Q4). Mean over the whole run (AUC) of `return_balanced`: "
                     f"{v99['return_balanced'].mean():.2f} vs {v98['return_balanced'].mean():.2f}.\n")
    lines.append("**Only two seeds:** sd = |d| / sqrt(2) has one degree of freedom (its 95% interval spans roughly 0.45 x to 32 x the estimate), "
                 "so this is a rough scale, not an estimate. The implied half-width of the 95% t-interval over 5 seeds is "
                 f"t(0.975, 4) = {t_quantile_975(4):.3f} times sd / sqrt(5) = {t_quantile_975(4) / math.sqrt(5) / math.sqrt(2):.3f} x |d|.\n")
    return "\n".join(lines), info


# ---------------------------------------------------------------------------------------------
# Q4: budget, validation, targets
# ---------------------------------------------------------------------------------------------
def own_return_col(reward: str) -> str:
    return f"return_{reward}"


def q4_plateaus(d: Data) -> tuple[str, dict]:
    rows, info = [], {}
    for (algo, reward, seed), run in sorted(d.runs.items()):
        v = run["val"]
        pl = st.plateau_point(v["env_step"], v[own_return_col(reward)])
        last_step = float(v["env_step"].iloc[-1])
        mid = v[(v["env_step"] >= 0.6 * last_step) & (v["env_step"] < 0.8 * last_step)][own_return_col(reward)].mean()
        info[(algo, reward, seed)] = pl
        rows.append([f"{ALGO_SHORT[algo]} {reward} s{seed}", pl["target"], mid, pl["plateau_step"], "yes" if pl["plateaued"] else "NO",
                     fmt(v[own_return_col(reward)].iloc[0], 1), fmt(v[own_return_col(reward)].max(), 1)])
    return md_table(rows, ["run", "mean own return, last 20%", "mean own return, 60-80%", "plateau point [env steps]", "plateaued before the last 20%?",
                           "first val. point", "best val. point"], nd=1), info


def q4_noise(d: Data) -> tuple[str, list[dict]]:
    rows, recs = [], []
    for (algo, reward, seed), run in sorted(d.runs.items()):
        n = st.val_noise(run["val_ep"])
        v = run["val"]
        fl_c = st.tail_fluctuation(v["env_step"], v["crash_rate"])
        fl_s = st.tail_fluctuation(v["env_step"], v["mean_speed"])
        fl_r = st.tail_fluctuation(v["env_step"], v[own_return_col(reward)])
        recs.append({"run": (algo, reward, seed), **n, "fl_crash": fl_c, "fl_speed": fl_s, "fl_return": fl_r})
        rows.append([f"{ALGO_SHORT[algo]} {reward} s{seed}", n["p_crash"], n["se_crash_20"], n["se_crash_30"], n["se_crash_50"],
                     n["se_speed_20"], n["se_speed_30"], n["se_speed_50"], fl_c, fl_s])
    header = ["run", "pooled crash rate (last 5 pts)", "SE crash n=20", "n=30", "n=50", "SE speed n=20 [m/s]", "n=30", "n=50",
              "point-to-point sd crash (last 20%)", "point-to-point sd speed (last 20%)"]
    return md_table(rows, header, nd=3), recs


def level_table(d: Data, levels: dict[str, list[dict]]) -> tuple[str, dict]:
    """Final / best performance and first step at which every level holds for TARGET_CONSECUTIVE = 1, 2, 3."""
    rows, info = [], {}
    for (algo, reward, seed), run in sorted(d.runs.items()):
        v = run["val"]
        last3_c, last3_s = last3(v, "crash_rate"), last3(v, "mean_speed")
        row = [f"{ALGO_SHORT[algo]} {reward} s{seed}", last3_c, last3_s, v["crash_rate"].min(), v["mean_speed"].max()]
        for name, conds in levels.items():
            for k in (1, 2, 3):
                r = steps_to_threshold(v, conds, consecutive=k)
                row.append(r["env_step"])
                info[((algo, reward, seed), name, k)] = r["env_step"]
        rows.append(row)
    header = ["run", "crash (mean last 3)", "speed (mean last 3)", "best crash", "best speed"]
    for name in levels:
        header += [f"{name} k={k}" for k in (1, 2, 3)]
    return md_table(rows, header, nd=2), info


def hours(total_sim_steps: float, rate: float) -> float:
    return total_sim_steps / rate / 3600



# ---------------------------------------------------------------------------------------------
# additional Q4 evidence: progress at fixed budgets, training-time behaviour, candidate target levels
# ---------------------------------------------------------------------------------------------
def window_mean(v: pd.DataFrame, col: str, centre: float, half: float = 10_000) -> float:
    sub = v[(v["env_step"] >= centre - half) & (v["env_step"] <= centre + half)]
    return float(sub[col].mean()) if len(sub) else math.nan


def progress_table(d: Data) -> str:
    """Own return, crash rate and mean speed averaged over validation points within +-10k steps of 50k / 100k / 150k, and over
    the last 20% of the run; the last column is the share of the improvement (first 3 points -> last 20%) already made at 100k."""
    rows = []
    for (algo, reward, seed), run in sorted(d.runs.items()):
        v, col = run["val"], own_return_col(reward)
        last = float(v["env_step"].iloc[-1])
        tail = v[v["env_step"] >= 0.8 * last]
        first = float(v[col].head(3).mean())
        fin = float(tail[col].mean())
        r100 = window_mean(v, col, 100_000)
        share = (r100 - first) / (fin - first) if abs(fin - first) > 1e-9 else math.nan
        rows.append([f"{ALGO_SHORT[algo]} {reward} s{seed}", window_mean(v, col, 50_000), r100, window_mean(v, col, 150_000), fin,
                     window_mean(v, "crash_rate", 100_000), float(tail["crash_rate"].mean()), window_mean(v, "mean_speed", 100_000),
                     float(tail["mean_speed"].mean()), share])
    return md_table(rows, ["run", "own return @50k", "@100k", "@150k", "last 20%", "crash @100k", "crash last 20%", "speed @100k",
                           "speed last 20%", "share of improvement made at 100k"], nd=2)


def training_behaviour(d: Data) -> str:
    """Behaviour of the stochastic TRAINING policy (train_log.csv) in three windows of the run: crash rate, mean speed, and the share
    of SLOWER / FASTER among the chosen actions; for PPO also the policy entropy from update_log.csv (maximum ln 5 = 1.61)."""
    rows = []
    for (algo, reward, seed) in sorted(d.runs):
        rd = sb3_cal.run_dir(algo, reward, seed, d.root)
        tl = pd.read_csv(rd / "train_log.csv")
        total = float(tl["env_step"].max())
        if algo == "ppo":
            ul = pd.read_csv(rd / "update_log.csv")
        for name, lo, hi in (("first 10%", 0, 0.1), ("40-60%", 0.4, 0.6), ("last 10%", 0.9, 1.01)):
            w = tl[(tl["env_step"] > lo * total) & (tl["env_step"] <= hi * total)]
            if not len(w):
                continue
            n_steps = float(w["length"].sum())
            ent_txt = "-"
            if algo == "ppo":
                uw = ul[(ul["env_step"] > lo * total) & (ul["env_step"] <= hi * total)]
                ent_txt = f"{uw['entropy'].mean():.2f}" if len(uw) else "-"
            rows.append([f"{ALGO_SHORT[algo]} {reward} s{seed}", name, len(w), float(w["crashed"].mean()), float(w["mean_speed"].mean()),
                         pct(w["act_SLOWER"].sum() / n_steps), pct(w["act_FASTER"].sum() / n_steps), ent_txt])
    return md_table(rows, ["run", "window", "episodes", "crash rate", "mean speed", "SLOWER share", "FASTER share", "PPO entropy"])


def candidate_levels(d: Data) -> str:
    """How many of the runs reach a grid of candidate `strong` levels (crash <= c and speed >= v), for k = 1, 2, 3."""
    rows = []
    runs = sorted(d.runs.items())
    for c in (0.2, 0.3, 0.4):
        for v_ in (24.8, 25.5, 26.8):
            conds = [{"metric": "crash_rate", "operator": "<=", "value": c}, {"metric": "mean_speed", "operator": ">=", "value": v_}]
            cells = []
            for k in (1, 2, 3):
                hit = [1 for (a, r, sd), run in runs
                       if not math.isnan(steps_to_threshold(run["val"], conds, consecutive=k)["env_step"])]
                cells.append(f"{len(hit)}/{len(runs)}")
            rows.append([f"crash <= {c:.1f} and speed >= {v_:.1f}", *cells])
    return md_table(rows, ["candidate level", "runs reaching it, k = 1", "k = 2", "k = 3"])


def resume_summary(d: Data) -> str:
    rows = []
    for (algo, reward, seed), run in sorted(d.runs.items()):
        rs = run["meta"]["resumes"]
        if rs:
            rows.append([f"{ALGO_SHORT[algo]} {reward} s{seed}", len(rs), ", ".join(f"{r['env_step']:,}" for r in rs)])
    if not rows:
        return "No run was resumed.\n"
    return (md_table(rows, ["run", "resumes", "resumed at env step"]) + "\nEach resume restarted from the last checkpoint (written every ~10 min of wall "
            "time) with the in-flight episodes restarted under new derived seeds, so these runs are not bit-identical to an uninterrupted run. All resumes were "
            "caused by `PermissionError` on `os.replace` of `progress.json` (OneDrive / the orchestrator reading the file) in processes started before the "
            "retry fix; the PPO runs, started later, never failed. The lost work was at most one checkpoint interval per resume.\n")

def main_grid_plateau_table(d: Data) -> str:
    """The plateau rule of the main grid (``common.aggregate.steps_to_plateau``) applied to the calibration curves, for reporting only:
    plateau env step and whether it lies inside the last 20% of the curve (censored)."""
    rows = []
    for (algo, reward, seed), run in sorted(d.runs.items()):
        r = steps_to_plateau(run["val"], "return_own", reward=reward)
        rows.append([f"{ALGO_SHORT[algo]} {reward} s{seed}", len(run["val"]), r["env_step"], "yes" if r["censored"] else "no",
                     r["final_level"], r["improvement"]])
    return md_table(rows, ["run", "validation points", "plateau env_step", "censored", "final level (own return)", "improvement"], nd=2)


# ---------------------------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------------------------
def make_plots(d: Data, plot_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": C_GRID, "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.titlecolor": C_INK, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.grid": True, "grid.color": C_GRID,
                         "grid.linewidth": 0.6})
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    metrics = [("own return", None, "return under the training reward"), ("crash_rate", "crash_rate", "crash rate"),
               ("mean_speed", "mean_speed", "mean speed [m/s]"), ("completion_rate", "completion_rate", "completion rate")]
    rewards = [r for r in REWARD_ORDER if any(k[1] == r for k in d.runs)]
    for name, col, ylabel in metrics:
        fig, axes = plt.subplots(1, len(rewards), figsize=(3.6 * len(rewards), 3.4), squeeze=False, sharey=(col is not None))
        for ax, reward in zip(axes[0], rewards):
            column = own_return_col(reward) if col is None else col
            for algo, colour in (("dqn", C_DQN), ("ppo", C_PPO)):
                for seed in CALIBRATION_SEEDS:
                    run = d.runs.get((algo, reward, seed))
                    if run is None:
                        continue
                    v = run["val"]
                    ax.plot(v["env_step"] / 1000, v[column], color=colour, lw=1.6 if seed == 99 else 1.2, ls="-" if seed == 99 else "--",
                            label=f"{ALGO_SHORT[algo]} seed {seed}")
            for pol in BASELINES:
                s = d.base_summary(pol, "main")
                if s is None:
                    continue
                y = s[column]
                ax.axhline(y, color=C_BASE, lw=0.9, ls=":")
                ax.annotate(pol, (1.0, y), xycoords=("axes fraction", "data"), xytext=(-3, 2), textcoords="offset points",
                            ha="right", va="bottom", fontsize=7, color="#52514e")
            ax.set_title(f"reward: {reward}", fontsize=9, loc="left")
            ax.set_xlabel("env steps [thousand]")
        axes[0][0].set_ylabel(ylabel)
        handles, labels = axes[0][0].get_legend_handles_labels()
        for ax in axes[0]:
            h2, l2 = ax.get_legend_handles_labels()
            for h, l in zip(h2, l2):
                if l not in labels:
                    handles.append(h), labels.append(l)
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        p = plot_dir / f"curves_{(col or 'return_own')}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    # crash rate by condition, final models of seed 99 and the baselines
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    labels = [(a, r) for a in sb3_cal.ALGOS for r in REWARD_ORDER if (a, r, 99) in d.runs] + [("base", p) for p in BASELINES]
    width = 0.8 / max(len(labels), 1)
    for i, (a, r) in enumerate(labels):
        ys = []
        for cond in CONDITIONS:
            s = d.base_summary(r, cond) if a == "base" else d.summary(a, r, 99, cond)
            ys.append(s["crash_rate"] if s else math.nan)
        colour = C_BASE if a == "base" else (C_DQN if a == "dqn" else C_PPO)
        alpha = 0.35 if a == "base" else {"default": 1.0, "efficiency": 0.8, "balanced": 0.6, "safety": 0.4}[r]
        ax.bar(np.arange(len(CONDITIONS)) + i * width, ys, width * 0.9, color=colour, alpha=alpha,
               label=f"baseline {r}" if a == "base" else f"{ALGO_SHORT[a]} {r}")
    ax.set_xticks(np.arange(len(CONDITIONS)) + 0.4 - width / 2)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel("crash rate (20 VAL_SEEDS)")
    ax.legend(ncol=4, fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    p = plot_dir / "conditions_crash_rate.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    # behaviour of the reward presets: speed vs crash rate
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for (algo, reward, seed), _ in sorted(d.runs.items()):
        s = d.summary(algo, reward, seed)
        if s:
            ax.scatter(s["mean_speed"], s["crash_rate"], color=C_DQN if algo == "dqn" else C_PPO, s=36, zorder=3,
                       marker="o" if seed == 99 else "s", edgecolor="#fcfcfb", linewidth=1.2)
            ax.annotate(f"{reward}" + ("" if seed == 99 else " s98"), (s["mean_speed"], s["crash_rate"]), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="#52514e")
    for pol in BASELINES:
        s = d.base_summary(pol, "main")
        if s:
            ax.scatter(s["mean_speed"], s["crash_rate"], color=C_BASE, s=28, marker="x", zorder=2)
            ax.annotate(pol, (s["mean_speed"], s["crash_rate"]), xytext=(4, -9), textcoords="offset points", fontsize=7, color=C_BASE)
    ax.set_xlabel("mean speed [m/s]")
    ax.set_ylabel("crash rate")
    ax.set_title("final models, main condition (blue DQN, orange PPO; square = seed 98)", fontsize=8, loc="left")
    fig.tight_layout()
    p = plot_dir / "rewards_speed_vs_crash.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------
def build_report(d: Data, plots: list[Path]) -> str:
    L: list[str] = []
    A = L.append
    n_runs = len(d.runs)
    A("# Calibration with Stable-Baselines3\n")
    A("Generated by `python -m scripts.calibration.analyze` from the files in `results/calibration/` (nothing here is typed by hand "
      "except section 6). **Calibration only.** Reference implementations from Stable-Baselines3 2.9.2 were "
      "trained in our environment to check, before anything is frozen and before the main grid, what the scripted-baseline pilot could "
      "not check. They are never part of the DDQN vs PPO comparison and never enter the main tables. **SB3 DQN is vanilla DQN with a "
      "target network (max target), not Double DQN**; SB3 PPO is a reference PPO. All numbers use the 20 `VAL_SEEDS` (never `TEST_SEEDS`); "
      "with 20 episodes one episode moves the crash rate by 5 points. **The interpretation, the recommendations and the answers to Q1-Q4 are in section 6 at the end.**\n")
    A("## 0. What was run\n")
    runs_rows = []
    for (algo, reward, seed), run in sorted(d.runs.items()):
        t = run["meta"]["timing"]
        runs_rows.append([ALGO_SHORT[algo], reward, seed, t["final_env_step"], t["total_wall_s"] / 60, t["train_steps_per_s"],
                          100 * t["val_time_s"] / max(t["total_wall_s"], 1e-9), len(run["val"])])
    A(md_table(runs_rows, ["algorithm", "reward", "seed", "env steps", "wall [min]", "training steps/s (no validation)", "validation share of wall [%]",
                           "val. points"], nd=1))
    if d.plan:
        dropped = d.plan.get("dropped") or []
        A(f"\nPlan limit {d.plan['limit_h']} h; estimated {d.plan['estimated_total_h']:.2f} h; dropped before the start: "
          f"{', '.join(dropped) if dropped else 'none'}.")
    if d.actual:
        A(f" Actual: training {d.actual['training_wall_s'] / 3600:.2f} h, whole plan {d.actual['total_wall_s'] / 3600:.2f} h.")
        bad = {k: v for k, v in d.actual["runs"].items() if str(v.get("status", "")).startswith(("failed", "stopped"))}
        if bad:
            A(" Runs not finished: " + ", ".join(f"{k} ({v['status']})" for k, v in bad.items()) + ".")
    A("\n\n### Resumes\n")
    A(resume_summary(d))
    A("\n### Hyperparameters (no tuning, identical for all rewards)\n")
    for algo in sb3_cal.ALGOS:
        A(f"**{ALGO_LABEL[algo]}**\n")
        hp = sb3_cal.hyperparameters(algo)
        A(md_table([[k, json.dumps(v), s] for k, (v, s) in hp.items()], ["parameter", "value", "source"]))
        A("")
    A(f"PPO collects whole rollouts of 8 x 256 = 2048 steps; 200,000 is not a multiple, so PPO ran {sb3_cal.total_steps('ppo'):,} env steps "
      "(98 rollouts) and DQN exactly 200,000. Weight initialisation is SB3's own (orthogonal for PPO), not the main grid's. "
      "Validation every 5,000 env steps (PPO: at the first rollout boundary after each multiple, after the update), 20 deterministic "
      "episodes, `val_log.csv` in exactly the main-grid schema, plus the per-episode records in `val_episodes.csv`.\n")

    # ---- Q1
    A("## 1. Q1: do the four reward presets lead to different learned behaviour?\n")
    tables, per_reward = q1_tables(d)
    A(tables)
    A("\n**Intended ordering** (efficiency fastest and riskiest, safety slowest and safest, balanced in between, safety with the fewest lane changes):\n")
    for algo, pr in per_reward.items():
        chk = intended_ordering(pr)
        if chk:
            A(f"- {ALGO_SHORT[algo]}: " + "; ".join(f"{k}: **{'yes' if v else 'NO'}**" for k, v in chk.items()))
    cmp_text, indist = q1_spread_comparison(d, per_reward)
    A("\n**Differences between presets vs the seed-to-seed spread.** The spread is |seed 99 - seed 98| of the balanced final model "
      "(two seeds, rough, see Q3); the second column is sampling noise of the 20-episode means (2 sqrt(2) x SE, the 95% band of a difference "
      "of two independent 20-episode means). A difference counts as larger than the spread only if it exceeds both.\n")
    A(cmp_text)
    A("\n**Pairs of presets that are practically indistinguishable** (no difference in crash rate, mean speed or lane changes exceeds "
      "both thresholds): " + (", ".join(indist) if indist else "none") + ".\n")
    A("\n### Does a higher return mean better driving?\n")
    A(q1_return_vs_behaviour(d))

    # ---- Q2
    A("## 2. Q2: are the density levels and test conditions informative for a learned agent?\n")
    A("Final models of seed 99 and the scripted baselines on the five `TEST_CONDITIONS`, 20 `VAL_SEEDS` episodes each "
      "(`long`: 60 s episodes, 30 vehicles; `precise`: `highway-v0`, 15 Hz).\n")
    qtext, flags, qinfo = q2_tables(d)
    A(qtext)
    A("\n**Uninformative conditions** (all learned models within 10 points of each other on crash rate, or all near 0% / all near 100%): "
      + ("; ".join(flags) if flags else "none flagged") + ".\n")
    A("**Gaps relative to `main`, and the `long` / `main` crashes-per-km ratio** (as in `docs/evaluation_protocol.md` section 3: an agent's "
      "`long` result is judged against the baselines' ratio, not against its own `main`):\n")
    A(q2_gaps(d))
    A("")

    # ---- Q3
    A("## 3. Q3: seed-to-seed spread\n")
    q3_text, q3_info = q3_section(d)
    A(q3_text)

    # ---- Q4
    A("## 4. Q4: learning speed, validation noise, achievable performance\n")
    A("Learning curves (validation, `main`, 20 episodes; baselines as dotted horizontal lines; seed 98 dashed):\n")
    for p in plots:
        if p.name.startswith("curves_"):
            A(f"![{p.stem}](calibration_evidence/plots/{p.name})")
    A("\n### Plateau points\n")
    A("Earliest validation point after which the trailing mean over 3 points of the own-reward return stays within 5% of the mean of the last 20% "
      "of the run (`stats.plateau_point`). \"plateaued\" = that point lies before the last 20% of the run; a run whose plateau point falls "
      "inside the last 20% has not shown a plateau within the budget (still improving, or too noisy for a 5% band).\n")
    ptab, pinfo = q4_plateaus(d)
    A(ptab)
    unplat = [f"{ALGO_SHORT[k[0]]} {k[1]} s{k[2]}" for k, v in pinfo.items() if not v["plateaued"]]
    A("\n**Runs that have not plateaued by the end of the budget:** " + (", ".join(unplat) if unplat else "none") + ".\n")
    A("\n### Plateau rule of the main grid applied to these curves (sanity check, reported only)\n")
    A(f"`common.aggregate.steps_to_plateau` (post-hoc, pre-registered for the main grid): the final level is the mean of the raw own-reward return over the last 20% of "
      f"the validation points, the improvement is the final level minus the first value, and the plateau point is the earliest point from which the rolling mean over "
      f"{PLATEAU_WINDOW} points (the window starting at the point) stays within {PLATEAU_TOL:.0%} of the improvement of the final level. \"censored\" = the point lies "
      "inside the last 20% of the curve, or there is no improvement / no stable plateau (n/a). These curves have 40 validation points 5,000 steps apart, so the window "
      "spans 25,000 steps here (the main grid will have 25 points 12,000 steps apart, 60,000 steps per window). Nothing was changed because of this table.\n")
    A(main_grid_plateau_table(d))
    A("\n### Progress at fixed budgets\n")
    A("Validation values averaged over the points within +-10,000 steps of the budget, own-reward return (crash rate and speed at 100k and over the last 20%). "
      "The last column is the share of the improvement from the first three points to the last 20% that was already made at 100k.\n")
    A(progress_table(d))
    A("\n### Behaviour of the training policy (stochastic, `train_log.csv`)\n")
    A(training_behaviour(d))
    A("### Validation noise\n")
    A("From the per-episode validation records at the last 5 validation points: binomial SE of the crash rate (pooled crash probability) and "
      "SE of the mean speed, for 20 episodes and scaled to 30 and 50 by sqrt(20 / n); the last two columns are the point-to-point standard "
      "deviation of the validation value over the last 20% of the run (noise plus any remaining drift).\n")
    ntab, nrecs = q4_noise(d)
    A(ntab)
    A("\n### Achieved performance and targets\n")
    A("`beats_idle` is the decided level (crash rate < 0.6 and speed > 24.8 m/s). `strong_placeholder` is the current placeholder (crash rate <= 0.2, "
      "speed >= 25.5). `strong_baseline_anchor` is the candidate midway between idle (24.76) and faster (28.9): crash rate <= 0.2 and speed >= 26.8 "
      "m/s. The columns `level k=n` are the env step of the first validation point of `n` consecutive points at which the level holds (n/a = never).\n")
    ltab, linfo = level_table(d, PRESENT_TARGETS)
    A(ltab)
    A("\nCandidate `strong` levels, number of the 10 runs that reach them (any reward; the `safety` runs cannot reach a speed bound by design):\n")
    A(candidate_levels(d))
    A("\n### Measured cost\n")
    crow = []
    for algo in sb3_cal.ALGOS:
        rs = [(k, v) for k, v in d.runs.items() if k[0] == algo]
        if not rs:
            continue
        tw = np.mean([v["meta"]["timing"]["total_wall_s"] for _, v in rs])
        sp = np.mean([v["meta"]["timing"]["train_steps_per_s"] for _, v in rs])
        vs = np.mean([v["meta"]["timing"]["val_time_s"] / v["meta"]["timing"]["total_wall_s"] for _, v in rs])
        us = np.mean([v["meta"]["timing"]["update_time_s"] / v["meta"]["timing"]["total_wall_s"] for _, v in rs])
        q = (d.bench or {}).get("dqn_train_steps_per_s" if algo == "dqn" else "ppo_train_steps_per_s")
        crow.append([ALGO_SHORT[algo], len(rs), tw / 60, sp, q, 100 * us, 100 * vs])
    A(md_table(crow, ["algorithm", "runs", "mean wall per run [min]", "training steps/s in the plan (parallel load)",
                      "quiet benchmark steps/s", "update share of wall [%]", "validation share of wall [%]"], nd=1))
    A("\nSpeeds of runs trained in parallel are lower than the quiet benchmarks (shared machine, `docs/performance_analysis.md` C5, C8), so wall "
      "times are not comparable across algorithms; the update and validation shares are.\n")

    # ---- proposals
    A(proposals(d, pinfo, nrecs, linfo, per_reward, q3_info, qinfo, flags, indist))
    notes = ROOT / "docs" / "calibration_notes.md"
    if notes.exists():
        A("\n## 6. Interpretation and recommendations (written by hand after reading the results)\n")
        A(notes.read_text(encoding="utf-8").strip() + "\n")
    return "\n".join(L)


def proposals(d, pinfo, nrecs, linfo, per_reward, q3_info, qinfo, flags, indist) -> str:
    L: list[str] = []
    A = L.append
    A("\n## 5. Proposals (for the team; nothing was written into `common/config.py` or `common/train_config.py`)\n")
    A("**Caveats first.** One seed per reward (two for balanced); SB3 DQN is not Double DQN; our implementations and tuned hyperparameters will "
      "differ (different initialisation, update rules, tuned learning rates); validation uses 20 episodes. All numbers are rough.\n")

    # --- total_env_steps
    plats = {k: v["plateau_step"] for k, v in pinfo.items() if not math.isnan(v["plateau_step"])}
    late = max(plats.values()) if plats else math.nan
    A("### total_env_steps\n")
    if plats:
        worst = max(plats, key=plats.get)
        lo, hi = 1.3 * late, 1.5 * late
        prop = st.round_up(1.4 * late, 25_000)
        A(f"Latest plateau point over all runs: {late:,.0f} env steps ({ALGO_SHORT[worst[0]]} {worst[1]} seed {worst[2]}); x 1.3-1.5 = "
          f"{lo:,.0f}-{hi:,.0f}; rounded up to a multiple of 25,000 at 1.4x: **{prop:,.0f}** (the formula result; see section 6 for the recommendation). ")
        nonplat = [k for k, v in pinfo.items() if not v["plateaued"]]
        if nonplat:
            A(f"{len(nonplat)} of {len(pinfo)} runs did not plateau before the last 20% of the 200k budget (listed in section 4); for those the "
              "plateau point is a lower bound and the proposal is capped by the calibration budget, so it is not evidence that more training is useless.")
        A("The PPO rollout size (`n_envs x n_steps`) of the final configuration must divide the budget.\n")
    else:
        prop = 200_000
        A("No plateau point could be computed.\n")
    # costs
    bench = d.bench or {}
    steps_total = sum(v["meta"]["timing"]["final_env_step"] + len(v["val"]) * 20 * 30 for v in d.runs.values())
    wall = d.actual["training_wall_s"] if d.actual else None
    a_meas = steps_total / wall if wall else math.nan
    a_bench = bench.get("ppo_env_phase_steps_per_s", math.nan)
    A("\n**Cost of the full plan** (40 main-grid models + 24 sensitivity runs = 64 runs, each `T` training steps + validations of 20 episodes x 30 "
      "steps with about 25 validation points; tuning adds up to 8 runs at the default budget). Rates: "
      f"`A_plan` = {a_meas:.0f} simulated steps/s aggregate, measured in this calibration (all simulated steps incl. validation / wall time of the "
      f"whole training phase, so it includes SB3's update cost and the idle tail), and `A_peak` = {a_bench:.0f} steps/s (8-process simulator throughput "
      "of the quiet PPO benchmark, simulation only):\n")
    rows = []
    for T in sorted({100_000, 150_000, 200_000, int(prop)}):
        per_model = T + 25 * 20 * 30
        rows.append([f"{T:,}" + (" (proposed)" if T == int(prop) else ""), per_model, 64 * per_model / 1e6,
                     hours(64 * per_model, a_meas), hours(64 * per_model, a_bench)])
    A(md_table(rows, ["T", "simulated steps per model", "64 runs [M steps]", "hours at A_plan", "hours at A_peak"], nd=1))
    A("\nOur own DDQN and PPO add their own update cost (SB3's is included in `A_plan`); `A_peak` is the optimistic bound.\n")

    # --- validation
    A("### val_every_env_steps and val_n_episodes\n")
    ve = st.round_up(prop / 25, 1000) if prop else 5000
    A(f"20-30 points per curve: with T = {prop:,.0f}, `val_every_env_steps` = **{ve:,.0f}** gives {prop / ve:.0f} points; for the other budgets: "
      + ", ".join(f"T = {T:,} -> {st.round_up(T / 25, 1000):,.0f} ({T / st.round_up(T / 25, 1000):.0f} points)" for T in (100_000, 150_000, 200_000))
      + " (PPO validates at the first rollout boundary after each multiple, so keep the interval above the rollout size).\n")
    se20 = [r["se_crash_20"] for r in nrecs]
    sp20 = [r["se_speed_20"] for r in nrecs]
    if se20:
        A(f"Validation noise with 20 episodes: SE of the crash rate {np.mean(se20):.3f} on average (max {np.max(se20):.3f}) = about "
          f"{100 * np.mean(se20):.0f} points, SE of the mean speed {np.mean(sp20):.2f} m/s. With 30 episodes the crash-rate SE would be "
          f"{np.mean(se20) * math.sqrt(20 / 30):.3f}, with 50 episodes {np.mean(se20) * math.sqrt(20 / 50):.3f}; the noise at the worst case (p = 0.5) is "
          f"{math.sqrt(0.25 / 20):.3f} / {math.sqrt(0.25 / 30):.3f} / {math.sqrt(0.25 / 50):.3f}. ")
    A("Recommendation: **keep 20 episodes for the 64 runs only if the target rule below (consecutive points) is accepted as the noise control**; "
      "enlarging `VAL_SEEDS` to 50 halves the variance of every validation point (SE x 0.63) but "
      f"costs 2.5x the validation simulation: {64 * 25 * 50 * 30 / 1e6:.2f} M instead of {64 * 25 * 20 * 30 / 1e6:.2f} M simulated steps for the 64 runs "
      f"(+{hours(64 * 25 * 30 * 30, a_meas):.1f} h at `A_plan`). It would need 30 new episode seeds that are disjoint from every other set "
      "(for instance 20020..20049, which collide with none of the existing sets nor with the derived vector-env seeds), and **`beats_idle` must be "
      "re-derived**: its constants come from the idle baseline on the current 20 `VAL_SEEDS`. This is a proposal only; no seed set was changed.\n")

    # --- targets
    A("### The `strong` target and TARGET_CONSECUTIVE\n")
    finals = {}
    for algo in sb3_cal.ALGOS:
        for reward in REWARD_ORDER:
            vs = [last3(d.runs[(algo, reward, s)]["val"], "crash_rate") for s in d.seeds(algo, reward)]
            ss = [last3(d.runs[(algo, reward, s)]["val"], "mean_speed") for s in d.seeds(algo, reward)]
            if vs:
                finals[(algo, reward)] = (float(np.mean(vs)), float(np.mean(ss)))
    if finals:
        bal = {a: finals[(a, "balanced")] for a in sb3_cal.ALGOS if (a, "balanced") in finals}
        if bal:
            worse_c = max(v[0] for v in bal.values())
            worse_s = min(v[1] for v in bal.values())
            cc, ss_ = st.round_up(worse_c, 0.05), st.round_down(worse_s, 0.1)
            A(f"- Baseline anchor (midway between idle and faster): crash rate <= 0.2 and mean speed >= 26.8 m/s.\n"
              f"- \"Worse of the SB3 finals, rounded slightly down\" (balanced reward, mean of the last 3 validation points, worse of DQN and PPO): crash "
              f"rate {worse_c:.2f} -> <= {cc:.2f}; speed {worse_s:.2f} -> >= {ss_:.1f} m/s.\n")
        A("Final values per model (mean of the last 3 validation points):\n")
        A(md_table([[f"{ALGO_SHORT[a]} {r}", v[0], v[1]] for (a, r), v in finals.items()], ["model", "crash rate", "mean speed [m/s]"]))
        A("\nTrade-off: a target no model reaches gives NaN for everything (no comparison at all); one that every model reaches at the first "
          "validation points does not separate the algorithms. The table of section 4 shows, per run and level, whether and when it was reached "
          "(`n/a` = never). Because the levels are applied to every run regardless of the reward, a level that only the speed-oriented rewards can reach "
          "leaves the `safety` models without a value; check the rows of the `safety` and `balanced` runs in that table before choosing.\n")
    # consecutive
    A("**TARGET_CONSECUTIVE.** A single validation point is a noisy 20-episode estimate. Probability that a policy whose TRUE crash rate is `p` "
      "passes a crash bound of 0.2 (<= 4 crashes of 20) at `k` independent consecutive points (crash rate only; speed adds a little more noise):\n")
    rows = []
    for p in (0.10, 0.20, 0.30, 0.40):
        rows.append([p, *[st.pass_probability(p, 0.2, 20, k) for k in (1, 2, 3)]])
    A(md_table(rows, ["true crash rate", "k = 1", "k = 2", "k = 3"], nd=3))
    A("\nA policy that is clearly better than the bound (true 0.10) passes one point with probability "
      f"{st.pass_probability(0.10, 0.2):.2f}, two in a row {st.pass_probability(0.10, 0.2, 20, 2):.2f}, three in a row {st.pass_probability(0.10, 0.2, 20, 3):.2f}; a policy "
      f"clearly worse (true 0.40) passes by luck {st.pass_probability(0.40, 0.2):.3f} / {st.pass_probability(0.40, 0.2, 20, 2):.4f} / {st.pass_probability(0.40, 0.2, 20, 3):.5f}. "
      "A policy exactly at the bound (0.20) passes a single point in 63% of the cases, so the reached step of a borderline run is itself noisy. ")
    A("The k-sensitivity of the measured runs is in the `k=1/2/3` columns of the target table above. Recommendation: see the notes section.\n")

    # --- rewards
    A("### Rewards\n")
    A("Evidence is in section 1 (ordering checks, differences vs the seed spread, pairs that are practically indistinguishable"
      + (": " + ", ".join(indist) if indist else "") + ") and in the return-vs-behaviour table. Any weight change is a proposal for the team.\n")
    # --- conditions
    A("### Densities and conditions\n")
    A("Evidence is in section 2. Flagged as uninformative: " + ("; ".join(flags) if flags else "none") + ".\n")
    return "\n".join(L)


def copy_evidence(d: Data, plots: list[Path], dest: Path = EVIDENCE_DIR) -> int:
    """Copy the plots and, for every run, ``val_log.csv`` and ``run_meta.json`` to ``dest/<algo>/<reward>/seed<s>/``
    (nothing else: no checkpoints, buffers, models, train/update logs, per-episode records or evaluation CSVs). Returns the bytes copied."""
    import shutil

    total = 0
    (dest / "plots").mkdir(parents=True, exist_ok=True)
    for p in plots:
        shutil.copyfile(p, dest / "plots" / p.name)
        total += p.stat().st_size
    for (algo, reward, seed) in sorted(d.runs):
        src = sb3_cal.run_dir(algo, reward, seed, d.root)
        out = dest / algo / reward / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        for name in EVIDENCE_RUN_FILES:
            shutil.copyfile(src / name, out / name)
            total += (src / name).stat().st_size
    return total


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=str(ROOT / "docs" / "calibration.md"))
    ap.add_argument("--no-evidence", action="store_true", help="do not refresh docs/calibration_evidence/")
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else sb3_cal.CAL_ROOT
    d = Data(root)
    if not d.runs:
        raise SystemExit(f"no finished runs under {root}")
    plots = make_plots(d, root / "plots")
    Path(args.out).write_text(build_report(d, plots), encoding="utf-8")
    print(f"written {args.out} and {len(plots)} plots in {root / 'plots'}")
    if not args.no_evidence:
        n_bytes = copy_evidence(d, plots)
        print(f"evidence copy in {EVIDENCE_DIR}: {n_bytes / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
