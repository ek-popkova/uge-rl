"""Reward pilot v2, steps 2-3: returns under each reward, Knox et al. (2023)-style checks, plot.

Usage (from the repo root, after reward_v2_collect.py):
    .venv/Scripts/python.exe scripts/pilot/reward_v2_analyze.py

Reads results/pilot/reward_v2/{steps,episodes}.csv, writes returns.csv, returns_summary.csv,
knox_checks.csv and plots/knox_references.png.

Reference outcomes per (config, reward):
  G_crash  mean return of crashed episodes, all policies pooled
  G_slow   mean return of non-crashed 'slower' episodes
  G_succ   mean return of non-crashed 'faster' episodes (>= 5 needed, else non-crashed 'idle')
Ordering test: G_crash < G_slow < G_succ. Indifference point p solves
G_slow = p*G_succ + (1-p)*G_crash, i.e. p/(1-p) successful episodes are worth one crash.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pilot_common import make_env, OUT_DIR

V2_DIR = OUT_DIR / "reward_v2"
CONFIGS = ["low", "medium", "high"]
DENSITY = {"low": 0.4, "medium": 0.7, "high": 1.0}
POLICIES = ["random", "idle", "slower", "faster"]
VARIANTS = ["default", "A", "B", "C"]
LABELS = {"default": "default (normalized)", "A": "A efficiency", "B": "B safety", "C": "C balanced"}
SHORT = {"default": "default", "A": "A efficiency", "B": "B safety", "C": "C balanced"}
WEIGHTS = {
    "A": dict(w_speed=1.0, w_right=0.0, w_collision=1.0, w_lane=0.0),
    "B": dict(w_speed=0.4, w_right=0.1, w_collision=5.0, w_lane=0.2),
    "C": dict(w_speed=0.7, w_right=0.1, w_collision=2.0, w_lane=0.05),
}
MIN_SUCC = 5  # minimum number of non-crashed 'faster' episodes for G_succ


def custom_step_reward(df: pd.DataFrame, w: dict) -> pd.Series:
    """r = w_speed*high_speed + w_right*right_lane - w_collision*crashed - w_lane*lane_changed."""
    return (w["w_speed"] * df["high_speed_reward"] + w["w_right"] * df["right_lane_reward"]
            - w["w_collision"] * df["crashed"] - w["w_lane"] * df["lane_changed"])


def build_returns(steps: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    key = ["config", "policy", "seed"]
    tmp = steps[key + ["reward"]].copy()
    for v, w in WEIGHTS.items():
        tmp[f"ret_{v}"] = custom_step_reward(steps, w)
    g = tmp.groupby(key).sum().rename(columns={"reward": "ret_default"}).reset_index()
    out = eps[key + ["length", "crashed", "truncated", "mean_speed", "distance", "return_default"]].merge(g, on=key)
    # Sanity: the sum of logged step rewards must equal the env's episode return.
    assert np.allclose(out["ret_default"], out["return_default"]), "step rewards do not sum to episode return"
    return out.drop(columns="return_default")


def analytic_references(cfg: dict, T: int) -> dict[str, dict]:
    """Idealized trajectories over T steps, right lane, no lane changes.

    crash: T/2 steps at 25 m/s, the last of them is the crash step;
    slow: T steps at 20 m/s; succ: T steps at 30 m/s.
    """
    lo, hi = cfg["reward_speed_range"]
    hs = lambda v: float(np.clip((v - lo) / (hi - lo), 0, 1))
    n_c = T // 2

    def default_step(c: float, speed: float) -> float:
        raw = cfg["collision_reward"] * c + cfg["high_speed_reward"] * hs(speed) + cfg["right_lane_reward"] * 1.0
        return (raw - cfg["collision_reward"]) / (cfg["high_speed_reward"] + cfg["right_lane_reward"] - cfg["collision_reward"])

    out = {}
    out["default"] = dict(
        G_crash=(n_c - 1) * default_step(0, 25) + default_step(1, 25),
        G_slow=T * default_step(0, 20), G_succ=T * default_step(0, 30),
        cost=-cfg["collision_reward"] / (cfg["high_speed_reward"] + cfg["right_lane_reward"] - cfg["collision_reward"]),
    )
    for v, w in WEIGHTS.items():
        step = lambda c, speed: w["w_speed"] * hs(speed) + w["w_right"] * 1.0 - w["w_collision"] * c
        out[v] = dict(G_crash=(n_c - 1) * step(0, 25) + step(1, 25), G_slow=T * step(0, 20),
                      G_succ=T * step(0, 30), cost=w["w_collision"])
    return out


def knox_row(g_crash: float, g_slow: float, g_succ: float) -> tuple[bool, float, float]:
    ok = bool(np.all(np.isfinite([g_crash, g_slow, g_succ])) and g_crash < g_slow < g_succ)
    if not ok:
        return False, np.nan, np.nan
    p = (g_slow - g_crash) / (g_succ - g_crash)
    return True, p, p / (1 - p)


def bootstrap_order(crash: np.ndarray, slow: np.ndarray, succ: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Resample episodes within each group: share of resamples with G_crash < G_slow < G_succ and a
    90% interval of p among the resamples where the ordering holds."""
    if min(len(crash), len(slow), len(succ)) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1) for x in (crash, slow, succ)]
    ok = (m[0] < m[1]) & (m[1] < m[2])
    if not ok.any():
        return 0.0, np.nan, np.nan
    pb = (m[1][ok] - m[0][ok]) / (m[2][ok] - m[0][ok])
    return float(ok.mean()), float(np.quantile(pb, 0.05)), float(np.quantile(pb, 0.95))


def required_penalty(cfg: dict, T: int) -> pd.DataFrame:
    """Analytic collision weight needed for a target indifference point p (idealized trajectories).

    G_crash = base - w_c, so p = (G_slow - base + w_c) / (G_succ - base + w_c)
    => w_c = (p*(G_succ - base) - (G_slow - base)) / (1 - p).
    For 'default' the weight is on the env's own scale (reward divided by the normalization range).
    """
    lo, hi = cfg["reward_speed_range"]
    hs = lambda v: float(np.clip((v - lo) / (hi - lo), 0, 1))
    n_c = T // 2
    rows = []
    for v in VARIANTS:
        if v == "default":
            rng_ = cfg["high_speed_reward"] + cfg["right_lane_reward"] - cfg["collision_reward"]
            step = lambda speed: (cfg["high_speed_reward"] * hs(speed) + cfg["right_lane_reward"] - cfg["collision_reward"]) / rng_
            scale = rng_  # env collision weight = w_eff * range
            g_slow, g_succ, base = T * step(20), T * step(30), n_c * (cfg["high_speed_reward"] * hs(25) + cfg["right_lane_reward"] - cfg["collision_reward"]) / rng_
            current = -cfg["collision_reward"]
        else:
            w = WEIGHTS[v]
            step = lambda speed: w["w_speed"] * hs(speed) + w["w_right"]
            scale, g_slow, g_succ, base, current = 1.0, T * step(20), T * step(30), n_c * step(25), w["w_collision"]
        p_zero = (g_slow - base) / (g_succ - base)  # indifference point with NO collision term at all
        for p in (0.5, 0.9, 0.99):
            wc = max(0.0, (p * (g_succ - base) - (g_slow - base)) / (1 - p))  # 0 = already met
            rows.append(dict(reward=v, target_p=p, required_w_collision=wc * scale, current_w_collision=current,
                             factor_vs_current=wc * scale / current, p_if_collision_weight_zero=p_zero))
    return pd.DataFrame(rows)


def main() -> None:
    steps = pd.read_csv(V2_DIR / "steps.csv")
    eps = pd.read_csv(V2_DIR / "episodes.csv")
    ret = build_returns(steps, eps)
    ret.to_csv(V2_DIR / "returns.csv", index=False)

    # ---- mean returns by config x policy x outcome (reference table for the report)
    rows = []
    for (c, p), g in ret.groupby(["config", "policy"]):
        for outcome, gg in (("all", g), ("crashed", g[g.crashed == 1]), ("not_crashed", g[g.crashed == 0])):
            row = dict(config=c, policy=p, outcome=outcome, n=len(gg), mean_length=gg["length"].mean())
            for v in VARIANTS:
                row[f"ret_{v}"] = gg[f"ret_{v}"].mean() if len(gg) else np.nan
            rows.append(row)
    pd.DataFrame(rows).to_csv(V2_DIR / "returns_summary.csv", index=False)

    # ---- Knox checks
    env = make_env({})
    cfg = env.unwrapped.config
    T = int(cfg["duration"])
    env.close()
    ideal = analytic_references(cfg, T)

    srows = steps.merge(ret[["config", "policy", "seed", "crashed"]].rename(columns={"crashed": "ep_crashed"}),
                        on=["config", "policy", "seed"])
    out = []
    for c in CONFIGS:
        rc = ret[ret.config == c]
        crashed = rc[rc.crashed == 1]
        slow = rc[(rc.policy == "slower") & (rc.crashed == 0)]
        faster_ok = rc[(rc.policy == "faster") & (rc.crashed == 0)]
        sc = srows[srows.config == c]
        # "empirical": spec rule (faster, fallback idle). "empirical_idle_succ": idle at every density,
        # so that G_succ means the same thing at all densities (sensitivity check).
        for kind, force_idle in (("empirical", False), ("empirical_idle_succ", True)):
          if not force_idle and len(faster_ok) >= MIN_SUCC:
            succ, src = faster_ok, "faster"
          else:
            succ, src = rc[(rc.policy == "idle") & (rc.crashed == 0)], ("idle" if force_idle else "idle (fallback)")
          grp = {
            "crash": sc[sc.ep_crashed == 1],
            "slow": sc[(sc.policy == "slower") & (sc.ep_crashed == 0)],
            "succ": sc[(sc.policy == ("faster" if src == "faster" else "idle")) & (sc.ep_crashed == 0)],
          }
          for v in VARIANTS:
            col = f"ret_{v}"
            gc = crashed[col].mean() if len(crashed) else np.nan
            gs = slow[col].mean() if len(slow) else np.nan
            gu = succ[col].mean() if len(succ) else np.nan
            ok, p, per = knox_row(gc, gs, gu)
            ok_frac, p_lo, p_hi = bootstrap_order(crashed[col].values, slow[col].values, succ[col].values)
            cost = ideal[v]["cost"]
            row = dict(
                kind=kind, config=c, density=DENSITY[c], reward=v,
                G_crash=gc, G_slow=gs, G_succ=gu, order_ok=ok, p=p, successes_per_crash=per,
                order_boot_frac=ok_frac, p_boot_lo=p_lo, p_boot_hi=p_hi,
                collision_cost=cost, cost_over_G_succ=cost / gu if np.isfinite(gu) and gu > 0 else np.nan,
                n_crash=len(crashed), mean_crash_step=crashed["length"].mean(),
                n_slow=len(slow), n_succ=len(succ), succ_source=src, low_n=bool(len(succ) < MIN_SUCC or len(slow) < MIN_SUCC),
                crash_hs=grp["crash"]["high_speed_reward"].mean(), crash_right=grp["crash"]["right_lane_reward"].mean(),
                slow_hs=grp["slow"]["high_speed_reward"].mean(), slow_right=grp["slow"]["right_lane_reward"].mean(),
                succ_hs=grp["succ"]["high_speed_reward"].mean(), succ_right=grp["succ"]["right_lane_reward"].mean(),
                slow_lane_changes=grp["slow"]["lane_changed"].sum() / max(len(slow), 1),
                succ_lane_changes=grp["succ"]["lane_changed"].sum() / max(len(succ), 1),
                crash_lane_changes=grp["crash"]["lane_changed"].sum() / max(len(crashed), 1),
            )
            for p_ in POLICIES:
                x = crashed[crashed.policy == p_][col]
                row[f"G_crash_{p_}"] = x.mean() if len(x) else np.nan
                row[f"n_crash_{p_}"] = len(x)
            out.append(row)
    for v in VARIANTS:
        a = ideal[v]
        ok, p, per = knox_row(a["G_crash"], a["G_slow"], a["G_succ"])
        out.append(dict(kind="analytic", config="ideal", density=np.nan, reward=v, G_crash=a["G_crash"],
                        G_slow=a["G_slow"], G_succ=a["G_succ"], order_ok=ok, p=p, successes_per_crash=per,
                        collision_cost=a["cost"], cost_over_G_succ=a["cost"] / a["G_succ"]))
    knox = pd.DataFrame(out)
    knox.to_csv(V2_DIR / "knox_checks.csv", index=False)
    required_penalty(cfg, T).to_csv(V2_DIR / "required_penalty.csv", index=False)
    plot(knox)
    pd.set_option("display.width", 250)
    print(knox[["kind", "config", "reward", "G_crash", "G_slow", "G_succ", "order_ok", "p", "successes_per_crash",
                "order_boot_frac", "p_boot_lo", "p_boot_hi", "cost_over_G_succ", "n_crash", "mean_crash_step", "n_slow", "n_succ", "succ_source", "low_n"]].round(2).to_string(index=False))


def plot(knox: pd.DataFrame) -> None:
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    colors = {"G_crash": "#eb6834", "G_slow": "#2a78d6", "G_succ": "#1baf7a"}  # fixed categorical order
    names = {"G_crash": "G_crash (crashed episodes)", "G_slow": "G_slow (non-crashed, slower)",
             "G_succ": "G_succ (non-crashed faster; idle if too few)"}
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "axes.edgecolor": grid,
                         "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted, "text.color": ink,
                         "axes.grid": True, "grid.color": grid, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5})
    emp = knox[knox.kind == "empirical"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    w = 0.26
    for ax, c in zip(axes, CONFIGS):
        sub = emp[emp.config == c].set_index("reward").loc[VARIANTS]
        x = np.arange(len(VARIANTS))
        for k, key in enumerate(("G_crash", "G_slow", "G_succ")):
            vals = sub[key].values
            for i, v in enumerate(vals):
                pos = x[i] + (k - 1) * (w + 0.02)
                if not np.isfinite(v):
                    ax.text(pos, 0.4, "n/a", ha="center", fontsize=8, color=muted)
                    continue
                fallback = key == "G_succ" and sub["succ_source"].iloc[i] != "faster"
                ax.bar(pos, v, width=w, color=colors[key], edgecolor=surface, linewidth=1.5,
                       hatch="///" if fallback else None, label=names[key] if (i == 0 and c == CONFIGS[0]) else None)  # legend from the first panel
                ax.text(pos, v + (0.5 if v >= 0 else -0.5), f"{v:.1f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7.5, color=muted)
        ax.axhline(0, color=muted, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{SHORT[v]}\norder {'OK' if sub.loc[v, 'order_ok'] else 'FAILS'}\n(holds in {sub.loc[v, 'order_boot_frac']:.0%}\nof resamples)"
             for v in VARIANTS], fontsize=8)
        n_succ = int(sub["n_succ"].iloc[0])
        ax.set_title(f"{c}, density {DENSITY[c]}" + (" (= env default)" if c == "high" else "")
                     + f"   [G_succ: n={n_succ}]", loc="left", fontsize=10.5)
    axes[0].set_ylabel("mean episode return")
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle("Reference outcomes per reward variant", x=0.01, y=0.985, ha="left", fontsize=12, fontweight="bold")
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.945), ncol=3, frameon=False, labelcolor=ink)
    fig.text(0.01, 0.01, "Hatched G_succ = non-crashed idle episodes (fewer than 5 non-crashed faster episodes at that density). "
             "Return = undiscounted episode return; 'default' = the env's normalized reward.", fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    (V2_DIR / "plots").mkdir(exist_ok=True)
    fig.savefig(V2_DIR / "plots" / "knox_references.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
