"""Re-check the final reward weights (common/config.py) on the regenerated pilot logs (results/pilot/reward_v2).

Usage (from the repo root):
    .venv/Scripts/python.exe scripts/pilot/reward_final_analyze.py

Reads results/pilot/reward_v2/{steps,episodes}.csv. For every density (low/medium/high), reward
(default, efficiency, balanced, safety) and discount factor gamma it computes the Knox et al.
(2023) reference outcomes exactly as in reward_v2_analyze.py:
  G_crash  mean discounted return of crashed episodes, all four policies pooled
  G_slow   mean discounted return of non-crashed 'slower' episodes
  G_succ   mean discounted return of non-crashed 'faster' episodes (>= 5 needed, else non-crashed 'idle')
the ordering test G_crash < G_slow < G_succ (bootstrap, 2000 resamples) and the empirical
indifference point p, next to the analytic p (common.config.indifference_point; column target_p). Discounted return from the start of the episode:
G = sum_t gamma^(t-1) * r_t.

For the "penalty felt near the decision" check, each episode's last min(3, length) steps (for a
crashed episode: the crash step and the two steps before it) are discounted from the first step of that
window: W = sum_k gamma^k * r_(L-2+k). Crashed windows are compared with the same last-3-step window of
non-crashed episodes (the G_slow group and the G_succ group).
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pilot_common import OUT_DIR, ROOT

sys.path.insert(0, str(ROOT))
from common.config import REWARDS, indifference_point  # noqa: E402  (final weights)
from reward_v2_analyze import MIN_SUCC, bootstrap_order, knox_row  # noqa: E402  (same definitions as reward_v2)

V2 = OUT_DIR / "reward_v2"
OUT = OUT_DIR / "reward_final_v2"
CONFIGS = ["low", "medium", "high"]
DENSITY = {"low": 0.4, "medium": 0.7, "high": 1.0}
VARIANTS = ["default", "efficiency", "balanced", "safety"]
GAMMAS = [1.0, 0.99, 0.9, 0.8]
T = 30  # episode length, == BASE_CONFIG["duration"]
WINDOW = 3


def step_rewards(steps: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-step reward of every variant (default = the logged env reward)."""
    out = {"default": steps["reward"]}
    for name in VARIANTS[1:]:
        w = REWARDS[name]
        out[name] = (w["w_speed"] * steps["high_speed_reward"] + w["w_right"] * steps["right_lane_reward"]
                     - w["w_collision"] * steps["crashed"] - w["w_lane"] * steps["lane_changed"])
    return out


def analytic_p(name: str, gamma: float) -> float:
    """Indifference point of the idealized trajectories (crash at T/2, 25 m/s; slow 20; success 30)."""
    disc = lambda n0, n1: sum(gamma ** (t - 1) for t in range(n0, n1 + 1))  # sum of gamma^(t-1), t=n0..n1
    if name == "default":
        # env reward: (-1*c + 0.4*hs + 0.1*right + 1) / 1.5
        r = lambda c, hs: (-1.0 * c + 0.4 * hs + 0.1 + 1.0) / 1.5
        g_crash = r(0, 0.5) * disc(1, T // 2 - 1) + r(1, 0.5) * gamma ** (T // 2 - 1)
        g_slow, g_succ = r(0, 0.0) * disc(1, T), r(0, 1.0) * disc(1, T)
    else:
        w = REWARDS[name]
        r25, r20, r30 = 0.5 * w["w_speed"] + w["w_right"], w["w_right"], w["w_speed"] + w["w_right"]
        g_crash = r25 * disc(1, T // 2) - w["w_collision"] * gamma ** (T // 2 - 1)
        g_slow, g_succ = r20 * disc(1, T), r30 * disc(1, T)
    ok, p, _ = knox_row(g_crash, g_slow, g_succ)
    return p if ok else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    steps = pd.read_csv(V2 / "steps.csv")
    eps = pd.read_csv(V2 / "episodes.csv")
    key = ["config", "policy", "seed"]
    rewards = step_rewards(steps)

    rows = []
    for gamma in GAMMAS:
        disc = gamma ** (steps["step"].values - 1)
        # per-episode length-aware window start: last min(3, L) steps
        length = steps.groupby(key)["step"].transform("max")
        in_win = steps["step"] > length - WINDOW
        win_disc = gamma ** (steps["step"].values - (length - np.minimum(WINDOW, length) + 1)).values
        for variant in VARIANTS:
            r = rewards[variant]
            tmp = steps[key].copy()
            tmp["g"], tmp["w"] = (r * disc).values, (r * win_disc * in_win).values
            per_ep = tmp.groupby(key).sum().reset_index().merge(eps[key + ["length", "crashed"]], on=key)
            target = np.nan  # analytic p of the idealized episodes (gamma = 1), None if the ordering fails
            if variant != "default":
                w = REWARDS[variant]
                target = indifference_point(w["w_speed"], w["w_right"], w["w_collision"], T=T) or np.nan
            for c in CONFIGS:
                rc = per_ep[per_ep.config == c]
                crashed = rc[rc.crashed == 1]
                slow = rc[(rc.policy == "slower") & (rc.crashed == 0)]
                faster_ok = rc[(rc.policy == "faster") & (rc.crashed == 0)]
                if len(faster_ok) >= MIN_SUCC:
                    succ, src = faster_ok, "faster"
                else:
                    succ, src = rc[(rc.policy == "idle") & (rc.crashed == 0)], "idle (fallback)"
                gc, gs, gu = (x["g"].mean() if len(x) else np.nan for x in (crashed, slow, succ))
                ok, p, per = knox_row(gc, gs, gu)
                frac, lo, hi = bootstrap_order(crashed["g"].values, slow["g"].values, succ["g"].values)
                rows.append(dict(
                    gamma=gamma, config=c, density=DENSITY[c], reward=variant, target_p=target,
                    G_crash=gc, G_slow=gs, G_succ=gu, succ_source=src,
                    n_crash=len(crashed), n_slow=len(slow), n_succ=len(succ),
                    low_n=bool(len(succ) < MIN_SUCC or len(slow) < MIN_SUCC),
                    order_ok=ok, order_boot_frac=frac, p=p, p_boot_lo=lo, p_boot_hi=hi,
                    successes_per_crash=per,
                    target_successes_per_crash=target / (1 - target) if np.isfinite(target) else np.nan,
                    p_minus_target=p - target if np.isfinite(target) and np.isfinite(p) else np.nan,
                    analytic_p_at_gamma=analytic_p(variant, gamma),
                    mean_crash_step=crashed["length"].mean(),
                    win_crash=crashed["w"].mean(), win_slow=slow["w"].mean(), win_succ=succ["w"].mean(),
                    win_gap_succ_minus_crash=succ["w"].mean() - crashed["w"].mean(),
                    collision_term_in_window=(REWARDS[variant]["w_collision"] * gamma ** (WINDOW - 1)
                                              if variant != "default" else 1.0 / 1.5 * gamma ** (WINDOW - 1)),
                ))
    knox = pd.DataFrame(rows)
    knox.to_csv(OUT / "knox_final.csv", index=False)
    plot(knox)
    pd.set_option("display.width", 250)
    show = knox[["gamma", "config", "reward", "target_p", "G_crash", "G_slow", "G_succ", "order_ok", "order_boot_frac",
                 "p", "p_boot_lo", "p_boot_hi", "analytic_p_at_gamma", "win_crash", "win_succ"]]
    print(show.round(2).to_string(index=False))


def plot(knox: pd.DataFrame) -> None:
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    colors = {"G_crash": "#eb6834", "G_slow": "#2a78d6", "G_succ": "#1baf7a"}
    names = {"G_crash": "G_crash (crashed episodes)", "G_slow": "G_slow (non-crashed, slower)",
             "G_succ": "G_succ (non-crashed faster; idle if too few)"}
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "axes.edgecolor": grid,
                         "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted, "text.color": ink,
                         "axes.grid": True, "grid.color": grid, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    (OUT / "plots").mkdir(exist_ok=True)
    for c in CONFIGS:
        fig, axes = plt.subplots(1, len(GAMMAS), figsize=(17, 5.0), sharey=False)
        for ax, gamma in zip(axes, GAMMAS):
            sub = knox[(knox.config == c) & (knox.gamma == gamma)].set_index("reward").loc[VARIANTS]
            x, w = np.arange(len(VARIANTS)), 0.26
            for k, key in enumerate(("G_crash", "G_slow", "G_succ")):
                for i, v in enumerate(sub[key].values):
                    if not np.isfinite(v):
                        continue
                    pos = x[i] + (k - 1) * (w + 0.02)
                    fallback = key == "G_succ" and sub["succ_source"].iloc[i] != "faster"
                    ax.bar(pos, v, width=w, color=colors[key], edgecolor=surface, linewidth=1.2,
                           hatch="///" if fallback else None, label=names[key] if (i == 0 and gamma == GAMMAS[0]) else None)
            ax.axhline(0, color=muted, lw=0.8)
            ax.set_xticks(x)
            labels = []
            for v in VARIANTS:
                s = sub.loc[v]
                head = "OK" if s.order_ok else "FAILS"
                pe = "n/a" if pd.isna(s.p) else f"{s.p:.2f}"
                tgt = "" if pd.isna(s.target_p) else f"/{s.target_p:.2f}"
                labels.append(f"{v}\norder {head}\np={pe}{tgt}")
            ax.set_xticklabels(labels, fontsize=7.5)
            ax.set_title(f"gamma = {gamma:g}", loc="left", fontsize=10)
        axes[0].set_ylabel("mean discounted episode return")
        h, l = axes[0].get_legend_handles_labels()
        fig.suptitle(f"Final reward weights, {c} density ({DENSITY[c]}): reference outcomes", x=0.01, y=0.985,
                     ha="left", fontsize=12, fontweight="bold")
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.94), ncol=3, frameon=False, labelcolor=ink)
        fig.text(0.01, 0.01, "Labels: p = empirical indifference point / target p. Hatched G_succ = non-crashed idle "
                 "episodes (fewer than 5 non-crashed faster episodes). Return discounted from the episode start.",
                 fontsize=8, color=muted)
        fig.tight_layout(rect=(0, 0.04, 1, 0.9))
        fig.savefig(OUT / "plots" / f"knox_final_{c}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
