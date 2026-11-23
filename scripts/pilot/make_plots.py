"""Plots for the environment pilot (reads the CSVs written by exp1).

Usage (from the repo root):
    .venv/Scripts/python.exe scripts/pilot/make_plots.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pilot_common import OUT_DIR, POLICIES

PLOT_DIR = OUT_DIR / "plots"

# Fixed-order categorical palette (light surface); one hue per density level.
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]  # blue, orange, aqua, magenta
CRASH_COLOR = "#eb6834"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 10.5,
    }
)


def grid_plot(df: pd.DataFrame, col: str, ylabel: str, title: str, fname: str, ylim=None, pct=False) -> None:
    """One panel per (lanes_count row, policy column); one line per vehicles_density."""
    lanes = sorted(df["lanes_count"].unique())
    densities = sorted(df["vehicles_density"].unique())
    fig, axes = plt.subplots(len(lanes), len(POLICIES), figsize=(12, 3.6 * len(lanes)),
                             sharex=True, sharey=True, squeeze=False)
    for i, l in enumerate(lanes):
        for j, p in enumerate(POLICIES):
            ax = axes[i][j]
            for k, d in enumerate(densities):
                s = df[(df.lanes_count == l) & (df.policy == p) & (df.vehicles_density == d)].sort_values("vehicles_count")
                y = s[col] * (100 if pct else 1)
                ax.plot(s["vehicles_count"], y, color=SERIES[k], lw=2, marker="o", ms=6,
                        mec=SURFACE, mew=1.5, label=f"density {d}")
            ax.set_title(f"{p} policy, lanes_count={l}", loc="left")
            ax.set_xticks(sorted(df["vehicles_count"].unique()))
            if ylim:
                ax.set_ylim(*ylim)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(lanes) - 1:
                ax.set_xlabel("vehicles_count")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(densities), frameon=False, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PLOT_DIR / fname, dpi=140)
    plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT_DIR / "density_sweep.csv")
    grid_plot(df, "crash_rate", "crash rate (%)", "Crash rate vs vehicles_count",
              "crash_rate_vs_vehicles.png", ylim=(-3, 103), pct=True)
    grid_plot(df, "length_mean", "mean episode length (steps)", "Mean episode length vs vehicles_count",
              "episode_length_vs_vehicles.png")
    grid_plot(df, "ahead_gt4_share_mean", "steps with >4 vehicles ahead within 100 m (%)",
              "Share of steps where the 4 observed slots could not cover all vehicles ahead (<=100 m)",
              "nearby_gt4_ahead_share.png", ylim=(-3, 103), pct=True)
    grid_plot(df, "near_gt4_share_mean", "steps with >4 vehicles within 100 m, any side (%)",
              "Share of steps with >4 vehicles within 100 m (ahead and behind)",
              "nearby_gt4_all_share.png", ylim=(-3, 103), pct=True)
    print("plots written to", PLOT_DIR)


if __name__ == "__main__":
    main()
