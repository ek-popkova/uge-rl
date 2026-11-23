"""Markdown tables for PILOT_REPORT.md, generated from the pilot CSVs.

Usage (from the repo root):
    .venv/Scripts/python.exe scripts/pilot/report_tables.py   # prints all tables
Imported by build_report.py.
"""

from __future__ import annotations

import json
import pprint

import numpy as np
import pandas as pd

from pilot_common import OUT_DIR, make_env

NA = "n/a"


def md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if i else ":---" for i in range(len(cols))) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def pct(x: float) -> str:
    return NA if pd.isna(x) else f"{100 * x:.0f}%"


def num(x: float, d: int = 1) -> str:
    return NA if pd.isna(x) else f"{x:.{d}f}"


def default_config() -> str:
    env = make_env()
    cfg = pprint.pformat(env.unwrapped.config, sort_dicts=True, width=100)
    idx = env.unwrapped.action_type.actions_indexes
    env.close()
    return f"{cfg}\n\nactions_indexes = {idx}"


def density_table(lanes: int, vc: int = 30) -> str:
    main = pd.read_csv(OUT_DIR / "density_sweep.csv")
    extra = pd.read_csv(OUT_DIR / "density_extra_sweep.csv")
    main["src"], extra["src"] = "", "*"
    d = pd.concat([extra, main])
    d = d[(d.lanes_count == lanes) & (d.vehicles_count == vc)]
    cov = pd.read_csv(OUT_DIR / "obs_coverage.csv")
    cov = cov[(cov.lanes_count == lanes) & (cov.policy == "idle")].set_index("vehicles_density")
    rows = []
    for dens in sorted(d.vehicles_density.unique()):
        g = d[d.vehicles_density == dens].set_index("policy")
        i, r, s = g.loc["idle"], g.loc["random"], g.loc["slower"]
        hidden = cov.loc[dens, "share_steps_same_lane_leader_hidden"] if dens in cov.index else np.nan
        rows.append(
            {
                "density": f"{dens}{i['src']}",
                "idle crash": pct(i.crash_rate), "random crash": pct(r.crash_rate), "slower crash": pct(s.crash_rate),
                "idle reaches limit": pct(i.timelimit_rate),
                "idle mean len": num(i.length_mean),
                "veh within 100 m (idle)": num(i.near_mean_mean),
                "veh ahead 100 m": num(i.ahead_mean_mean),
                "steps >4 ahead": pct(i.ahead_gt4_share_mean),
                "steps >4 any side": pct(i.near_gt4_share_mean),
                "same-lane leader hidden": pct(hidden),
                "idle min gap (m)": num(i.min_gap_mean),
            }
        )
    return md(pd.DataFrame(rows))


def vc_table(lanes: int = 3) -> str:
    d = pd.read_csv(OUT_DIR / "density_sweep.csv")
    d = d[(d.lanes_count == lanes) & (d.policy == "idle")]
    rows = []
    for dens in sorted(d.vehicles_density.unique()):
        row = {"density": dens}
        for vc in sorted(d.vehicles_count.unique()):
            g = d[(d.vehicles_density == dens) & (d.vehicles_count == vc)].iloc[0]
            row[f"{vc} veh"] = f"{100 * g.crash_rate:.0f}% / {g.near_mean_mean:.1f}"
        rows.append(row)
    return md(pd.DataFrame(rows))


def speed_table() -> str:
    s = pd.read_csv(OUT_DIR / "speed.csv")
    s = s[s.policy == "idle"]
    rows = []
    for lanes in (3, 4):
        row = {"lanes": lanes}
        for vc in sorted(s.vehicles_count.unique()):
            row[f"{vc} veh"] = num(s[(s.lanes_count == lanes) & (s.vehicles_count == vc)].steps_per_s.mean(), 0)
        rows.append(row)
    return md(pd.DataFrame(rows))


def duration_table() -> str:
    d = pd.read_csv(OUT_DIR / "duration.csv")
    d = d[~d.config.str.contains("vc50")]
    rows = []
    for cfg in d.config.unique():
        g = d[d.config == cfg].set_index("duration")
        r0 = g.iloc[0]
        row = {"config": f"{cfg} ({int(r0.vehicles_count)} veh, dens {r0.vehicles_density})"}
        for t in (20, 30, 40, 60):
            row[f"{t} s"] = f"{100 * g.loc[t, 'crash_rate']:.0f}% / {100 * g.loc[t, 'timelimit_rate']:.0f}%"
        row["steps/s"] = num(g.steps_per_s.mean(), 0)
        rows.append(row)
    return md(pd.DataFrame(rows))


def runtime_table() -> tuple[str, float]:
    t = json.loads((OUT_DIR / "timing.json").read_text())
    names = {
        "exp1_density": "Exp. 1 main grid (benchmark + sweep)",
        "exp1_density_extra": "Supplementary low-density sweep",
        "obs_coverage": "Observation-coverage check",
        "exp2_duration": "Exp. 2 durations",
        "record_videos": "Videos",
        "exp1_speed_rebenchmark": "Clean speed re-benchmark",
    }
    rows, total = [], 0.0
    for k, label in names.items():
        sec = t[k]["seconds"]
        total += sec
        rows.append({"step": label, "wall time": f"{sec / 60:.1f} min"})
    rows.append({"step": "**total compute**", "wall time": f"**{total / 60:.1f} min**"})
    return md(pd.DataFrame(rows)), total


if __name__ == "__main__":
    for name in ("density_table", "vc_table", "speed_table", "duration_table"):
        print(f"\n### {name}\n")
        print(globals()[name]() if name != "density_table" else density_table(3) + "\n\n" + density_table(4))
    print(runtime_table()[0])
