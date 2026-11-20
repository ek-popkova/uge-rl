"""Assemble results/pilot/PILOT_REPORT.md: prose + tables generated from the pilot CSVs.

Usage (from the repo root):
    .venv/Scripts/python.exe scripts/pilot/build_report.py
"""

from __future__ import annotations

from pilot_common import OUT_DIR
import report_tables as T

TEMPLATE = r"""# Environment pilot report

Pilot for the Double DQN vs PPO comparison on HighwayEnv (`highway-env==1.12.1`,
`highway-fast-v0`, default Kinematics observation, `DiscreteMetaAction`). **No RL agent was
trained and `common/config.py` was not changed.** Everything below was measured with three
scripted baseline policies: `random` (uniform over the 5 actions), `idle` (always action 1)
and `slower` (always action 4). Action indices were verified against
`env.unwrapped.action_type.actions_indexes`. All seeds are 1000..1049 (50 episodes per
cell, identical for every config), so no cell was reduced or skipped.

## 0. Proposal in one table

| | low | medium | high |
|---|---:|---:|---:|
| `vehicles_density` | **0.4** | **0.7** | **1.0** |
| `vehicles_count` | 20 | 20 | 20 |
| idle crash rate (3 lanes, 30 s) | 8% | 54% | 92% |
| random crash rate | 44% | 70% | 100% |
| `slower` crash rate | 0% | 0% | 2% |
| vehicles within 100 m (idle) | 2.0 | 3.9 | 5.5 |
| steps with >4 vehicles ahead within 100 m | 0% | 0% | 34% |

(Numbers are for 20 vehicles; tables 2.1/2.2 use 30 vehicles and differ by at most 2 points.)

Recommended base config: `lanes_count=3`, `duration=30`, `simulation_frequency=5` (the
`highway-fast-v0` default), observation `vehicles_count=5` (unchanged). Reasons in sections 3-4.
These are proposals only: `common/config.py` still holds the old placeholders (4 lanes,
duration 40, `simulation_frequency=15`, densities 0.8/1.0/1.4 with 20/50/80 vehicles), and
`simulation_frequency=15` alone makes the simulator about 2.5x slower
(see `docs/performance_analysis.md`, C2).

Main surprises:

1. **The requested grid has no "low" level.** Its sparsest cell (density 0.7) already makes
   `idle` crash in 54% of episodes (3 lanes). I added a clearly marked supplementary sweep
   below the grid (densities 0.3-0.6, rows marked `*`) to find a low level.
2. **`vehicles_count` above ~20 changes nothing.** All other vehicles spawn *ahead* of the
   ego (none behind) and `vehicles_count` only lengthens that platoon; the ego never
   reaches the far end in 30 s. `vehicles_density` is the real difficulty knob.

## 1. Actual default config of `highway-fast-v0` (`env.unwrapped.config`)

```
@@CONFIG@@
```

Notes: defaults differ from `common/config.py` (lanes 3 vs 4, duration 30 vs 40,
`simulation_frequency` 5 vs 15, 20 vehicles, density 1.0). The ego can only track target
speeds 20/25/30 m/s (`MDPVehicle.DEFAULT_TARGET_SPEEDS`), so `slower` bottoms out at 20 m/s.
With `reward_speed_range=[20, 30]` that is `high_speed_reward = 0`.

## 2. Key numbers

Definitions. *Crash rate* = share of episodes with `info["crashed"]` at the end. *Reaches limit*
= share of episodes that were `truncated` (note: an episode that crashes on its very last
step is both crashed and truncated, so the two shares can add up to more than 100%).
*Vehicles within 100 m* = mean number of other vehicles with |dx| <= 100 m, any lane, ahead or
behind, averaged over steps. *Ahead* counts only dx > 0. The default observation has
`see_behind=False` and `vehicles_count=5` = ego + the **4** nearest vehicles ahead, so
"steps >4 ahead" is the share of steps with more vehicles ahead than observation slots, and
*same-lane leader hidden* is the share of steps where the nearest vehicle ahead in the ego lane
(within 100 m) is not among those 4 slots (replicates `KinematicObservation.observe`).
*Min gap* = bumper-to-bumper distance to the leader in the ego lane (center distance - 5 m),
minimum over the episode, so it is about 0 or slightly negative after a crash.

### 2.1 Traffic density, 3 lanes, `vehicles_count=30` (idle statistics; `*` = supplementary rows)

@@DENSITY3@@

### 2.2 Traffic density, 4 lanes, `vehicles_count=30`

@@DENSITY4@@

"same-lane leader hidden" was only computed for densities 0.3, 0.5, 0.7, 1.0, 1.5, 2.0
(`obs_coverage.py`); it is 0% at both 0.3 and 0.5, so 0.4 is 0% as well.

### 2.3 Effect of `vehicles_count` (3 lanes, idle): crash rate / mean vehicles within 100 m

@@VC@@

From 20 vehicles upward the rows are the same within the statistical noise of 50 episodes
(the standard error of a crash rate near 50% is about 7 points). Only 10-15 vehicles differ
slightly (e.g. 64% vs 54% at density 0.7, 86% vs 94% at 1.0), mostly within that noise: a
shorter platoon, so fewer vehicles to meet.

### 2.4 Environment speed (steps/s, idle policy, resets included, clean re-benchmark)

@@SPEED@@

Cost grows roughly linearly with `vehicles_count` and is almost independent of density.
(The first benchmark, taken while other processes were disturbing the machine, was
discarded; these numbers are from a clean re-run of the full grid, 10 episodes per cell.
Per-config values for all 3 policies: `results/pilot/density_sweep.csv`, columns
`steps_per_s`, `step_only_per_s`; reset is about 6 ms.)

### 2.5 Episode duration (idle policy; crash rate / reaches time limit)

@@DURATION@@

`default` is the unmodified env (20 vehicles, density 1.0, 3 lanes); the other rows are the
proposed levels at 30 vehicles. The 60 s run reproduces the shorter durations exactly
(same seeds, same trajectories up to truncation), and `vehicles_count=50` gave identical
results to 30 at every duration. The `steps/s` column is noisy (see the performance
analysis) and not comparable across rows.

### 2.6 Does 5 observed vehicles suffice? (idle, 30 vehicles)

The two columns "steps >4 ahead" and "same-lane leader hidden" in tables 2.1/2.2 answer this.
Short version: at density <= 0.7 there are practically never more than 4 vehicles ahead
within 100 m (0% of steps with 3 lanes, 4% with 4 lanes at 0.7) and the ego-lane leader is
hidden in at most 1% of steps; at density 1.0 the leader is hidden in 2% (3 lanes)
or 9% (4 lanes) of steps; from density 1.5 it is hidden in 15-32% of steps.

## 3. Proposed low / medium / high setting

All three use `lanes_count=3`, `duration=30`, `vehicles_count=20` (identical results to
30-40 at 30 s, and about 1.8x faster, section 2.4).

| level | density | why |
|---|---:|---|
| low | 0.4 | `idle` survives 92% of episodes and the nearest vehicle ahead stays 59 m away on average, so the task is mostly "go fast". Random still crashes in 44%, so it is not trivial. Lower densities (0.3: 12% idle crash) are not easier in a useful way, and 0.5 (26%) is too close to medium. About 2.0 vehicles within 100 m. |
| medium | 0.7 | Hard for random (70-72% crash), not impossible for idle (46% reach the time limit), while `slower` never crashes (0%), so a competent policy has real room: speed vs safety. About 3.9 vehicles within 100 m, none of the observation slots are ever overfull. It is the sparsest cell of the requested grid. |
| high | 1.0 | Idle crashes in 92% (94% with 30 vehicles) and random in 100%, but `slower` still survives 98%, so it is hard yet feasible: any policy that brakes in time survives. About 5.5 vehicles within 100 m, more than 4 vehicles ahead in 34% of steps. This is also the default density of `highway-fast-v0`. |

The three levels differ clearly: idle crash 8% -> 54% -> 92% (gaps of 46 and 38 points, much
larger than the +/-7 point standard error with n=50), vehicles within 100 m 2.0 -> 3.9 -> 5.5,
vehicles ahead 1.6 -> 2.9 -> 4.3, idle minimum gap 59 m -> 18 m -> 0.7 m.

Why not higher densities: at 1.5 even `slower` crashes in 64% (3 lanes; 30% with 4 lanes)
and at 2.0 in 90%, while idle and random crash in 100% within about 4-8 steps on average. The ego cannot go
below 20 m/s, so these may be close to infeasible; the baselines cannot prove either way. I
would only use 1.5 as an optional stress test of a trained agent, not for training.

Caveat: the baselines only bound the difficulty from above. No baseline here drives well, so
"feasible" at high density is argued from `slower` (a bad but safe policy) only.

## 4. Recommendations

**`lanes_count`: 3 is enough.** With 3 lanes the idle crash rate separates the levels more
sharply (about 8/54/92% vs 14/35/85% with 4 lanes, 20-30 vehicles) and the ego can still overtake on both sides
from the middle lane. The cost per step is the same within noise (section 2.4). 4 lanes only
makes everything easier (idle crash at density 0.7: 34% vs 54%) and does not change what the
observation or the actions look like. Keep 4 lanes as an optional generalization test.

**Observation `vehicles_count`: keep 5.** At low and medium density (3 lanes) there are never more
than 4 vehicles ahead within 100 m (0% of steps) and the ego-lane leader is hidden in 0% of steps.
At high density (1.0, 3 lanes) it is hidden in 2% of steps (9% with 4 lanes), although more than
4 vehicles are ahead in 34% (the extra ones are in other lanes or farther away). If a density
of 1.5 or more is ever used, raise it to about 8-10 (leader hidden in 15-32% of steps).
(Caveat: the share is weighted by steps, and crash episodes are short, so it is computed on
the early part of episodes more often than on late ones.)

**Episode duration: 30 s (30 steps at `policy_frequency=1`).** For `idle`: crash/limit
rates are 8%/94% (low), 54%/46% (medium), 94%/6% (high), measured at 30 vehicles. At 20 s the levels are easier
(medium 24% crash, 84% reach the limit) and an agent gets fewer decisions; at 40 s medium
is already 76% and low 24%; at 60 s idle always crashes at medium and high, and low 52%.
30 s keeps all three levels informative, matches the env default and costs the least per
episode among the informative options. If duration is raised above 30, use
`vehicles_count=30`, since 20 was only verified at 30 s (30 and 50 were verified up to 60 s).

**`vehicles_count` (environment): 20.** Results for 20, 30 and 40 are the same within noise,
and 20 is 1.8x faster than 30 and 2.4x faster than 40 (3 lanes, section 2.4).

## 5. Deviations, caveats, what was added

- **Nothing skipped.** The full 40-config grid with 50 episodes per cell for all three
  policies (6,000 episodes) was run. The estimated wall time (9.5 min with 12 workers) was far
  below the 45 min limit.
- **Added beyond the request** (clearly separate): supplementary low-density sweep (densities
  0.3-0.6, `vehicles_count` 20/30, `density_extra_*.csv`), observation-coverage check
  (`obs_coverage.csv`), and extra rows in Experiment 2 (proposed levels and a
  `vehicles_count=50` check).
- The speed benchmark was run twice; the first run (inside the grid sweep) was disturbed by
  something unidentified and was replaced by a clean re-run. Speed values are from the
  re-run (resets included, idle policy, 10 episodes per cell) and vary run to run by up to 2-5x
  on this laptop; use them only as relative guidance.
- Lane changes are counted as a change of `lane_index[2]` between steps, which can lag the
  decision by a step (the vehicle must cross the lane boundary). The random policy
  is seeded per episode (`np.random.default_rng(seed)`), so results are reproducible.
- Videos (`videos/<config>.mp4` and `.gif`, idle policy, seed 1000, 3 lanes) all end in a
  crash after 16, 14, 3 and 3 steps for (10, 0.7), (20, 1.0), (30, 1.5) and (40, 2.0):
  that is what the idle policy does at these densities, and it is a quick visual check of
  the spawn layout (all traffic ahead of the ego).
- Only one ego spawn lane/speed distribution was seen (the env default, speed 25 m/s).
  The conclusions apply to baseline policies; learned agents may behave differently.

## 6. Total runtime of the pilot

@@RUNTIME@@

Total compute **@@TOTAL@@ min** (wall clock, sequential sum). Plotting and report
assembly are not included (a few minutes). Most of the time went to Experiment 2 (single
process, 4x redundant) and to the speed benchmarks; a leaner design would take about 22 min.
Details, measurements and the list of avoidable costs: `docs/performance_analysis.md`.

## Files

`results/pilot/`: `density_sweep.csv` (one row per config x policy), `density_episodes.csv`
(one row per episode), `density_extra_*.csv`, `speed.csv`, `duration.csv`,
`obs_coverage.csv`, `timing.json`, `plots/` (crash rate, episode length and ">4 nearby vehicles" per policy x
lanes), `videos/`. Code: `scripts/pilot/`.
"""


def main() -> None:
    runtime_md, total = T.runtime_table()
    text = (
        TEMPLATE.replace("@@CONFIG@@", T.default_config())
        .replace("@@DENSITY3@@", T.density_table(3))
        .replace("@@DENSITY4@@", T.density_table(4))
        .replace("@@VC@@", T.vc_table(3))
        .replace("@@SPEED@@", T.speed_table())
        .replace("@@DURATION@@", T.duration_table())
        .replace("@@RUNTIME@@", runtime_md)
        .replace("@@TOTAL@@", f"{total / 60:.1f}")
    )
    (OUT_DIR / "PILOT_REPORT.md").write_text(text, encoding="utf-8")
    print("wrote", OUT_DIR / "PILOT_REPORT.md", f"({len(text)} chars)")


if __name__ == "__main__":
    main()
