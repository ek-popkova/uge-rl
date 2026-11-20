# Performance analysis of the environment pilot (`scripts/pilot/`)

Analysis date: 2026-11-20. Read-only analysis: nothing in `scripts/pilot/` or
`common/` was changed, and the pilot was not interrupted. The pilot's compute had
finished (all experiments done) before the last two diagnostic tests below.

Issue IDs (`C1`, `C2`, ...) are referenced from `CLAUDE.md` / `AGENTS.md`.

## 1. Summary

The pilot took **99 min** of compute. Its speed is limited by two things that no
amount of code tidying removes, plus avoidable waste on top:

| # | Cause | Kind | Impact |
|---|-------|------|--------|
| C1 | highway-env step cost: pure-Python, ~50% of time in `neighbour_vehicles` / `local_coordinates` | intrinsic | 13-55 steps/s per process (default config 45-55; 40 vehicles 20) |
| C5 | Parallel scaling ceiling on this laptop CPU: ~3.5-5x one process, plateau at ~8 workers, *worse* at 12-16 | hardware | aggregate 140-180 steps/s at best |
| C3 | Experiment 2 ran 24 runs sequentially, 4x redundant (shorter durations are prefixes of the 60 s run) | avoidable | 48 min = 48% of all pilot time |
| C4 | Speed benchmark ran sequentially over the whole 120-cell grid, twice | avoidable | 17 min |
| C6 | Extra scripts re-simulate episodes the sweep already ran (obs_coverage, exp3) | avoidable | 39k steps (~14% of all steps) |
| C5b | Coarse task granularity in the extra sweep / obs_coverage (16 and 12 tasks on 12 workers) | avoidable | extra sweep at 92 steps/s vs 140 achievable |
| C2 | `common/config.py` placeholders cost 2.4-5 steps/s (latent, not used by the pilot) | latent | 5-20x slower than the pilot env |
| C7 | No incremental saving / resume / ETA | risk | a crash at minute 47 of Experiment 2 would lose everything |
| C8 | Unexplained 2-5x run-to-run speed variance on this machine | unknown | invalidated the first speed benchmark |

Expected speed-up if fixed and parallelized: **99 min -> about 22 min (about 4.5x)**;
about 44 min (2.3x) with parallelization alone (details in section 5). The ceiling is
set by C1 and C5, not by the code.

## 2. Setup and how things were measured

- Machine: Intel Core Ultra 9 285H (hybrid laptop CPU, 16 logical cores, no
  hyper-threading: 6 P + 8 E + 2 low-power E), `os.cpu_count() = 16`, Windows 11,
  Python 3.13.3, gymnasium 1.4.0, highway-env 1.12.1.
- Item 7 benchmark: throw-away script in the scratchpad (not in the
  repo) that imports the pilot's own `pilot_common.make_env`, one process, `idle`
  policy, 200 steps per measurement, `reset(seed=...)` when an episode ends, 3
  repetitions, step time and reset time timed separately. cProfile on the heaviest
  config. The scaling test uses spawned processes with a start barrier.
- Caveat: the 200-step probes ran while one pilot process (Experiment 2) was running
  on another core, so they are somewhat pessimistic; the quiet-machine numbers
  (sections 3.1 and 3.5) come from after the pilot had finished.

## 3. Confirmed issues

### C1. The simulator is the bottleneck (intrinsic)

Evidence (`pilot_common.make_env`, exact pilot env creation, 200 steps, 3 repetitions):

| Config | step-only steps/s | ms/step | reset (ms) |
|--------|------------------:|--------:|-----------:|
| default (3 lanes, 20 veh, density 1.0) | 31, 32, 45 (quiet machine: 42-55) | 22-32 | 7-10 |
| heaviest in grid, 3 lanes (40 veh, density 2.0) | 24, 25, 27 (single run alone: 13) | 37-42 | 6-7 |
| heaviest in grid, 4 lanes | 13, 14, 19 | 53-75 | 8-13 |

The clean re-benchmark of the full grid (`results/pilot/speed.csv`, 500 s, nothing else
running) gives idle-policy steps/s by `vehicles_count`: 10 -> 90, 15 -> 69, 20 -> 51,
30 -> 30, 40 -> 21, i.e. cost grows roughly linearly with the number of vehicles.

cProfile of the heaviest config (200 steps, 20.1 s with profiler): `Lane.local_coordinates`
is called 2.49 M times (6.0 s own time, 30%); `Road.neighbour_vehicles` 51 k times
(10.5 s cumulative, 52%); `IDMVehicle.act` 15.1 s cumulative. All of it is pure
Python inside `highway_env/road/` and `highway_env/vehicle/behavior.py`: each of the
5 simulation frames per step makes every vehicle scan all other vehicles.

Impact: this is the floor. Reset is small: median 5.6 ms, 1.4% of benchmark wall time.
Fix: cannot be removed; budget for it (fewer vehicles, default `simulation_frequency=5`,
parallel runs). See C2 and C5.

Related finding that makes extra vehicles pure waste: all other vehicles spawn *ahead*
of the ego (0 behind; checked for 10/20/40 vehicles), and
`vehicles_count` only lengthens the platoon (farthest vehicle at density 1.0: 236 m for
10 vehicles, 472 m for 20, 946 m for 40). With 30-step episodes, results for 20/30/40
vehicles are practically identical (idle crash rate, density 0.7, 3 lanes: 54%/54%/54%),
and `vehicles_count=50` gave bit-identical crash rates to 30 for all durations up to
60 s (Experiment 2). Paying 2x per step for them buys nothing.

### C2. `common/config.py` placeholders would be very slow (latent, not used by the pilot)

`common/config.py:24` `simulation_frequency: 15`, `:30-31` presets `medium` = 50
vehicles, `high` = 80 vehicles (plus 4 lanes, duration 40). Measured through
`common.envs.make_env`, idle policy, 60 steps (short because it is so slow):

| Preset | vehicles / density | sim freq | steps/s | ms/step |
|--------|--------------------|---------:|--------:|--------:|
| low | 20 / 0.8 | 15 | 19.0 | 52.7 |
| medium | 50 / 1.0 | 15 | 5.0 | 198.7 |
| high | 80 / 1.4 | 15 | 2.4 | 423.7 |

versus 45-55 steps/s for the pilot's default config (20 vehicles, sim freq 5). A PPO run
of 1 M steps on the `medium` preset would need about 56 h in one process, `high` about
116 h. Fix: do not train with these values; after the team agrees on them, re-measure.
Evidence from the pilot says 20-30 vehicles and `simulation_frequency=5` (the
`highway-fast-v0` default) are enough. Values were left unchanged on purpose.

### C3. Experiment 2 is sequential and 4x redundant (48% of pilot time)

`scripts/pilot/exp2_duration.py:38-44`: nested loop over 6 configs x 4 durations, each
creating an env and running 50 episodes with `run_lean`, single process; results
written once at the end (`:63`). Measured: 24 runs, 26,658 steps, 2,874 s = 9.3 steps/s,
i.e. 48% of the 5,966 s total pilot compute.

Redundancy: the duration only changes when an episode is *truncated*; with the same
seeds the trajectories are identical until then. Verified from `results/pilot/duration.csv`:
the cumulative crash rate by step 20/30/40 computed from the 60 s run equals the
separately simulated 20/30/40 s runs exactly, for all four configs
(e.g. medium: 0.24/0.54/0.76 in both). `medium_vc50` and `high_vc50` equal `medium` and
`high` exactly. So only the 60 s run of each config was needed: 5,713 steps instead of
26,658 (4.7x fewer), and the runs are independent so they parallelize.
Impact: about 45 min saved. Fix: simulate the longest duration once and derive the
others; run configs in a pool; write per-task results as they finish.

### C4. Speed benchmark is sequential over the whole grid and ran twice

`scripts/pilot/exp1_density.py:62` `benchmark(configs)` runs 120 cells x 10 episodes
sequentially (19,964 steps), called at `:121` (re-benchmark) and `:133`. It took 514 s
the first time and 500 s the second time (the first was invalid, see C8). Step cost is
~linear in `vehicles_count` and barely depends on density, so about 6-8 cells
(vehicles in {10, 20, 40} x lanes {3, 4}, a few policies) and a fitted cost model would
have given the same information in about 1-2 min. Impact: about 15 min. Fix: benchmark a
representative subset, once, on a quiet machine.

### C5. Parallel scaling is capped at ~3.5-5x; 12 workers is past the knee

Scaling test (spawned processes, start barrier, `idle`, default config, 250 steps each):

| workers | aggregate steps/s | speed-up | efficiency |
|--------:|------------------:|---------:|-----------:|
| 1 | 51.8 | 1.0x | 100% |
| 4 | 141.9 | 2.7x | 68% |
| 8 | 181.8 | 3.5x | 44% |
| 12 | 164.2 | 3.2x | 26% |
| 16 | 147.7 | 2.9x | 18% |

Heaviest config (40 veh, density 2.0, 3 lanes): 13.0 / 44.4 / 65.0 / 71.2 / 66.4 steps/s
for 1/4/8/12/16 workers (5.5x at best). The same test with `OMP_NUM_THREADS=1` is within
noise (1/4/8/12/16 workers: 36 / 116 / 179 / 131 / 135 steps/s). The pilot's own sweep
agrees: 99,947 steps in 712 s with 12 workers = 140 steps/s, i.e. 11.7 steps/s per
worker versus about 39 for one bare process. The default `--workers` of the pilot,
`min(12, cpu_count - 2)` (`exp1_density.py:109`, `obs_coverage.py:85`), was never tuned.
Throughput peaks around 8 workers; adding more only adds contention (likely the shared
power/thermal budget and the slower E cores; the cause was not isolated).

Impact: the 12-worker main sweep ran at the machine's plateau, so it is not the issue.
Rule of thumb for this machine: expect at most 3.5-5x; use about 8 workers, and measure.
Fix: make `--workers` explicit, default to a measured value, never `cpu_count`.

### C5b. Coarse tasks leave workers idle

- Extra sweep (`exp1_density.py:154`, `--extra-grid`): 16 tasks (config x 3 policies x
  50 episodes) on 12 workers = two waves, the second with 4 busy workers. 66,024 steps in
  715 s = 92 steps/s versus the ~140 achieved by the main sweep (ideal about 470 s).
- `scripts/pilot/obs_coverage.py:84-86`: `pool.map` over 12 tasks (2 lane settings x 6
  densities) on 12 workers; dense configs take much longer than sparse ones, so the wall
  time is the slowest task. 36,148 steps in 454 s = 80 steps/s.
- `scripts/pilot/exp3_reward_scale.py:47-49`: 150 episodes, single process, 2,828 steps in
  179 s = 15.8 steps/s.

Impact: about 4 + 3.5 + 2 min. Fix: tasks per (config, policy) or per chunk of seeds,
heaviest first, `imap_unordered`; chunk results independent of worker count.

### C6. Experiments re-simulate episodes the sweep already ran

`obs_coverage.py` (36,148 steps) and `exp3_reward_scale.py` (2,828 steps) simulate
episodes with the same seeds and policies as the main sweep, only to log extra per-step
quantities (hidden-leader flags, reward components). Adding those fields to
`run_episode` (`pilot_common.py:76`) would have cost nothing extra. Instrumentation cost
was measured at x1.00-1.06 over the bare loop, so logging more per step is cheap.
Impact: about 39k steps = 14% of all steps (about 5 min). Fix: one pass, many metrics.

### C7. No incremental saving, resume or ETA

All experiments write their CSV once at the end (`exp2_duration.py:63`,
`exp1_density.py:170-174`, `exp3_reward_scale.py:97-99`); a crash or a kill at minute 47
of Experiment 2 would have lost everything. Progress output is a counter without ETA
(`exp1_density.py:162`). No `--resume`. Impact: risk, not speed. Fix: write one file per
finished task, skip existing ones on restart, print ETA.

### C8. Unexplained 2-5x run-to-run speed variance (not resolved)

- The first benchmark (514 s) has erratic 3-lane cells (e.g. `vehicles_count=20`: 19.5
  steps/s at density 0.7 but 64.8 at density 1.5; 3 lanes slower than 4 lanes at the
  same vehicle count) while the 4-lane half is smooth. The clean re-run is smooth
  (see C1), so the first run was disturbed by something not identified.
- Experiment 2 lean-loop speeds for the same env swing, e.g. `default` 14.7-17.0
  steps/s while the same config measured 42-55 steps/s on a quiet machine; `medium`
  rose from 8.1 to 23.9 steps/s across its four durations.
- Ruled out by a single-process test (150 steps, default config, 2 repetitions):
  P-core vs E-core vs low-power-core affinity (steps/s, two repetitions: P 51.7/47.5,
  E 48.4/41.9, LP-E 46.9/44.3 vs unpinned 55.2/46.3; all within noise), `HIGH` priority, and explicit opt-out of Windows power throttling
  (the opt-out call itself returned failure, so that variant is inconclusive).
- Not ruled out: thermal / power limits of the laptop after sustained 12-worker load,
  OS background activity (OneDrive sync of the project folder, antivirus).
Impact: timings from this machine are only reliable on a quiet system. Fix: benchmark
immediately before a long run, record steps/s in the run metadata, never compare
timings taken at different times.

## 4. Checked, not an issue

1. **Environment id and config.** Every `gym.make` goes through `pilot_common.make_env`
   (`pilot_common.py:20,38`): `highway-fast-v0`, overrides only `lanes_count`,
   `vehicles_count`, `vehicles_density` (and `duration` in Experiment 2). `simulation_frequency`
   stays at the fast default of 5, `policy_frequency` 1. (`highway-v0` was not
   measured; the pilot never used it.) `common/envs.py:41` also uses `highway-fast-v0`,
   but its config is a problem, see C2.
2. **Rendering.** `render_mode=None` for every experiment. `render_mode="rgb_array"` and
   `RecordVideo` appear only in `scripts/pilot/record_videos.py` (4 videos, 18 s total
   including GIF conversion).
3. **Environment lifecycle.** The env is created once per config (and once per duration
   in Experiment 2) and reused with `reset(seed=...)` for every episode
   (`pilot_common.py:run_episode`, `run_lean`). Resets are 4-10 ms, 1.4% of wall time.
4. **I/O and logging.** No per-step printing or disk writes. Per-step logs exist only in
   Experiment 3 (2,828 rows, in memory, written once). pandas is used only for final
   aggregation, never inside step loops. Result CSVs are small (0.8 MB largest).
5. **Instrumentation overhead.** `run_episode` metrics vs the bare loop, 15 episodes:
   x1.02 (default), x1.00 (heaviest), x1.06 (30 vehicles, density 0.7).
6. **Parallelism of the main sweep.** It was already multi-process (12 workers) and at the
   machine's plateau (140 steps/s, section C5).
7. **Threading.** The pilot imports only numpy, pandas, gymnasium, highway-env,
   matplotlib and moviepy; no torch. One Python process shows 16 threads (an idle
   OpenBLAS pool) and `OMP_NUM_THREADS` is not set anywhere, but `OMP_NUM_THREADS=1`
   changed nothing measurable (C5). It will matter once torch is imported; the rule
   stays in `CLAUDE.md` as a cheap precaution.
8. **GPU.** Not applicable to the pilot (no networks); for DDQN/PPO see the rules: the
   simulator (the bottleneck, C1) is CPU-only Python.
9. **Core type / priority.** Process affinity to P/E/LP cores and priority class make no
   measurable difference (C8).

## 5. Progress and remaining time; estimate with the issues fixed

Pilot compute (wall clock, from `results/pilot/timing.json`):

| Step | Wall | Steps | Steps/s | Mode |
|------|-----:|------:|--------:|------|
| Exp 1 benchmark (first, invalid) | 514 s | 19,964 | 38.8 | sequential |
| Exp 1 sweep, 50 eps x 120 cells | 712 s | 99,947 | 140 | 12 workers |
| Exp 3 reward scale | 179 s | 2,828 | 15.8 | sequential |
| Videos | 18 s | - | - | sequential |
| Extra low-density sweep | 715 s | 66,024 | 92 | 12 workers |
| Observation coverage | 454 s | 36,148 | 80 | 12 workers |
| Exp 2 duration | 2,874 s | 26,658 | 9.3 | sequential |
| Exp 1 re-benchmark (clean) | 500 s | 19,964 | 40 | sequential |
| **Total** | **5,966 s = 99.4 min** | 271,533 | | |

Progress when this analysis was written: all experiments have finished; only plots, the
report and the commit remain (no heavy compute left). Remaining compute: none.

Estimate for the *same* pilot, assuming the realistic aggregate throughput of 140
steps/s that the main sweep achieved (plateau, C5), uncertainty about +/-25% (C8):

| Variant | Work | Time |
|---------|------|-----:|
| As run | 271,533 steps, partly sequential | 99 min |
| Parallelize everything, keep all redundancy (and both benchmarks) | 231,605 steps / 140 + 17 min benchmarks | about 44 min (2.3x) |
| Fixed: parallel + fine tasks, one pass for sweep + obs + reward metrics, Exp 2 as one 60 s run per config (5,713 steps), benchmark on 6-8 cells | (99,947 + 66,024 + 5,713) / 140 + about 1.5 min benchmark + 0.3 min videos | **about 22 min (4.5x)** |

The 4.5x is the ceiling for this machine and these configs: C1 and C5 cannot be fixed in
code. Further gains must come from the experiment design (fewer vehicles, fewer cells).

## 6. Consequences for the real training runs

- Per-process throughput is 20-55 steps/s on the pilot configs, 2-5 steps/s on the
  `common/config.py` placeholders. At 140 steps/s aggregate (8-12 workers) 1 M env
  steps take about 2 h, so budget training in steps, not episodes, and parallelize
  across independent runs (seeds, algorithms) first.
- Keep `vehicles_count` at 20-30 and `simulation_frequency` at 5 unless a measurement
  says otherwise (C1, C2).
- A GPU will not help: the simulator is the bottleneck and the networks are small MLPs.
