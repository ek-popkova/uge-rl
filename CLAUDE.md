# Project guide for coding agents

Double DQN vs PPO (both implemented from scratch in PyTorch) on HighwayEnv
(`highway-fast-v0`, `highway-env==1.12.1`, gymnasium). Shared code lives in `common/`.
See `README.md` for layout and setup.

## Conventions

- Python: use the project venv (`.venv/Scripts/python.exe` on Windows). Code comments in English.
- `common/config.py` holds team-agreed placeholders. Do not change its values unless asked.
- Experiment outputs go to `results/` (git-ignored); do not commit results.

## Performance rules

Background and measurements: `docs/performance_analysis.md` (issue IDs C1-C8 below).
The simulator is slow (13-55 env steps/s per process on this machine) and parallel
scaling is capped at about 3.5-5x, so wasted simulation is the main cost. Follow these
for all experiment code (DDQN/PPO training, evaluation, sweeps):

1. **Measure before any long run.** In a quiet process, time >= 200 steps of the exact
   env/config you will use, then print `steps/s`, total planned steps and the estimated
   runtime before starting. If the estimate exceeds ~45 min, reduce the plan or ask.
   Never run a timing benchmark while other jobs are running, and never compare timings
   taken at different times: speed varies 2-5x run to run on this machine (C8).
2. **Budget the per-step cost.** Cost grows roughly linearly with `vehicles_count`
   (10 -> ~90 steps/s, 40 -> ~21) and `simulation_frequency=15` is ~2.5x slower than the
   `highway-fast-v0` default of 5 (C1, C2). Keep `simulation_frequency=5` and
   `vehicles_count <= 30`: extra vehicles spawn far ahead of the ego and never influence
   30 s episodes. The values in `common/config.py` follow this rule (5 Hz, 20 vehicles); re-measure if you change them.
3. **Never render outside `scripts/record_video.py`.** Use `render_mode=None` for training,
   evaluation and sweeps; no `RecordVideo` or other frame-capturing wrapper outside the
   video script.
4. **Reuse environments.** Create the env once per worker/config and call
   `reset(seed=...)` for every episode; never `gym.make` per episode.
5. **Parallelize independent runs across cores.** One process per run (seed, algorithm,
   config). Expose `--workers`; do not default to `os.cpu_count()`: aggregate throughput
   plateaus at ~8 workers here and drops at 12-16 (C5). Measure the scaling once on a
   new machine. Split work into fine-grained tasks (config x policy x seed chunk),
   heaviest first, and use `imap_unordered`, so no worker idles at the end (C5b).
6. **PPO: use `gymnasium.vector.AsyncVectorEnv`** for rollout collection (n envs = workers,
   each env created once). DDQN: one env per run, parallelize across runs, not inside one.
7. **Limit threads per worker.** Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
   `OPENBLAS_NUM_THREADS=1` before importing numpy/torch in every worker, and call
   `torch.set_num_threads(1)` in each worker. (No effect was measured for the
   simulator-only pilot, but torch will oversubscribe cores without it.)
8. **GPU does not help.** The simulator is pure-Python CPU code and the networks are small
   MLPs, so host-device transfers cost more than they save. Default to `device="cpu"`;
   switch only with a measurement that shows a gain.
9. **Do not re-simulate.** Compute all per-step metrics in one pass over the episodes
   (cheap: the pilot's instrumentation cost <= 6%). Derive shorter-duration results from
   the longest episodes: with the same seed trajectories are identical until truncation,
   so crash rates for 20/30/40 s are prefixes of the 60 s run (C3, C6). Do not run
   configs that are provably equivalent.
10. **Buffer I/O.** No printing or disk writes inside the step loop. Collect per-episode
    records in memory, flush per finished task/episode batch, aggregate with pandas only
    after the loops. Print progress at most every few seconds, with an ETA.
11. **Checkpoint and resume.** Write each finished task to its own file and skip finished
    tasks on restart (`--resume`); save training checkpoints (weights, optimizer, step
    counter, RNG state) every N steps. A crash must not lose more than one task (C7).
12. **Deterministic and worker-independent.** Derive every seed from
    `(global_seed, run id, episode index)`, never from worker id or scheduling order;
    each task builds its own RNGs. Results must be identical for `--workers 1` and
    `--workers N` (test this on a small case).
13. **Log speed.** Store measured steps/s and wall time in the run metadata so slow or
    disturbed runs can be recognized later (C8).

## Known issues

The scripts in `scripts/pilot/` were left unfixed on purpose because that run had to finish.
**Do not copy them as a template without fixing these** (details and measurements in
`docs/performance_analysis.md`):

- `scripts/pilot/exp2_duration.py:38-44,63` (C3): sequential, 24 runs where 4 would do:
  the 20/30/40 s runs repeat prefixes of the 60 s run, the `*_vc50` configs duplicate the
  30-vehicle ones, and results are written only at the very end. 48 min, 48% of the pilot.
- `scripts/pilot/exp1_density.py:62,121,133` (C4): `benchmark()` times the whole
  120-cell grid sequentially (500 s) and was run twice because the first run was
  disturbed. Benchmark 6-8 cells once and fit a cost model.
- `scripts/pilot/exp1_density.py:109,154` and `scripts/pilot/obs_coverage.py:84-86` (C5, C5b):
  `--workers` defaults to `min(12, cpu_count - 2)` (past the scaling knee) and tasks are
  coarse (16 and 12 tasks), leaving workers idle; the extra sweep ran at 92 steps/s vs
  140 achievable.
- `scripts/pilot/obs_coverage.py` (C5b, C6): re-simulates episodes that the main sweep already
  ran (36k steps). Fold such metrics into one `run_episode` pass.
- All `scripts/pilot/exp*.py` (C7): results are written once at the end, no `--resume`,
  progress counters without ETA.
- `scripts/pilot/pilot_common.py` and the worker pools: no `OMP_NUM_THREADS` /
  `torch.set_num_threads(1)` handling (harmless for the pilot, required once torch is
  imported).
- `common/config.py` (C2): **resolved** in the final configuration commit. The former placeholders
  (`simulation_frequency=15`, 50/80-vehicle presets, 5.0 and 2.4 steps/s) were replaced by the pilot-based values
  (`simulation_frequency=5`, 20 vehicles; see `docs/config_decisions.md`).
