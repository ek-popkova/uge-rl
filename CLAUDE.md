# Project guide for coding agents

Double DQN vs PPO (both implemented from scratch in PyTorch) on HighwayEnv
(`highway-fast-v0`, `highway-env==1.12.1`, gymnasium). Shared code lives in `common/`.
See `README.md` for layout and setup.

## Conventions

- Python: use the project venv (`.venv/Scripts/python.exe` on Windows). Code comments in English.
- `common/config.py` holds team-agreed placeholders. Do not change its values unless asked.
- Experiment outputs go to `results/` (git-ignored); do not commit results.
- A new metric, log column or algorithm component gets a test on a hand-built case in the same change (see "Tests: why they exist and when to run them").

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
6. **PPO: vectorized envs for rollout collection, each env created once.** `n_envs` is a
   hyperparameter (`PPOConfig.n_envs`), not the worker count; `common.train_env.make_vector_env`
   runs the envs in this process for `--workers 1` and in `AsyncVectorEnv` subprocesses for
   `--workers > 1` (identical data, tested). DDQN: one env per run, parallelize across runs, not
   inside one.
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

## Experiment tracking

Infrastructure for DDQN and PPO is in place and the algorithms are still stubs. A training run is
`common.training.TrainingRun` + an algorithm `Trainer` (`ddqn/train.py`, `ppo/train.py`); the harness
writes every number needed for the report, identically for both algorithms. Details: `README.md`
(Training), `docs/evaluation_protocol.md` (sections 5, 7, 9, 11).

Rules:

- **One path for all training runs.** Every training script uses `common.training.TrainingRun` and
  `common.train_logging`. No ad-hoc CSVs, prints or metric formulas. Metrics are computed only in
  `common/metrics.py` (the same `EpisodeRecorder` runs in evaluation and, through
  `common.train_env.TrainEnvWrapper`, in training). The x-axis is always `env_step` (transitions summed
  over vector envs), never episodes or updates.
- **Schemas live in one place.** `common/train_logging.py` defines `train_log.csv`, `update_log.csv` and
  `val_log.csv`; `common/metrics.py` defines the episode columns. Adding a column means updating the
  schema, `tests/test_training.py` / `tests/test_metrics.py` and `docs/evaluation_protocol.md`.
- **Fixed before training, never changed after seeing results:** `TARGET_PERFORMANCE` (both levels), `TARGET_CONSECUTIVE`, `REVERSAL_WINDOW`,
  `NEAR_MISS_TTC`, the budgets (`TrainBudget`), the validation interval and number of validation
  episodes, the seed sets and test conditions. Any deviation is reported as such.
- **Benchmarks only in a quiet process.** `common.benchmark.measure_env_speed` and `measure_inference`
  run when nothing else is running (rule 1, C8). Wall times of runs that ran in parallel are comparable only
  between runs that ran under the same load; compare `update_time_s` per env step and env steps to the target
  rather than raw wall time.
- **Variants.** Sensitivity runs use `--set key=value` (recorded in `run_meta.json`) and go to
  `runs/<algo>/<reward>/<variant>/seed<s>`; the main grid is `main`. Never write a sensitivity run into `main`.

### TO DECIDE BEFORE TRAINING

Everything below is a PLACEHOLDER (`# PLACEHOLDER: decide before training`): a conventional value that only
keeps the code runnable, not a tuned one. The harness prints `placeholders_in_use` at start and stores it
in `run_meta.json`. Replace `_ph(...)` by a plain default in `common/train_config.py` once decided.

| Item | Where | What it affects |
|---|---|---|
| `total_env_steps` (100,000) | `TrainBudget` | training length of every model; same for both algorithms; must be a multiple of the PPO rollout size |
| `val_every_env_steps` (5,000), `val_n_episodes` (20 = all `VAL_SEEDS`) | `TrainBudget` | resolution and noise of the learning curves; validation cost |
| `checkpoint_every_env_steps` (10,000), `milestone_fractions` (0.1, 0.5) | `TrainBudget` | resume granularity (at most this much work lost); which early checkpoints exist for "early vs late" videos |
| `TARGET_PERFORMANCE` levels `beats_idle` (`crash_rate < 0.6` AND `mean_speed > 24.8`, idle on `VAL_SEEDS`) and `strong` (`crash_rate <= 0.2` AND `mean_speed >= 25.5`), plus `TARGET_CONSECUTIVE` (2) | `common/config.py` | the "env steps to reach a target" metric (computational cost comparison). Both levels are reported for every run, so an unreachable `strong` does not leave the comparison empty; the target must hold at `TARGET_CONSECUTIVE` validation points in a row (one lucky point of 20 episodes must not count). Joint on purpose: `slower` has a 0% crash rate at 20 m/s, so a crash bound alone is met without learning. Both levels must stay unreachable for every scripted baseline (`test_no_scripted_baseline_meets_target_performance`); re-run that test after any edit |
| DDQN: `lr, gamma, buffer_size, batch_size, learning_starts, train_freq, gradient_steps, target_update_interval, tau, eps_start, eps_end, eps_decay_steps, max_grad_norm, hidden_sizes` | `DDQNConfig` | final performance and stability; `gamma >= MIN_GAMMA` is enforced |
| PPO: `lr, gamma, gae_lambda, clip_range, n_envs, n_steps, n_epochs, minibatch_size, ent_coef, vf_coef, max_grad_norm, target_kl, hidden_sizes` | `PPOConfig` | same; `n_envs * n_steps` is the rollout size and the step granularity |
| `hidden_sizes` must be equal for both (`check_comparable`) | both configs | fair comparison (network size) |
| Sensitivity grid per algorithm (not defined yet): lr, gamma, DDQN epsilon schedule (`eps_decay_steps`, `eps_end`), PPO `clip_range` | `--set` runs, 3 seeds each | assignment point 7; each setting costs `3 x (T + validation)` steps |
| `update_log_every` K (DDQN 100, PPO 1) | CLI default (`common.training.DEFAULT_UPDATE_LOG_EVERY`) | size of `update_log.csv` only |

Runtime implication. Aggregate throughput here is about 140 env steps/s with 8-12 processes (80-180 depending on
the session, `docs/performance_analysis.md` C5, C8), so every 1 M simulated steps cost about 2 h. One
model costs `T + n_val x val_n_episodes x 30` simulated steps, with `n_val = T / val_every` (`TrainBudget.n_val_points`).
With the placeholders (T = 100,000, 20 validations x 20 episodes x 30 steps = 12,000 steps) that is 112,000
steps; the 40 main-grid models are 4.5 M steps, about 9 h at 140 steps/s (13 h at 92). The sensitivity runs come
on top: e.g. 4 hyperparameters x 2 non-default values x 3 seeds = 24 runs per algorithm, 2.7 M steps each,
about 11 h for both. Network updates and the `EpisodeRecorder` overhead (measured: see `README.md`, Training)
are not included. `TrainingRun` prints its own simulation-only estimate from a quiet benchmark before
starting and refuses above `--max-minutes` (default 45) without `--force`.

### Not expressible in code yet (checklist for the report)

- [ ] Qualitative behaviour (lane changes, overtaking, speed adaptation, failure cases): record videos of
  crash seeds (`python -m scripts.record_video --from-eval-csv ... --outcome crashed --n 3`) from the final checkpoint
  AND from an early milestone checkpoint of the same run. Needs trained agents and the `load_agent` TODO in
  `scripts/record_video.py`. `overtakes` counts vehicles passed, not manoeuvres: confirm on video.
- [ ] Evaluation driver for trained agents (the analogue of `scripts/eval_baselines.py`: `evaluate()` for the five
  conditions on `TEST_SEEDS`, `save_eval`, with `variant`); not written because no agent can be loaded yet.
- [ ] "Does a higher return mean better driving?": needs trained models. Compare the ranking by `return_<preset>`
  with the behavioural metrics (crash rate, completion, mean speed, near-miss, lane changes) across the four reward presets.
- [ ] Hyperparameter sensitivity: choose the grid (table above), run 3 seeds per setting, report separately from the main grid.
- [ ] Cost comparison: run the quiet benchmarks (`quiet_benchmark_steps_per_s`, `inference` in `run_meta.json`) in a quiet process;
  runs trained in parallel have disturbed wall times (C8).
- [ ] Critical discussion (limitations: 5 Hz fast simulator, one training density, 5 seeds, placeholder-derived hyperparameters).

## Tests: why they exist and when to run them

Why:

- The project is a controlled comparison of DDQN and PPO. The costly failure is not a crash but a silent change
  that corrupts numbers and is only found after hours of training, or never.
- What each group protects:
  - **Config and seed invariants** (`test_config.py`, seed and layout tests): disjoint seed sets, env config equal to
    the documented one, action indices. Results stay comparable with the baselines and the docs.
  - **Reward and metric formulas on hand-built cases** (`test_rewards.py`, `test_metrics.py`, `test_evaluate.py`): TTC,
    lane changes, reversals, overtakes, return recomputation under every preset. The report rests on these.
  - **Determinism** (`test_evaluate.py`, `test_training.py`): results independent of `--workers`, episode order and
    fresh vs reused env. Seeds stay reproducible.
  - **Training harness via `RandomTrainer`** (`test_training.py`): files and schemas, resume equals an uninterrupted run,
    monotonic `env_step`. DDQN and PPO are written by different people against the same harness; these tests catch
    either implementation breaking it.
  - **`TARGET_PERFORMANCE` (both levels) unreachable for scripted baselines**: the steps-to-target metric stays meaningful.

When:

- **Fast tier** (`python -m pytest -m "not slow"`): during development, after any change. Pure-function and
  fake-env tests only; no real environment is created.
- **Full suite** (`python -m pytest`): before every commit or merge that touches `common/`, after pulling a teammate's
  changes, after changing package versions, and always before launching the main 40-model grid.
- **Never while training runs or timing benchmarks are running**: the tests load the CPU and distort steps/s and wall
  times in `run_meta.json` (C8).
- A test that creates a real highway-env environment or runs episodes must carry `@pytest.mark.slow`.

Still to add when the algorithms are implemented (each needs the real `DDQNTrainer` / `PPOTrainer`):

- [ ] Time-limit handling, for DDQN and PPO: bootstrap from `s'` when `truncated`, value 0 only when `terminated`; test both
  cases on hand-built transitions (`docs/evaluation_protocol.md` section 1).
- [ ] Double DQN target `r + gamma * Q_target(s', argmax_a Q_online(s', a))` on a hand-computed example, and that it differs from
  the vanilla DQN max target `r + gamma * max_a Q_target(s', a)`; the target network updates on schedule (`target_update_interval`,
  `tau`); epsilon follows its schedule; a hand-built batch gives the expected TD loss.
- [ ] GAE on a short hand-computed trajectory, including a truncation in the middle of the rollout (bootstrap from `final_obs`,
  the GAE recursion is cut there) and a terminal state (no bootstrap).
- [ ] PPO clipped objective on a toy case (ratios inside and outside the clip range, both advantage signs), the `approx_kl` /
  `clip_fraction` definitions, and the KL early stop (`n_epochs_done < n_epochs` when `approx_kl` exceeds `target_kl`).
- [ ] `save` / `load` round trip gives identical actions for a fixed observation batch; `act(obs, deterministic=True)` is
  deterministic; parameter count equals `n_parameters()`; `hidden_sizes` equal for both algorithms (`check_comparable`).
- [ ] The real trainers pass the same harness tests as `RandomTrainer` (files, schemas, resume equals an uninterrupted run,
  monotonic `env_step`, `--workers 1` equals `--workers 2`: parametrize the existing harness tests over the trainer class), and a run
  is reproducible for a fixed seed.
- [ ] `make_eval_agent()` returns a light, picklable `Agent` (no replay or rollout buffer) that `validate()` can send to worker
  processes; `load_agent` in `scripts/record_video.py` loads both agents from `final.pt` and from a milestone checkpoint.
