# Project guide for coding agents

Double DQN vs PPO (both implemented from scratch in PyTorch) on HighwayEnv
(`highway-fast-v0`, `highway-env==1.12.1`, gymnasium). Shared code lives in `common/`.
See `README.md` for layout and setup.

## Conventions

- Python: use the project venv (`.venv/Scripts/python.exe` on Windows). Code comments in English.
- `common/config.py` holds team-agreed placeholders. Do not change its values unless asked.
- Experiment outputs go to `results/` (git-ignored); do not commit results.
- SB3 calibration runs (`scripts/calibration/`, report `docs/calibration.md`) are calibration only: run seeds `CALIBRATION_SEEDS = [99, 98]` (98 is reserved for them), outputs in `results/calibration/`, evaluated on `VAL_SEEDS` only; never part of the DDQN vs PPO comparison.
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
- **Shared training decisions are frozen** (table below, `docs/config_decisions.md` "Shared training decisions"): they may
  only change by team agreement, and a change after the main grid has started is reported as a deviation.
- **Benchmarks only in a quiet process.** `common.benchmark.measure_env_speed` and `measure_inference`
  run when nothing else is running (rule 1, C8). Wall times of runs that ran in parallel are comparable only
  between runs that ran under the same load; compare `update_time_s` per env step and env steps to the target
  rather than raw wall time.
- **Variants.** Sensitivity runs use `--set key=value` (recorded in `run_meta.json`) and go to
  `runs/<algo>/<reward>/<variant>/seed<s>`; the main grid is `main`. Never write a sensitivity run into `main`.
  The 24 sensitivity runs are `common.train_config.expand_sensitivity_grid()` (`SensitivityRun.cli_args()` gives the CLI);
  their base point is the main-grid run, guarded by `check_sensitivity_run`. Tuning runs are `--variant tune-<name>
  --seed 99` (`PILOT_RUN_SEED`; reward `balanced`; algorithm-specific keys only; default budget): `load_eval_episodes` and
  `per_model_table` exclude them, the harness refuses seed 99 outside `tune-*`; selection with
  `common.aggregate.select_tuning_config`, log in `docs/tuning_log.md`.

### Decided (shared by DDQN and PPO)

Frozen by the team; `common/train_config.py`, `common/networks.py`, `common/config.py`; justifications in
`docs/config_decisions.md` ("Shared training decisions"). Change only by team agreement.

| Decision | Value |
|---|---|
| `gamma` | 0.99 for both |
| network | MLP `hidden_sizes = (256, 256)`, `activation = "relu"`, input = 5x5 Kinematics flattened to 25 features (`obs_flatten`), no extra normalization; built by `common.networks.build_mlp`; weight initialization is algorithm-specific |
| same size | DDQN Q-network = PPO actor; the PPO critic is an extra network of the same size, separate from the actor (`ppo_shared_trunk = False`); `check_comparable` checks gamma, `hidden_sizes`, `activation` |
| tuning | at most 4 configurations per algorithm (defaults + up to 3), algorithm-specific keys only; `PILOT_RUN_SEED = 99`, reward `balanced`, density `medium`, `VAL_SEEDS` only, variant `tune-<name>`; criterion = mean `return_balanced` over the last 3 validation points, ties by lower `crash_rate` |
| sensitivity | reward `balanced`, density `medium`, main-grid `total_env_steps`, seeds `[0, 1, 2]`, one parameter at a time, every grid RELATIVE to the base value (base is in the grid by construction): lr base x1/3, x1, x3 (both); DDQN `eps_decay_fraction` base -0.2, base, base +0.2; PPO `clip_range` base -0.1, base, base +0.1 (a value outside (0, 1] / (0, 1) raises, never clipped: eps base in (0.2, 0.8], clip base in (0.1, 0.9)); base value = main-grid run (not re-run); 12 runs per algorithm, 24 in total. DDQN decays epsilon over `eps_decay_fraction` x `total_env_steps` (resolved to steps at run start, stored in `run_meta.json` `resolved`) |
| videos | `milestone_fractions = (0.1, 0.25, 0.5)` + the always-kept final checkpoint; candidate episode seeds shared by both algorithms: both crashed, both completed, and the two disagreement cases (only DDQN / only PPO crashed), `common.failure_cases.video_episode_seeds`; the report video uses training seed 0 of each reward, `main` condition, plus the early / middle milestones (`docs/evaluation_protocol.md` section 11) |
| derived / engineering (no team decision) | `TARGET_PERFORMANCE["beats_idle"]` = `crash_rate < 0.6` and `mean_speed > 24.8` (idle on `VAL_SEEDS`: 0.6 and 24.7564 m/s, source next to the constant); `checkpoint_every_env_steps = 10,000` (about 10 min of wall time at 12-17 env steps/s per process with 8-12 parallel runs, does not affect results); `update_log_every` K = 100 (DDQN) / 1 (PPO) |

### TO DECIDE BEFORE TRAINING

Everything below is still a PLACEHOLDER (`# PLACEHOLDER: decide before training`): a conventional value that only
keeps the code runnable, not a tuned one. The harness prints `placeholders_in_use` at start and stores it
in `run_meta.json`. Replace `_ph(...)` by a plain default in `common/train_config.py` once decided.

Shared open items (the team decides them after the pilot runs):

| Item | Where | What it affects |
|---|---|---|
| `total_env_steps` (100,000) | `TrainBudget` | training length of every model; same for both algorithms; must be a multiple of the PPO rollout size. The DDQN epsilon schedule and its sensitivity grid are fractions of it and follow it |
| `val_every_env_steps` (5,000), `val_n_episodes` (20 = all `VAL_SEEDS`) | `TrainBudget` | resolution and noise of the learning curves; validation cost |
| `TARGET_PERFORMANCE["strong"]` (`crash_rate <= 0.2` AND `mean_speed >= 25.5`) and `TARGET_CONSECUTIVE` (2) | `common/config.py` | the "env steps to reach a target" metric (computational cost comparison). Both levels are reported for every run, so an unreachable `strong` does not leave the comparison empty; the target must hold at `TARGET_CONSECUTIVE` validation points in a row (one lucky point of 20 episodes must not count). Joint on purpose: `slower` has a 0% crash rate at 20 m/s, so a crash bound alone is met without learning. Both levels must stay unreachable for every scripted baseline (`test_no_scripted_baseline_meets_target_performance`); re-run that test after any edit |

These five are the only shared open items; everything else below is algorithm-specific.

Algorithm-specific hyperparameters stay with each algorithm's owner (tuned within the tuning budget above):

| Item | Where | What it affects |
|---|---|---|
| DDQN: `lr, buffer_size, batch_size, learning_starts, train_freq, gradient_steps, target_update_interval, tau, eps_start, eps_end, eps_decay_fraction (0.3), max_grad_norm` | `DDQNConfig` | final performance and stability. `lr` and `eps_decay_fraction` are the bases of their sensitivity grids (eps base must stay in (0.2, 0.8]) |
| PPO: `lr, gae_lambda, clip_range, n_envs, n_steps, n_epochs, minibatch_size, ent_coef, vf_coef, max_grad_norm, target_kl` | `PPOConfig` | same; `n_envs * n_steps` is the rollout size and the step granularity. `lr` and `clip_range` are the bases of their sensitivity grids (clip base must stay in (0.1, 0.9)) |

Runtime implication. Aggregate throughput here is about 140 env steps/s with 8-12 processes (80-180 depending on
the session, `docs/performance_analysis.md` C5, C8), so every 1 M simulated steps cost about 2 h. One
model costs `T + n_val x val_n_episodes x 30` simulated steps, with `n_val = T / val_every` (`TrainBudget.n_val_points`).
With the placeholders (T = 100,000, 20 validations x 20 episodes x 30 steps = 12,000 steps) that is 112,000
steps; the 40 main-grid models are 4.5 M steps, about 9 h at 140 steps/s (13 h at 92). The sensitivity runs come
on top: 12 runs per algorithm, 24 in total (`expand_sensitivity_grid`), 2.7 M steps, about 5.4 h for both; tuning adds at most
4 runs per algorithm, 8 in total, 0.9 M steps, about 1.8 h. Network updates and the `EpisodeRecorder` overhead (measured: see `README.md`, Training)
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
- [ ] Hyperparameter tuning (`docs/tuning_log.md`) and sensitivity (the 24 runs of `expand_sensitivity_grid()`, 3 seeds per
  setting; run `check_sensitivity_runs` before building the table), reported separately from the main grid.
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
  deterministic; parameter count equals `n_parameters()` (DDQN: 73,733 = `build_action_network`; PPO: actor 73,733 + critic 72,705,
  both from `common.networks`); `hidden_sizes` equal for both algorithms (`check_comparable`).
- [ ] The real trainers pass the same harness tests as `RandomTrainer` (files, schemas, resume equals an uninterrupted run,
  monotonic `env_step`, `--workers 1` equals `--workers 2`: parametrize the existing harness tests over the trainer class), and a run
  is reproducible for a fixed seed.
- [ ] `make_eval_agent()` returns a light, picklable `Agent` (no replay or rollout buffer) that `validate()` can send to worker
  processes; `load_agent` in `scripts/record_video.py` loads both agents from `final.pt` and from a milestone checkpoint.
