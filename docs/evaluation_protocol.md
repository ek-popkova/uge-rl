# Evaluation protocol

This protocol is fixed before any agent is trained. It describes how DDQN and PPO are trained, tested and compared, so
that the text can be copied into the report. Code: `common/evaluate.py`, `common/metrics.py`, `common/aggregate.py`,
`common/baselines.py`, `common/paths.py`, `common/seeding.py`; training-time tracking: `common/training.py`,
`common/train_logging.py`, `common/train_env.py`, `common/run_meta.py`, `common/benchmark.py`,
`common/train_config.py`; values: `common/config.py`.

## 1. Models

- Two algorithms (DDQN, PPO) x four rewards (`default`, `efficiency`, `balanced`, `safety`) x five training seeds
  (`TRAIN_SEEDS = [0, 1, 2, 3, 4]`) = 40 models. Hyperparameter sensitivity runs use three seeds
  (`SENSITIVITY_SEEDS = [0, 1, 2]`) and are reported separately.
- All models train on the same environment: `highway-fast-v0`, medium density (`TRAIN_DENSITY`), `BASE_CONFIG`.
- Same number of environment steps and the same network for both algorithms: MLP `hidden_sizes = (256, 256)` with ReLU on
  the 5 x 5 Kinematics observation flattened to 25 features (DDQN Q-network = PPO actor; the PPO critic is an extra
  network of the same size; `common/networks.py`). Discount factor gamma = 0.99 for both (`MIN_GAMMA = 0.99`): a late crash
  still matters (0.99^29 is about 0.75), whereas with a smaller gamma the lighter rewards can rank a crash above slow safe
  driving (`docs/config_decisions.md`, "Shared training decisions").
- **Final checkpoint only.** The model evaluated is the one at the end of the step budget. If a checkpoint is ever
  selected (for example the best validation return), the selection uses `VAL_SEEDS`, never `TEST_SEEDS`.
- **Time limit is not termination.** An episode that ends because the 30 s (or 60 s) limit is reached is `truncated`, not
  `terminated`. Training code bootstraps from the next state when `truncated` is true (PPO: `r + gamma * V(s')` in the
  GAE; DDQN: `r + gamma * Q_target(s', argmax_a Q_online(s', a))`), and treats only a crash
  (`terminated`) as a terminal state with value 0. Treating the time limit as terminal would make the value of a state
  depend on the remaining time, which the observation does not contain.

## 2. Seed sets

| Set | Seeds | Role |
|---|---|---|
| `PILOT_SEEDS` | 1000..1049 | environment pilot with scripted policies (done) |
| `TRAIN_SEEDS` | 0..4 | run seeds of the main grid |
| `SENSITIVITY_SEEDS` | 0, 1, 2 | run seeds of the sensitivity runs (reuse training seeds 0..2: the base point is the main-grid run) |
| `PILOT_RUN_SEED` | 99 | run seed of the hyperparameter tuning runs (`tune-<name>` variants), never a result seed |
| `CALIBRATION_SEEDS` | 99, 98 | run seeds of the SB3 calibration runs (`scripts/calibration/`, `docs/calibration.md`; 99 is `PILOT_RUN_SEED`, 98 is used nowhere else and refused by the training harness); never result seeds |
| `VAL_SEEDS` | 20000..20019 | episode seeds of the periodic evaluation during training (learning curves) |
| `TEST_SEEDS` | 10000..10099 | episode seeds of the final evaluation, used once |

Pilot, train, validation and test sets and `PILOT_RUN_SEED` are pairwise disjoint (asserted in `common/config.py`, tested in
`tests/test_config.py`). Seeding rules for training runs are in `common/seeding.py`: global seeds from the run seed,
`env.reset(seed=s)` once and unseeded resets afterwards, exploration from `np.random.default_rng(s)`, vectorized env `i`
gets seed `100000 + s * 1000 + i` (`derived_env_seeds(s, n_envs)`, at most 1000 envs per run). The offset keeps derived
seeds disjoint from the pilot, validation and test seeds for every run seed (tested for all `TRAIN_SEEDS` and
`SENSITIVITY_SEEDS` with up to 64 envs, and for `PILOT_RUN_SEED` with all 1000 env indices against every other run seed).

## 3. Test conditions

Every condition is an environment id plus config overrides on top of `BASE_CONFIG` and the density preset
(`common.config.TEST_CONDITIONS`, `common.envs.make_test_env`). All are evaluation-only and are never used for training.

| Condition | Environment | Differs from training | Why it exists |
|---|---|---|---|
| `main` | `highway-fast-v0`, density 0.7 | nothing (new episodes) | the headline result |
| `low` | same, density 0.4 | traffic density | generalization to easier traffic |
| `high` | same, density 1.0 | traffic density | generalization to harder traffic |
| `long` | same, 60 s, 30 vehicles | episode length (60 s instead of 30 s) | longer drives; does the policy keep driving safely? 30 vehicles because 20 was only verified up to 30 s in the pilot (30 and 50 gave identical results up to 60 s) |
| `precise` | `highway-v0`, 15 Hz | physics frequency, and collisions between other vehicles are checked | simulator precision: does the policy depend on the coarse 5 Hz physics of the fast variant? |

**Reading `long`.** Crash risk per km is not constant over a 60 s episode, even for fixed policies. Idle has 1.28
crashes/km on `main` but 1.59 on `long` (ratio 1.24; random 1.02 and faster 1.01; slower never crashes, so no ratio), and
idle crashes at 3.2 per 100 steps at risk during the first 30 s but 9.8 during the second 30 s, while random stays near 5
in both halves. Where the risk comes from depends on the policy and on where the ego is relative to the traffic.
Therefore:

- agents on `long` are judged relative to the baselines on the same condition: compare the agent's `long`/`main` ratio of
  crashes per km with the baselines' ratio, and compare models with each other within `long`;
- an agent's `long` result is never compared directly with its own `main` result as if they measured the same thing.

`ego_spacing = 1.5` is pinned in `BASE_CONFIG` (it is the `highway-fast-v0` default; `highway-v0` defaults to 2), so
`precise` starts from the same initial layout as training. `highway-fast-v0` and `highway-v0` differ otherwise only in
the defaults that `BASE_CONFIG` pins and in the collision check of non-ego vehicles.

## 4. Test episodes

- 100 episodes per model and condition, one per seed in `TEST_SEEDS`, `env.reset(seed=seed)` before every episode, so
  every model faces identical starting situations in every condition.
- The policy is deterministic (`agent.act(obs, deterministic=True)`: greedy for DDQN, argmax or mean action for PPO); no
  learning and no exploration during evaluation.
- The environment is created once per worker and reused (`reset(seed=...)` per episode). Results do not depend on the
  number of workers, on the episode order, or on whether the env is fresh (tested).
- Evaluation runs with the rendering off and one thread per worker.
- The test set is used once per model and condition. Nothing (hyperparameters, checkpoint, number of steps) is chosen
  on `TEST_SEEDS`.

## 5. Metrics

Recorded per episode in a single pass (`common.metrics`):

| Metric | Definition |
|---|---|
| `episode_return` | undiscounted return under the reward preset of the env; `return_env` = return of the env's own default reward |
| `return_<preset>` | return under any preset, recomputed from `sum_high_speed`, `sum_right_lane`, `lane_changes`, `crashed` (`recompute_return`) |
| `crashed` | `info["crashed"]` at the end; crash rate = mean over episodes |
| `completed` | time limit reached without a crash (truncated and not crashed) |
| `length` | number of steps |
| `mean_speed` | mean ego speed over the steps [m/s] |
| `distance_m` | ego x at the end minus at the start [m] |
| `crashes_per_km` | `crashed / distance_km` per episode; in every aggregate the **ratio of sums** (total crashes / total km), because a mean of per-episode ratios is dominated by early crashes with a tiny distance |
| `lane_changes` | number of steps in which the real lane index changed (not the chosen action) |
| `min_gap` | minimum bumper-to-bumper gap to the leader (nearest vehicle ahead in the ego lane) over the episode [m] |
| `min_ttc` | minimum time-to-collision `gap / (v_ego - v_leader)` over the steps in which the ego closes in on the leader [s]; a gap of 0 or less counts as 0; undefined if it never closes in |
| `near_miss` | not crashed and `min_ttc < 1.5 s` (`NEAR_MISS_TTC`); fixed before training |
| `near_miss_steps` | number of steps whose state has TTC < `NEAR_MISS_TTC`, a rate in addition to the per-episode flag; the final crash step is not counted (a crash is not a near miss) |
| `truncated` | the time limit ended the episode (`completed` = `truncated` and not `crashed`) |
| `act_LANE_LEFT`, `act_IDLE`, `act_LANE_RIGHT`, `act_FASTER`, `act_SLOWER` | how often the agent chose each meta-action (action histogram); the five counts sum to `length` |
| `lane0_steps` ... `lane2_steps` | steps spent in each lane after the step (0 = leftmost); sum to `length` |
| `rightmost_lane_share` | `lane2_steps / length` |
| `lane_change_reversals` | lane changes that undo the immediately preceding lane change within `REVERSAL_WINDOW = 3` steps (A to B, then B to A; every change in A-B-A-B counts, so that is 2); fixed before training, next to `NEAR_MISS_TTC` in `common/metrics.py` |
| `speed_std` | standard deviation (ddof 0) of the ego speed over the steps [m/s] |
| `n_speed_changes` | steps in which the ego target speed changed (a FASTER / SLOWER that took effect; at the speed limits it does not) |
| `overtakes` | vehicles that were ahead of the ego at the start and are behind it at the end, any lane (net passes). **Compared per km, not per episode:** `overtakes_per_km` = total overtakes / total km, the ratio of sums exactly like `crashes_per_km`, because the per-episode count grows with speed and survival time (an episode that crashes after 5 s has almost none) |

Gap and TTC use the state after every step and the initial state. Mean `min_gap` and `min_ttc` are averages over the
episodes where they are defined. Counts (`act_*`, `lane*_steps`, `near_miss_steps`, ...) are summarized as means of
per-episode counts, so they depend on the episode length; divide by `length` for shares.

**`overtakes` is robust here, but it counts passes, not manoeuvres.** In `highway-env` 1.12.1 the road is never
emptied of vehicles (tested for `main` and `precise`) and every other vehicle spawns ahead of the ego (tested), so
"ahead at the start" is a well-defined set that is compared with the ego position only twice (start, end), with no
extra simulation. It does not tell an active overtake from a vehicle that changed lane or slowed down; the videos
(section 11) decide that. A vehicle that passes the ego again is not counted. It is a behavioural indicator, not a
measure of skill. The per-episode `overtakes` column is kept; summaries and aggregated tables use `overtakes_per_km`.

## 6. Aggregation

- **Level 1, per model** (algorithm, reward, variant, seed, condition): mean of the metrics over its 100 test episodes.
  `variant` is `main` for the main grid and the sensitivity-run name otherwise (section 11).
- **Level 2, per (algorithm, reward, variant, condition):** mean over the five training seeds with a two-sided 95% Student-t
  interval, `mean +/- t(0.975, n-1) * sd / sqrt(n)` with `n` the number of seeds (`n-1 = 4` in the main grid, t = 2.776)
  and `sd` the sample standard deviation. With fewer than two seeds no interval is reported. The reason: a bootstrap over
  five values underestimates the uncertainty (it cannot leave the observed range and ignores that the standard deviation
  itself is estimated from few runs), and the t-interval is the standard choice for few runs. The training seed is the
  unit of analysis; pooled episodes are never treated as independent samples. Individual runs (the level-1 values) are
  always shown next to the mean.
- Returns are only comparable within one reward preset (a crash costs 15, 20 or 30 depending on the preset). Reward
  presets are compared by behavioural metrics (crash rate, completion rate, mean speed, lane changes, near-miss rate,
  crashes per km). `return_own` is the return under the preset the model was trained with.

## 7. Learning curves

During training, `common.evaluate.validate(agent, n=val_n_episodes, reward=<training reward>)` runs the agent on the
first `val_n_episodes` `VAL_SEEDS` episodes (at most 20) in the `main` condition every `val_every_env_steps`
environment steps (same schedule for both algorithms) and appends one row to `val_log.csv` (section 11). The x-axis is
always `env_step` (transitions summed over vector envs), never episodes or updates. A validation runs at the first update
boundary at or after each multiple of the interval (DDQN: every step, PPO: after every rollout of `n_envs * n_steps`
steps) and the actual `env_step` is stored; a last validation always happens at the end of the budget.

- **Both reward views from one run.** `validate()` returns `episode_return` under the reward the validation env was
  built with (the training reward) and `return_<preset>` for every preset (`default`, `efficiency`, `balanced`,
  `safety`), recomputed from the per-episode components, so the curve under the training reward and under the default
  reward are both available without re-running anything. Returns of different presets are not comparable with each
  other (section 6).
- **Alignment across seeds.** Curves are shown per seed together with the mean over seeds. `learning_curve_table`
  aligns the k-th validation of every seed (by validation index `val_idx`, not by `env_step`, because the actual steps of
  PPO differ slightly between runs) and returns the mean and the 95% Student-t interval over seeds per point (`n - 1`
  degrees of freedom with `n` the seeds that have that point). The reported x is the mean of the aligned `env_step`
  values (min and max are given too).
- **Area under the curve** (`curve_auc`, `auc_across_seeds`): trapezoid rule over `env_step` from the first to the last
  validation point, divided by the step span (so it is the average value of the metric during training, in the units of
  the metric); per seed first, then mean and t-interval over seeds.
- **Steps to a target** (`steps_to_threshold`, targets in `common.config.TARGET_PERFORMANCE`): the `env_step` and the
  training wall time (without validation time, `train_wall_s`) at which a run reaches a target level. A level is a list
  of behavioural conditions (comparable across reward presets) that must ALL hold at the same validation point, with
  at least one safety and one efficiency condition (validated at import). It is reached at the **first of
  `TARGET_CONSECUTIVE = 3` consecutive validation points** (for both levels) at which all its conditions hold, and that first point's
  `env_step` and `train_wall_s` are reported; NaN if the level is never reached. Validation uses 20 episodes, so
  `crash_rate` moves in steps of 0.05 and one lucky point must not count (an isolated passing point does not).
- **Two pre-registered levels**, both reported for every run and fixed before training (section 10):
  `beats_idle` (strictly better than idle on both metrics on `VAL_SEEDS`: crash rate below 0.60 and mean speed above
  24.8 m/s) and `strong` (crash rate <= 0.4 and mean speed >= 25.5 m/s). If `strong` turns out unreachable (NaN for every
  model), `beats_idle` still allows a comparison, and no threshold is changed after seeing results.
- **The levels are joint and must stay unreachable for the scripted baselines.** `slower` has a 0% crash rate at 20 m/s
  on `main`, so a crash bound alone would reward an agent that only learns to brake; `idle` crashes in 71% of the
  episodes at 24.7 m/s on `main` (VAL_SEEDS: 0.60 and 24.8 m/s). `tests/test_training.py::test_no_scripted_baseline_meets_target_performance`
  runs the four baselines through `validate()` on `VAL_SEEDS` and fails if one meets either level, which guards the values
  against later edits. `beats_idle` is derived from the idle baseline (crash rate 0.6 exactly and mean speed 24.7564 m/s on
  `VAL_SEEDS`, the latter rounded up to 24.8) and needs no team decision; `strong` and `TARGET_CONSECUTIVE = 3` were decided by the team after the SB3 calibration
  (`docs/config_decisions.md`, "Values frozen after the calibration"; evidence `docs/calibration.md` section 4).
- **Steps to the plateau** (`common.aggregate.steps_to_plateau`, post-hoc: it is computed from the finished validation curve and never
  used to stop a run; every run trains for the full budget). Pre-registered constants `PLATEAU_WINDOW = 5` and `PLATEAU_TOL = 0.10`.
  For the metric `return_own` (the return under the run's own training reward, `return_<reward>`): final level = mean of the raw metric
  over the last 20% of the validation points; improvement = final level minus the first validation value; the curve is smoothed with a
  rolling mean over `window` validation points (the window that starts at the point, so a clean step is found at the step, not
  `window - 1` points later); the plateau point is the earliest validation point from which the smoothed curve stays within
  `tol * improvement` of the final level until the end. The tolerance is relative to the improvement, not to the return, because a fixed
  5% of the return was smaller than the validation noise of DQN in the calibration. The result holds the `env_step` and `train_wall_s`
  of that point and `censored = True` if it lies inside the last 20% of the points (the run may not have converged); with no improvement
  (or no stable plateau) the result is NaN with `censored = True`. Per model, then mean and 95% t-interval over seeds, NaN-aware, with the
  number of seeds that reached it and the number of censored ones (`cost_per_model`, `cost_across_seeds`). Report censored runs as such.

## 8. Baselines

Four scripted policies (`random`, seeded per episode; `idle`; `slower`; `faster`) run through the same pipeline: the same
`TEST_SEEDS`, conditions and metrics, with the return recomputed under every reward preset
(`python -m scripts.eval_baselines`, outputs in `results/eval/baselines/`). They calibrate the scale of the metrics
(what a trivial policy achieves, what is the floor and what is feasible), and `precise` shows how much harder or easier
the precise physics is before any agent is tested on it (Patterson et al., 2024). The baselines have no training seed, so
their uncertainty is a Wilson interval over episodes.

### Baseline reference values (`TEST_SEEDS`, 100 episodes each; Wilson 95% interval over episodes)

| Condition | Policy | Crash rate (95% CI) | Completed | Mean speed [m/s] | Crashes / km | Near-miss rate |
|---|---|---|---:|---:|---:|---:|
| main | random | 80% (71%-87%) | 20% | 24.9 | 2.07 | 5% |
| main | idle | 71% (61%-79%) | 29% | 24.7 | 1.28 | 4% |
| main | slower | 0% (0%-4%) | 100% | 20.0 | 0.00 | 0% |
| main | faster | 99% (95%-100%) | 1% | 28.8 | 2.83 | 1% |
| low | random | 49% (39%-59%) | 51% | 25.1 | 0.84 | 10% |
| low | idle | 11% (6%-19%) | 89% | 25.0 | 0.15 | 3% |
| low | slower | 0% (0%-4%) | 100% | 20.0 | 0.00 | 0% |
| low | faster | 59% (49%-68%) | 41% | 29.7 | 0.81 | 5% |
| high | random | 97% (92%-99%) | 3% | 24.5 | 3.65 | 1% |
| high | idle | 96% (90%-98%) | 4% | 24.2 | 2.80 | 0% |
| high | slower | 2% (1%-7%) | 98% | 20.0 | 0.03 | 0% |
| high | faster | 100% (96%-100%) | 0% | 28.3 | 4.39 | 0% |
| long | random | 97% (92%-99%) | 3% | 24.9 | 2.11 | 1% |
| long | idle | 100% (96%-100%) | 0% | 24.6 | 1.59 | 0% |
| long | slower | 0% (0%-4%) | 100% | 20.0 | 0.00 | 0% |
| long | faster | 100% (96%-100%) | 0% | 28.8 | 2.86 | 0% |
| precise | random | 77% (68%-84%) | 23% | 24.7 | 1.86 | 6% |
| precise | idle | 70% (60%-78%) | 30% | 24.6 | 1.24 | 5% |
| precise | slower | 0% (0%-4%) | 100% | 20.0 | 0.00 | 0% |
| precise | faster | 98% (93%-99%) | 2% | 28.8 | 2.74 | 2% |

The baseline numbers on `TEST_SEEDS` are the reference values for the report. The pilot numbers (50 seeds,
`PILOT_SEEDS`) were used only to choose the densities and the reward weights; densities and rewards are not re-tuned on
test results. The test-seed baselines differ from the pilot within sampling noise (idle on medium: 71% on 100 test seeds
vs 54% on 50 pilot seeds).

Checks: on the pilot seeds the pipeline reproduces the pilot's idle episodes exactly (50 of 50 crash outcomes, lengths
and mean speeds identical at medium density). The idle crash rate on `TEST_SEEDS` (71% / 11% / 96% for medium / low /
high) is higher than the pilot's 54% / 8% / 92% because the 50 pilot seeds and the 100 test seeds are different
samples (the pilot interval at medium is about 40-67%), not because the setup differs.
`precise` (`highway-v0`, 15 Hz) is almost the same task as `main` for these policies: crash rates differ by at most 3
points, mean speed by at most 0.12 m/s, and the crash outcome flips in only 5 (random), 11 (idle), 0 (slower) and 1
(faster) of 100 paired episodes, in both directions. The physics precision therefore does not noticeably change the
difficulty before any agent is tested; a trained agent can still be sensitive to it.

## 9. File layout

```
results/runs/<algo>/<reward>/<variant>/seed<s>/final.pt              final checkpoint (agent weights)
results/runs/<algo>/<reward>/<variant>/seed<s>/milestone_<p>pct.pt   weights at p% of the budget (early vs late)
results/runs/<algo>/<reward>/<variant>/seed<s>/resume.pt             full state for --resume (overwritten)
results/runs/<algo>/<reward>/<variant>/seed<s>/train_log.csv         one row per training episode
results/runs/<algo>/<reward>/<variant>/seed<s>/update_log.csv        one row per (K) gradient update(s)
results/runs/<algo>/<reward>/<variant>/seed<s>/val_log.csv           validation metrics, one row per validation
results/runs/<algo>/<reward>/<variant>/seed<s>/run_meta.json         identity, config, timing, versions
results/eval/<algo>/<reward>/<variant>/seed<s>/<condition>.csv       per-episode test results
results/eval/baselines/<condition>.csv, summary.csv                  scripted baselines
```

`<variant>` is `main` for the 40 main-grid models. A sensitivity run is a variant named from its overrides by
`common.train_config.variant_name` (keys sorted, e.g. `clip_range=0.1`, `gamma=0.995,lr=0.0003`, `hidden_sizes=32x32`),
so it can never overwrite a main run. The layout without the variant level was never used for a trained run and is not
supported (`load_eval_episodes` raises if it finds one). Hyperparameter tuning runs are the variants `tune-<name>` (reward
`balanced`, run seed `PILOT_RUN_SEED = 99`); `load_eval_episodes` and `per_model_table` exclude them by default, so they never
reach a result table. Supplementary runs are the variants `supp-<name>` of the pre-declared registry
`common.train_config.SUPPLEMENTARY_RUNS`; a run with a supplementary reward preset (`common.config.SUPPLEMENTARY_REWARDS`) has that
preset as `<reward>`. `load_eval_episodes` and `per_model_table` exclude them unless `include_supplementary=True`, so they never
reach a DDQN vs PPO table. Every table carries a `role` column (`common.paths.variant_role`: `main`, `sensitivity`, `tuning`,
`supplementary`). Helpers in `common/paths.py`. `results/` is git-ignored.

## 10. Fixed before training, not to be changed after seeing results

Reward weights, densities and `BASE_CONFIG` (`common/config.py`); the seed sets; the test conditions and the number of test
episodes; the metric definitions including the near-miss threshold of 1.5 s and the lane-change reversal window of 3 steps;
`TARGET_PERFORMANCE` (both levels) and `TARGET_CONSECUTIVE`; deterministic evaluation of the final checkpoint; the aggregation (seed as unit, 95% Student-t
interval over seeds); gamma >= 0.99; the same step budget (`TrainBudget`: total steps, validation interval and number of
validation episodes) and network size for both algorithms; the treatment of time-limit truncation. Any deviation is reported
as such. No shared item is open any more: the values below were decided after the SB3 calibration (`docs/config_decisions.md`, "Values
frozen after the calibration"); only the algorithm-specific hyperparameters are still placeholders (list in `CLAUDE.md`, "TO DECIDE
BEFORE TRAINING") that their owners choose during tuning, before the first main-grid run.

Shared training decisions, agreed by the team (`docs/config_decisions.md`, "Shared training decisions"); they may only change by
team agreement, and a change after the main grid has started is reported as a deviation:

- gamma = 0.99 for both algorithms; network: MLP `hidden_sizes = (256, 256)`, ReLU, input = the 5 x 5 Kinematics observation
  flattened to 25 features without extra normalization, built by `common.networks.build_mlp` for both; DDQN Q-network = PPO
  actor, the PPO critic is an extra network of the same size, separate from the actor (`ppo_shared_trunk = False`); weight
  initialization is algorithm-specific. `check_comparable` checks gamma, `hidden_sizes` and `activation`.
- **Tuning budget:** at most 4 configurations per algorithm (the defaults + up to 3 variations), only algorithm-specific
  hyperparameters varied (never the shared values, the budget or the validation settings); one run per configuration with
  `PILOT_RUN_SEED = 99`, reward `balanced`, density `medium`, validation on `VAL_SEEDS` only, variant `tune-<name>`.
  **Selection criterion:** mean `return_balanced` over the last 3 validation points, ties broken by the lower `crash_rate`
  (`common.aggregate.select_tuning_config`). Log: `docs/tuning_log.md`. Tuning never touches `TEST_SEEDS`.
- **Sensitivity framework:** reward `balanced`, density `medium`, the main grid's `total_env_steps`, `SENSITIVITY_SEEDS`,
  one parameter at a time with the others at their frozen base values. Grid (`SENSITIVITY_GRID`), every value relative to the
  base: learning rate base x1/3, x1, x3 (both algorithms); DDQN `eps_decay_fraction` (the fraction of `total_env_steps` over
  which epsilon decays) base -0.2, base, base +0.2; PPO `clip_range` base -0.1, base, base +0.1; `gamma` is not varied. A
  resolved value outside (0, 1] (`eps_decay_fraction`) or (0, 1) (`clip_range`) raises, it is never clipped. The base value
  is in the grid by construction and is not re-run: its point in a sensitivity table is the main-grid
  run (reward `balanced`, seeds 0, 1, 2, variant `main`), so 4 settings x 3 seeds = 12 runs per algorithm, 24 in total
  (`expand_sensitivity_grid`). `check_sensitivity_run` / `check_sensitivity_runs` fail if a sensitivity run differs from the
  main run in anything but its varied parameter. Relative values are resolved from the base config at expansion time; the
  absolute values are in the run's `run_meta.json` (`hyperparameters`, plus `resolved.eps_decay_steps` for the DDQN epsilon
  schedule). The 24 runs for the current placeholder bases are listed in `docs/config_decisions.md`.
- **Supplementary runs** (outside the DDQN vs PPO comparison, reported separately): declared before training in
  `common.train_config.SUPPLEMENTARY_RUNS` (name -> algorithm, exact overrides, rewards, seeds of `TRAIN_SEEDS`, purpose; validated
  at import), run as `--variant supp-<name>` with the frozen budget and the training density. `check_run_roles` refuses a `supp-`
  run that does not match its entry exactly, a `supp-` name that is not registered, and every other variant that is neither
  `main`, a sensitivity point of `expand_sensitivity_grid()` nor `tune-*` (debug runs, whose total steps / validation settings differ from the frozen budget, excepted). The base point
  is the main-grid run of the same algorithm, reward and seed; `check_supplementary_run` / `check_supplementary_runs` fail if
  anything but the declared overrides differs. Supplementary reward presets (`SUPPLEMENTARY_REWARDS`, experiment (b); weights fixed
  analytically before the runs) are usable by registered `supp-` runs only, never by main, tuning or sensitivity runs; their base
  run is the `balanced` main run (the reward weights excepted) and their `return_own` is the return under their own preset
  (`episode_return` in `val_log.csv`, recomputed from the components in the test tables). Entries: `docs/config_decisions.md`.
- **Budget (frozen after the calibration):** `total_env_steps = 300,000`, `val_every_env_steps = 12,000` (25 validation points per
  learning curve), `val_n_episodes = 20` (all `VAL_SEEDS`), the same for DDQN and PPO, tuning and sensitivity runs. PPO updates in rollouts of
  `n_envs * n_steps` env steps, so this product must divide both 300,000 and 12,000 (any divisor of 12,000, e.g. 1,000, 2,000, 3,000,
  4,000, 6,000, 12,000); `check_budget_matches_rollout` raises otherwise and names the nearest valid values.
- **Target levels (frozen after the calibration):** `beats_idle` (crash rate < 0.6, mean speed > 24.8 m/s) and `strong` (crash rate <= 0.4,
  mean speed >= 25.5 m/s), each reached at `TARGET_CONSECUTIVE = 3` consecutive validation points; both stay unreachable for every scripted
  baseline. Every run trains for the full budget (no early stopping).
- **Plateau rule (pre-registered):** `steps_to_plateau` with `PLATEAU_WINDOW = 5` and `PLATEAU_TOL = 0.10` (section 7), applied after
  training; not changed after the main grid has started.
- **Two machines, split by algorithm:** all DDQN runs (main grid, tuning, sensitivity) are trained on one machine and all PPO runs on the
  other (deliberate deviation from the earlier split by training seed; algorithm and hardware are confounded, reported as a limitation).
  DDQN vs PPO cost is compared only by hardware-independent metrics (env steps to the target levels and to the plateau, gradient updates,
  parameter count); wall times, steps/s, update time and inference time are reported per machine and never compared between the
  algorithms (section 11).
- **Video milestones** `milestone_fractions = (0.1, 0.25, 0.5)` plus the always-kept final checkpoint, and the video episode
  rule in section 11.

## 11. Training logs, computational cost, variants and videos

Every training run (`common/training.py`, the same for DDQN and PPO) writes, with `env_step` as the x-axis everywhere:

- `train_log.csv`: one row per finished training episode: `env_step` (at episode end), `wall_s`, `vec_env` (index of the
  vector env, 0 for DDQN) and every per-episode metric of section 5, computed by the same `EpisodeRecorder` as the test
  episodes (`episode_return` is the training reward, `return_env` the default reward). The leader / TTC part costs one
  neighbour search per step; it is on in training (measured overhead: `README.md`, Training).
- `update_log.csv`: one row per `K` `log_update` calls (`update_idx` counts gradient updates, i.e. optimizer steps, for both
  algorithms; `updates_in_row` is the number of updates in the row. DDQN makes one call per gradient step; PPO one call per
  rollout reporting all its minibatch steps, `n_epochs_done x rollout_size / minibatch_size`): `lr`, `loss`, `grad_norm` (before
  clipping), `grad_norm_max`, `update_time_s`; DDQN: `td_loss, q_mean, q_max, target_q_mean, epsilon,
  buffer_size`; PPO: `policy_loss, value_loss, entropy, approx_kl, clip_fraction, explained_variance, n_epochs_done, ratio_max,
  kl_beta, kl_after_update` (`n_epochs_done`: early stop on KL; `ratio_max`: largest probability ratio of the update; `kl_beta`: PPO-KL
  penalty coefficient after the update, NaN for PPO-Clip; `kl_after_update`: exact mean KL(pi_old || pi_new) over the rollout after
  all epochs, for both losses). For PPO, `grad_norm` is the mean over the rollout's minibatch steps, so `grad_norm_max` equals
  `grad_norm` in PPO rows (`ppo/README.md`). A column of the other algorithm is not allowed; a missing one is NaN. **Aggregation of the K calls
  of a row** (one rule per column, the same for both algorithms, defined in `UPDATE_AGG`; with K = 1 every rule returns the
  single value): `max` for `q_max` (overestimation spikes must survive), `ratio_max` (PPO: the largest ratio of the window) and `grad_norm_max` (derived by the logger from
  `grad_norm`: gradient spikes are the usual sign of instability); `sum` for `update_time_s`; `last` for the state variables
  `lr`, `epsilon`, `buffer_size`, `kl_beta` (and `env_step`, `update_idx`, `wall_s`); `mean` for every other metric (`loss`,
  `grad_norm`, `td_loss`, `q_mean`, `target_q_mean`, `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction`,
  `explained_variance`, `n_epochs_done`, `kl_after_update`). NaN values are dropped before a rule is applied; a column with no
  value in the window stays NaN.
- `val_log.csv`: `val_idx`, `env_step`, `wall_s`, `train_wall_s` (wall time without validation), cumulative
  `cum_env_time_s`, `cum_update_time_s`, `cum_val_time_s`, `cum_checkpoint_time_s`, then the full `summarize_metrics`
  output and `return_<preset>` for every preset (section 7).
- `run_meta.json`: algorithm, training reward and density, seed, variant and overridden keys; the full hyperparameters
  and `TrainBudget`; the fully resolved env config actually used (`env.unwrapped.config`), env id, reward weights, the
  action index mapping (checked against `common.actions.ACTION_INDEX`, the run fails loudly if highway-env changes it),
  observation shape, `NEAR_MISS_TTC`, `REVERSAL_WINDOW`; git commit and dirty flag, library versions, OS, machine identifier (`hostname`), CPU model, logical
  cores, thread settings, device, `--workers`, `onedrive_path` (whether the project or the output directory lies inside a OneDrive folder); timestamps, wall time and its split into env / update / validation /
  checkpoint seconds, training steps/s without validation, the quiet benchmark steps/s measured before training, final env
  step, episodes, gradient updates (`n_updates`) and per-sample gradient evaluations (`n_gradient_samples`, the sum of the
  minibatch sizes of all gradient updates; None if the trainer did not report it), status (`running` / `finished` / `failed`, also on a crash or Ctrl-C), the number of trainable
  parameters and the inference time.

**Inference time** (`common.benchmark.measure_inference`): milliseconds per `agent.act(obs, deterministic=True)` at batch
size 1, CPU, one thread, on observations visited by the final agent on the first five `VAL_SEEDS` episodes; 100 untimed
warm-up calls, then 1,000 timed calls; median and 90th percentile are stored (also mean). Measured once at the end of
training and only meaningful in a quiet process: other running jobs inflate it (CLAUDE.md rule 1, C8), so for runs trained
in parallel the benchmark is repeated on the saved checkpoints in a quiet process. It is a per-machine figure and is not compared
between DDQN and PPO (they were trained on different machines, see below).

**Computational cost.** The DDQN vs PPO comparison uses only hardware-independent metrics: (a) env steps to the `TARGET_PERFORMANCE`
levels and to the plateau (section 7), (b) the number of gradient updates and of samples processed by them, per env step
(`gradient_updates_per_env_step` = `timing.n_updates / final_env_step`, `samples_per_env_step` = `timing.n_gradient_samples /
final_env_step` in `common.aggregate.cost_per_model`; a gradient update is one optimizer step for both algorithms) and (c) the
parameter count. Everything measured in seconds
(`train_wall_s`, wall time to a target or plateau, steps/s, `update_time_s` and its split from `env_time_s`, inference time) depends on the
machine and is NOT compared between DDQN and PPO, because the two algorithms were trained on different machines; these quantities are
reported per machine, with the machine described (`cpu_model`, `logical_cores` from `run_meta.json`), and used only for within-algorithm
comparisons (for example one algorithm across reward presets or variants).

**Two machines, split by algorithm.** All DDQN runs (main grid, tuning, sensitivity) are trained on one computer and all PPO runs on the
other. This replaces the earlier plan (split by training seed, both algorithms on each machine); it is a deliberate deviation, reported as a
limitation (algorithm and hardware are confounded). Consequently DDQN and PPO are compared only by hardware-independent metrics (env steps
to the targets and to the plateau, gradient updates, parameter count). Everything measured in seconds (`train_wall_s`, wall time to a target
or plateau, steps/s, update time per env step, inference time) is NOT compared between the algorithms: it is reported per machine (with
`cpu_model` and `logical_cores`) and used only within one algorithm. `common.aggregate.cost_across_seeds` groups these columns by the machine
identifier (`reproducibility.hostname`) and never pools them across machines, while env steps to a target or plateau are pooled over all runs
of the group. The inference benchmark is repeated on the saved checkpoints in a quiet process on each machine and is likewise a per-machine
figure. Resource limit, identical on both machines: at most 8 busy simulator processes at the same time (`docs/performance_analysis.md` C5;
counting rule in `CLAUDE.md`, "Two machines, split BY ALGORITHM"). Both machines use the library versions of `requirements-lock.txt`.

**OneDrive.** During the calibration, OneDrive locked files and 6 runs crashed with `PermissionError`; a resumed run is not bit-identical to an
uninterrupted one. At the start of a run the harness prints a prominent warning if the project or the output directory is inside a
OneDrive folder (a path component named like OneDrive, or a path below the OneDrive environment variables on Windows) and records
`onedrive_path` in `run_meta.json`; it does not block the run. Move the project out of OneDrive (or pause sync) and disable sleep before long runs.

**Resume and determinism.** `resume.pt` holds the trainer state, the simulator state of every env (the running episode
included), the RNG states, the timers and the logger position; rows logged beyond the checkpoint `env_step` are dropped.
A resumed run therefore reproduces the uninterrupted run row by row (tested, up to wall-clock columns). `--workers` only
chooses between an in-process and a subprocess vector env (PPO) and the validation pool: the logs are identical (tested).

**Failure cases and videos.** `common.failure_cases` lists the episode seeds of a test CSV by outcome (`crashed`, `near_miss`,
`completed`), sorted by return (worst first for failures); `scripts/record_video.py --from-eval-csv <csv> --outcome crashed
--n 3` renders them (`env.reset(seed=seed)` reproduces the measured episode). The qualitative analysis shows crash seeds of
the final checkpoint and of an early milestone checkpoint (`milestone_<p>pct.pt`, `TrainBudget.milestone_fractions`) of the
same run. `scripts/record_video.py` is the only script that renders.

**Video episodes (fixed rule).** Checkpoints: the milestones 10% / 25% / 50% (`TrainBudget.milestone_fractions`) and the final
checkpoint, which is always kept. Episodes: for every (reward, training seed) the rule gives candidate `TEST_SEEDS` episodes
that are the same for both algorithms and for every checkpoint of both runs, so the algorithms (and early / middle / final
behaviour) are compared on identical traffic. They are computed from the `main` test results of the two FINAL models (same
reward and training seed) by `common.failure_cases.video_episode_seeds`, always the lowest seed number of the category
(deterministic, not picked by looking at a video):

| Category | Definition |
|---|---|
| `both_crashed` | both models crashed on the seed |
| `both_completed` | both models reached the time limit |
| `ddqn_crashed_only` (disagreement) | DDQN crashed, PPO did not: the same starting situation, opposite outcomes |
| `ppo_crashed_only` (disagreement) | PPO crashed, DDQN did not |

A category is empty (`None`) when no seed qualifies; the report then says so and, for `both_*`, shows each final model's first
seed of `list_episodes_by_outcome` instead. **The rule yields candidates for every (reward, training seed), but the report
video uses only training seed 0 of each reward, on the `main` condition, plus the early and middle milestone checkpoints of
the same models.** That keeps the video short; the other candidates stay available.

## References

- Patterson, A., Neumann, S., White, M., White, A. (2024). *Empirical Design in Reinforcement Learning.* Journal of
  Machine Learning Research. (Multiple independent runs, the training seed as the unit of analysis, and calibration
  baselines.)
- Cobbe, K., Klimov, O., Hesse, C., Kim, T., Schulman, J. (2019). *Quantifying Generalization in Reinforcement Learning.*
  Proceedings of the 36th International Conference on Machine Learning (ICML). (Separate training and test conditions.)
