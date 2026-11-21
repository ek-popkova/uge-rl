# Evaluation protocol

This protocol is fixed before any agent is trained. It describes how DDQN and PPO are trained, tested and compared, so
that the text can be copied into the report. Code: `common/evaluate.py`, `common/metrics.py`, `common/aggregate.py`,
`common/baselines.py`, `common/paths.py`, `common/seeding.py`; values: `common/config.py`.

## 1. Models

- Two algorithms (DDQN, PPO) x four rewards (`default`, `efficiency`, `balanced`, `safety`) x five training seeds
  (`TRAIN_SEEDS = [0, 1, 2, 3, 4]`) = 40 models. Hyperparameter sensitivity runs use three seeds
  (`SENSITIVITY_SEEDS = [0, 1, 2]`) and are reported separately.
- All models train on the same environment: `highway-fast-v0`, medium density (`TRAIN_DENSITY`), `BASE_CONFIG`.
- Same number of environment steps and the same network size for both algorithms; discount factor gamma >= 0.99
  (`MIN_GAMMA`), because with a smaller gamma a late crash is discounted so much that the lighter rewards can rank a
  crash above slow safe driving (`docs/config_decisions.md`).
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
| `SENSITIVITY_SEEDS` | 0, 1, 2 | run seeds of the sensitivity runs |
| `VAL_SEEDS` | 20000..20019 | episode seeds of the periodic evaluation during training (learning curves) |
| `TEST_SEEDS` | 10000..10099 | episode seeds of the final evaluation, used once |

Pilot, train, validation and test sets are pairwise disjoint (asserted in `common/config.py`, tested in
`tests/test_config.py`). Seeding rules for training runs are in `common/seeding.py`: global seeds from the run seed,
`env.reset(seed=s)` once and unseeded resets afterwards, exploration from `np.random.default_rng(s)`, vectorized env `i`
gets seed `s * 1000 + i`.

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

Gap and TTC use the state after every step and the initial state. Mean `min_gap` and `min_ttc` are averages over the
episodes where they are defined.

## 6. Aggregation

- **Level 1, per model** (algorithm, reward, seed, condition): mean of the metrics over its 100 test episodes.
- **Level 2, per (algorithm, reward, condition):** mean over the five training seeds with a 95% percentile bootstrap
  confidence interval over seeds (10,000 resamples). The training seed is the unit of analysis; pooled episodes are
  never treated as independent samples. Individual runs (the level-1 values) are plotted next to the mean.
  With five seeds the bootstrap interval is narrow and cannot exceed the observed range, so it is always read together
  with the individual runs.
- Returns are only comparable within one reward preset (a crash costs 15, 20 or 30 depending on the preset). Reward
  presets are compared by behavioural metrics (crash rate, completion rate, mean speed, lane changes, near-miss rate,
  crashes per km). `return_own` is the return under the preset the model was trained with.

## 7. Learning curves

During training, `common.evaluate.validate(agent, n=20)` runs the agent on 20 `VAL_SEEDS` episodes in the `main`
condition every N environment steps (same N for both algorithms) and appends the mean metrics and the step count to
`val_log.csv`. Curves are shown per seed together with the mean over seeds.

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
results/runs/<algo>/<reward>/seed<s>/final.pt          final checkpoint
results/runs/<algo>/<reward>/seed<s>/train_log.csv     training log
results/runs/<algo>/<reward>/seed<s>/val_log.csv       step + validation metrics
results/eval/<algo>/<reward>/seed<s>/<condition>.csv   per-episode test results
results/eval/baselines/<condition>.csv, summary.csv    scripted baselines
```

Helpers in `common/paths.py`. `results/` is git-ignored.

## 10. Fixed before training, not to be changed after seeing results

Reward weights, densities and `BASE_CONFIG` (`common/config.py`); the seed sets; the test conditions and the number of test
episodes; the metric definitions including the near-miss threshold of 1.5 s; deterministic evaluation of the final
checkpoint; the aggregation (seed as unit, 95% bootstrap CI over seeds); gamma >= 0.99; the same step budget and network
size for both algorithms; the treatment of time-limit truncation. Any deviation is reported as such.

## References

- Patterson, A., Neumann, S., White, M., White, A. (2024). *Empirical Design in Reinforcement Learning.* Journal of
  Machine Learning Research. (Multiple independent runs, the training seed as the unit of analysis, and calibration
  baselines.)
- Cobbe, K., Klimov, O., Hesse, C., Kim, T., Schulman, J. (2019). *Quantifying Generalization in Reinforcement Learning.*
  Proceedings of the 36th International Conference on Machine Learning (ICML). (Separate training and test conditions.)
