# Configuration decisions

Every value in `common/config.py` with the value chosen, the `highway-fast-v0` default, and the evidence it rests on.
Sources: **P** = `results/pilot/PILOT_REPORT.md`, **RF2** = `results/pilot/reward_final_v2/REWARD_FINAL_V2.md`, **PA** = `docs/performance_analysis.md`.
All pilot numbers use scripted baseline policies (random, idle, slower, faster), 50 episodes per cell, seeds 1000..1049;
no learning agent was trained, so the calibration is independent of the two compared algorithms.

Status labels: **not a studied factor, kept at default** = the pilot did not vary it; **studied, equals default** = the pilot
compared alternatives and the default won; **chosen** = value differs from the default or has no default.

## Why the configuration was calibrated this way

- **Calibrating with non-learning baselines, independent of the compared algorithms.** The environment settings
  (traffic density, episode length, observation size) were set by measuring scripted policies, not by tuning on DDQN
  or PPO, so neither algorithm is favoured by the settings and both see identical conditions
  (Patterson et al., 2024, sect. 4.2 and 5.3).

## BASE_CONFIG and environment

| Parameter | Chosen | Default | Status | Justification (pilot number) |
|---|---|---|---|---|
| `ENV_ID` | `highway-fast-v0` | n/a (no default; README names `highway-fast-v0` / `highway-v0`) | chosen | The fast variant simulates 5 Hz; a 15 Hz simulation is 2.5x slower per step (19 vs 45-55 steps/s at 20 vehicles, PA C2), and the pilot only measured the fast variant. |
| `observation` | `{"type": "Kinematics"}` (5 x 5 features, `vehicles_count=5`) | same | not a studied factor, kept at default | The feature set was not varied. |
| observation `vehicles_count` | 5 (ego + 4 nearest ahead) | 5 | studied, equals default | The ego-lane leader is hidden by the 4-slot limit in 0% of steps at low and medium density and 2% at high density (3 lanes); more than 4 vehicles ahead within 100 m in 0 / 0 / 34% of steps (P sect. 4, 2.1). |
| `action` | `{"type": "DiscreteMetaAction"}` (5 actions: left, idle, right, faster, slower) | same | not a studied factor, kept at default | Action set not varied; the baselines use indices IDLE=1, FASTER=3, SLOWER=4. |
| `lanes_count` | 3 | 3 | studied, equals default | Idle crash rate separates the levels more sharply with 3 lanes (8 / 54 / 92%) than with 4 (about 14 / 35 / 85%) (P sect. 3-4). |
| `duration` | 30 (s = steps) | 30 | studied, equals default | Idle crash / reaches-limit at 30 s: 8% / 92%, 54% / 46%, 92% / 8% for low / medium / high; at 40 s medium is already 76% crashed and at 60 s 100% (P sect. 2.5, 4). |
| `policy_frequency` | 1 | 1 | not a studied factor, kept at default | One decision per second (30 decisions per episode); not varied. |
| `simulation_frequency` | 5 | 5 | not a studied factor, kept at default | Only cost was measured: the former placeholder 15 is about 2.5x slower per step (PA C2). |
| `vehicles_count` (env) | 20 | 20 | studied, equals default | Idle crash rates for 20 / 30 / 40 vehicles are 54 / 54 / 54% at density 0.7 and 92 / 94 / 94% at 1.0 (noise, n = 50), but 20 vehicles run at 49 steps/s vs 27 and 20 (P sect. 2.3-2.4). All traffic spawns ahead of the ego, so the far end of the platoon is never reached. |
| `ego_spacing` | 1.5 | 1.5 (`highway-v0`: 2) | not a studied factor, kept at default (pinned) | Pinned so that evaluation on `highway-v0` (`precise` condition) starts from the same layout as training. |
| other keys (`vehicles_density` base 1.0, `normalize_reward`, rewards of the env, `reward_speed_range`, ...) | not overridden | highway-fast-v0 defaults | not a studied factor, kept at default | `tests/test_config.py` checks that the env config equals the default except `vehicles_density`. |

## DENSITIES and training density

| Parameter | Chosen | Default | Status | Justification (pilot number) |
|---|---|---|---|---|
| `DENSITIES["low"]["vehicles_density"]` | 0.4 | 1.0 | chosen | Idle survives 92% of episodes (crash 8%), random crashes 44%; about 2.0 vehicles within 100 m (P sect. 3). The requested grid had no cell this easy, so it comes from the supplementary sweep. |
| `DENSITIES["medium"]["vehicles_density"]` | 0.7 | 1.0 | chosen | Hard for random (70% crash) but not impossible for idle (46% reach the time limit); `slower` never crashes (0%); about 3.9 vehicles within 100 m (P sect. 3). |
| `DENSITIES["high"]["vehicles_density"]` | 1.0 | 1.0 | studied, equals default | Idle crashes in 92%, random in 100%, but `slower` survives 98% (hard yet feasible); about 5.5 vehicles within 100 m; 1.5 and above are probably infeasible (`slower` crashes 64% at 1.5 with 3 lanes) (P sect. 3). |
| `TRAIN_DENSITY` | `"medium"` | n/a | chosen | Medium is the level where the safety/speed trade-off exists (idle crashes 54% of the time and random 70%, yet `slower` never crashes). Evaluation covers all three densities. |

## REWARDS

Custom rewards are computed by `common.rewards.CustomRewardWrapper` from the reward components of the env, **unnormalized**:
`r = w_speed*high_speed + w_right*right_lane - w_collision*crashed - w_lane*lane_changed`, with `high_speed`, `right_lane` in
[0, 1] from `info["rewards"]`, `crashed = info["crashed"]` and `lane_changed` = real change of `lane_index[2]` during the step
(not the chosen action). `REWARDS["default"]` keeps the env's own normalized reward unchanged.

| Variant (assignment) | w_speed | w_right | w_collision | w_lane | analytic p | Meaning |
|---|---:|---:|---:|---:|---:|---|
| `efficiency` (Reward A) | 0.8 | 0.0 | 15 | 0.0 | 0.27 | Speed first, lane position irrelevant, risk-tolerant. |
| `balanced` (Reward B) | 0.4 | 0.1 | 20 | 0.1 | 0.61 | Moderate speed, keep right, mild lane-change penalty. |
| `safety` (Reward C) | 0.2 | 0.1 | 30 | 0.5 | 0.83 | Weak speed incentive, harsh crash penalty, costly lane changes. |

`python -m common.config` prints the weights and the analytic indifference points (`common.config.indifference_point`).

- **`w_right`** encodes the keep-right traffic rule. **`w_lane`** is applied by the wrapper because highway-env 1.12.1 ignores
  the config key `lane_change_reward` (its `_rewards()` returns only the collision, right_lane, high_speed and on_road components).
- **Why unnormalized.** The env's normalization maps [collision, speed + right_lane] to [0, 1]. This adds a per-step survival
  bonus (an alive step that earns nothing already gets 0.667 with the default weights) and shrinks the speed signal as the
  collision weight grows, so the collision weight no longer acts as a penalty. The default reward illustrates this: in the pilot,
  crawling (`slower`) out-earned driving at normal speed (`idle`), mean return 20.9 > 12.9 (P, sect. 5 of the earlier report).
- **Why these collision penalties.** 15 / 20 / 30 are the smallest round values that pass the Knox et al. (2023, sect. 4.3)
  ordering test G_crash < G_slow < G_succ at every density (low density is the binding case, because crashes happen later
  there and so cost less relative to the episode), while keeping a clear caution gradient efficiency < balanced < safety.
- **Replaced derivation.** The earlier target-p derivation (`collision_weight(target_p, ...)` with p = 0.25 / 0.50 / 0.75,
  giving w_collision = 14.25 / 32.25 / 86.25 and w_speed = 0.9, w_right = 0.1, w_lane = 0.0 / 0.1 / 0.3) was replaced by the
  fixed weights above. It is kept only as history in git (branch `backup/pre-reward-removal`, if still present).
- **Discount factor.** Train with gamma >= 0.99 (`MIN_GAMMA`). With gamma <= 0.9 a late crash is discounted so much that
  the lighter variants can rank a crash above slow safe driving.
- **Caveat.** The lane-change penalty of `safety` (0.5) exceeds its per-step maximum reward (0.2 + 0.1 = 0.3), so a safety
  agent is expected to change lanes rarely, even when a lane change would be useful in dense traffic.
- **Empirical check.** `results/pilot/reward_final_v2/REWARD_FINAL_V2.md` (regenerated pilot data, scripted baselines only,
  2,000-resample bootstrap; commands in `README.md`, "Reproducing the pilot"): at gamma = 1 and 0.99 all three variants pass
  the ordering at every density, with empirical p (low / medium / high) of 0.11 / 0.36 / 0.46 (efficiency), 0.55 / 0.73 / 0.74
  (balanced) and 0.82 / 0.90 / 0.89 (safety). At gamma = 0.9 and 0.8 only `safety` passes everywhere.

## Seeds

| Parameter | Chosen | Default | Status | Justification (pilot number) |
|---|---|---|---|---|
| `TRAIN_SEEDS` | `[0, 1, 2, 3, 4]` | n/a | chosen | Five independent training runs per (algorithm, reward); the seed is the unit of analysis (`docs/evaluation_protocol.md`). |
| `SENSITIVITY_SEEDS` | `[0, 1, 2]` | n/a | chosen | Three runs per hyperparameter sensitivity setting. |
| `VAL_SEEDS` | `range(20000, 20020)` (20 episodes) | n/a | chosen | Periodic evaluation during training (learning curves); never used for the final result. |
| `TEST_SEEDS` (alias `EVAL_SEEDS`) | `range(10000, 10100)` (100 episodes) | n/a (earlier placeholder `range(100, 120)`) | chosen | 100 episodes bring the standard error of a crash rate near 50% to about 5 points (7 points with the pilot's n = 50); used once for the final evaluation. |
| `PILOT_SEEDS` | `range(1000, 1050)` | n/a | chosen | Kept only to assert disjointness. |

Pilot, train, validation and test seed sets are pairwise disjoint (asserted in `common/config.py`, tested in `tests/test_config.py`).

## TEST_CONDITIONS

Evaluation-only conditions (never used for training), each an env id plus config overrides on top of `BASE_CONFIG` and the density
preset: `main` (training condition), `low` / `high` (other densities), `long` (60 s, 30 vehicles because 20 was only verified up
to 30 s in the pilot), `precise` (`highway-v0`, `simulation_frequency=15`, collisions between other vehicles checked). Rationale
and metrics: `docs/evaluation_protocol.md`.

## TARGET_PERFORMANCE and TARGET_CONSECUTIVE (placeholders)

Used by `common.aggregate.steps_to_threshold` for "env steps to target performance" in the cost comparison. Behavioural
metrics only, so the levels are comparable across reward presets. Two **pre-registered, named levels**, both reported for
every run, fixed before any training run and not changed after seeing results (`docs/evaluation_protocol.md` sections 7
and 10); all values below are **placeholders, decide before training**.

| Parameter | Value | Meaning |
|---|---|---|
| `TARGET_PERFORMANCE["beats_idle"]` | `crash_rate < 0.6` and `mean_speed > 24.8` | strictly better than `idle` on both metrics on `VAL_SEEDS` (idle: 0.60 and 24.76 m/s): the agent has learned something over the default policy |
| `TARGET_PERFORMANCE["strong"]` | `crash_rate <= 0.2` and `mean_speed >= 25.5` | driving fast and safely; may turn out unreachable, in which case `beats_idle` still allows a comparison |
| `TARGET_CONSECUTIVE` | 2 | a level is reached at the first of this many consecutive validation points at which all its conditions hold (20 validation episodes: crash rate moves in steps of 0.05, one lucky point must not count) |

Each level is joint and must stay unreachable for the scripted baselines: `slower` has a 0% crash rate at 20 m/s, so a
crash bound alone would reward braking; `idle` crashes in 71% of the episodes at 24.7 m/s (`docs/evaluation_protocol.md`
section 8). On `VAL_SEEDS` the baselines have crash rate / mean speed: random 0.70 / 24.2, idle 0.60 / 24.8, slower 0.00 / 20.0,
faster 1.00 / 28.9 m/s; `tests/test_training.py::test_no_scripted_baseline_meets_target_performance` checks both levels. The
budget and the hyperparameters (`common/train_config.py`) are placeholders too.

## Limitations

- **Training only at medium density.** Generalization to low and high density is evaluated, not trained.

## References

- Patterson, A., Neumann, S., White, M., White, A. (2024). *Empirical Design in Reinforcement Learning.* Journal of Machine
  Learning Research (sections 4.2 and 5.3 as cited above).
- Knox, W. B., Allievi, A., Banzhaf, H., Schmitt, F., Stone, P. (2023). *Reward (Mis)design for Autonomous Driving.*
  Artificial Intelligence (sections 4.2-4.3 as cited above).
