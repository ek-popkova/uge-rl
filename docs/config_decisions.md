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
| `PILOT_RUN_SEED` | 99 | n/a | chosen | Run seed of the hyperparameter tuning runs (see "Shared training decisions"); not a result seed. |

Pilot, train, validation, test seed sets and `PILOT_RUN_SEED` are pairwise disjoint (asserted in `common/config.py`, tested in
`tests/test_config.py`); the sensitivity seeds reuse training seeds 0..2 on purpose (their base point is the main-grid run).

## TEST_CONDITIONS

Evaluation-only conditions (never used for training), each an env id plus config overrides on top of `BASE_CONFIG` and the density
preset: `main` (training condition), `low` / `high` (other densities), `long` (60 s, 30 vehicles because 20 was only verified up
to 30 s in the pilot), `precise` (`highway-v0`, `simulation_frequency=15`, collisions between other vehicles checked). Rationale
and metrics: `docs/evaluation_protocol.md`.

## TARGET_PERFORMANCE and TARGET_CONSECUTIVE

Used by `common.aggregate.steps_to_threshold` for "env steps to target performance" in the cost comparison. Behavioural
metrics only, so the levels are comparable across reward presets. Two **pre-registered, named levels**, both reported for
every run, fixed before any training run and not changed after seeing results (`docs/evaluation_protocol.md` sections 7
and 10). `beats_idle` is decided (derived from the idle baseline, no team decision needed); `strong` and `TARGET_CONSECUTIVE` were
decided by the team after the SB3 calibration (section "Values frozen after the calibration" below).

| Parameter | Value | Meaning |
|---|---|---|
| `TARGET_PERFORMANCE["beats_idle"]` (decided) | `crash_rate < 0.6` and `mean_speed > 24.8` | strictly better than `idle` on both metrics on `VAL_SEEDS`: the agent has learned something over the default policy. Source: `idle` through `validate` on all 20 `VAL_SEEDS` gives crash rate 0.6 exactly and mean speed 24.7564 m/s; 24.8 is that speed rounded up to 0.1 m/s (slightly stricter than "faster than idle"). Re-derive only if `VAL_SEEDS` or the env change |
| `TARGET_PERFORMANCE["strong"]` (decided after the calibration) | `crash_rate <= 0.4` and `mean_speed >= 25.5` | driving fast without crashing in most episodes; unreachable for every scripted baseline; `beats_idle` stays the fallback if a model never reaches it (the earlier placeholder `crash_rate <= 0.2` was reached by 1 of 10 calibration runs at `TARGET_CONSECUTIVE = 1` and by none at 2 or 3) |
| `TARGET_CONSECUTIVE` (decided after the calibration) | 3, for both levels | a level is reached at the first of this many consecutive validation points at which all its conditions hold (20 validation episodes: crash rate moves in steps of 0.05, one lucky point must not count) |

Each level is joint and must stay unreachable for the scripted baselines: `slower` has a 0% crash rate at 20 m/s, so a
crash bound alone would reward braking; `idle` crashes in 71% of the episodes at 24.7 m/s (`docs/evaluation_protocol.md`
section 8). On `VAL_SEEDS` the baselines have crash rate / mean speed: random 0.70 / 24.24, idle 0.60 / 24.76, slower 0.00 / 20.03,
faster 1.00 / 28.90 m/s; `tests/test_training.py::test_no_scripted_baseline_meets_target_performance` checks both levels. The
budget (`total_env_steps`, validation interval and episodes) is frozen as well (below); only the DDQN algorithm-specific hyperparameters (`common/train_config.py`) are still placeholders (the PPO ones were decided by tuning, "PPO hyperparameters decided by tuning" below).

## Shared training decisions

Agreed by the team and identical for DDQN and PPO; code: `common/train_config.py`, `common/networks.py`, `common/config.py`
(tuning and sensitivity constants), `common/aggregate.py`. **These may only change by team agreement. A change after the main
grid has started is reported as a deviation** (`docs/evaluation_protocol.md` section 10). The budget and the target values were frozen after
the SB3 calibration (next section); no shared item is open any more. Everything else that is still a placeholder
(`CLAUDE.md`, "TO DECIDE BEFORE TRAINING") is an algorithm-specific hyperparameter and stays with its owner; after tuning only DDQN's are left.

| Decision | Value | Justification |
|---|---|---|
| discount factor `gamma` | 0.99 for both | 0.99^29 is about 0.75 at the end of a 30-step episode, so a late crash still matters (and `MIN_GAMMA = 0.99` rules out the smaller values that rank a crash above slow safe driving) |
| network input | the 5 x 5 Kinematics observation flattened to 25 features, no extra normalization (`obs_flatten = True`) | the env already normalizes; one flattening (`common.networks.flatten_obs`) for both algorithms |
| network | MLP, `hidden_sizes = (256, 256)`, ReLU (`activation = "relu"`), built by `common.networks.build_mlp` | 25 inputs and a small MLP: the cost is negligible next to the simulator (13-55 steps/s); [256, 256] as in the HighwayEnv examples. Weight initialization is left to each algorithm (algorithm-specific) |
| same network size | DDQN Q-network = PPO actor (25 -> 256 -> 256 -> 5, 73,733 parameters); the PPO critic (25 -> 256 -> 256 -> 1, 72,705) is an extra network of the same size that belongs to the algorithm | the compared policy / value function has the same capacity; `check_comparable` checks gamma, `hidden_sizes`, `activation` (and `obs_flatten`, `device`) |
| separate actor and critic (`ppo_shared_trunk = False`) | PPO only | the critic loss does not interfere with the policy features |
| tuning budget | at most 4 configurations per algorithm (the defaults + up to 3 variations), only algorithm-specific hyperparameters varied (never the shared values above, the budget or the validation settings); one run per configuration with the pilot run seed `PILOT_RUN_SEED = 99`, reward `balanced`, density `medium`, validation on `VAL_SEEDS` only | equal for both algorithms, so the comparison is not "who tuned more"; seed 99 is disjoint from every seed set and its derived vector-env seeds collide with none (tested) |
| selection criterion | mean `return_balanced` over the last 3 validation points; ties (within 1e-9) broken by the lower `crash_rate` over the same points, then the first listed (the defaults); `common.aggregate.select_tuning_config` | fixed before any tuning; a mean over 3 points damps the noise of a single 20-episode validation |
| tuning runs | variant `tune-<name>` (`runs/<algo>/balanced/tune-<name>/seed99/`); `load_eval_episodes` and `per_model_table` exclude them by default; the harness refuses a `tune-*` run with another seed / reward / density, a shared key or a budget override, and refuses seed 99 outside `tune-*` | they never mix with the main grid or the sensitivity runs. Log: `docs/tuning_log.md` |
| supplementary runs | pre-declared registry `SUPPLEMENTARY_RUNS` (`common/train_config.py`), variant `supp-<name>` (`runs/<algo>/<reward>/supp-<name>/seed<s>/`), frozen budget, training density, seeds of `TRAIN_SEEDS`; supplementary reward presets in `common.config.SUPPLEMENTARY_REWARDS`, usable by registered `supp-` runs only; `check_run_roles` refuses every unregistered variant (debug runs, whose total steps / validation settings differ from the frozen budget, excepted); excluded from every DDQN vs PPO table (`include_supplementary=False`) | declared before training like the sensitivity grid, so no extra run can be added after seeing results without showing up as a registry change; a variant name always tells the role. Entries: "Supplementary runs" below |
| sensitivity framework | reward `balanced`, density `medium`, the same `total_env_steps` as the main grid, `SENSITIVITY_SEEDS = [0, 1, 2]`, one parameter varied at a time, the others at their frozen base values | one-at-a-time around frozen values is the cheapest design that shows which hyperparameter matters |
| sensitivity grid (`SENSITIVITY_GRID`), **relative to the base value** | learning rate: base x1/3, x1, x3 (both algorithms); DDQN `eps_decay_fraction`: base -0.2, base, base +0.2; PPO `clip_range`: base -0.1, base, base +0.1. `gamma` is not varied (>= 0.99 is required, little effect on 30-step episodes) | the base is in the grid by construction, so the grid follows whatever base the owner chooses and whatever `total_env_steps` the team decides. A value outside its valid range (`eps_decay_fraction` in (0, 1], `clip_range` in (0, 1)) raises instead of being clipped: the `eps_decay_fraction` base must lie in (0.2, 0.8], the `clip_range` base in (0.1, 0.9). 4 non-base settings x 3 seeds = 12 runs per algorithm, 24 in total (`expand_sensitivity_grid`) |
| DDQN epsilon decay | `eps_decay_fraction` (placeholder base 0.3) of `total_env_steps`, linear from `eps_start` to `eps_end` | a fraction instead of absolute steps, so the schedule and its sensitivity grid follow the budget. Resolved to `eps_decay_steps = round(fraction x total_env_steps)` at run start (`DDQNConfig.eps_decay_steps`) and stored in `run_meta.json` under `resolved` (PPO: `rollout_size`) |
| sensitivity base point | NOT re-run: it is the main-grid run with reward `balanced`, seeds 0, 1, 2, variant `main` | saves 6 runs per algorithm. Only valid if the non-varied hyperparameters, the budget and the env are identical, so `check_sensitivity_run` (run at the start of a sensitivity run when the main run exists, and by `check_sensitivity_runs` before a table is built) fails otherwise |
| video milestones | `milestone_fractions = (0.1, 0.25, 0.5)`; the final checkpoint `final.pt` is always kept | early / middle / final behaviour of the same run |
| video episodes | candidate `TEST_SEEDS` episodes shared by both algorithms, from the final models' `main` test results (`common.failure_cases.video_episode_seeds`): both crashed, both completed and the two disagreement cases (only DDQN / only PPO crashed); the report video uses training seed 0 of each reward only (rule in `docs/evaluation_protocol.md` section 11) | identical starting situations make the two algorithms (and early / middle / final checkpoints) directly comparable; one training seed keeps the video short |
| engineering settings (no effect on results) | `checkpoint_every_env_steps = 10,000`; `update_log_every` K = 100 for DDQN, 1 for PPO | checkpoint: about 10 min of wall time at the measured speed, assuming 12-17 env steps/s per process with 8-12 runs in parallel (aggregate 140 steps/s, PA C5), network updates make it a bit slower; at most this much work is lost after a crash. K only sets the size of `update_log.csv` |

### Sensitivity runs for the current bases

The absolute values depend on the base values and on `total_env_steps` (here the DDQN placeholder bases lr 5e-4,
`eps_decay_fraction` 0.3, the PPO bases decided by tuning lr 3e-4, `clip_range` 0.2, and the frozen `total_env_steps` 300,000; the PPO rows are
final, the DDQN rows change if tuning changes the DDQN bases); `expand_sensitivity_grid()` recomputes them
and `resolve_hyperparameters` resolves `eps_decay_fraction` to env steps (the base 0.3 is 90,000 steps; tested in `tests/test_frozen_values.py`). Every
setting runs with seeds 0, 1, 2 (reward `balanced`, density `medium`), 3 runs per setting; the base setting is the main-grid run.

| Algorithm | Parameter | Grid | Resolved value | Variant directory | Runs |
|---|---|---|---|---|---:|
| DDQN | `lr` | base x1/3 | 0.000166667 | `lr=0.000166667` | 3 |
| DDQN | `lr` | base x3 | 0.0015 | `lr=0.0015` | 3 |
| DDQN | `eps_decay_fraction` | base -0.2 | 0.1 (30,000 steps) | `eps_decay_fraction=0.1` | 3 |
| DDQN | `eps_decay_fraction` | base +0.2 | 0.5 (150,000 steps) | `eps_decay_fraction=0.5` | 3 |
| PPO | `lr` | base x1/3 | 0.0001 | `lr=0.0001` | 3 |
| PPO | `lr` | base x3 | 0.0009 | `lr=0.0009` | 3 |
| PPO | `clip_range` | base -0.1 | 0.1 | `clip_range=0.1` | 3 |
| PPO | `clip_range` | base +0.1 | 0.3 | `clip_range=0.3` | 3 |
| | | | | **total** | **24** |

### Supplementary runs (registry, decided 2026-11-27)

Outside the DDQN vs PPO comparison; reported separately. `common.train_config.SUPPLEMENTARY_RUNS` is the source; adding an entry is
one line there (validated at import: algorithm-specific keys only, a valid config whose rollout fits the budget, values different from
the base, rewards of `REWARDS` / `SUPPLEMENTARY_REWARDS`, seeds of `TRAIN_SEEDS`, not a sensitivity point). `FromBase("rollout_size")`
takes the value from the base config, so the entry follows the PPO defaults chosen after tuning. Base point: the main-grid run with the
same algorithm, reward and seed.

| Variant | Algorithm | Overrides | Rewards | Seeds | Runs | Purpose |
|---|---|---|---|---|---:|---|
| `supp-ppo-kl` | PPO | `policy_loss = "kl"` (`kl_target`, `kl_beta_init` as decided below) | `balanced` | 0, 1, 2 | 3 | PPO-KL vs PPO-Clip (report section 3.3) |
| `supp-a2c` | PPO | `n_epochs = 1`, `minibatch_size = n_envs * n_steps` (`FromBase`; 2,000 with the tuned defaults) | `balanced` | 0, 1, 2 | 3 | A2C-like: one full-batch gradient step per rollout at ratio 1 |
| (after tuning) `supp-ent-<value>` | PPO | `ent_coef = <value>` | all 4 `REWARDS` | 0-4 | 20 | stronger exploration; added once its `ent_coef` is known |
| | | | | | **6 (+20)** | |

Supplementary reward presets (`common.config.SUPPLEMENTARY_REWARDS`, experiment (b)): no entry yet; the weights are fixed analytically
before the runs (`indifference_point`; the ordering G_crash < G_slow < G_succ must hold, checked at import). A run with one needs a
registry entry that lists the preset; its base run is the `balanced` main run (reward weights excepted). DDQN: the registry works for it,
no entry. Cost: 6 runs x 315,000 = 1.9 M simulated steps, about 3.7 h on the PPO machine at 143 steps/s.

## Values frozen after the calibration

Decided by the team after the SB3 calibration (`docs/calibration.md`, in particular sections 4 and 6; plots and per-run
`val_log.csv` / `run_meta.json` in `docs/calibration_evidence/`). The calibration trained SB3 DQN (vanilla, not Double DQN) and SB3 PPO
for 200,000 steps in the project env and was never part of the DDQN vs PPO comparison. Its report only proposed values; the team
accepted them with one change, the budget (300,000 instead of the recommended 200,000). The values are now plain defaults in
`common/train_config.py` and `common/config.py` (no PLACEHOLDER marker) and may only change by team agreement; a change after the main
grid has started is reported as a deviation (`docs/evaluation_protocol.md` section 10).

| Value | Frozen as | Justification |
|---|---|---|
| `TrainBudget.total_env_steps` | 300,000 | 8 of the 10 calibration runs had not plateaued by 200k and the plateau rule gives 300k (the calibration itself recommended 200k because that figure is censored by the budget: it is a lower bound). The team accepted the cost: 64 runs (40 main + 24 sensitivity) x 315,000 simulated steps = 20.2 M, about 39 h on one machine at the measured 143 steps/s, about 20 h per machine on two (one algorithm per machine, 32 runs each) |
| `TrainBudget.val_every_env_steps` | 12,000 | 25 points per learning curve at 300k (the calibration proposed 25 points; its 8,000 was computed for a 200k budget). Validation costs 25 x 20 x 30 = 15,000 steps per run, 5% of the training steps |
| `TrainBudget.val_n_episodes` | 20 | all `VAL_SEEDS`. 50 episodes would need 30 new seed values, a re-derived `beats_idle` (its constants come from the idle baseline on these 20 seeds) and about 4 h more machine time |
| `TARGET_CONSECUTIVE` | 3, for both levels | the chance that a policy 10 points worse than the bound passes it by luck at `k` consecutive points of 20 episodes is about 25% / 6% / 1.5% at `k` = 1 / 2 / 3. At `k` = 1 `beats_idle` was reached by transient lucky points at 10,000 steps |
| `TARGET_PERFORMANCE["strong"]` | `crash_rate <= 0.4` and `mean_speed >= 25.5` | the only candidate level that more than one calibration run reached at `k` = 3 (2 of 10, both DQN); unreachable for every scripted baseline (slower 20.0 m/s, idle crash rate 0.6, random 0.70, faster 1.00 on `VAL_SEEDS`); `beats_idle` stays the fallback. Reward `safety` agents cannot reach a speed bound by design, so steps to target are reported per reward |

The tuning runs (seed 99, `tune-<name>` variants, default budget only) therefore also train for 300,000 steps, so their curves match
the main grid; `check_run_roles` still refuses budget overrides in a tuning run. The sensitivity runs use the default budget and
their base points are the main-grid runs; the 24 runs and the epsilon schedule resolved for 300,000 steps are listed above.

### PPO rollout size must divide the budget

PPO collects `n_envs * n_steps` transitions per update, so it can only stop or validate between updates. `total_env_steps` and
`val_every_env_steps` must be multiples of the rollout size, otherwise PPO overshoots the budget or validates at shifted steps, and the two
algorithms no longer get exactly the same number of env steps. `common.train_config.check_budget_matches_rollout` raises with the nearest
valid budget values and the nearest valid rollout sizes; it is called by `check_comparable(ddqn_cfg, ppo_cfg, budget)`, by `resolve_setup`
and at the start of `TrainingRun.run`. Since 300,000 = 25 x 12,000, the rule is: **`n_envs * n_steps` must be a divisor of 12,000**, e.g. 1,000,
2,000, 3,000, 4,000, 6,000 or 12,000 (the SB3 calibration used 2,048, which would be refused). DDQN advances one transition at a time and
has no such constraint. Debug and test budgets (explicit `--total-steps` ... overrides, never used for results) only need a `total_env_steps` that
fits; they may validate at the next rollout boundary. `minibatch_size` must in turn divide `n_envs * n_steps` (`PPOConfig` raises), so
every minibatch has the same size.

### PPO-KL settings (PPO only, decided, not tuned)

Algorithm-specific values of `PPOConfig` that are decided rather than placeholders (owner: PPO; `common/train_config.py`). They are not
tuned and are not part of the sensitivity grid.

| Setting | Value | Justification |
|---|---|---|
| `policy_loss` | `"clip"` (PPO-Clip, report `eq:ppo-clip`) for the main grid, tuning and sensitivity runs; `"kl"` (adaptive KL penalty) only for the supplementary run `supp-ppo-kl` | PPO-Clip is the variant the assignment and the literature treat as standard PPO; PPO-KL is the alternative trust-region mechanism of the same paper, run as a separate variant (report section 3.3) |
| `kl_target` (d_targ) | 0.01 | the middle value of the three tried in the PPO paper (Schulman et al. 2017, Table 1: 0.003, 0.01, 0.03; adaptive KL scored 0.68, 0.74, 0.71) |
| `kl_beta_init` | 1.0 | the PPO paper's initial value ("beta was initialized at 1", Table 1 caption); the paper notes the initial value "is not important in practice because the algorithm quickly adjusts it" (section 4) |

### PPO hyperparameters decided by tuning (2026-11-28)

PPO tuning is finished (4 of 4 configurations, seed 99, reward `balanced`, commit `9685a02`; table and notes in `docs/tuning_log.md`).
`select_tuning_config` chose `nsteps500`, i.e. the former defaults with `n_steps = 500`. Its whole hyperparameter set (from the run's
`run_meta.json`) is now the plain default of `PPOConfig` (owner: PPO; `common/train_config.py`), with no PPO placeholder left; pinned by
`tests/test_frozen_values.py::test_ppo_defaults_are_frozen_after_tuning`. These values are frozen for the main grid, the sensitivity runs and
the supplementary runs; a change is reported as a deviation.

| Setting | Value | How it was decided |
|---|---|---|
| `n_steps` | 500 (rollout `n_envs * n_steps` = 2,000, divides 300,000 and 12,000) | varied in tuning: 250 (criterion 3.82), 500 (4.86, chosen), 1,000 (4.06) |
| `ent_coef` | 0.01 | varied in tuning: 0.05 (3.93) was not chosen |
| `lr` | 3e-4 | tuning default, not varied; base of the lr sensitivity grid (1e-4 / 9e-4) |
| `clip_range` | 0.2 | tuning default, not varied; base of the clip sensitivity grid (0.1 / 0.3) |
| `gae_lambda`, `n_envs`, `n_epochs`, `minibatch_size`, `vf_coef`, `max_grad_norm`, `target_kl` | 0.95, 4, 10, 100, 0.5, 0.5, None | tuning defaults, not varied |

Every tuning configuration drove at about 21 m/s with 0 crashes and reached neither `TARGET_PERFORMANCE` level, so the frozen defaults are
the best of four slow-safe policies, not a policy that drives fast. Report this as a limitation of the 4-configuration tuning budget.

### Post-hoc plateau metric

"How fast an algorithm learns" is measured after training from the validation curve, never used to stop a run: every run trains for the
full budget. `common.aggregate.steps_to_plateau(val_log, metric="return_own", reward, window=5, tol=0.10)` (constants `PLATEAU_WINDOW`,
`PLATEAU_TOL`, pre-registered next to the function): the final level is the mean of the raw metric over the last 20% of the validation
points, the improvement is the final level minus the first validation value, the curve is smoothed with a rolling mean over `window`
points (the window that starts at the point, so a clean step is found at the step and not `window - 1` points late), and the plateau point is
the earliest validation point from which the smoothed curve stays within `tol` x the improvement of the final level until the end. The
band is relative to the improvement because a fixed 5% of the return was smaller than the validation noise of DQN in the calibration.
`censored` is True if the plateau point lies in the last 20% of the curve (the run may not have converged); no improvement, or no stable
plateau, gives NaN with `censored = True`. For `return_own` the return under the run's own training reward is used. The aggregation
(`cost_per_model`, `cost_across_seeds`) reports it next to `steps_to_threshold`: per model, then mean and 95% t-interval over seeds,
NaN-aware, with the number of seeds that reached it and how many of them are censored. Sanity check on the calibration curves (reporting
only): `docs/calibration.md`, section 4.

### Two machines and OneDrive

The runs are split over two computers BY ALGORITHM: all DDQN runs (main grid, tuning, sensitivity) on one machine, all PPO runs on the
other, and each owner tunes their algorithm-specific hyperparameters on their own machine. This replaces the earlier plan (split by training
seed, each machine running both algorithms and all rewards). It is a deliberate deviation and is reported as a limitation: algorithm and
hardware are confounded, so a wall-clock difference between DDQN and PPO cannot be attributed to the algorithm. `run_meta.json` records the
hostname, the CPU model and the number of logical cores.

- **Compared between DDQN and PPO:** only hardware-independent metrics: env steps to the `TARGET_PERFORMANCE` levels (`steps_to_threshold`),
  env steps to the plateau (`steps_to_plateau`), the number of gradient updates and the parameter count.
- **Not compared between DDQN and PPO:** wall times, steps/s, `update_time_s`, inference ms. They are reported per machine, with the machine
  described (`cpu_model`, `logical_cores`), and used only for within-algorithm comparisons (`cost_across_seeds` groups them by machine and
  never pools them).
- **Resource limit, identical on both machines:** at most 8 busy simulator processes at the same time per machine. Reason: aggregate
  throughput peaks at 8 processes on the reference Intel Core Ultra 9 285H (16 logical cores) and drops at 12-16, and process affinity makes
  no measurable difference (`docs/performance_analysis.md` C5, item 9). Every process that steps an env counts: one per DDQN run (validation
  with `--workers 1` runs in-process); for PPO with `--workers > 1` one subprocess per vector env plus the validation pool, which alternate
  (validation runs between rollouts) and therefore peak at `max(n_envs, workers)` processes; a PPO run with `--workers 1` is one process.
  Nothing else heavy runs during training (C8). If the second machine has a different CPU, the scaling is measured once there (rule 5) and
  recorded in `docs/performance_analysis.md`; the limit stays 8 unless that measurement shows the plateau elsewhere, and any change is
  reported as a deviation.
- **Same library versions** on both machines (`requirements-lock.txt`, checked by `scripts/check_env.py`).

During the calibration OneDrive locked files and 6 runs
crashed with `PermissionError`, and resumed runs are not bit-identical; the harness warns and records `onedrive_path` if the project or output
directory is inside OneDrive. Move the project out of OneDrive (or pause sync) and disable sleep before long runs.

## Limitations

- **Training only at medium density.** Generalization to low and high density is evaluated, not trained.
- **Algorithm and hardware are confounded.** DDQN and PPO are trained on different machines (split by algorithm, not by training seed, a
  deliberate deviation from the earlier plan), so no wall-clock quantity can be compared between them. The cost comparison rests on env
  steps to the targets and to the plateau, gradient updates and parameter count; wall times, steps/s, update time and inference time are
  reported per machine and used only within one algorithm.

## References

- Patterson, A., Neumann, S., White, M., White, A. (2024). *Empirical Design in Reinforcement Learning.* Journal of Machine
  Learning Research (sections 4.2 and 5.3 as cited above).
- Knox, W. B., Allievi, A., Banzhaf, H., Schmitt, F., Stone, P. (2023). *Reward (Mis)design for Autonomous Driving.*
  Artificial Intelligence (sections 4.2-4.3 as cited above).
