# Configuration decisions

Every value in `common/config.py` with the value chosen, the `highway-fast-v0` default, and the evidence it rests on.
Sources: **P** = `results/pilot/PILOT_REPORT.md`, **PA** = `docs/performance_analysis.md`.
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
| other keys (`vehicles_density` base 1.0, `ego_spacing`, `normalize_reward`, rewards of the env, `reward_speed_range`, ...) | not overridden | highway-fast-v0 defaults | not a studied factor, kept at default | `tests/test_config.py` checks that the env config equals the default except `vehicles_density`. |

## DENSITIES and training density

| Parameter | Chosen | Default | Status | Justification (pilot number) |
|---|---|---|---|---|
| `DENSITIES["low"]["vehicles_density"]` | 0.4 | 1.0 | chosen | Idle survives 92% of episodes (crash 8%), random crashes 44%; about 2.0 vehicles within 100 m (P sect. 3). The requested grid had no cell this easy, so it comes from the supplementary sweep. |
| `DENSITIES["medium"]["vehicles_density"]` | 0.7 | 1.0 | chosen | Hard for random (70% crash) but not impossible for idle (46% reach the time limit); `slower` never crashes (0%); about 3.9 vehicles within 100 m (P sect. 3). |
| `DENSITIES["high"]["vehicles_density"]` | 1.0 | 1.0 | studied, equals default | Idle crashes in 92%, random in 100%, but `slower` survives 98% (hard yet feasible); about 5.5 vehicles within 100 m; 1.5 and above are probably infeasible (`slower` crashes 64% at 1.5 with 3 lanes) (P sect. 3). |
| `TRAIN_DENSITY` | `"medium"` | n/a | chosen | Medium is the level where the safety/speed trade-off exists (idle crashes 54% of the time and random 70%, yet `slower` never crashes). Evaluation covers all three densities. |

## REWARDS

The reward presets in `common/config.py` are still the team-agreed placeholders and are not covered by the pilot.

## Seeds

| Parameter | Chosen | Default | Status | Justification (pilot number) |
|---|---|---|---|---|
| `TRAIN_SEEDS` | `[0, 1, 2]` | n/a | chosen | Three independent training runs per algorithm and setting; disjoint from the evaluation and pilot seeds. |
| `EVAL_SEEDS` | `range(10000, 10100)` (100 episodes) | n/a (earlier placeholder `range(100, 120)`) | chosen | 100 episodes bring the standard error of a crash rate near 50% to about 5 points (7 points with the pilot's n = 50); disjoint from training seeds 0-2 and pilot seeds 1000-1049 (asserted in `common/config.py`, tested in `tests/test_config.py`). |
| `PILOT_SEEDS` | `range(1000, 1050)` | n/a | chosen | Kept only to assert disjointness. |

## Limitations

- **Training only at medium density.** Generalization to low and high density is evaluated, not trained.

## References

- Patterson, A., Neumann, S., White, M., White, A. (2024). *Empirical Design in Reinforcement Learning.* Journal of Machine
  Learning Research (sections 4.2 and 5.3 as cited above).
