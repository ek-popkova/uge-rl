# Tuning log

The team fills this in during hyperparameter tuning. The rules are fixed before any tuning
(`docs/config_decisions.md`, "Shared training decisions"):

- at most **4 configurations per algorithm**: the defaults + up to 3 variations; only algorithm-specific hyperparameters
  may be overridden (never `gamma`, `hidden_sizes`, `activation`, `obs_flatten`, `ppo_shared_trunk`, the budget or the
  validation settings);
- one run per configuration: run seed `PILOT_RUN_SEED = 99`, reward `balanced`, density `medium`, validation on `VAL_SEEDS`
  only, variant `tune-<name>` (`results/runs/<algo>/balanced/tune-<name>/seed99/`); these runs never enter a result table;
- **selection criterion** (fixed): mean `return_balanced` over the last 3 validation points; ties broken by the lower
  `crash_rate` over the same points (then the configuration listed first, the defaults).

Start a configuration, then select (names without the `tune-` prefix):

```bash
python -m ddqn.train --reward balanced --seed 99 --variant tune-default
python -m ddqn.train --reward balanced --seed 99 --variant tune-lr-high --set lr=0.001
```

```python
from common.aggregate import select_tuning_config, tuning_table
from common.paths import load_tuning_val_logs

logs = load_tuning_val_logs("ddqn")  # {name: val_log} of every tune-* run, seed 99
print(tuning_table(logs))            # config, criterion, crash_rate, last_env_step, chosen
print(select_tuning_config(logs))    # name of the chosen configuration
```

Tuning runs use the frozen default budget (300,000 steps, validation every 12,000 steps; `check_run_roles` refuses budget overrides), so
their curves match the main grid. For PPO, `n_envs * n_steps` must divide both 300,000 and 12,000 (for example 1,000, 2,000, 3,000, 4,000,
6,000, 12,000), otherwise the run is refused (`check_budget_matches_rollout`). Start them with `--force` (a 300,000-step run exceeds the
default 45-minute estimate limit).

Copy the chosen configuration's values into the algorithm's defaults in `common/train_config.py` (and remove the
`PLACEHOLDER` marker of every field that is decided this way); the main grid and the sensitivity base point use them.

## DDQN

| Algorithm | Config name | Overridden keys | Criterion value | crash_rate | Chosen |
|---|---|---|---:|---:|---|
| ddqn | default | (none) |  |  |  |
| ddqn |  |  |  |  |  |
| ddqn |  |  |  |  |  |
| ddqn |  |  |  |  |  |

## PPO

| Algorithm | Config name | Overridden keys | Criterion value | crash_rate | Chosen |
|---|---|---|---:|---:|---|
| ppo | default | (none) | 3.8179 | 0.00 | no |
| ppo | ent005 | `ent_coef=0.05` | 3.9326 | 0.00 | no |
| ppo | nsteps1000 | `n_steps=1000` | 4.0565 | 0.00 | no |
| ppo | nsteps500 | `n_steps=500` | 4.8560 | 0.00 | yes |

PPO notes. Rows in the order of `tuning_table(load_tuning_val_logs("ppo"))`; `select_tuning_config` → `nsteps500`. All four runs: commit
`9685a02` (`dirty: false`), seed 99, reward `balanced`, density `medium`, frozen budget (300,000 / 12,000 / 20), `--workers 1`, no resume,
4 of 4 configurations used. Selected configuration = the defaults with `n_steps = 500` (rollout 4 x 500 = 2,000); its full
hyperparameter set is now the default of `PPOConfig` (`common/train_config.py`), no PPO placeholder left.

- Dates. `default`, `ent005` and `nsteps500` ran in parallel overnight on 2026-11-28 (00:19-04:27 local time, UTC+1). `nsteps1000` ran alone on
  2026-11-28 (10:29-14:59); its wall time (16,180 s) is distorted by about 32 min of sleep and by a stretch on battery power, at about 1/3 of the
  speed. Wall time does not affect the result. The three parallel start-up benchmarks (`quiet_benchmark_steps_per_s` = 14.9) were not quiet either.
- Behaviour (mean of the last 3 validation points): all four drive at about 21 m/s with 0 crashes, completion 1.00, 1.25 lane changes and
  ~0.98 of the time in the rightmost lane; neither `beats_idle` nor `strong` is reached at any validation point (`mean_speed` never above
  21.95 m/s). This is the slow-safe mode seen in the SB3 calibration (`docs/calibration.md` section 6), but without the entropy collapse.
  - `default`: 20.81 m/s, SLOWER is 0.65 of the actions; entropy falls to about 0.9-1.0.
  - `ent005`: 20.93 m/s; entropy stays at about 1.2 and SLOWER drops to 0.28 (IDLE / LANE_RIGHT instead), but no speed gain; the noisiest curve.
  - `nsteps500`: 21.67 m/s, IDLE 0.58 instead of SLOWER; above `default` at 13 of the 14 points from 144k on; explained variance never <= 0.
  - `nsteps1000`: 21.00 m/s, LANE_RIGHT 0.60 / IDLE 0.29; the longer rollout did not continue the gain of `nsteps500`.
- All four did the same number of Adam steps (`timing.n_updates` = 30,000, `minibatch_size` 100 fixed): a longer rollout means fewer,
  larger policy updates, not fewer gradient steps.
- Deterministic vs stochastic evaluation (20 `VAL_SEEDS` x 5 sampling streams, final models of the first three): sampling drives
  0.3-0.9 m/s faster and earns +0.2 to +0.8 `return_balanced`, still at about 21-22 m/s; the protocol stays deterministic.
- Analyses (git-ignored): `results/_analysis/tuning_round1/` (selection, curves, update diagnostics) and `results/_analysis/stochastic_eval/`.

Criterion value = mean `return_balanced` over the last 3 validation points; `crash_rate` = mean over the same points;
`Chosen` = yes / no (exactly one yes per algorithm). Date, commit and any deviation (for example an interrupted run) go
below the tables.
