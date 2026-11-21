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
| ppo | default | (none) |  |  |  |
| ppo |  |  |  |  |  |
| ppo |  |  |  |  |  |
| ppo |  |  |  |  |  |

Criterion value = mean `return_balanced` over the last 3 validation points; `crash_rate` = mean over the same points;
`Chosen` = yes / no (exactly one yes per algorithm). Date, commit and any deviation (for example an interrupted run) go
below the tables.
