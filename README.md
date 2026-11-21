# Safe and Efficient Autonomous Driving with Deep Reinforcement Learning

University reinforcement learning project. We train autonomous driving agents in
[HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) (Gymnasium,
`highway-fast-v0` / `highway-v0`) and compare two algorithms implemented from
scratch in PyTorch: Double DQN (value-based) and PPO (policy-based, actor-critic).
Both are trained and evaluated under identical environment, reward and evaluation
conditions, so all shared code lives in the `common/` package.

## Repository structure

```
.
├── common/       # shared code: env config, env factory, rewards, evaluation, metrics, Agent interface
├── ddqn/         # Double DQN implementation
├── ppo/          # PPO implementation
├── scripts/      # random baseline, video recording, pilot experiments (scripts/pilot/)
├── tests/        # pytest tests for the config and the reward wrapper
├── docs/         # config_decisions.md, performance_analysis.md
├── notebooks/    # analysis notebooks
└── results/      # local outputs (ignored by git)
```

## Setup

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Configuration

The shared settings live in `common/config.py` and were chosen from the environment pilot
(`docs/config_decisions.md` gives the value, the default and the evidence for every parameter).

- **Environment:** `highway-fast-v0`, Kinematics observation (5 vehicles), `DiscreteMetaAction`,
  3 lanes, 30 steps per episode (`policy_frequency=1`), `simulation_frequency=5`, 20 vehicles; everything else
  at the environment defaults.
- **Traffic density** (only `vehicles_density` differs): low 0.4, medium 0.7, high 1.0. Training uses
  `TRAIN_DENSITY = "medium"`; evaluation covers all three.
- **Rewards** (`--reward`): `default` is the environment's own normalized reward. The custom variants are unnormalized,
  `r = w_speed*high_speed + w_right*right_lane - w_collision*crashed - w_lane*lane_changed`:

  | reward | w_speed | w_right | w_collision | w_lane |
  |---|---:|---:|---:|---:|
  | `efficiency` | 0.8 | 0.0 | 15 | 0.0 |
  | `balanced` | 0.4 | 0.1 | 20 | 0.1 |
  | `safety` | 0.2 | 0.1 | 30 | 0.5 |

  Train with gamma >= 0.99. Print the weights and analytic indifference points with `python -m common.config`.
- **Seeds:** training `[0..4]`, sensitivity `[0, 1, 2]`, validation `20000..20019`, test `10000..10099`, all disjoint from
  each other and from the pilot seeds 1000..1049.
- **Test conditions** (evaluation only): `main`, `low`, `high`, `long` (60 s), `precise` (`highway-v0`, 15 Hz); see
  `docs/evaluation_protocol.md`.

## Tests

```bash
pip install -r requirements.txt   # includes pytest
python -m pytest                  # about 15 s: config and real-env checks
```

## Reproducing the pilot

`results/` is git-ignored, so the reward pilot data is regenerated from scratch (about 2 minutes with 8 workers; only
scripted baseline policies are simulated, no agent is trained). Run from the repository root:

```bash
# 1. Simulate random / idle / slower / faster on low / medium / high density, seeds 1000..1049 (600 episodes)
.venv/Scripts/python.exe scripts/pilot/reward_v2_collect.py --workers 8
#    -> results/pilot/reward_v2/steps.csv, episodes.csv
# 2. Knox et al. check of the reward weights in common/config.py (2,000-resample bootstrap, gamma 1 / 0.99 / 0.9 / 0.8)
.venv/Scripts/python.exe scripts/pilot/reward_final_analyze.py
#    -> results/pilot/reward_final_v2/knox_final.csv, plots/
# 3. Build the report
.venv/Scripts/python.exe scripts/pilot/reward_final_report.py
#    -> results/pilot/reward_final_v2/REWARD_FINAL_V2.md
```

All seeds are fixed (episode seeds 1000..1049, bootstrap seed 0), so the data is identical on every run and does not
depend on `--workers` (checked byte for byte with 8 and 4 workers). Use `--resume` with step 1 to continue an
interrupted run. On Linux/macOS use `.venv/bin/python`.

## Scripted baselines

Random, idle, slower and faster policies on every test condition (writes `results/eval/baselines/`):

```bash
python -m scripts.eval_baselines --workers 8 [--resume]
```

## Training

TODO (placeholders until the algorithms are implemented):

```bash
python -m ddqn.train --reward default --density medium --seed 0 --total-steps 100000 --out-dir results/ddqn
python -m ppo.train  --reward default --density medium --seed 0 --total-steps 100000 --out-dir results/ppo
```

## Evaluation

The protocol is in `docs/evaluation_protocol.md`. In code:

```python
from common.evaluate import evaluate, validate
df = evaluate(agent, condition="main", reward="balanced", workers=4)   # 100 TEST_SEEDS episodes
curve_point = validate(agent, n=20)                                    # VAL_SEEDS, for learning curves
```

`common/paths.py` defines where models, logs and test results go; `common/aggregate.py` turns per-episode results into
per-model values and means over seeds with 95% Student-t intervals. Video recording: `scripts/record_video.py`.

## Team

- TBD (Double DQN)
- TBD (PPO)
