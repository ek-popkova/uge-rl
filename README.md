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
- **Seeds:** training `[0, 1, 2]`, evaluation `10000..10099` (disjoint from the pilot seeds 1000..1049).

## Tests

```bash
pip install -r requirements.txt   # includes pytest
python -m pytest                  # about 15 s: config and real-env checks
```

## Random baseline

```bash
python -m scripts.random_baseline --episodes 3 --density medium
```

## Training

TODO (placeholders until the algorithms are implemented):

```bash
python -m ddqn.train --reward default --density medium --seed 0 --total-steps 100000 --out-dir results/ddqn
python -m ppo.train  --reward default --density medium --seed 0 --total-steps 100000 --out-dir results/ppo
```

## Evaluation

TODO: evaluation command via `common.evaluate`, and video recording via
`scripts/record_video.py`.

## Team

- TBD (Double DQN)
- TBD (PPO)
