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
├── scripts/      # random baseline, video recording
├── notebooks/    # analysis notebooks
├── report/       # LaTeX report (see report/README.md)
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

## Report

The LaTeX report lives in `report/`. Build it with `reportuild.ps1` (Windows) or `report/build.sh`;
see [report/README.md](report/README.md).

## Team

- TBD (Double DQN)
- TBD (PPO)
