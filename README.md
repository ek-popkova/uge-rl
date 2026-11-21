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
├── common/       # shared code (see below): env config and factory, rewards, evaluation, metrics, training harness
├── ddqn/         # Double DQN implementation (stub: agent.py, train.py with the algorithm still to write)
├── ppo/          # PPO implementation (stub)
├── scripts/      # scripted baselines, video recording (the only script that renders), pilot experiments (scripts/pilot/)
├── tests/        # pytest: config, rewards, evaluation protocol, metrics, training harness (RandomTrainer end to end)
├── docs/         # config_decisions.md, evaluation_protocol.md, performance_analysis.md
├── notebooks/    # analysis notebooks
└── results/      # local outputs (ignored by git)
```

`common/` in groups:

| Group | Modules |
|---|---|
| environment and rewards | `config.py` (team-agreed values), `envs.py`, `rewards.py`, `actions.py`, `seeding.py` |
| evaluation | `evaluate.py` (`evaluate`, `validate`), `metrics.py` (`EpisodeRecorder`, summaries), `aggregate.py` (seed-level statistics, learning curves, steps to target), `baselines.py`, `failure_cases.py`, `paths.py` |
| training harness | `train_config.py` (budget, `DDQNConfig`, `PPOConfig`, `--set` overrides), `training.py` (`TrainingRun`, `Trainer` interface), `train_env.py`, `train_logging.py`, `run_meta.py`, `benchmark.py`, `random_trainer.py` (test double), `parallel.py` |

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
python -m pytest -m "not slow"    # fast tier: 51 tests in 3.6 s (pure functions and fake envs; use during development)
python -m pytest                  # full suite: 110 tests in 255 s (4 min 15 s; 198 s in an earlier quiet run, wall time varies 2-5x here, C8)
```

The 59 `slow` tests create a real highway-env environment or run episodes (evaluation determinism, spawned workers, the
training harness end to end with the `RandomTrainer`, the scripted baselines against `TARGET_PERFORMANCE`). Durations were
measured on this machine with nothing else running; do not run the tests while training or timing benchmarks are running
(they load the CPU and distort the measured steps/s). When to run which tier: `CLAUDE.md`, "Tests: why they exist and when
to run them".

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

The algorithms are still stubs, but the whole experiment-tracking infrastructure is in place: a run is the shared
harness `common.training.TrainingRun` plus an algorithm `Trainer` (`ddqn/train.py`, `ppo/train.py`; they currently
stop with `NotImplementedError` before anything is written). Everything the comparisons need is then logged
automatically and identically for both algorithms, with `env_step` (environment transitions, summed over vector envs)
as the x-axis.

```bash
python -m ddqn.train --reward balanced --seed 0
python -m ppo.train  --reward balanced --seed 0 --workers 4
python -m ppo.train  --reward balanced --seed 0 --set clip_range=0.1      # sensitivity run (variant clip_range=0.1)
python -m ddqn.train --reward balanced --seed 0 --resume                   # continue from <run dir>/resume.pt
```

| Option | Meaning |
|---|---|
| `--reward`, `--density`, `--seed` | training reward preset, density (protocol: medium), run seed (main grid: 0..4) |
| `--set key=value` | override one hyperparameter (repeatable); recorded in `run_meta.json`; the run goes to the variant directory named from the overrides |
| `--variant NAME` | explicit variant directory (default: `main`, or generated from `--set`); `main` cannot carry overrides |
| `--workers N` | PPO: 1 = vector env in this process, > 1 = one subprocess per env; also the validation pool. Results do not depend on it. |
| `--resume` / `--overwrite` | continue from the last checkpoint / delete an existing run and start over (a run directory with results is otherwise refused) |
| `--out-dir` | results root (default `results/`); the run is written to `<out-dir>/runs/<algo>/<reward>/<variant>/seed<s>/` |
| `--total-steps`, `--val-every`, `--val-episodes`, `--checkpoint-every` | budget overrides for debugging and tests only; the main grid uses the `TrainBudget` defaults |
| `--benchmark-steps`, `--max-minutes`, `--force` | quiet speed benchmark before the run (0 = skip), refuse to start above the estimate unless `--force` |

Before the first step the harness measures the simulator speed in a quiet process and prints the runtime estimate
(simulation only; network updates are not included). Hyperparameters and the budget are **placeholders** until the team
decides them (`common/train_config.py`; list in `CLAUDE.md`, "TO DECIDE BEFORE TRAINING"); the run warns about every
placeholder it uses.

Files written per run (schemas in `common/train_logging.py`, described in `docs/evaluation_protocol.md` section 11):

| File | Content |
|---|---|
| `train_log.csv` | one row per training episode: all per-episode metrics of the evaluation (return under the training and the default reward, crash, speed, lane changes, action histogram, lane occupancy, reversals, near-miss steps, overtakes, ...) |
| `update_log.csv` | one row per K gradient updates: lr, loss, grad norm, update time; DDQN: TD loss, Q statistics, epsilon, buffer size; PPO: policy / value loss, entropy, approx KL, clip fraction, explained variance, epochs done |
| `val_log.csv` | one row per validation on `VAL_SEEDS`: all summary metrics, `return_<preset>` for every reward preset, wall time without validation, cumulative env / update / validation / checkpoint time |
| `run_meta.json` | identity, hyperparameters, budget, the resolved env config, git commit, versions, hardware, threads, timing split, steps/s, benchmark, parameter count, inference time, status |
| `final.pt`, `milestone_<p>pct.pt`, `resume.pt` | final weights, weights at fractions of the budget (for early vs late videos), full state for `--resume` |

`python -m common.random_trainer --algo ppo --set n_envs=2 --set n_steps=10 --set minibatch_size=10 --total-steps 80
--val-every 40 --val-episodes 2 --checkpoint-every 40 --benchmark-steps 0 --out-dir results/_smoke` runs the whole
harness with random actions and fake updates (the `RandomTrainer` test double; no learning) and is what the tests
use to check the files, the schemas, exact resume and worker independence.

**Overhead of the training-time metrics** (the same `EpisodeRecorder` as in evaluation, run inside every training
env): 98 microseconds per step with the leader / time-to-collision part, 6 microseconds without it, against 24.7 ms per
simulator step: **0.40%** and 0.02% of the step time (one quiet process, identical seeds and actions, variants
interleaved, 645 steps each, three repetitions: 0.39-0.41%). The leader part therefore stays on.

## Evaluation

The protocol is in `docs/evaluation_protocol.md`. In code:

```python
from common.evaluate import evaluate, validate
df = evaluate(agent, condition="main", reward="balanced", workers=4)   # 100 TEST_SEEDS episodes
curve_point = validate(agent, n=20, reward="balanced")                 # VAL_SEEDS; incl. return_<preset> for every preset
```

`common/paths.py` defines where models, logs and test results go (`runs/<algo>/<reward>/<variant>/seed<s>`, variant `main`
for the main grid); `common/aggregate.py` turns per-episode results into per-model values and means over seeds with 95%
Student-t intervals, and the validation logs into learning curves (`learning_curve_table`, aligned by validation index),
areas under the curve and env steps to the `TARGET_PERFORMANCE` levels (`steps_to_threshold`). `common/failure_cases.py` lists episode
seeds by outcome (crashed / near miss / completed) from a test CSV; `scripts/record_video.py` renders them:

```bash
python -m scripts.record_video --algo ppo --checkpoint results/runs/ppo/balanced/main/seed0/final.pt --reward balanced \
    --from-eval-csv results/eval/ppo/balanced/main/seed0/main.csv --outcome crashed --n 3     # or --seeds 10003 10017
```

Loading a trained agent (`load_agent` in `scripts/record_video.py`) is a TODO until the agents exist; the scripted
baselines (`--algo idle`, ...) can be used to try the pipeline.

## Team

- TBD (Double DQN)
- TBD (PPO)
