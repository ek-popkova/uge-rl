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
├── ppo/          # PPO implementation: PPO-Clip and PPO-KL (core.py math, train.py trainer, agent.py; see ppo/README.md)
├── scripts/      # scripted baselines, video recording (the only script that renders), pilot experiments (scripts/pilot/)
├── tests/        # pytest: config, rewards, evaluation protocol, metrics, training harness (RandomTrainer end to end)
├── docs/         # config_decisions.md, evaluation_protocol.md, performance_analysis.md
├── notebooks/    # analysis notebooks
├── report/       # LaTeX report (see report/README.md)
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

pip install -r requirements-lock.txt
python scripts/check_env.py       # exits non-zero if Python or any package differs from the lock
```

Both machines (one trains DDQN, the other PPO) must have **identical versions**, so both install from `requirements-lock.txt`
(the full `pip freeze` of the reference environment, exact versions). `requirements.txt` stays the readable top-level list
(same exact versions for the direct dependencies). Python must be **3.13** (reference environment 3.13.3) and torch the **CPU
build** (on Linux: `pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu` first). Run
`scripts/check_env.py` once on every machine before the first training run: it compares the Python minor version and every
package of the lock with what is installed and prints each mismatch. `requirements-calibration.txt` (SB3) is already part of
the lock; it is only used by `scripts/calibration/`.

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
- **Seeds:** training `[0..4]`, sensitivity `[0, 1, 2]` (reuse training seeds 0..2: the base point is the main-grid run),
  tuning run seed `PILOT_RUN_SEED = 99`, validation `20000..20019`, test `10000..10099`, all disjoint from each other and from
  the pilot seeds 1000..1049.
- **Test conditions** (evaluation only): `main`, `low`, `high`, `long` (60 s), `precise` (`highway-v0`, 15 Hz); see
  `docs/evaluation_protocol.md`.

## Tests

```bash
pip install -r requirements-lock.txt   # includes pytest
python -m pytest -m "not slow"    # fast tier: 126 tests in about 4 s (pure functions and fake envs; use during development)
python -m pytest                  # full suite: 198 tests in 243 s (4 min 3 s; wall time varies 2-5x here, C8)
```

The 72 `slow` tests create a real highway-env environment or run episodes (evaluation determinism, spawned workers, the
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

PPO is implemented (`ppo/README.md`); DDQN is still a stub. A run is the shared harness `common.training.TrainingRun`
plus an algorithm `Trainer` (`ddqn/train.py`, `ppo/train.py`; the DDQN stub stops with `NotImplementedError` before
anything is written). Everything the comparisons need is then logged
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
| `--benchmark-steps`, `--max-minutes`, `--force` | quiet speed benchmark before the run (0 = skip), refuse to start above the estimate unless `--force`; a run with the frozen 300,000-step budget is estimated at hours, so real runs need `--force` (or a larger `--max-minutes`) |

Before the first step the harness measures the simulator speed in a quiet process and prints the runtime estimate
(simulation only; network updates are not included). The shared decisions are frozen (gamma 0.99; MLP 256x256 ReLU on the 25
flattened observation features, built by `common/networks.py`; tuning and sensitivity protocol; milestones 10 / 25 / 50%:
`docs/config_decisions.md`, "Shared training decisions"). The budget (`total_env_steps = 300,000`, `val_every_env_steps = 12,000`, `val_n_episodes = 20`), `TARGET_PERFORMANCE["strong"]` and `TARGET_CONSECUTIVE = 3` were frozen after the SB3 calibration (`docs/config_decisions.md`, "Values frozen after the calibration"); only the algorithm-specific hyperparameters are still
**placeholders** until decided (`common/train_config.py`; list in `CLAUDE.md`, "TO DECIDE BEFORE TRAINING"); the run warns
about every placeholder it uses.

Tuning and sensitivity runs (rules in `docs/config_decisions.md`; log in `docs/tuning_log.md`):

```bash
python -m ddqn.train --reward balanced --seed 99 --variant tune-lr-high --set lr=0.001   # tuning run, seed 99, at most 4 per algorithm
python -c "from common.train_config import expand_sensitivity_grid as g; [print(r.algo, r.variant, r.seed) for r in g()]"   # the 24 sensitivity runs
python -c "from common.train_config import expand_supplementary_runs as g; [print(' '.join(r.cli_args())) for r in g()]"   # the registered supplementary runs
```

Every run has exactly one role (`common.train_config.check_run_roles`): `main`, a sensitivity point, `tune-<name>` or a
supplementary run `supp-<name>` declared in `SUPPLEMENTARY_RUNS`; any other variant is refused unless its budget really differs from the frozen `TrainBudget`
(debug runs). Supplementary runs and supplementary reward presets (`common.config.SUPPLEMENTARY_REWARDS`) never enter a DDQN vs PPO
table.

Files written per run (schemas in `common/train_logging.py`, described in `docs/evaluation_protocol.md` section 11):

| File | Content |
|---|---|
| `train_log.csv` | one row per training episode: all per-episode metrics of the evaluation (return under the training and the default reward, crash, speed, lane changes, action histogram, lane occupancy, reversals, near-miss steps, overtakes, ...) |
| `update_log.csv` | one row per K gradient updates: lr, loss, grad norm, update time; DDQN: TD loss, Q statistics, epsilon, buffer size; PPO: policy / value loss, entropy, approx KL, clip fraction, explained variance, epochs done, max ratio, KL penalty coefficient, exact KL after the update. `update_idx` counts optimizer steps for both algorithms (PPO: all minibatch steps of a rollout) |
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

## Calibration with Stable-Baselines3

Before anything is frozen, reference implementations (SB3 DQN, which is vanilla DQN with a target network and **not** Double DQN, and SB3 PPO)
are trained in our environment to check the reward presets, the density levels and test conditions, the seed-to-seed spread and the
budget / validation / target settings. **Calibration only**: these runs are never part of the DDQN vs PPO comparison and never enter the main
tables. Code in `scripts/calibration/`, extra dependency in `requirements-calibration.txt`, run seeds `CALIBRATION_SEEDS = [99, 98]` (98 is used
nowhere else), outputs in `results/calibration/` (deliberately outside `results/runs` and `results/eval`, so the main-grid loaders never see them),
evaluation on `VAL_SEEDS` only (`TEST_SEEDS` stay untouched). Report: `docs/calibration.md`.

```bash
pip install -r requirements-calibration.txt
python -m scripts.calibration.run_plan --dry-run           # quiet benchmark + estimate + schedule, starts nothing
python -m scripts.calibration.run_plan --dqn-slots 5        # whole plan; run again after a crash / sleep to resume
python -m scripts.calibration.train_sb3 --algo ppo --reward balanced --seed 99 [--resume]   # one run
python -m scripts.calibration.evaluate_final --workers 8    # P4: final models + baselines, 5 conditions, VAL_SEEDS
python -m scripts.calibration.analyze                       # docs/calibration.md and results/calibration/plots/
```

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

## Report

The LaTeX report lives in `report/`. Build it with `reportuild.ps1` (Windows) or `report/build.sh`;
see [report/README.md](report/README.md).

## Team

- NAME_1: Double DQN (implemented, tuned and trained on machine 1)
- NAME_2: PPO (implemented, tuned and trained on machine 2)

Work split: the shared code (`common/`, protocol, tests) is common; the runs are split BY ALGORITHM, not by training seed. All DDQN
runs (main grid, tuning, sensitivity) are trained on one machine and all PPO runs on the other; each owner chooses their own
algorithm-specific hyperparameters during tuning. Because algorithm and hardware are therefore confounded, DDQN vs PPO cost is compared
only by hardware-independent metrics (env steps to the target levels and to the plateau, gradient updates, parameter count); wall times,
steps/s and inference time are reported per machine and never compared between the algorithms. Both machines use the same library
versions (`requirements-lock.txt`, `scripts/check_env.py`) and the same resource limit: at most 8 busy simulator processes at the same
time (`CLAUDE.md`, "Two machines, split BY ALGORITHM"; `docs/performance_analysis.md` C5).
