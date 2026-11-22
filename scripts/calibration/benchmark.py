"""Quiet-process speed measurements for the SB3 calibration plan (CLAUDE.md rule 1) -> results/calibration/benchmark.json.

Run it BEFORE anything else is running (``run_plan.py`` does this first). It measures, with the exact training setup
(env + episode recorder + SB3 + networks):

    env_steps_per_s          one process, random actions, no network (``common.benchmark.measure_env_speed``)
    dqn_train_steps_per_s    one SB3 DQN training process, 1 gradient step per env step, after the learning_starts phase
    dqn_update_s_per_step    seconds of the SB3 update per env step
    ppo_train_steps_per_s    SB3 PPO with 8 env subprocesses: one full rollout (2048 steps) + update
    ppo_update_s_per_rollout seconds of the PPO update (10 epochs over 2048 samples, batch 64)
    ppo_env_phase_steps_per_s  2048 / (rollout wall - update): the aggregate simulator throughput of 8 processes
    dqn_val_s / ppo_val_s    seconds of one validation (20 episodes x 30 steps): in-process (DQN), 4-worker pool (PPO)
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import json
import sys
import tempfile
import time
from functools import partial
from pathlib import Path


def measure(env_steps: int = 300) -> dict:
    import torch

    torch.set_num_threads(1)
    from common.benchmark import measure_env_speed
    from common.run_meta import cpu_model
    from common.train_logging import Timers, TrainLogger
    from scripts.calibration import sb3_cal
    from scripts.calibration.sb3_train import CalDQN, CalPPO, _Hooks, build_model, light_agent_from_model, make_vec_env

    out: dict = {"cpu_model": cpu_model(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "env_steps_measured": env_steps}
    out["env_steps_per_s"] = measure_env_speed(partial(sb3_cal.make_cal_env, "balanced"), env_steps)
    print(f"env only (1 process): {out['env_steps_per_s']:.1f} steps/s", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        # --- DQN: warm-up fills learning_starts, then 400 timed training steps (>= 200)
        timers = Timers()
        logger = TrainLogger(Path(tmp) / "dqn", "ddqn", timers, update_log_every=100)
        venv = make_vec_env("dqn", "balanced", 99)
        model = build_model("dqn", venv)
        model._cal = _Hooks(timers, logger)
        model.learn(250, reset_num_timesteps=True, log_interval=10**9)
        u0, t0 = timers.seconds["update"], time.perf_counter()
        n = 400
        model.learn(n, reset_num_timesteps=False, log_interval=10**9)
        dt = time.perf_counter() - t0
        out["dqn_train_steps_per_s"] = n / dt
        out["dqn_update_s_per_step"] = (timers.seconds["update"] - u0) / n
        print(f"DQN training (1 process): {out['dqn_train_steps_per_s']:.1f} steps/s, update "
              f"{1000 * out['dqn_update_s_per_step']:.2f} ms/step", flush=True)
        light = light_agent_from_model(model, "dqn")
        validator = sb3_cal.Validator("balanced", workers=1)
        validator.run(light)  # builds the validation env
        t0 = time.perf_counter()
        validator.run(light)
        out["dqn_val_s"] = time.perf_counter() - t0
        print(f"DQN validation (20 episodes, in-process): {out['dqn_val_s']:.1f} s", flush=True)
        venv.close()

        # --- PPO: 8 env subprocesses; warm-up rollout, then one timed rollout + update (2048 steps)
        timers = Timers()
        logger = TrainLogger(Path(tmp) / "ppo", "ppo", timers, update_log_every=1)
        venv = make_vec_env("ppo", "balanced", 99)
        model = build_model("ppo", venv)
        model._cal = _Hooks(timers, logger)
        model.learn(sb3_cal.PPO_ROLLOUT, reset_num_timesteps=True, log_interval=10**9)
        u0, t0 = timers.seconds["update"], time.perf_counter()
        model.learn(sb3_cal.PPO_ROLLOUT, reset_num_timesteps=False, log_interval=10**9)
        dt = time.perf_counter() - t0
        upd = timers.seconds["update"] - u0
        out["ppo_train_steps_per_s"] = sb3_cal.PPO_ROLLOUT / dt
        out["ppo_update_s_per_rollout"] = upd
        out["ppo_env_phase_steps_per_s"] = sb3_cal.PPO_ROLLOUT / max(dt - upd, 1e-9)
        print(f"PPO training (8 env processes): {out['ppo_train_steps_per_s']:.1f} steps/s, update {upd:.2f} s per rollout, "
              f"simulator phase {out['ppo_env_phase_steps_per_s']:.1f} steps/s aggregate", flush=True)
        light = light_agent_from_model(model, "ppo")
        validator = sb3_cal.Validator("balanced", workers=sb3_cal.VAL_POOL_WORKERS)
        try:
            validator.run(light)  # spawns the pool and builds the validation envs
            t0 = time.perf_counter()
            validator.run(light)
            out["ppo_val_s"] = time.perf_counter() - t0
        finally:
            validator.close()
        print(f"PPO validation (20 episodes, {sb3_cal.VAL_POOL_WORKERS}-worker pool): {out['ppo_val_s']:.1f} s", flush=True)
        venv.close()
    out["aggregate_over_single"] = out["ppo_env_phase_steps_per_s"] / out["env_steps_per_s"]
    return out


def main() -> None:
    from scripts.calibration import sb3_cal

    ap = __import__("argparse").ArgumentParser(description=__doc__, formatter_class=__import__("argparse").RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=sb3_cal.CAL_ROOT / "benchmark.json")
    ap.add_argument("--env-steps", type=int, default=300)
    args = ap.parse_args()
    res = measure(args.env_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"written {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
