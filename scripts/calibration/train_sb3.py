"""Train one SB3 calibration run (calibration only, never part of the DDQN vs PPO comparison).

    python -m scripts.calibration.train_sb3 --algo ppo --reward balanced --seed 99 [--resume]

``--resume`` continues from the last checkpoint (written about every 10 min of wall time) and does nothing for a
finished run. Heavy imports are lazy so that the PPO env subprocesses (spawn re-imports this module) stay light.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):  # CLAUDE.md rule 7, before numpy / torch
    os.environ.setdefault(_name, "1")

import argparse


def main(argv=None) -> None:
    from common.config import CALIBRATION_SEEDS, REWARDS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    ap.add_argument("--reward", choices=list(REWARDS), required=True)
    ap.add_argument("--seed", type=int, choices=CALIBRATION_SEEDS, required=True)
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint; skip a finished run")
    ap.add_argument("--overwrite", action="store_true", help="delete the files of an existing run and start over")
    ap.add_argument("--root", default=None, help="results root (default results/calibration; tests use a temp dir)")
    ap.add_argument("--total-steps", type=int, default=None, help="debug / tests only (the calibration budget is 200,000)")
    ap.add_argument("--val-every", type=int, default=None, help="debug / tests only")
    ap.add_argument("--val-episodes", type=int, default=None, help="debug / tests only")
    ap.add_argument("--checkpoint-seconds", type=float, default=None, help="debug / tests only")
    args = ap.parse_args(argv)

    from pathlib import Path

    from scripts.calibration import sb3_cal
    from scripts.calibration.sb3_train import CalibrationRun

    kw = {}
    if args.total_steps is not None:
        kw["total_env_steps"] = args.total_steps
    if args.val_every is not None:
        kw["val_every"] = args.val_every
    if args.val_episodes is not None:
        kw["val_n"] = args.val_episodes
    if args.checkpoint_seconds is not None:
        kw["checkpoint_seconds"] = args.checkpoint_seconds
    root = Path(args.root) if args.root else sb3_cal.CAL_ROOT
    CalibrationRun(args.algo, args.reward, args.seed, resume=args.resume, overwrite=args.overwrite, root=root, **kw).run()


if __name__ == "__main__":
    main()
