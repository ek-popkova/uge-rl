"""CLI entry point for training PPO (stub)."""

from __future__ import annotations

import argparse

from common.config import DENSITIES, REWARDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent.")
    parser.add_argument("--reward", choices=list(REWARDS), default="default")
    parser.add_argument("--density", choices=list(DENSITIES), default="medium")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--out-dir", type=str, default="results/ppo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: create env via common.envs.make_env, build PPOAgent, run the
    # training loop, save checkpoints and logs to args.out_dir.
    raise NotImplementedError(f"TODO: implement PPO training loop ({args})")


if __name__ == "__main__":
    main()
