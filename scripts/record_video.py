"""Load a saved agent and record evaluation episodes as videos (stub)."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record videos of a saved agent.")
    parser.add_argument("--algo", choices=["ddqn", "ppo"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--density", type=str, default="medium")
    parser.add_argument("--reward", type=str, default="default")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default="results/videos")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: build the env with render_mode="rgb_array" via common.envs.make_env,
    # wrap it in gymnasium.wrappers.RecordVideo(args.out_dir), load the agent
    # from args.checkpoint and run args.episodes episodes.
    raise NotImplementedError(f"TODO: implement video recording ({args})")


if __name__ == "__main__":
    main()
