"""Run a uniformly random policy through ``common.evaluate``.

Usage (from the repository root):
    python -m scripts.random_baseline --episodes 3 --density medium
"""

from __future__ import annotations

import argparse
from typing import Any

from common.agent_base import Agent
from common.config import DENSITIES, REWARDS
from common.envs import make_env
from common.evaluate import evaluate


class RandomAgent(Agent):
    """Agent that samples uniformly from the action space."""

    def __init__(self, action_space: Any) -> None:
        self.action_space = action_space

    def act(self, obs: Any, deterministic: bool = True) -> Any:
        return self.action_space.sample()

    def save(self, path: str) -> None:  # nothing to save
        pass

    def load(self, path: str) -> None:  # nothing to load
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-policy baseline.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--density", choices=list(DENSITIES), default="medium")
    parser.add_argument("--reward", choices=list(REWARDS), default="default")
    args = parser.parse_args()

    env_kwargs = {"density": args.density, "reward": args.reward}

    probe = make_env(**env_kwargs)
    agent = RandomAgent(probe.action_space)
    agent.action_space.seed(0)
    probe.close()

    summary, _ = evaluate(agent, env_kwargs, n_episodes=args.episodes)
    print(f"Random baseline ({args.episodes} episodes, density={args.density}, reward={args.reward})")
    print(f"  mean return : {summary['mean_return']:.3f}")
    print(f"  mean length : {summary['mean_length']:.1f}")
    print(f"  crash rate  : {summary['crash_rate']:.2%}")


if __name__ == "__main__":
    main()
