"""Record evaluation episodes of a saved agent as videos (the ONLY script that may render).

Which episodes: explicit seeds (``--seeds 10003 10017``), or picked by outcome from a per-episode test CSV
(``--from-eval-csv results/eval/ppo/balanced/main/seed0/main.csv --outcome crashed --n 3``), or the first
``--episodes`` ``TEST_SEEDS``. ``env.reset(seed=seed)`` reproduces a test episode exactly, so the video shows
the episode the test CSV measured. For the qualitative analysis (assignment point 8) record failure cases
of the final checkpoint AND of an early milestone checkpoint (``milestone_<p>pct.pt``) of the same run.

    python -m scripts.record_video --algo ppo --checkpoint results/runs/ppo/balanced/main/seed0/final.pt \
        --reward balanced --condition main --from-eval-csv results/eval/ppo/balanced/main/seed0/main.csv \
        --outcome crashed --n 3 --out-dir results/videos

Agent loading: ``load_agent`` is a registry. ``ppo`` loads ``final.pt`` / milestone checkpoints with ``PPOAgent.load``;
``ddqn`` raises NotImplementedError until the DDQN agent exists (the scripted baselines ``random``, ``idle``,
``slower``, ``faster`` work for testing the pipeline; ``--checkpoint`` is ignored for them).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional, Sequence

from common.agent_base import Agent
from common.baselines import BASELINE_NAMES, make_baseline
from common.config import TEST_CONDITIONS, TEST_SEEDS
from common.failure_cases import OUTCOMES, seeds_by_outcome


def _load_ddqn(checkpoint: str) -> Agent:
    # TODO: from ddqn.agent import DDQNAgent; agent = DDQNAgent(...); agent.load(checkpoint); return agent
    raise NotImplementedError("TODO: load a trained DDQN agent from a checkpoint")


def _load_ppo(checkpoint: str) -> Agent:
    from ppo.agent import PPOAgent

    agent = PPOAgent()  # the network spec is read from the checkpoint by load()
    agent.load(checkpoint)
    return agent


AGENT_LOADERS: dict[str, Callable[[str], Agent]] = {"ddqn": _load_ddqn, "ppo": _load_ppo}


def load_agent(algo: str, checkpoint: str) -> Agent:
    """Registry: build the agent of ``algo`` and load ``checkpoint`` (baselines ignore the checkpoint)."""
    if algo in BASELINE_NAMES:
        return make_baseline(algo)
    if algo not in AGENT_LOADERS:
        raise ValueError(f"Unknown algo {algo!r}; choose from {[*AGENT_LOADERS, *BASELINE_NAMES]}")
    return AGENT_LOADERS[algo](checkpoint)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=[*AGENT_LOADERS, *BASELINE_NAMES], required=True)
    p.add_argument("--checkpoint", type=str, default="", help="final.pt or milestone_<p>pct.pt (not needed for baselines)")
    p.add_argument("--condition", choices=list(TEST_CONDITIONS), default="main", help="test condition to render")
    p.add_argument("--reward", type=str, default="default", help="reward preset the env is built with")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="episode seeds to record")
    p.add_argument("--from-eval-csv", type=Path, default=None, help="per-episode test CSV to pick seeds from")
    p.add_argument("--outcome", choices=list(OUTCOMES), default="crashed", help="outcome to pick with --from-eval-csv")
    p.add_argument("--n", type=int, default=3, help="number of episodes with --from-eval-csv")
    p.add_argument("--episodes", type=int, default=3, help="first N TEST_SEEDS if neither --seeds nor --from-eval-csv")
    p.add_argument("--out-dir", type=str, default="results/videos")
    return p


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    """Episode seeds to record: ``--seeds``, else by outcome from ``--from-eval-csv``, else the first TEST_SEEDS."""
    if args.seeds is not None and args.from_eval_csv is not None:
        raise ValueError("give either --seeds or --from-eval-csv, not both")
    if args.seeds is not None:
        return [int(s) for s in args.seeds]
    if args.from_eval_csv is not None:
        seeds = seeds_by_outcome(args.from_eval_csv, args.outcome, args.n)
        if not seeds:
            raise ValueError(f"no {args.outcome} episodes in {args.from_eval_csv}")
        return seeds
    return [int(s) for s in TEST_SEEDS[: args.episodes]]


def record(agent: Agent, name: str, condition: str, reward: str, seeds: Sequence[int], out_dir: str) -> list[Path]:
    """Render one deterministic episode per seed into ``out_dir/<name>-<condition>-seed<seed>-episode-0.mp4``."""
    from gymnasium.wrappers import RecordVideo

    from common.envs import make_test_env

    env = make_test_env(condition, reward, render_mode="rgb_array")  # the only place that renders
    try:
        for seed in seeds:
            video_env = RecordVideo(env, out_dir, episode_trigger=lambda _: True, name_prefix=f"{name}-{condition}-seed{seed}")
            obs, _ = video_env.reset(seed=seed)
            agent.start_episode(seed)
            done = False
            while not done:  # the video is written when the episode ends
                obs, _, terminated, truncated, _ = video_env.step(agent.act(obs, deterministic=True))
                done = terminated or truncated
    finally:
        env.close()
    return sorted(Path(out_dir).glob(f"{name}-{condition}-seed*.mp4"))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    seeds = resolve_seeds(args)
    agent = load_agent(args.algo, args.checkpoint)
    name = args.algo if not args.checkpoint else f"{args.algo}-{Path(args.checkpoint).stem}"
    print(f"recording seeds {seeds} ({args.condition}, reward {args.reward}) into {args.out_dir}")
    for path in record(agent, name, args.condition, args.reward, seeds, args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
