"""Evaluation protocol shared by all agents."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from common.agent_base import Agent
from common.config import EVAL_SEEDS
from common.envs import make_env
from common.metrics import EpisodeMetrics, MetricsCollector, summarize


def evaluate(
    agent: Agent,
    env_kwargs: Optional[dict[str, Any]] = None,
    n_episodes: int = 20,
    seeds: Optional[Sequence[int]] = None,
) -> tuple[dict[str, float], list[EpisodeMetrics]]:
    """Evaluate an agent with deterministic actions on fixed seeds.

    Args:
        agent: Agent implementing ``common.agent_base.Agent``.
        env_kwargs: Keyword arguments for ``common.envs.make_env``
            (density, reward, render_mode).
        n_episodes: Number of evaluation episodes.
        seeds: One seed per episode. Defaults to ``common.config.EVAL_SEEDS``.

    Returns:
        A tuple ``(summary, episodes)`` with aggregated statistics and the
        per-episode metrics.
    """
    env_kwargs = dict(env_kwargs or {})
    seeds = list(seeds if seeds is not None else EVAL_SEEDS)
    if len(seeds) < n_episodes:
        raise ValueError(f"Need at least {n_episodes} seeds, got {len(seeds)}")

    env = make_env(**env_kwargs)
    episodes: list[EpisodeMetrics] = []
    try:
        for seed in seeds[:n_episodes]:
            obs, info = env.reset(seed=seed)
            collector = MetricsCollector(seed=seed)
            done = False
            while not done:
                action = agent.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                collector.step(reward, info)
                done = terminated or truncated
            episodes.append(collector.finish())
    finally:
        env.close()

    return summarize(episodes), episodes
