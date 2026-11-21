"""Evaluation protocol shared by all agents (see ``docs/evaluation_protocol.md``).

Every episode starts with ``env.reset(seed=seed)`` and the agent acts deterministically
(``agent.act(obs, deterministic=True)``), with no learning and no exploration, so all models face
identical starting situations. The environment is created once per process and reused. Results do
not depend on the number of workers or on how the seeds are split into chunks.

Agents are passed either as an ``Agent`` instance (pickled to the workers, so keep it light: no
replay buffer) or as a picklable zero-argument factory that builds the agent inside the worker,
e.g. ``functools.partial(load_agent, path)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

import gymnasium as gym
import pandas as pd

from common.agent_base import Agent
from common.config import REWARDS, TEST_CONDITIONS, TEST_SEEDS, VAL_SEEDS
from common.envs import make_test_env
from common.metrics import EpisodeRecorder, summarize_metrics
from common.parallel import limit_threads, run_tasks

AgentSpec = Union[Agent, Callable[[], Agent]]

# One env per (condition, reward) and process, reused for every episode (CLAUDE.md rule 4).
_ENV_CACHE: dict[tuple[str, str], gym.Env] = {}


def _get_env(condition: str, reward: str) -> gym.Env:
    key = (condition, reward)
    if key not in _ENV_CACHE:
        _ENV_CACHE[key] = make_test_env(condition, reward)
    return _ENV_CACHE[key]


def run_episode(env: gym.Env, agent: Agent, seed: int) -> dict[str, Any]:
    """Run one deterministic episode on ``env`` reset with ``seed``; return its metrics record."""
    obs, _ = env.reset(seed=seed)
    agent.start_episode(seed)
    rec = EpisodeRecorder(env)
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        action = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rec.step(reward, info)
    return rec.finish(seed, truncated, info)


def run_chunk(task: tuple[AgentSpec, str, str, tuple[int, ...]]) -> dict[str, Any]:
    """Worker task ``(agent, condition, reward, seeds)``: returns ``records``, ``steps`` and ``wall_s``."""
    import time

    agent_spec, condition, reward, seeds = task
    t0 = time.perf_counter()
    agent = agent_spec if isinstance(agent_spec, Agent) else agent_spec()
    limit_threads()  # also limits torch if the agent just imported it
    env = _get_env(condition, reward)
    records = [run_episode(env, agent, seed) for seed in seeds]
    return {
        "records": records,
        "steps": sum(r["length"] for r in records),
        "wall_s": time.perf_counter() - t0,
    }


def make_chunks(seeds: Sequence[int], chunk_size: int) -> list[tuple[int, ...]]:
    """Split ``seeds`` into consecutive chunks of at most ``chunk_size``."""
    seeds = [int(s) for s in seeds]
    return [tuple(seeds[i : i + chunk_size]) for i in range(0, len(seeds), chunk_size)]


def records_to_frame(records: list[dict[str, Any]], condition: str, reward: str) -> pd.DataFrame:
    """Build the per-episode DataFrame (sorted by seed) from worker records."""
    df = pd.DataFrame(records).sort_values("seed").reset_index(drop=True)
    df.insert(0, "condition", condition)
    df.insert(1, "reward", reward)
    return df


def evaluate(
    agent: AgentSpec,
    condition: str = "main",
    reward: str = "default",
    seeds: Optional[Sequence[int]] = None,
    workers: int = 1,
    chunk_size: int = 5,
    progress: bool = False,
) -> pd.DataFrame:
    """Evaluate an agent on a test condition, one deterministic episode per seed.

    Args:
        agent: An ``Agent`` or a picklable factory returning one (see module docstring).
        condition: Key of ``common.config.TEST_CONDITIONS``.
        reward: Reward preset the env is built with; fills ``episode_return``. The return under
            any other preset can be recomputed with ``common.metrics.recompute_return``.
        seeds: One episode seed per episode; defaults to ``common.config.TEST_SEEDS``.
        workers: Number of processes (1 = run in this process).
        chunk_size: Seeds per task; smaller chunks balance load, larger ones amortize start-up.
        progress: Print progress with ETA to stderr.

    Returns:
        DataFrame with one row per episode (columns documented in ``common.metrics``), sorted by
        seed, with the ``condition`` and ``reward`` columns first.
    """
    if condition not in TEST_CONDITIONS:
        raise ValueError(f"Unknown test condition {condition!r}; choose from {list(TEST_CONDITIONS)}")
    if reward not in REWARDS:
        raise ValueError(f"Unknown reward {reward!r}; choose from {list(REWARDS)}")
    seeds = list(TEST_SEEDS if seeds is None else seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    tasks = [(agent, condition, reward, chunk) for chunk in make_chunks(seeds, chunk_size)]
    results = run_tasks(run_chunk, tasks, workers=workers, desc=f"eval {condition}", progress=progress)
    records = [r for res in results for r in res["records"]]
    return records_to_frame(records, condition, reward)


def validate(
    agent: AgentSpec,
    n: int = 20,
    reward: str = "default",
    workers: int = 1,
) -> dict[str, float]:
    """Mean metrics on the first ``n`` ``VAL_SEEDS`` under the "main" condition.

    Intended for learning curves during training: call it every N steps and append the result to
    ``val_log.csv``. Never selects or tunes anything on ``TEST_SEEDS``.
    """
    if not 0 < n <= len(VAL_SEEDS):
        raise ValueError(f"n must be in 1..{len(VAL_SEEDS)}, got {n}")
    df = evaluate(agent, "main", reward, VAL_SEEDS[:n], workers=workers, chunk_size=max(n // max(workers, 1), 1))
    return summarize_metrics(df, extra_prefix=None)
