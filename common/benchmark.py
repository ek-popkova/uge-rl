"""Computational-cost measurements: simulator speed and inference time.

ONLY MEANINGFUL IN A QUIET PROCESS: run them when no other job is running on the machine and compare
numbers only if they were taken in the same session (speed varies 2-5x run to run here, CLAUDE.md rule 1
and C8). Both are called by ``common.training.TrainingRun`` and stored in ``run_meta.json``.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from common.agent_base import Agent
from common.config import VAL_SEEDS


def measure_env_speed(make_env_fn: Callable[[], Any], n_steps: int = 200, seed: int = 1000) -> float:
    """Env steps per second of one env in this process, over at least ``n_steps`` steps.

    Uses its own throw-away env (so the training env's RNG and state are not touched) and random
    actions; the env is reset with ``seed``, ``seed + 1``, ... so the measurement is repeatable.
    """
    env = make_env_fn()
    rng = np.random.default_rng(seed)
    steps, ep = 0, 0
    t0 = time.perf_counter()
    while steps < n_steps:
        env.reset(seed=seed + ep)
        ep += 1
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(int(rng.integers(env.action_space.n)))
            done = terminated or truncated
            steps += 1
    elapsed = time.perf_counter() - t0
    env.close()
    return steps / elapsed


def record_val_observations(agent: Agent, n_episodes: int = 5, condition: str = "main", reward: str = "default") -> np.ndarray:
    """Observations visited by ``agent`` (deterministic) on the first ``n_episodes`` ``VAL_SEEDS``.

    Shape ``(N, *obs_shape)``. Uses the shared per-process env cache of ``common.evaluate``.
    """
    from common.evaluate import _get_env

    env = _get_env(condition, reward)
    observations = []
    for seed in VAL_SEEDS[:n_episodes]:
        obs, _ = env.reset(seed=seed)
        agent.start_episode(seed)
        done = False
        while not done:
            observations.append(np.array(obs, copy=True))
            obs, _, terminated, truncated, _ = env.step(agent.act(obs, deterministic=True))
            done = terminated or truncated
    return np.stack(observations)


def measure_inference(agent: Agent, obs_batch: Sequence[Any], n: int = 1000, warmup: int = 100) -> dict[str, float]:
    """Milliseconds per ``agent.act(obs, deterministic=True)`` at batch size 1, on CPU with one thread.

    The observations of ``obs_batch`` (recorded from ``VAL_SEEDS`` episodes, see
    ``record_val_observations``) are cycled; ``warmup`` untimed calls come first. Returns
    ``median_ms``, ``p90_ms``, ``mean_ms`` and the settings used. Only meaningful in a quiet process:
    other running jobs inflate the numbers.
    """
    from common.parallel import limit_threads

    if len(obs_batch) == 0:
        raise ValueError("obs_batch is empty")
    limit_threads()  # one thread, also for torch if it is already imported
    for i in range(warmup):
        agent.act(obs_batch[i % len(obs_batch)], deterministic=True)
    times = np.empty(n)
    for i in range(n):
        obs = obs_batch[i % len(obs_batch)]
        t0 = time.perf_counter_ns()
        agent.act(obs, deterministic=True)
        times[i] = (time.perf_counter_ns() - t0) / 1e6
    return {
        "median_ms": float(np.median(times)),
        "p90_ms": float(np.percentile(times, 90)),
        "mean_ms": float(times.mean()),
        "n": n,
        "warmup": warmup,
        "batch_size": 1,
        "threads": 1,
        "n_distinct_observations": len(obs_batch),
    }
