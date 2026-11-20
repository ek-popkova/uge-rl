"""Per-episode metrics collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeMetrics:
    """Metrics of a single evaluation episode."""

    seed: int = 0
    episode_return: float = 0.0
    length: int = 0
    crashed: bool = False
    # TODO: add more metrics (mean speed, lane changes, ...) once agreed.
    extra: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Accumulates metrics step by step for one episode."""

    def __init__(self, seed: int = 0) -> None:
        self.metrics = EpisodeMetrics(seed=seed)

    def step(self, reward: float, info: dict[str, Any]) -> None:
        """Update the metrics with the result of one environment step."""
        self.metrics.episode_return += float(reward)
        self.metrics.length += 1
        if info.get("crashed", False):
            self.metrics.crashed = True
        # TODO: track additional per-step quantities (speed, ...).

    def finish(self) -> EpisodeMetrics:
        """Return the metrics of the finished episode."""
        return self.metrics


def summarize(episodes: list[EpisodeMetrics]) -> dict[str, float]:
    """Aggregate per-episode metrics into summary statistics."""
    n = max(len(episodes), 1)
    return {
        "mean_return": sum(e.episode_return for e in episodes) / n,
        "mean_length": sum(e.length for e in episodes) / n,
        "crash_rate": sum(e.crashed for e in episodes) / n,
    }
