"""Unified training logger used by both ``ddqn/train.py`` and ``ppo/train.py`` (via ``common.training``).

The x-axis of every log is ``env_step``: total environment transitions, summed over vector envs
(never episodes or updates). Three CSV files per run, schemas defined HERE and nowhere else (a new
column means: change the constants, ``tests/test_training.py`` and ``docs/evaluation_protocol.md``):

``train_log.csv``   one row per finished TRAINING episode: ``TRAIN_LOG_COLUMNS`` = ``env_step`` (at
                    episode end), ``wall_s``, ``vec_env`` (env index, 0 for a single env) and every column
                    of ``common.metrics.EPISODE_COLUMNS`` (same code as the test CSVs).
``update_log.csv``  one row per ``update_log_every`` (K) calls of ``log_update``: ``UPDATE_LOG_COLUMNS[algo]``.
                    ``update_idx`` counts gradient updates cumulatively, ``updates_in_row`` is the number of
                    gradient updates in the row. The K calls of a row are combined by the rule of each column in
                    ``UPDATE_AGG`` (one rule per column, the same for both algorithms; see below).
``val_log.csv``     one row per validation: ``VAL_LOG_COLUMNS`` = ``val_idx``, ``env_step``, ``wall_s``,
                    ``train_wall_s`` (wall time without validation), cumulative env / update / val /
                    checkpoint seconds, the full ``summarize_metrics`` output and ``return_<preset>``
                    for every reward preset.

Aggregation of the K calls of one ``update_log.csv`` row (``UPDATE_AGG``; with K = 1 every rule returns the
single value, so PPO rows are exact):

    env_step, update_idx, wall_s   value at the last call of the window
    updates_in_row                 sum of ``n_updates``
    mean   loss, grad_norm, td_loss, q_mean, target_q_mean, policy_loss, value_loss, entropy, approx_kl,
           clip_fraction, explained_variance, n_epochs_done   (typical level over the window)
    max    q_max            overestimation spikes must survive: the max over the window of the per-call max
           grad_norm_max    DERIVED by the logger from ``grad_norm`` (algorithms report only ``grad_norm``):
                            gradient spikes are the usual sign of instability; the mean ``grad_norm`` next to it
                            shows the typical level
    sum    update_time_s    total time spent updating in the window
    last   lr, epsilon, buffer_size   state variables, the value at the end of the window

Rows are buffered in memory and written by ``maybe_flush`` / ``flush`` (called by the harness between
training chunks, never inside the step loop). On ``--resume``, ``truncate_to(env_step)`` drops every row
beyond the checkpoint's env_step, so the resumed run has no duplicates. Unknown keys raise; missing
algorithm-specific keys are NaN.
"""

from __future__ import annotations

import csv
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from common.config import REWARDS
from common.metrics import EPISODE_COLUMNS, SUMMARY_COLUMNS

# ---------------------------------------------------------------------------------------------
# Schemas (single place)
# ---------------------------------------------------------------------------------------------
TRAIN_LOG_COLUMNS = ["env_step", "wall_s", "vec_env", *EPISODE_COLUMNS]

# Metrics every algorithm reports in log_update, plus the algorithm-specific ones.
UPDATE_COMMON_METRICS = ["lr", "loss", "grad_norm", "update_time_s"]  # grad_norm: before clipping
UPDATE_ALGO_METRICS = {
    "ddqn": ["td_loss", "q_mean", "q_max", "target_q_mean", "epsilon", "buffer_size"],
    "ppo": [
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance",
        "n_epochs_done",
    ],
}
# Columns computed by the logger from a reported metric (not accepted as input): column -> (source, rule).
UPDATE_DERIVED = {"grad_norm_max": ("grad_norm", "max")}
UPDATE_LOG_COLUMNS = {
    algo: [
        "env_step", "update_idx", "updates_in_row", "wall_s", "lr", "loss", "grad_norm", "grad_norm_max",
        "update_time_s", *extra,
    ]
    for algo, extra in UPDATE_ALGO_METRICS.items()
}
# How the metrics of the K calls of one row are combined: "mean" (NaN ignored), "max", "sum" or "last".
# One deliberate rule per metric, independent of the algorithm (see the module docstring); an import-time check
# below makes sure no metric is left without a rule.
UPDATE_AGG = {
    "lr": "last",
    "loss": "mean",
    "grad_norm": "mean",
    "update_time_s": "sum",
    "td_loss": "mean",
    "q_mean": "mean",
    "q_max": "max",
    "target_q_mean": "mean",
    "epsilon": "last",
    "buffer_size": "last",
    "policy_loss": "mean",
    "value_loss": "mean",
    "entropy": "mean",
    "approx_kl": "mean",
    "clip_fraction": "mean",
    "explained_variance": "mean",
    "n_epochs_done": "mean",
}
_AGG_RULES = {
    "mean": lambda v: sum(v) / len(v),
    "max": max,
    "sum": sum,
    "last": lambda v: v[-1],
}
assert set(UPDATE_AGG.values()) <= set(_AGG_RULES), "unknown aggregation rule"
assert set(UPDATE_AGG) == set(UPDATE_COMMON_METRICS) | {m for ms in UPDATE_ALGO_METRICS.values() for m in ms}, (
    "every update metric needs exactly one aggregation rule in UPDATE_AGG"
)
assert all(rule in _AGG_RULES and src in UPDATE_AGG for src, rule in UPDATE_DERIVED.values())

VAL_RETURN_COLUMNS = [f"return_{preset}" for preset in REWARDS]
VAL_METRIC_COLUMNS = [*SUMMARY_COLUMNS, *[c for c in VAL_RETURN_COLUMNS if c not in SUMMARY_COLUMNS]]
VAL_LOG_COLUMNS = [
    "val_idx", "env_step", "wall_s", "train_wall_s",
    "cum_env_time_s", "cum_update_time_s", "cum_val_time_s", "cum_checkpoint_time_s",
    *VAL_METRIC_COLUMNS,
]

LOG_FILES = {"train": "train_log.csv", "update": "update_log.csv", "val": "val_log.csv"}


# ---------------------------------------------------------------------------------------------
# Timers
# ---------------------------------------------------------------------------------------------
class Timers:
    """Accumulates env / update / val / checkpoint seconds and the wall clock of the whole run.

    Usage: ``with timers.env(): ...``. The wall clock continues across ``--resume`` (``state_dict``);
    time between an interruption and the resume is not counted.
    """

    PARTS = ("env", "update", "val", "checkpoint")

    def __init__(self) -> None:
        self.seconds = {p: 0.0 for p in self.PARTS}
        self._wall_offset = 0.0
        self._t0 = time.perf_counter()

    @contextmanager
    def _span(self, part: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[part] += time.perf_counter() - t0

    def env(self):
        return self._span("env")

    def update(self):
        return self._span("update")

    def val(self):
        return self._span("val")

    def checkpoint(self):
        return self._span("checkpoint")

    def wall(self) -> float:
        """Seconds since the start of the run (cumulative over resumes)."""
        return self._wall_offset + time.perf_counter() - self._t0

    def train_wall(self) -> float:
        """Wall time without validation (what the learning curves use as computational cost)."""
        return self.wall() - self.seconds["val"]

    def state_dict(self) -> dict[str, Any]:
        return {"seconds": dict(self.seconds), "wall": self.wall()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.seconds = {p: float(state["seconds"][p]) for p in self.PARTS}
        self._wall_offset = float(state["wall"])
        self._t0 = time.perf_counter()


# ---------------------------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------------------------
def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        return repr(value)  # round-trips exactly, nan -> "nan"
    return value


class _Table:
    """Buffered CSV table with a fixed schema (appends; header written for a new file)."""

    def __init__(self, path: Path, columns: list[str]) -> None:
        self.path, self.columns = path, columns
        self.buffer: list[list[Any]] = []

    def add(self, row: Mapping[str, Any]) -> None:
        unknown = set(row) - set(self.columns)
        if unknown:
            raise KeyError(f"{self.path.name}: unknown column(s) {sorted(unknown)}; schema: {self.columns}")
        self.buffer.append([_fmt(row.get(c, math.nan)) for c in self.columns])

    def flush(self) -> None:
        if not self.buffer and self.path.exists():
            return
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(self.columns)
            writer.writerows(self.buffer)
        self.buffer.clear()

    def truncate_to(self, env_step: int) -> int:
        """Keep the rows with ``env_step <= env_step``; return how many were dropped. Checks the header."""
        if not self.path.exists():
            return 0
        with open(self.path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            return 0
        if rows[0] != self.columns:
            raise ValueError(f"{self.path}: header does not match the current schema; cannot resume")
        col = self.columns.index("env_step")
        kept = [rows[0]] + [r for r in rows[1:] if float(r[col]) <= env_step]
        tmp = self.path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(kept)
        os.replace(tmp, self.path)
        return len(rows) - len(kept)


class TrainLogger:
    """Buffered logger for ``train_log.csv``, ``update_log.csv`` and ``val_log.csv`` of one run.

    Args:
        run_dir: Directory of the run (created).
        algo: ``"ddqn"`` or ``"ppo"`` (selects the update-log schema).
        timers: The run's ``Timers`` (wall clock and cumulative seconds).
        update_log_every: K, ``log_update`` calls per ``update_log.csv`` row (1 = every call).
        flush_rows / flush_seconds: ``maybe_flush`` writes when this many rows are buffered or this
            long since the last write.
    """

    def __init__(
        self,
        run_dir: Path,
        algo: str,
        timers: Timers,
        update_log_every: int = 1,
        flush_rows: int = 200,
        flush_seconds: float = 30.0,
    ) -> None:
        if algo not in UPDATE_LOG_COLUMNS:
            raise ValueError(f"Unknown algorithm {algo!r}; choose from {list(UPDATE_LOG_COLUMNS)}")
        if update_log_every < 1:
            raise ValueError("update_log_every must be >= 1")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.algo, self.timers, self.update_log_every = algo, timers, update_log_every
        self.flush_rows, self.flush_seconds = flush_rows, flush_seconds
        self.tables = {
            "train": _Table(self.run_dir / LOG_FILES["train"], TRAIN_LOG_COLUMNS),
            "update": _Table(self.run_dir / LOG_FILES["update"], UPDATE_LOG_COLUMNS[algo]),
            "val": _Table(self.run_dir / LOG_FILES["val"], VAL_LOG_COLUMNS),
        }
        self._allowed_update_keys = set(UPDATE_COMMON_METRICS) | set(UPDATE_ALGO_METRICS[algo])
        self.n_episodes = 0
        self.update_idx = 0  # gradient updates so far
        self.val_idx = 0
        self._pending: list[tuple[int, int, dict[str, float]]] = []  # (env_step, n_updates, metrics) of the open window
        self._last_update_seconds = 0.0  # timers.seconds["update"] at the last log_update
        self._last_flush = time.perf_counter()

    # -- logging -------------------------------------------------------------------------------
    def log_episode(self, env_step: int, vec_env: int, record: Mapping[str, Any]) -> None:
        """One finished training episode (``record`` from ``common.train_env.finished_episodes``)."""
        self.n_episodes += 1
        self.tables["train"].add({"env_step": env_step, "wall_s": self.timers.wall(), "vec_env": vec_env, **record})

    def log_update(self, env_step: int, n_updates: int = 1, **metrics: float) -> None:
        """Report one gradient update (or ``n_updates`` of them) with its metrics.

        ``update_time_s`` is filled from the ``timers.update()`` time since the previous call unless
        given. Call it after the ``with timers.update():`` block it describes.
        """
        unknown = set(metrics) - self._allowed_update_keys
        if unknown:
            raise KeyError(
                f"unknown update metric(s) {sorted(unknown)} for {self.algo}; allowed: {sorted(self._allowed_update_keys)}"
            )
        spent = self.timers.seconds["update"]
        metrics.setdefault("update_time_s", spent - self._last_update_seconds)
        self._last_update_seconds = spent
        self.update_idx += n_updates
        self._pending.append((env_step, n_updates, {k: float(v) for k, v in metrics.items()}))
        if len(self._pending) >= self.update_log_every:
            self._emit_update_row()

    def _emit_update_row(self) -> None:
        window, self._pending = self._pending, []
        row: dict[str, Any] = {
            "env_step": window[-1][0],
            "update_idx": self.update_idx,
            "updates_in_row": sum(n for _, n, _ in window),
            "wall_s": self.timers.wall(),
        }
        rules = [(key, key, UPDATE_AGG[key]) for key in self._allowed_update_keys]
        rules += [(col, src, rule) for col, (src, rule) in UPDATE_DERIVED.items()]
        for col, src, rule in rules:
            vals = [m[src] for _, _, m in window if src in m and not math.isnan(m[src])]
            if vals:  # a metric nobody reported stays NaN
                row[col] = _AGG_RULES[rule](vals)
        self.tables["update"].add(row)

    def log_val(self, env_step: int, metrics: Mapping[str, float]) -> None:
        """One validation (``metrics`` = ``common.evaluate.validate`` output); timers give the cost columns."""
        unknown = set(metrics) - set(VAL_METRIC_COLUMNS)
        if unknown:
            raise KeyError(f"unknown validation metric(s) {sorted(unknown)}; schema: {VAL_METRIC_COLUMNS}")
        sec = self.timers.seconds
        self.tables["val"].add(
            {
                "val_idx": self.val_idx,
                "env_step": env_step,
                "wall_s": self.timers.wall(),
                "train_wall_s": self.timers.train_wall(),
                "cum_env_time_s": sec["env"],
                "cum_update_time_s": sec["update"],
                "cum_val_time_s": sec["val"],
                "cum_checkpoint_time_s": sec["checkpoint"],
                **metrics,
            }
        )
        self.val_idx += 1

    # -- I/O ---------------------------------------------------------------------------------------
    def buffered_rows(self) -> int:
        return sum(len(t.buffer) for t in self.tables.values())

    def maybe_flush(self) -> None:
        """Write the buffers if ``flush_rows`` rows are waiting or ``flush_seconds`` have passed."""
        if self.buffered_rows() >= self.flush_rows or time.perf_counter() - self._last_flush >= self.flush_seconds:
            self.flush()

    def flush(self) -> None:
        for table in self.tables.values():
            table.flush()
        self._last_flush = time.perf_counter()

    def close(self, emit_partial: bool = True) -> None:
        """Flush everything. ``emit_partial`` writes an open K-update window as a short row (normal end of a run)."""
        if emit_partial and self._pending:
            self._emit_update_row()
        self.flush()

    # -- resume ------------------------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Position of the logger for the resume checkpoint (call ``flush`` first)."""
        return {
            "n_episodes": self.n_episodes,
            "update_idx": self.update_idx,
            "val_idx": self.val_idx,
            "pending": list(self._pending),
            "last_update_seconds": self._last_update_seconds,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.n_episodes = int(state["n_episodes"])
        self.update_idx = int(state["update_idx"])
        self.val_idx = int(state["val_idx"])
        self._pending = [(int(e), int(n), dict(m)) for e, n, m in state["pending"]]
        self._last_update_seconds = float(state["last_update_seconds"])

    def truncate_to(self, env_step: int) -> dict[str, int]:
        """Drop every logged row with ``env_step`` beyond the checkpoint's; returns the dropped counts."""
        return {name: table.truncate_to(env_step) for name, table in self.tables.items()}


def read_logs(run_dir: Path) -> dict[str, Optional["Any"]]:
    """The three logs of a run as DataFrames (``None`` for a missing file); convenience for analysis and tests."""
    import pandas as pd

    out = {}
    for name, fname in LOG_FILES.items():
        path = Path(run_dir) / fname
        out[name] = pd.read_csv(path) if path.exists() else None
    return out
