"""Process-pool helper for experiment code: thread limits, heaviest-first tasks, ETA progress.

Rules from CLAUDE.md implemented here: one process per task with ``imap_unordered`` (no worker
idles at the end), thread limits set in every worker, throttled progress output with an ETA, and
no default worker count based on ``os.cpu_count()`` (the caller passes ``workers``).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from typing import Any, Callable, Optional, Sequence

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def limit_threads() -> None:
    """Limit BLAS/OpenMP/torch to one thread. Call at the start of every worker.

    The environment variables only take effect for libraries imported afterwards, so scripts
    should also set them before importing numpy/torch; ``torch.set_num_threads(1)`` works at any
    time but is only called if torch is already imported, so call this again after creating a
    torch agent.
    """
    for name in THREAD_ENV_VARS:
        os.environ.setdefault(name, "1")
    torch = sys.modules.get("torch")  # never import torch just to limit it (slow start-up)
    if torch is not None:
        torch.set_num_threads(1)


def _format_eta(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    return f"{seconds // 60}m{seconds % 60:02d}s"


class Progress:
    """Prints ``done/total`` with an ETA to stderr at most every ``min_interval`` seconds."""

    def __init__(self, total: int, desc: str = "", min_interval: float = 5.0, enabled: bool = True) -> None:
        self.total, self.desc, self.min_interval, self.enabled = total, desc, min_interval, enabled
        self.done = 0
        self.t0 = time.perf_counter()
        self.last_print = self.t0

    def update(self, n: int = 1) -> None:
        self.done += n
        now = time.perf_counter()
        if not self.enabled or (now - self.last_print < self.min_interval and self.done < self.total):
            return
        self.last_print = now
        elapsed = now - self.t0
        eta = elapsed / self.done * (self.total - self.done) if self.done else float("nan")
        print(
            f"[{self.desc}] {self.done}/{self.total} tasks, elapsed {_format_eta(elapsed)}, "
            f"ETA {_format_eta(eta)}",
            file=sys.stderr,
            flush=True,
        )


def run_tasks(
    func: Callable[[Any], Any],
    tasks: Sequence[Any],
    workers: int = 1,
    desc: str = "tasks",
    progress: bool = False,
    on_result: Optional[Callable[[int, Any], None]] = None,
) -> list[Any]:
    """Run ``func(task)`` for every task and return the results in the order of ``tasks``.

    Tasks are dispatched in the given order (put the heaviest first) with ``imap_unordered``.
    ``func`` and the tasks must be picklable. With ``workers == 1`` everything runs in this
    process. ``on_result(index, result)`` is called in the parent as soon as a task finishes
    (use it to write each finished task to disk).
    """
    results: list[Any] = [None] * len(tasks)
    bar = Progress(len(tasks), desc, enabled=progress)
    indexed = list(enumerate(tasks))
    if workers <= 1 or len(tasks) <= 1:
        limit_threads()
        for i, task in indexed:
            results[i] = func(task)
            if on_result is not None:
                on_result(i, results[i])
            bar.update()
        return results

    # "spawn" is the only start method on Windows and avoids forking torch state elsewhere.
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, len(tasks)), initializer=limit_threads) as pool:
        for i, res in pool.imap_unordered(_Indexed(func), indexed):
            results[i] = res
            if on_result is not None:
                on_result(i, res)
            bar.update()
    return results


class _Indexed:
    """Picklable wrapper returning ``(index, func(task))`` for ``(index, task)`` pairs."""

    def __init__(self, func: Callable[[Any], Any]) -> None:
        self.func = func

    def __call__(self, item: tuple[int, Any]) -> tuple[int, Any]:
        i, task = item
        return i, self.func(task)
