"""Check that this machine has the same Python minor version and package versions as requirements-lock.txt.

Run it once on every machine before the first training run (and after any pip change):

    python scripts/check_env.py [--lock requirements-lock.txt]

Exit code 0: everything matches. Exit code 1: at least one mismatch (every one is printed).
The comparison itself is the pure function ``find_mismatches`` (tested on a hand-built case in tests/test_check_env.py).
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Mapping, Optional

DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "requirements-lock.txt"
_PYTHON_LINE = re.compile(r"^#\s*python_version:\s*(\d+)\.(\d+)")
_PIN_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;]+)\s*(?:;\s*(.+))?$")


def normalize_name(name: str) -> str:
    """PEP 503 name normalization, so ``Farama-Notifications`` and ``farama_notifications`` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def strip_local_label(version: str) -> str:
    """``2.15.0+cpu`` -> ``2.15.0``: the CPU / CUDA build label is not part of the compared version."""
    return version.split("+", 1)[0]


def parse_lock(text: str, marker_env: Optional[Mapping[str, str]] = None) -> tuple[Optional[tuple[int, int]], dict[str, str]]:
    """Parse the lock file: ``(python (major, minor) or None, {normalized name: version})``.

    A pin may carry an environment marker (``colorama==0.4.6 ; sys_platform == "win32"``); a pin whose marker is false
    here is skipped (it is not installed on this platform).
    """
    python: Optional[tuple[int, int]] = None
    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _PYTHON_LINE.match(line)
        if m:
            python = (int(m.group(1)), int(m.group(2)))
            continue
        if line.startswith("#"):
            continue
        m = _PIN_LINE.match(line)
        if not m:
            raise ValueError(f"cannot parse lock line: {raw!r}")
        name, version, marker = m.groups()
        if marker:
            from packaging.markers import Marker  # packaging is part of the lock

            if not Marker(marker).evaluate(dict(marker_env) if marker_env else None):
                continue
        pins[normalize_name(name)] = version
    return python, pins


def find_mismatches(
    locked_python: Optional[tuple[int, int]],
    python: tuple[int, int],
    locked: Mapping[str, str],
    installed: Mapping[str, Optional[str]],
) -> list[str]:
    """Human-readable mismatches between the lock and this environment (empty list = identical).

    ``installed`` maps a normalized package name to its installed version, or None if it is not installed.
    """
    problems: list[str] = []
    if locked_python is None:
        problems.append("lock file has no '# python_version: X.Y.Z' line")
    elif tuple(locked_python) != tuple(python[:2]):
        problems.append(f"Python {python[0]}.{python[1]} installed, lock expects {locked_python[0]}.{locked_python[1]}")
    for name in sorted(locked):
        have = installed.get(name)
        if have is None:
            problems.append(f"{name}: not installed, lock expects {locked[name]}")
        elif strip_local_label(have) != strip_local_label(locked[name]):
            problems.append(f"{name}: installed {have}, lock expects {locked[name]}")
    return problems


def installed_versions(names) -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK, help="lock file (default: requirements-lock.txt in the repo root)")
    args = ap.parse_args(argv)
    locked_python, locked = parse_lock(args.lock.read_text(encoding="utf-8"))
    problems = find_mismatches(locked_python, sys.version_info[:2], locked, installed_versions(locked))
    if problems:
        print(f"ENVIRONMENT MISMATCH with {args.lock.name} ({len(problems)}):")
        for p in problems:
            print("  -", p)
        print("Fix: pip install -r requirements-lock.txt (Python 3.13, CPU torch) before any training run.")
        return 1
    print(f"OK: Python {sys.version_info[0]}.{sys.version_info[1]} and {len(locked)} packages match {args.lock.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
