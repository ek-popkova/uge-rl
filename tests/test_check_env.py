"""scripts/check_env.py: the lock-file comparison on a hand-built case (fast tier, no environment is created)."""

from __future__ import annotations

from scripts.check_env import find_mismatches, normalize_name, parse_lock

LOCK = """\
# comment
# python_version: 3.13.3
numpy==2.5.5
Farama-Notifications==0.0.6
torch==2.15.0
colorama==0.4.6 ; sys_platform == "win32"
only-on-an-os-that-is-not-this-one==1.0 ; sys_platform == "plan9"
"""


def test_parse_lock_reads_python_pins_normalizes_names_and_applies_markers():
    python, pins = parse_lock(LOCK, marker_env={"sys_platform": "win32"})
    assert python == (3, 13)
    assert pins == {"numpy": "2.5.5", "farama-notifications": "0.0.6", "torch": "2.15.0", "colorama": "0.4.6"}
    _, linux = parse_lock(LOCK, marker_env={"sys_platform": "linux"})
    assert "colorama" not in linux and "numpy" in linux  # a pin whose marker is false here is not expected
    assert normalize_name("Farama_Notifications") == "farama-notifications"


def test_find_mismatches_reports_every_difference_and_nothing_else():
    locked = {"numpy": "2.5.5", "torch": "2.15.0", "pandas": "3.1.1", "gymnasium": "1.4.0"}
    installed = {"numpy": "2.5.5", "torch": "2.15.0+cpu", "pandas": "3.1.0", "gymnasium": None}
    problems = find_mismatches((3, 13), (3, 13), locked, installed)
    assert len(problems) == 2  # the +cpu label is ignored; numpy matches
    assert any("pandas" in p and "3.1.0" in p and "3.1.1" in p for p in problems)
    assert any("gymnasium" in p and "not installed" in p for p in problems)
    assert find_mismatches((3, 13), (3, 13), {"numpy": "2.5.5"}, {"numpy": "2.5.5"}) == []  # identical: exit code 0 case


def test_find_mismatches_checks_the_python_minor_version_but_not_the_patch_level():
    assert find_mismatches((3, 13), (3, 13), {}, {}) == []
    wrong = find_mismatches((3, 13), (3, 12), {}, {})
    assert len(wrong) == 1 and "3.12" in wrong[0] and "3.13" in wrong[0]
    assert find_mismatches(None, (3, 13), {}, {})  # a lock without a python_version line is itself reported
