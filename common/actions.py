"""DiscreteMetaAction indices of highway-env 1.12.1, in one place.

``common.metrics`` (action histogram), ``common.baselines`` (scripted policies) and
``common.run_meta`` (loud check against the live env) all use this mapping.
``tests/test_evaluate.py`` and ``common.run_meta.check_action_mapping`` verify it against the env.
"""

from __future__ import annotations

ACTION_INDEX = {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2, "FASTER": 3, "SLOWER": 4}
N_ACTIONS = len(ACTION_INDEX)
# Names ordered by action index: ACTION_NAMES[i] is the name of action i.
ACTION_NAMES = tuple(sorted(ACTION_INDEX, key=ACTION_INDEX.get))
assert [ACTION_INDEX[n] for n in ACTION_NAMES] == list(range(N_ACTIONS))
