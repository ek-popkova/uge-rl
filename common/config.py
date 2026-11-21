"""Shared environment, traffic-density and reward configuration.

Environment and density values were chosen from the environment pilot (see
``docs/config_decisions.md`` and ``results/pilot/PILOT_REPORT.md``) so that both algorithms use
identical settings. Do not change them without updating the docs. The reward weights are fixed
values agreed by the team (see ``docs/config_decisions.md``).

Print the reward weights and their analytic indifference points with:  python -m common.config
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

ENV_ID = "highway-fast-v0"

# Base environment configuration passed to gym.make(..., config=...).
# Only parameters that are part of the study are listed; everything else stays at the
# highway-fast-v0 defaults on purpose (tests/test_config.py checks that these values equal
# the defaults, so the dict documents them without overriding anything).
BASE_CONFIG: dict[str, Any] = {
    "observation": {"type": "Kinematics"},  # defaults: vehicles_count=5 (ego + 4 nearest ahead)
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 3,
    "duration": 30,  # seconds == steps at policy_frequency=1
    "policy_frequency": 1,
    "simulation_frequency": 5,
    "vehicles_count": 20,
    # highway-fast-v0 default; pinned so that evaluation on highway-v0 (default 2) starts from the
    # same initial layout as training.
    "ego_spacing": 1.5,
}

# Traffic density presets, merged on top of BASE_CONFIG. Only vehicles_density differs.
DENSITIES: dict[str, dict[str, Any]] = {
    "low": {"vehicles_density": 0.4},
    "medium": {"vehicles_density": 0.7},
    "high": {"vehicles_density": 1.0},
}
TRAIN_DENSITY = "medium"  # both algorithms train here; evaluation covers all densities


# Custom per-step reward (unnormalized), computed by common.rewards.CustomRewardWrapper from the
# reward components of the env:
#   r = w_speed * high_speed + w_right * right_lane - w_collision * crashed - w_lane * lane_changed
# high_speed, right_lane in [0, 1] come from info["rewards"]; crashed = info["crashed"];
# lane_changed = 1 if the real lane index changed during the step. The env's own reward weights
# and normalize_reward are NOT used for the custom variants.
# w_right encodes the keep-right traffic rule. w_lane is applied by the wrapper because
# highway-env 1.12.1 ignores the config key "lane_change_reward" (its _rewards() returns only
# the collision, right_lane, high_speed and on_road components).
#   efficiency: speed first, lane position irrelevant, risk-tolerant.
#   balanced:   moderate speed, keep right, mild lane-change penalty.
#   safety:     weak speed incentive, harsh crash penalty, costly lane changes (w_lane = 0.5 exceeds
#               the per-step maximum reward 0.3, so lane changes should be rare).
# "default" (empty dict) keeps the env's own normalized reward unchanged.
REWARDS: dict[str, dict[str, float]] = {
    "default": {},
    "efficiency": {"w_speed": 0.8, "w_right": 0.0, "w_collision": 15.0, "w_lane": 0.0},
    "balanced": {"w_speed": 0.4, "w_right": 0.1, "w_collision": 20.0, "w_lane": 0.1},
    "safety": {"w_speed": 0.2, "w_right": 0.1, "w_collision": 30.0, "w_lane": 0.5},
}

# Minimum discount factor for training. With gamma <= 0.9 a late crash is discounted so much that
# the lighter reward variants can rank a crash above slow safe driving.
MIN_GAMMA = 0.99


# Target performance for the "env steps to reach a target" metric (common.aggregate.steps_to_threshold).
# Two PRE-REGISTERED, named levels, both fixed before any training run and never changed after seeing results
# (docs/evaluation_protocol.md section 10). Both are reported for every run, so that if the strong level turns out
# unreachable (NaN for every model) the weaker one still allows a comparison, without touching a threshold afterwards.
# A level is a list of conditions that must ALL hold, each a `metric <operator> value` on a column of val_log.csv
# (common.metrics.SUMMARY_COLUMNS); behavioural metrics only, so the levels are comparable across reward presets.
# Each level is JOINT (one safety and one efficiency condition) and UNREACHABLE for every scripted baseline,
# otherwise an agent that merely learns a trivial policy would win "steps to target"
# (tests/test_training.py::test_no_scripted_baseline_meets_target_performance checks both levels on VAL_SEEDS).
#
# Baselines on `main` (docs/evaluation_protocol.md section 8, TEST_SEEDS) and on VAL_SEEDS (20 episodes, the data
# the target is evaluated on; from the slow baseline test):
#   policy   crash rate (test / val)   mean speed [m/s] (test / val)
#   slower      0% / 0.00               20.0 / 20.0   safe but slow: a crash bound alone is met without learning
#   idle       71% / 0.60               24.7 / 24.8   the default cruising policy
#   random     80% / 0.70               24.9 / 24.2
#   faster     99% / 1.00               28.8 / 28.9   fast but crashes
# "beats_idle": strictly better than idle on both metrics on VAL_SEEDS (crash rate below idle's 0.60, mean speed
#   above idle's 24.8 m/s). The weak level: the agent has learned something over the default policy.
# "strong": crash rate <= 0.2 (far below idle) and mean speed >= 25.5 m/s (above idle, far above slower): driving fast
#   AND safely, i.e. avoiding slow traffic ahead by changing lanes, not by braking. May turn out unreachable.
TARGET_PERFORMANCE: dict[str, list[dict[str, Any]]] = {  # PLACEHOLDER: decide before training
    "beats_idle": [
        {"metric": "crash_rate", "operator": "<", "value": 0.6},  # safety: idle 0.60 on VAL_SEEDS
        {"metric": "mean_speed", "operator": ">", "value": 24.8},  # efficiency [m/s]: idle 24.8 on VAL_SEEDS
    ],
    "strong": [
        {"metric": "crash_rate", "operator": "<=", "value": 0.2},  # safety
        {"metric": "mean_speed", "operator": ">=", "value": 25.5},  # efficiency [m/s]
    ],
}

# A level counts as reached at the FIRST of TARGET_CONSECUTIVE consecutive validation points at which all its
# conditions hold. Validation uses 20 episodes, so crash_rate moves in steps of 0.05 and a single lucky point
# must not count. PLACEHOLDER: decide before training; fixed before training like the levels.
TARGET_CONSECUTIVE = 2  # PLACEHOLDER: decide before training


def indifference_point(w_speed: float, w_right: float, w_collision: float, T: int = 30) -> float | None:
    """Analytic indifference point p of a reward variant (analysis helper, not used in training).

    Knox et al. (2023, sect. 4.3); idealized episodes in the right lane, no lane changes:
    success = T steps at 30 m/s (high_speed = 1), slow = T steps at 20 m/s (high_speed = 0),
    crash at step T/2 at 25 m/s (high_speed = 0.5). p is the success probability at which a
    gamble between success and crash is worth exactly the slow safe episode.

    Returns None if the preference ordering G_crash < G_slow < G_succ fails.
    """
    g_succ = T * (w_speed + w_right)
    g_slow = T * w_right
    g_crash = (T / 2) * (0.5 * w_speed + w_right) - w_collision
    if not g_crash < g_slow < g_succ:
        return None
    return (g_slow - g_crash) / (g_succ - g_crash)


# Seeds. Four disjoint roles; see docs/evaluation_protocol.md and common/seeding.py.
PILOT_SEEDS: list[int] = list(range(1000, 1050))  # environment pilot (scripted baselines only)
TRAIN_SEEDS: list[int] = [0, 1, 2, 3, 4]  # main grid: 5 runs per (algorithm, reward)
SENSITIVITY_SEEDS: list[int] = [0, 1, 2]  # hyperparameter sensitivity: 3 runs
VAL_SEEDS: list[int] = list(range(20000, 20020))  # periodic evaluation during training (learning curves)
TEST_SEEDS: list[int] = list(range(10000, 10100))  # final evaluation, used once
EVAL_SEEDS = TEST_SEEDS  # backwards-compatible alias

_seed_sets = {"pilot": PILOT_SEEDS, "train": TRAIN_SEEDS, "val": VAL_SEEDS, "test": TEST_SEEDS}
for (_a, _sa), (_b, _sb) in combinations(_seed_sets.items(), 2):
    assert not set(_sa) & set(_sb), f"seed sets {_a} and {_b} overlap"
assert not set(SENSITIVITY_SEEDS) & (set(PILOT_SEEDS) | set(VAL_SEEDS) | set(TEST_SEEDS)), "seed sets overlap"


# Test conditions. Each = env id + config overrides on top of BASE_CONFIG + DENSITIES[density].
# All conditions are evaluation-only and never used for training (training: ENV_ID, TRAIN_DENSITY).
#   main:    the training condition, the headline result.
#   low/high: generalization to other traffic densities.
#   long:    60 s episodes. Uses 30 vehicles because 20 was only verified up to 30 s in the pilot
#            (30 and 50 gave identical results up to 60 s).
#   precise: differs from training only in physics frequency (15 Hz instead of 5 Hz) and in
#            collisions between other vehicles being enabled (highway-v0 checks them,
#            highway-fast-v0 does not).
TEST_CONDITIONS: dict[str, dict[str, Any]] = {
    "main": {"env_id": ENV_ID, "density": "medium", "overrides": {}},
    "low": {"env_id": ENV_ID, "density": "low", "overrides": {}},
    "high": {"env_id": ENV_ID, "density": "high", "overrides": {}},
    "long": {"env_id": ENV_ID, "density": "medium", "overrides": {"duration": 60, "vehicles_count": 30}},
    "precise": {"env_id": "highway-v0", "density": "medium", "overrides": {"simulation_frequency": 15}},
}


def describe_rewards() -> str:
    """Human-readable table of the reward presets with the analytic indifference point p."""
    lines = ["reward      w_speed w_right w_collision w_lane  p(analytic)"]
    for name, w in REWARDS.items():
        if not w:
            lines.append(f"{name:<11} (env's own normalized reward, unchanged)")
            continue
        p = indifference_point(w["w_speed"], w["w_right"], w["w_collision"], T=BASE_CONFIG["duration"])
        p_txt = "ordering fails" if p is None else f"{p:.2f}"
        lines.append(
            f"{name:<11} {w['w_speed']:>7.2f} {w['w_right']:>7.2f} {w['w_collision']:>11.2f} "
            f"{w['w_lane']:>6.2f}  {p_txt}"
        )
    lines.append(f"MIN_GAMMA = {MIN_GAMMA}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_rewards())
