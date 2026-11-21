"""Shared environment, traffic-density and reward configuration.

Environment and density values were chosen from the environment pilot (see
``docs/config_decisions.md`` and ``results/pilot/PILOT_REPORT.md``) so that both algorithms use
identical settings. Do not change them without updating the docs. The reward weights are fixed
values agreed by the team (see ``docs/config_decisions.md``).

Print the reward weights and their analytic indifference points with:  python -m common.config
"""

from __future__ import annotations

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


# Seeds. Evaluation seeds must not overlap with training seeds or the pilot seeds (1000..1049).
PILOT_SEEDS: list[int] = list(range(1000, 1050))
TRAIN_SEEDS: list[int] = [0, 1, 2]
EVAL_SEEDS: list[int] = list(range(10000, 10100))
assert not set(EVAL_SEEDS) & (set(TRAIN_SEEDS) | set(PILOT_SEEDS)), "seed sets overlap"


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
