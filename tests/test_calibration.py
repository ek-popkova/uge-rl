"""Tests of the SB3 calibration code (scripts/calibration/): cost model, run list, isolation from the main grid,
agent adapters, validation records, training files and resume, analysis helpers on hand-built cases.

Fast tier: pure functions. Slow tier (``@pytest.mark.slow``): real envs and tiny SB3 runs; needs stable-baselines3
(``requirements-calibration.txt``) and is skipped without it.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from common.config import REWARDS, TEST_CONDITIONS, VAL_SEEDS
from scripts.calibration import plan as planlib
from scripts.calibration import sb3_cal

BENCH = {  # hand-built quiet measurements: one process 50 steps/s, aggregate of 8 processes = 3.5 x one process
    "env_steps_per_s": 50.0, "dqn_train_steps_per_s": 40.0, "dqn_update_s_per_step": 0.002, "dqn_val_s": 15.0,
    "ppo_train_steps_per_s": 100.0, "ppo_update_s_per_rollout": 2.0, "ppo_val_s": 8.0, "aggregate_over_single": 3.5,
}


# ---------------------------------------------------------------------------------------------
# budget, hyper-parameters, layout
# ---------------------------------------------------------------------------------------------
def test_budget_and_hyperparameters_match_the_task():
    assert sb3_cal.TOTAL_ENV_STEPS == 200_000 and sb3_cal.VAL_EVERY_ENV_STEPS == 5_000 and sb3_cal.VAL_N_EPISODES == 20
    assert sb3_cal.total_steps("dqn") == 200_000
    assert sb3_cal.total_steps("ppo") == 98 * 2048 == 200_704  # smallest multiple of the rollout size >= 200,000
    d = {k: v for k, (v, _) in sb3_cal.DQN_HYPERPARAMETERS.items()}
    assert d["gamma"] == 0.99 and d["net_arch"] == [256, 256] and d["exploration_fraction"] == 0.3
    assert (d["learning_rate"], d["buffer_size"], d["learning_starts"], d["batch_size"]) == (5e-4, 15_000, 200, 32)
    assert (d["train_freq"], d["gradient_steps"], d["target_update_interval"]) == (1, 1, 50)
    p = {k: v for k, (v, _) in sb3_cal.PPO_HYPERPARAMETERS.items()}
    assert (p["learning_rate"], p["clip_range"], p["gae_lambda"], p["n_epochs"], p["batch_size"], p["ent_coef"]) == (
        3e-4, 0.2, 0.95, 10, 64, 0.0)
    assert (p["n_envs"], p["n_steps"], p["gamma"]) == (8, 256, 0.99)
    assert p["net_arch"] == {"pi": [256, 256], "vf": [256, 256]}
    for table in (sb3_cal.DQN_HYPERPARAMETERS, sb3_cal.PPO_HYPERPARAMETERS):
        assert all(isinstance(src, str) and src for _, src in table.values())  # every value has a recorded source


def test_calibration_outputs_are_invisible_to_the_main_grid_loaders(tmp_path):
    from common.paths import load_eval_episodes, load_val_logs

    assert sb3_cal.CAL_ROOT.parent.name == "results" and sb3_cal.CAL_ROOT.name == "calibration"
    root = tmp_path
    for sub in (sb3_cal.eval_dir("ppo", "balanced", 99, root), sb3_cal.eval_dir("dqn", "balanced", 98, root)):
        sub.mkdir(parents=True)
        pd.DataFrame({"seed": [1], "crashed": [True]}).to_csv(sub / "main.csv", index=False)
    for algo in ("ppo", "dqn"):
        rd = sb3_cal.run_dir(algo, "balanced", 99, root)
        rd.mkdir(parents=True)
        pd.DataFrame({"env_step": [1]}).to_csv(rd / "val_log.csv", index=False)
    assert load_eval_episodes(root).empty
    for algo in ("ddqn", "ppo"):
        assert load_val_logs(algo, "balanced", root) == {}


def test_run_dir_rejects_other_seeds_and_unknown_names(tmp_path):
    for bad in (0, 1, 5, 100):
        with pytest.raises(ValueError, match="calibration runs use the seeds"):
            sb3_cal.run_dir("dqn", "balanced", bad, tmp_path)
    with pytest.raises(ValueError):
        sb3_cal.run_dir("ddqn", "balanced", 99, tmp_path)
    with pytest.raises(ValueError):
        sb3_cal.run_dir("dqn", "nonsense", 99, tmp_path)


# ---------------------------------------------------------------------------------------------
# cost model and schedule
# ---------------------------------------------------------------------------------------------
def test_run_list_priority_order_and_trimming():
    runs = planlib.build_runs()
    assert [(r.algo, r.reward, r.seed, r.priority) for r in runs] == [
        ("dqn", "balanced", 99, 1), ("ppo", "balanced", 99, 1),
        ("dqn", "efficiency", 99, 2), ("ppo", "efficiency", 99, 2),
        ("dqn", "safety", 99, 2), ("ppo", "safety", 99, 2),
        ("dqn", "default", 99, 2), ("ppo", "default", 99, 2),
        ("dqn", "balanced", 98, 3), ("ppo", "balanced", 98, 3),
    ]
    assert len(planlib.build_runs(include_p3=False)) == 8
    no_default = planlib.build_runs(include_p3=False, include_default=False)
    assert [r.reward for r in no_default if r.algo == "dqn"] == ["balanced", "efficiency", "safety"]
    assert {r.weight for r in runs if r.algo == "ppo"} == {8} and {r.weight for r in runs if r.algo == "dqn"} == {1}


def test_speed_rel_follows_the_measured_curve():
    curve = planlib.aggregate_curve(BENCH)
    assert planlib.speed_rel(1, curve) == 1.0
    assert planlib.speed_rel(4, curve) == pytest.approx(2.7 / 4)
    assert planlib.speed_rel(8, curve) == pytest.approx(3.5 / 8)  # = BENCH aggregate_over_single / 8
    assert planlib.speed_rel(2, curve) == pytest.approx((1 + (2.7 - 1) / 3) / 2)  # linear between 1 and 4
    assert planlib.speed_rel(12, curve) == pytest.approx(3.2 / 12) and planlib.speed_rel(20, curve) == pytest.approx(2.9 / 20)
    assert planlib.aggregate_curve({"aggregate_over_single": 9.0})[8] == 5.0  # clamped


def test_quiet_durations_from_the_benchmark():
    dqn = planlib.quiet_duration("dqn", BENCH)
    assert dqn == pytest.approx(200_000 / 40 + 40 * 15.0)  # training + 40 validations
    ppo = planlib.quiet_duration("ppo", BENCH)
    assert ppo == pytest.approx(200_704 / 100 + 40 * 8.0)


def test_simulate_hand_built_cases():
    curve = planlib.aggregate_curve(BENCH)
    one = planlib.PlannedRun("dqn", "balanced", 99, 1, quiet_s=100.0)
    s = planlib.simulate([one], max_dqn=1, curve=curve)
    assert s.makespan_s == pytest.approx(100.0)  # alone: exactly its quiet duration
    two = [planlib.PlannedRun("dqn", "balanced", 99, 1, 100.0), planlib.PlannedRun("dqn", "safety", 99, 2, 100.0)]
    s = planlib.simulate(two, max_dqn=2, curve=curve)
    assert s.makespan_s == pytest.approx(100.0 / planlib.speed_rel(2, curve))  # both slowed down by 2 busy processes
    s = planlib.simulate(two, max_dqn=1, curve=curve)
    assert s.makespan_s == pytest.approx(200.0)  # one slot: run one after the other, no contention
    assert s.start_s["dqn/safety/seed99"] == pytest.approx(100.0)
    ppo = planlib.PlannedRun("ppo", "balanced", 99, 1, quiet_s=100.0)  # alone: its own 8 processes are already in the quiet time
    assert planlib.simulate([ppo], 1, curve).makespan_s == pytest.approx(100.0)
    s = planlib.simulate([ppo, planlib.PlannedRun("dqn", "balanced", 99, 1, 100.0)], 1, curve)
    assert s.peak_load == 9 and s.makespan_s > 100.0
    done = planlib.PlannedRun("dqn", "balanced", 99, 1, 100.0, done=0.75)  # resume: only 25% of the work left
    assert planlib.simulate([done], 1, curve).makespan_s == pytest.approx(25.0)


def test_make_plan_drops_p3_then_default_and_never_shortens_runs():
    fast = planlib.make_plan(BENCH, limit_h=100.0)
    assert fast.dropped == [] and len(fast.runs) == 10
    slow_bench = dict(BENCH, dqn_train_steps_per_s=9.0, ppo_train_steps_per_s=20.0)
    mid = planlib.make_plan(slow_bench, limit_h=fast_limit(slow_bench, 8))
    assert mid.dropped == ["P3: balanced seed 98 (DQN, PPO)"] and len(mid.runs) == 8
    low = planlib.make_plan(slow_bench, limit_h=fast_limit(slow_bench, 6))
    assert low.dropped[-1].startswith("P2: default") and len(low.runs) == 6
    assert all(r.quiet_s == planlib.quiet_duration(r.algo, slow_bench) for r in low.runs)  # budget untouched
    with pytest.raises(RuntimeError):
        planlib.make_plan(slow_bench, limit_h=0.1)


def fast_limit(bench, n_runs):
    """A limit just above the estimate of the plan with ``n_runs`` runs."""
    runs = planlib.build_runs(include_p3=n_runs > 8, include_default=n_runs > 6)
    for r in runs:
        r.quiet_s = planlib.quiet_duration(r.algo, bench)
    best = min(planlib.simulate(runs, k, planlib.aggregate_curve(bench)).makespan_s for k in (2, 3, 4, 5))
    return (best + planlib.eval_overhead_s(bench, len(runs))) / 3600 * 1.001


# ---------------------------------------------------------------------------------------------
# random reference of the early warning (hand-built episodes)
# ---------------------------------------------------------------------------------------------
def test_random_reference_from_hand_built_episodes(tmp_path):
    from scripts.calibration import evaluate_final

    # 4 random-policy episodes: two crashed; components chosen so every preset's return is easy to compute by hand
    df = pd.DataFrame({
        "policy": ["random"] * 4 + ["idle"] * 2,
        "crashed": [True, True, False, False, False, False],
        "sum_high_speed": [10.0, 10.0, 20.0, 20.0, 0.0, 0.0],
        "sum_right_lane": [10.0, 10.0, 10.0, 10.0, 0.0, 0.0],
        "lane_changes": [2, 2, 4, 4, 0, 0],
        "return_env": [1.0, 3.0, 5.0, 7.0, 0.0, 0.0],
    })
    out = tmp_path
    df.to_csv(out / "main.csv", index=False)
    evaluate_final.write_random_reference(out)
    ref = json.loads((out / "random_val_main.json").read_text())
    assert set(ref) == set(REWARDS)
    assert ref["balanced"]["crash_rate"] == 0.5
    # balanced: 0.4*hs + 0.1*rl - 20*crashed - 0.1*lc -> episodes: -15.8-0.0... computed explicitly
    rets = [0.4 * 10 + 0.1 * 10 - 20 - 0.1 * 2, 0.4 * 10 + 0.1 * 10 - 20 - 0.1 * 2,
            0.4 * 20 + 0.1 * 10 - 0.1 * 4, 0.4 * 20 + 0.1 * 10 - 0.1 * 4]
    assert ref["balanced"]["return"] == pytest.approx(np.mean(rets))
    assert ref["balanced"]["return_se"] == pytest.approx(np.std(rets, ddof=1) / 2)
    assert ref["default"]["return"] == pytest.approx(4.0)  # mean of return_env of the random rows only


def test_retry_on_permission_error_and_atomic_write(tmp_path):
    """Regression: on Windows os.replace raised PermissionError while the orchestrator / OneDrive had progress.json open and
    killed 6 DQN runs (restarted from their checkpoints). The helper retries a few times, then gives up."""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("locked")
        return "ok"

    assert sb3_cal.retry_on_permission_error(flaky, attempts=5, delay=0.0) == "ok" and calls["n"] == 3
    with pytest.raises(PermissionError):
        sb3_cal.retry_on_permission_error(lambda: (_ for _ in ()).throw(PermissionError("x")), attempts=2, delay=0.0)
    target = tmp_path / "a" / "x.json"
    sb3_cal.atomic_write_json(target, {"k": [1, 2]})
    assert json.loads(target.read_text()) == {"k": [1, 2]} and list(target.parent.glob("*.tmp")) == []


# ---------------------------------------------------------------------------------------------
# analysis helpers on hand-built cases
# ---------------------------------------------------------------------------------------------
def test_plateau_point_hand_built():
    from scripts.calibration import stats as st

    steps = [1000 * i for i in range(1, 11)]
    flat = st.plateau_point(steps, [0, 2, 5, 8, 9.6, 9.9, 10, 10, 10, 10])
    # last 20% = steps >= 8000 -> target 10; trailing 3-point means from index 2: 2.33, 5, 7.53, 9.17, 9.83, 9.97, 10, 10;
    # |roll - 10| <= 0.5 first holds (and keeps holding) at index 6, i.e. step 7000
    assert flat["target"] == 10 and flat["plateau_idx"] == 6 and flat["plateau_step"] == 7000 and flat["plateaued"]
    rising = st.plateau_point(steps, [float(i) for i in range(1, 11)])  # still improving: target 9, only the last roll (9) is inside
    assert rising["plateau_step"] == 10_000 and not rising["plateaued"]
    zero = st.plateau_point(steps, [0.0] * 10)  # tolerance 5% of 0 is empty but |0 - 0| <= 0 holds: plateau from the first full window
    assert zero["plateau_idx"] == 2 and zero["plateaued"]
    assert math.isnan(st.plateau_point([1, 2], [1, 2])["plateau_step"])  # too short


def test_pass_probability_binomial_values():
    from scripts.calibration import stats as st

    assert st.binom_cdf(4, 20, 0.1) == pytest.approx(0.95683, abs=1e-4)
    assert st.binom_cdf(4, 20, 0.2) == pytest.approx(0.62961, abs=1e-4)
    assert st.pass_probability(0.2, 0.2, 20, 1) == pytest.approx(0.62961, abs=1e-4)  # threshold 0.2 x 20 = 4 crashes allowed
    assert st.pass_probability(0.2, 0.2, 20, 2) == pytest.approx(0.62961**2, abs=1e-4)
    assert st.pass_probability(0.0, 0.2, 20, 3) == 1.0 and st.pass_probability(1.0, 0.2, 20, 1) == 0.0
    assert st.binom_cdf(-1, 20, 0.5) == 0.0


def test_val_noise_from_hand_built_episodes():
    from scripts.calibration import stats as st

    rows = []
    for v in range(7):  # 7 validation points of 20 episodes: 5 crashes each, speeds 24 (x10) and 26 (x10)
        for e in range(20):
            rows.append({"val_idx": v, "crashed": e < 5, "mean_speed": 24.0 if e % 2 == 0 else 26.0})
    n = st.val_noise(pd.DataFrame(rows), last_k=5)
    assert n["n_points"] == 5 and n["n_episodes"] == 20 and n["p_crash"] == 0.25
    assert n["se_crash_20"] == pytest.approx(math.sqrt(0.25 * 0.75 / 20))
    assert n["se_crash_50"] == pytest.approx(n["se_crash_20"] * math.sqrt(20 / 50))  # sqrt scaling
    sd = math.sqrt(20 / 19)  # sample sd of ten 24s and ten 26s
    assert n["sd_speed"] == pytest.approx(sd) and n["se_speed_30"] == pytest.approx(sd / math.sqrt(30))
    assert st.tail_fluctuation([1, 2, 3, 4, 5], [9, 9, 9, 1.0, 3.0]) == pytest.approx(math.sqrt(2))  # last 20%: steps >= 4 -> [1, 3]


def test_spearman_seed_spread_and_rounding():
    from scripts.calibration import stats as st

    assert st.spearman([1, 2, 3, 4], [10, 20, 30, 45]) == pytest.approx(1.0)
    assert st.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(st.spearman([1, 2], [1, 2])) and math.isnan(st.spearman([1, 1, 1], [1, 2, 3]))
    sp = st.seed_spread(0.5, 0.3)
    assert sp["abs_diff"] == pytest.approx(0.2) and sp["sd"] == pytest.approx(0.2 / math.sqrt(2))
    assert sp["halfwidth_ci"] == pytest.approx(2.7764451 * 0.2 / math.sqrt(2) / math.sqrt(5), abs=1e-6)  # t(0.975, 4) sd / sqrt 5
    assert st.round_up(0.31, 0.05) == pytest.approx(0.35) and st.round_up(0.30, 0.05) == pytest.approx(0.30)
    assert st.round_down(25.57, 0.1) == pytest.approx(25.5)


def test_intended_ordering_and_targets_on_hand_built_values():
    from common.aggregate import steps_to_threshold
    from scripts.calibration import analyze as an

    ok = {
        "efficiency": {"mean_speed": 27, "crash_rate": 0.6, "crashes_per_km": 3, "lane_changes": 5},
        "balanced": {"mean_speed": 25, "crash_rate": 0.3, "crashes_per_km": 1, "lane_changes": 3},
        "safety": {"mean_speed": 22, "crash_rate": 0.0, "crashes_per_km": 0, "lane_changes": 1},
    }
    assert all(an.intended_ordering(ok).values())
    bad = dict(ok, safety=dict(ok["safety"], lane_changes=4, mean_speed=26))
    chk = an.intended_ordering(bad)
    assert chk["safety has the fewest lane changes"] is False and chk["safety slowest"] is False
    assert chk["crash rate: efficiency > balanced > safety"] is True
    assert an.intended_ordering({"balanced": ok["balanced"]}) == {}
    val = pd.DataFrame({"env_step": [5000 * (i + 1) for i in range(8)],
                        "crash_rate": [0.8, 0.5, 0.2, 0.1, 0.1, 0.3, 0.1, 0.1],
                        "mean_speed": [24, 25, 27, 27, 27, 27, 27, 27]})
    anchor = an.PRESENT_TARGETS["strong_baseline_anchor"]  # crash <= 0.2 and speed >= 26.8
    assert steps_to_threshold(val, anchor, consecutive=1)["env_step"] == 15_000
    assert steps_to_threshold(val, anchor, consecutive=2)["env_step"] == 15_000  # points 3 and 4
    assert steps_to_threshold(val, anchor, consecutive=3)["env_step"] == 15_000  # points 3, 4, 5; the dip at point 6 ends that run
    assert math.isnan(steps_to_threshold(val, anchor, consecutive=4)["env_step"])  # the longest run of holds has 3 points
    assert steps_to_threshold(val, an.PRESENT_TARGETS["beats_idle"], consecutive=1)["env_step"] == 10_000  # crash < 0.6, speed > 24.8


def test_markdown_helpers():
    from scripts.calibration import analyze as an

    assert an.fmt(float("nan")) == "n/a" and an.fmt(1234) == "1,234" and an.fmt(0.12345, 3) == "0.123" and an.fmt(True) == "yes"
    table = an.md_table([["a", 1.0], ["b", float("nan")]], ["x", "y"])
    assert table.splitlines()[0] == "| x | y |" and table.splitlines()[-1] == "| b | n/a |"
    assert an.pct(0.456) == "46%" and an.pct(float("nan")) == "n/a"


def test_evidence_copy_takes_only_the_small_files(tmp_path):
    """docs/calibration_evidence/ must hold the plots, val_log.csv and run_meta.json and nothing else (no models, buffers, logs)."""
    from types import SimpleNamespace

    from scripts.calibration import analyze as an

    root, dest = tmp_path / "cal", tmp_path / "evidence"
    d = SimpleNamespace(root=root, runs={("dqn", "balanced", 99): None, ("ppo", "safety", 98): None})
    for key in d.runs:
        rd = sb3_cal.run_dir(*key, root)
        rd.mkdir(parents=True)
        for name in ("val_log.csv", "run_meta.json", "final.zip", "train_log.csv", "update_log.csv", "val_episodes.csv"):
            (rd / name).write_text("x" * 10)
    (root / "plots").mkdir()
    plot = root / "plots" / "curves.png"
    plot.write_bytes(b"png")
    n = an.copy_evidence(d, [plot], dest)
    copied = sorted(str(q.relative_to(dest)).replace("\\", "/") for q in dest.rglob("*") if q.is_file())
    assert copied == ["dqn/balanced/seed99/run_meta.json", "dqn/balanced/seed99/val_log.csv", "plots/curves.png",
                      "ppo/safety/seed98/run_meta.json", "ppo/safety/seed98/val_log.csv"]
    assert n == 3 + 4 * 10


def test_report_pins_the_placeholder_strong_level():
    """The calibration report analyses the level the calibration was planned with, whatever common.config freezes later."""
    from scripts.calibration import analyze as an

    assert an.PRESENT_TARGETS["strong_placeholder"] == [
        {"metric": "crash_rate", "operator": "<=", "value": 0.2}, {"metric": "mean_speed", "operator": ">=", "value": 25.5}]


# ---------------------------------------------------------------------------------------------
# slow tier: SB3 objects, real envs
# ---------------------------------------------------------------------------------------------
sb3 = pytest.importorskip("stable_baselines3", reason="pip install -r requirements-calibration.txt")


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["dqn", "ppo"])
def test_light_agent_equals_model_predict(algo):
    from scripts.calibration.sb3_train import build_model
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv(sb3_cal.env_fns("balanced", 1))
    model = build_model(algo, venv)
    light = sb3_cal.light_agent_from_model(model, algo)
    obs = np.random.default_rng(5).random((300, 5, 5)).astype(np.float32)
    assert sb3_cal.check_same_actions(model, light, obs) == 0
    assert {light.act(o) for o in obs} <= set(range(5))
    import pickle

    again = pickle.loads(pickle.dumps(light))  # picklable for the validation workers
    assert all(again.act(o) == light.act(o) for o in obs[:20])
    venv.close()


@pytest.mark.slow
def test_network_sizes_equal_the_frozen_shared_decision():
    from common.networks import build_action_network, build_value_network, count_parameters
    from common.train_config import DDQNConfig, PPOConfig
    from scripts.calibration.sb3_train import build_model, n_parameters
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv(sb3_cal.env_fns("balanced", 1))
    dqn, ppo = build_model("dqn", venv), build_model("ppo", venv)
    assert n_parameters(dqn, "dqn") == count_parameters(build_action_network(DDQNConfig())) == 73_733
    assert n_parameters(ppo, "ppo") == 73_733 + 72_705  # actor + critic, as in PPOConfig
    assert dqn.gamma == ppo.gamma == 0.99 and dqn.device.type == "cpu"
    assert dqn.exploration_fraction == 0.3 and ppo.n_steps == 256 and sb3_cal.PPO_N_ENVS == 8
    venv.close()


@pytest.mark.slow
def test_training_env_seeding_follows_common_seeding():
    from common.envs import make_env
    from common.seeding import vector_env_seed
    from scripts.calibration.sb3_train import make_vec_env

    venv = make_vec_env("dqn", "balanced", 98)
    first = venv.reset()[0]
    ref, _ = make_env(reward="balanced").reset(seed=98)
    np.testing.assert_array_equal(first, ref)  # the single env of DQN: initial reset seed = the run seed
    venv.close()
    venv = make_vec_env("dqn", "balanced", 98, resume_k=2)
    assert venv._seeds == [vector_env_seed(98, 120)]  # resume number 2: derived index 100 + 10 * 2
    venv.close()
    venv = make_vec_env("ppo", "efficiency", 99)  # 8 subprocess envs, seeds vector_env_seed(99, i)
    try:
        first = venv.reset()
        for i in range(8):
            ref, _ = make_env(reward="efficiency").reset(seed=vector_env_seed(99, i))
            np.testing.assert_array_equal(first[i], ref)
    finally:
        venv.close()


@pytest.mark.slow
def test_validator_equals_validate_and_keeps_episode_records():
    from common.baselines import make_baseline
    from common.evaluate import validate

    agent = make_baseline("idle")
    v = sb3_cal.Validator("balanced", workers=1, n_episodes=4)
    summary, df = v.run(agent)
    assert summary == validate(agent, n=4, reward="balanced")  # exactly the validate() output
    assert list(df["seed"]) == VAL_SEEDS[:4] and "return_safety" in df
    pooled = sb3_cal.Validator("balanced", workers=2, n_episodes=4, chunk=2)
    try:
        s2, df2 = pooled.run(agent)
        pd.testing.assert_frame_equal(df, df2)  # worker-independent
    finally:
        pooled.close()


@pytest.mark.slow
def test_dqn_run_files_schemas_and_resume(tmp_path):
    from common.train_logging import LOG_FILES, TRAIN_LOG_COLUMNS, UPDATE_LOG_COLUMNS, VAL_LOG_COLUMNS
    from scripts.calibration.sb3_train import CalibrationRun, SimulatedInterrupt

    kw = dict(root=tmp_path, total_env_steps=500, val_every=250, val_n=2, quiet=True)
    first = CalibrationRun("dqn", "safety", 98, checkpoint_seconds=0.0, interrupt_at_env_step=300, **kw)
    with pytest.raises(SimulatedInterrupt):
        first.run()
    rdir = sb3_cal.run_dir("dqn", "safety", 98, tmp_path)
    assert json.loads((rdir / "run_meta.json").read_text())["status"] == "failed"
    assert not sb3_cal.is_finished("dqn", "safety", 98, tmp_path)
    CalibrationRun("dqn", "safety", 98, resume=True, checkpoint_seconds=1e9, **kw).run()
    assert sb3_cal.is_finished("dqn", "safety", 98, tmp_path)
    meta = json.loads((rdir / "run_meta.json").read_text())
    assert len(meta["resumes"]) == 1 and meta["timing"]["final_env_step"] == 500
    assert meta["identity"]["algorithm"] == "sb3_dqn" and meta["model"]["n_parameters"] == 73_733
    assert "NOT Double DQN" in meta["calibration"]["algorithm_note"]
    assert meta["hyperparameters"]["gamma"] == 0.99 and meta["calibration"]["hyperparameter_sources"]["gamma"]
    val = pd.read_csv(rdir / LOG_FILES["val"])
    assert list(val.columns) == VAL_LOG_COLUMNS  # exactly the main-grid schema
    assert list(val["env_step"]) == [250, 500] and list(val["val_idx"]) == [0, 1]  # no duplicates after the resume
    assert list(pd.read_csv(rdir / LOG_FILES["train"]).columns) == TRAIN_LOG_COLUMNS
    assert list(pd.read_csv(rdir / LOG_FILES["update"]).columns) == UPDATE_LOG_COLUMNS["ddqn"]
    episodes = pd.read_csv(rdir / "val_episodes.csv")
    assert episodes.groupby("env_step").size().to_dict() == {250: 2, 500: 2}
    train = pd.read_csv(rdir / LOG_FILES["train"])
    assert train["env_step"].is_monotonic_increasing
    # the final model loads and acts like its light copy
    model = sb3_cal.load_sb3_model("dqn", rdir / "final.zip")
    assert sb3_cal.check_same_actions(model, sb3_cal.light_agent_from_model(model, "dqn"),
                                      np.random.default_rng(0).random((50, 5, 5)).astype(np.float32)) == 0


@pytest.mark.slow
def test_ppo_run_files_and_time_limit_handling(tmp_path):
    from common.train_logging import UPDATE_LOG_COLUMNS, VAL_LOG_COLUMNS
    from scripts.calibration.sb3_train import CalibrationRun

    rd = CalibrationRun("ppo", "efficiency", 99, root=tmp_path, total_env_steps=4096, val_every=2048, val_n=4,
                        quiet=True, val_workers=2, checkpoint_seconds=1e9).run()
    val = pd.read_csv(rd / "val_log.csv")
    assert list(val.columns) == VAL_LOG_COLUMNS and list(val["env_step"]) == [2048, 4096]  # rollout boundaries
    upd = pd.read_csv(rd / "update_log.csv")
    assert list(upd.columns) == UPDATE_LOG_COLUMNS["ppo"] and len(upd) == 2 and (upd["n_epochs_done"] == 10).all()
    train = pd.read_csv(rd / "train_log.csv")
    assert set(train["vec_env"]) <= set(range(8)) and train["env_step"].is_monotonic_increasing
    assert train["truncated"].any() or train["crashed"].any()
    # SB3 bootstraps from the next state when the 30 s limit ends an episode: DummyVecEnv / SubprocVecEnv mark it
    # "TimeLimit.truncated" for truncated and not terminated (checked on a hand-built env below)


@pytest.mark.slow
def test_sb3_marks_time_limit_truncation_not_termination():
    """highway-env ends an episode with truncated=True at the time limit and terminated=True on a crash; SB3's VecEnv sets
    ``TimeLimit.truncated`` only for the former, so the algorithms bootstrap from s' at the time limit (protocol, section 1)."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv(sb3_cal.env_fns("balanced", 1))
    venv.seed(7)
    venv.reset()
    slower = 4  # SLOWER: no crash within 30 s at medium density (baseline: 0% crashes)
    infos = None
    for _ in range(30):
        _, _, dones, infos = venv.step(np.array([slower]))
    assert dones[0] and infos[0]["TimeLimit.truncated"] is True and not infos[0]["crashed"]
    venv.close()


@pytest.mark.slow
def test_resume_without_checkpoint_restarts_but_keeps_the_orchestrator_log(tmp_path):
    """Regression: the first launch of a run passes --resume; with stray files and no checkpoint the run restarts cleanly,
    but train_stdout.log (open in the orchestrator) must never be deleted."""
    from scripts.calibration.sb3_train import CalibrationRun

    rdir = sb3_cal.run_dir("dqn", "balanced", 99, tmp_path)
    rdir.mkdir(parents=True)
    (rdir / "train_stdout.log").write_text("log")
    (rdir / "val_log.csv").write_text("stale")
    run = CalibrationRun("dqn", "balanced", 99, resume=True, root=tmp_path, quiet=True)
    assert run._prepare_dir() is None
    assert (rdir / "train_stdout.log").read_text() == "log" and not (rdir / "val_log.csv").exists()
    with pytest.raises(FileExistsError):  # without --resume existing results are protected
        (rdir / "val_log.csv").write_text("x")
        CalibrationRun("dqn", "balanced", 99, resume=False, root=tmp_path, quiet=True)._prepare_dir()
