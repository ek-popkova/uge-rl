"""Values frozen after the SB3 calibration (all fast, no simulation): the budget and target levels, the PPO rollout-size rule,
the post-hoc plateau metric, the cost tables with wall times grouped by machine, and the OneDrive / machine metadata.
"""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common import run_meta
from common.aggregate import (
    ALL_MACHINES,
    PLATEAU_TOL,
    PLATEAU_WINDOW,
    cost_across_seeds,
    cost_per_model,
    steps_to_plateau,
)
from common.config import TARGET_CONSECUTIVE, TARGET_PERFORMANCE
from common.metrics import check_target_performance
from common.random_trainer import RandomTrainer
from common.train_config import (
    DDQNConfig,
    PPOConfig,
    TrainBudget,
    check_budget_matches_rollout,
    check_comparable,
    expand_sensitivity_grid,
    placeholders_in_use,
    resolve_hyperparameters,
)
from common.training import TrainingRun, check_budget_fits_trainer, resolve_setup


# ---------------------------------------------------------------------------------------------
# frozen values
# ---------------------------------------------------------------------------------------------
def test_budget_and_target_values_are_frozen_and_no_longer_placeholders():
    b = TrainBudget()
    assert (b.total_env_steps, b.val_every_env_steps, b.val_n_episodes) == (300_000, 12_000, 20)
    assert b.n_val_points == 25  # 25 points per learning curve
    assert placeholders_in_use(b) == []
    assert TARGET_CONSECUTIVE == 3
    assert TARGET_PERFORMANCE["strong"] == [
        {"metric": "crash_rate", "operator": "<=", "value": 0.4},
        {"metric": "mean_speed", "operator": ">=", "value": 25.5},
    ]
    assert TARGET_PERFORMANCE["beats_idle"] == [  # decided earlier, unchanged
        {"metric": "crash_rate", "operator": "<", "value": 0.6},
        {"metric": "mean_speed", "operator": ">", "value": 24.8},
    ]
    check_target_performance(TARGET_PERFORMANCE, TARGET_CONSECUTIVE)


def test_the_strong_level_stays_out_of_reach_of_the_scripted_baselines():
    strong = {c["metric"]: c for c in TARGET_PERFORMANCE["strong"]}
    idle_crash, idle_speed, slower_speed = 0.60, 24.7564, 20.03  # baselines on VAL_SEEDS (docs/config_decisions.md)
    assert strong["crash_rate"]["value"] < idle_crash  # idle (crash 0.6) misses the safety condition
    assert strong["mean_speed"]["value"] > idle_speed > slower_speed  # slower misses the speed condition
    assert strong["mean_speed"]["value"] > TARGET_PERFORMANCE["beats_idle"][1]["value"]  # strictly harder than beats_idle on speed


def test_sensitivity_runs_follow_the_300k_budget():
    """The 24 sensitivity runs use the default (main-grid) budget; the DDQN epsilon schedule is resolved for 300k steps."""
    runs = expand_sensitivity_grid()
    assert len(runs) == 24
    for run in runs:
        setup = resolve_setup(run.algo, [*run.cli_args(), "--out-dir", "unused"])
        assert setup.budget == TrainBudget() and setup.budget.total_env_steps == 300_000 and not setup.budget_overrides
    eps = {r.overrides["eps_decay_fraction"] for r in runs if r.param == "eps_decay_fraction"}
    assert eps == {0.1, 0.5}  # base 0.3 -0.2 / +0.2
    steps = {f: resolve_hyperparameters(DDQNConfig(eps_decay_fraction=f), TrainBudget())["eps_decay_steps"] for f in (0.1, 0.3, 0.5)}
    assert steps == {0.1: 30_000, 0.3: 90_000, 0.5: 150_000}


def test_tuning_runs_use_the_same_300k_budget(tmp_path):
    for algo in ("ddqn", "ppo"):
        setup = resolve_setup(algo, ["--variant", "tune-x", "--seed", "99", "--reward", "balanced", "--out-dir", str(tmp_path)])
        assert setup.budget == TrainBudget() and setup.budget.total_env_steps == 300_000
    with pytest.raises(ValueError, match="budget"):  # still no budget override in a tuning run
        resolve_setup("ddqn", ["--variant", "tune-x", "--seed", "99", "--reward", "balanced", "--total-steps", "1000",
                               "--val-every", "500"])


# ---------------------------------------------------------------------------------------------
# PPO rollout size must divide the budget
# ---------------------------------------------------------------------------------------------
def test_rollout_that_does_not_divide_the_budget_raises_and_names_the_nearest_values():
    ppo = PPOConfig(n_envs=8, n_steps=256, minibatch_size=256)  # rollout 2,048 divides neither 300,000 nor 12,000
    with pytest.raises(ValueError) as exc:
        check_comparable(DDQNConfig(), ppo, TrainBudget())
    msg = str(exc.value)
    assert "2,048" in msg and "300,000" in msg and "12,000" in msg
    assert "299,008" in msg and "301,056" in msg  # nearest multiples of 2,048 around 300,000
    assert "10,240" in msg and "12,288" in msg  # ... and around 12,000
    assert "2,000" in msg and "2,400" in msg  # nearest divisors of gcd = 12,000 around 2,048
    with pytest.raises(ValueError, match="multiple"):
        check_budget_matches_rollout(TrainBudget(), 2048)


def test_rollouts_that_divide_both_values_pass():
    for n_envs, n_steps in ((4, 250), (8, 125), (4, 500), (6, 500), (8, 500), (6, 1000), (12, 1000)):  # 1,000 ... 12,000
        ppo = PPOConfig(n_envs=n_envs, n_steps=n_steps, minibatch_size=100)
        assert 12_000 % ppo.rollout_size == 0
        check_comparable(DDQNConfig(), ppo, TrainBudget())
    check_comparable(DDQNConfig(), PPOConfig())  # without a budget only the network spec is compared (as before)


def test_each_of_the_two_values_is_checked():
    only_validation = PPOConfig(n_envs=5, n_steps=5000, minibatch_size=1000)  # 25,000 divides 300,000 but not 12,000
    with pytest.raises(ValueError, match="val_every_env_steps") as exc:
        check_budget_matches_rollout(TrainBudget(), only_validation.rollout_size)
    assert "total_env_steps" not in str(exc.value).split("Alternatively")[0]
    only_total = TrainBudget(total_env_steps=301_000, val_every_env_steps=12_000)  # 1,000 divides both, 12,000 divides only one
    check_budget_matches_rollout(only_total, 1000)
    with pytest.raises(ValueError, match="total_env_steps"):
        check_budget_matches_rollout(only_total, 12_000)


def test_the_cli_and_the_run_start_apply_the_rule_to_the_frozen_budget(tmp_path):
    big = ["--set", "n_envs=8", "--set", "n_steps=256", "--set", "minibatch_size=256", "--out-dir", str(tmp_path)]
    with pytest.raises(ValueError, match="multiple"):
        resolve_setup("ppo", big)
    resolve_setup("ddqn", ["--out-dir", str(tmp_path)])  # DDQN steps one transition at a time: any budget fits
    # a setup that bypasses the CLI is refused at the start of the run, before anything is written
    ok = resolve_setup("ppo", ["--out-dir", str(tmp_path)])
    bad = dataclasses.replace(ok, cfg=PPOConfig(n_envs=8, n_steps=256, minibatch_size=256))
    with pytest.raises(ValueError, match="multiple"):
        TrainingRun(bad, RandomTrainer, quiet=True).run()
    assert not any(tmp_path.iterdir())
    # debug / test budgets (explicit overrides, never used for results) only need a total that fits
    debug = ["--set", "n_envs=3", "--set", "n_steps=10", "--set", "minibatch_size=10", "--out-dir", str(tmp_path)]
    resolve_setup("ppo", [*debug, "--total-steps", "90", "--val-every", "40"])
    with pytest.raises(ValueError, match="multiple"):
        resolve_setup("ppo", [*debug, "--total-steps", "100", "--val-every", "50"])
    check_budget_fits_trainer("ddqn", DDQNConfig(), TrainBudget(), debug_budget=False)


# ---------------------------------------------------------------------------------------------
# steps_to_plateau
# ---------------------------------------------------------------------------------------------
def _curve(values, metric="return_balanced"):
    n = len(values)
    steps = [1000 * (i + 1) for i in range(n)]
    return pd.DataFrame({"val_idx": range(n), "env_step": steps, "train_wall_s": [s / 10 for s in steps], metric: values})


def test_plateau_constants_are_the_pre_registered_ones():
    assert PLATEAU_WINDOW == 5 and PLATEAU_TOL == 0.10


def test_plateau_of_a_clean_step_curve_is_found_at_the_step():
    log = _curve([0.0] * 10 + [10.0] * 15)  # 25 points, the step is at index 10 (env step 11,000)
    r = steps_to_plateau(log, "return_own", reward="balanced")
    assert r["env_step"] == 11_000 and r["val_idx"] == 10 and r["train_wall_s"] == pytest.approx(1100.0)
    assert r["censored"] is False and r["final_level"] == pytest.approx(10.0) and r["improvement"] == pytest.approx(10.0)
    # the point before the step is outside the band: its window is (0 + 4 x 10) / 5 = 8 which is 2.0 away, band is 1.0
    shifted = steps_to_plateau(_curve([0.0] * 9 + [10.0] * 16), "return_own", reward="balanced")
    assert shifted["env_step"] == 10_000


def test_plateau_of_a_noisy_flat_curve_after_a_rise_ignores_the_noise():
    # noise of +-0.5 is far smaller than the band (10% of the improvement = 1.0) after smoothing
    values = [0.0] * 5 + [10.0 + (0.5 if i % 2 else -0.5) for i in range(5, 20)]
    r = steps_to_plateau(_curve(values), "return_own", reward="balanced")
    # last 4 points (indices 16..19) = 9.5, 10.5, 9.5, 10.5 -> final 10.0; the window starting at index 4 is
    # (0 + 10.5 + 9.5 + 10.5 + 9.5) / 5 = 8.0, 2.0 away; the window starting at index 5 is 10.1
    assert r["final_level"] == pytest.approx(10.0)
    assert r["env_step"] == 6_000 and r["censored"] is False


def test_a_curve_that_only_rises_in_the_last_fifth_is_censored():
    values = [0.0] * 20 + [1.0, 2.0, 4.0, 8.0, 16.0]  # final = mean of the last 5 = 6.2
    r = steps_to_plateau(_curve(values), "return_own", reward="balanced")
    assert r["env_step"] == 21_000 and r["censored"] is True  # the only window within 0.62 of 6.2 starts at index 20


def test_a_curve_without_improvement_gives_nan_and_censored():
    flat = steps_to_plateau(_curve([5.0, 4.0, 5.0, 4.0, 5.0, 4.0, 5.0, 4.0, 5.0, 4.0]), "return_own", reward="balanced")
    assert math.isnan(flat["env_step"]) and math.isnan(flat["train_wall_s"]) and flat["censored"] is True
    assert flat["improvement"] < 0
    down = steps_to_plateau(_curve(list(np.linspace(10, 0, 20))), "return_own", reward="balanced")
    assert math.isnan(down["env_step"]) and down["censored"] is True


def test_a_curve_whose_last_window_leaves_the_band_has_no_plateau():
    # 30 points: the final level is the mean of the last 6 values (0, 20, 0, 20, 0, 20) = 10, the improvement 10 (band 1.0),
    # but the last complete window (indices 25..29 = 20, 0, 20, 0, 20) averages 12: outside the band, so no point qualifies
    values = [0.0] * 24 + [0.0, 20.0, 0.0, 20.0, 0.0, 20.0]
    r = steps_to_plateau(_curve(values), "return_own", reward="balanced")
    assert r["final_level"] == pytest.approx(10.0) and r["improvement"] == pytest.approx(10.0)
    assert math.isnan(r["env_step"]) and r["censored"] is True


def test_plateau_metric_selection_validation_and_short_curves():
    log = _curve([0.0] * 10 + [10.0] * 15, metric="return_safety")
    assert steps_to_plateau(log, "return_own", reward="safety")["env_step"] == 11_000  # the run's own reward column
    assert steps_to_plateau(log, "return_safety")["env_step"] == 11_000  # or a column by name
    with pytest.raises(ValueError, match="reward"):
        steps_to_plateau(log, "return_own")
    with pytest.raises(ValueError, match="no column"):
        steps_to_plateau(log, "return_own", reward="balanced")
    assert math.isnan(steps_to_plateau(_curve([0.0, 1.0, 2.0]), "return_balanced")["env_step"])  # fewer points than the window
    shuffled = log.sample(frac=1.0, random_state=0)
    assert steps_to_plateau(shuffled, "return_safety")["env_step"] == 11_000  # ordered by env_step first
    with pytest.raises(ValueError):
        steps_to_plateau(log, "return_safety", tol=0.0)


# ---------------------------------------------------------------------------------------------
# cost tables: wall times grouped by machine
# ---------------------------------------------------------------------------------------------
def _run(machine, seed, speed_scale, rises_at=10, hostless=False):
    """(val_log, run_meta) of a hand-built run: an own-return step curve plus crash rate / speed that reach both levels."""
    n = 25
    steps = [12_000 * (i + 1) for i in range(n)]
    good = [i >= rises_at for i in range(n)]
    log = pd.DataFrame({
        "val_idx": range(n), "env_step": steps, "train_wall_s": [s / (100 * speed_scale) for s in steps],
        "return_balanced": [10.0 if g else 0.0 for g in good],
        "crash_rate": [0.1 if g else 0.9 for g in good], "mean_speed": [27.0 if g else 22.0 for g in good],
    })
    repro = {} if hostless else {"hostname": machine, "cpu_model": "cpu-" + machine}
    meta = {"reproducibility": repro, "timing": {"final_env_step": 300_000, "train_steps_per_s": 100.0 * speed_scale,
                                                  "update_time_s": 300.0 * speed_scale},
            "inference": {"median_ms": 0.5 * speed_scale}}
    return log, meta


def test_wall_time_columns_are_grouped_by_machine_and_never_pooled():
    runs = {
        ("ddqn", "balanced", "main", 0): _run("pcA", 0, 1.0),
        ("ddqn", "balanced", "main", 1): _run("pcA", 1, 1.0, rises_at=12),
        ("ddqn", "balanced", "main", 2): _run("pcA", 2, 1.0),
        ("ddqn", "balanced", "main", 3): _run("pcB", 3, 2.0),  # twice as fast machine
        ("ddqn", "balanced", "main", 4): _run("pcB", 4, 2.0, rises_at=14),
    }
    per_model = cost_per_model(runs)
    assert list(per_model["machine"]) == ["pcA", "pcA", "pcA", "pcB", "pcB"]
    assert per_model["steps_to_plateau"].tolist() == [132_000, 156_000, 132_000, 132_000, 180_000]
    across = cost_across_seeds(per_model)
    wall = across[across["metric"] == "train_steps_per_s"]
    assert sorted(wall["machine"]) == ["pcA", "pcB"]  # one row per machine, no pooled row
    by_machine = wall.set_index("machine")
    assert by_machine.loc["pcA", "mean"] == pytest.approx(100.0) and by_machine.loc["pcB", "mean"] == pytest.approx(200.0)
    assert by_machine.loc["pcA", "n_seeds"] == 3 and by_machine.loc["pcB", "n_seeds"] == 2
    assert math.isnan(by_machine.loc["pcB", "ci_low"]) is False  # 2 seeds on the machine: an interval exists
    for metric in ("train_wall_s_total", "wall_s_to_plateau", "wall_s_to_strong", "inference_median_ms", "update_time_s_per_env_step"):
        assert ALL_MACHINES not in set(across[across["metric"] == metric]["machine"]), metric
        assert set(across[across["metric"] == metric]["machine"]) == {"pcA", "pcB"}, metric
    inference = across[across["metric"] == "inference_median_ms"].set_index("machine")["mean"]
    assert inference["pcA"] == pytest.approx(0.5) and inference["pcB"] == pytest.approx(1.0)  # scale is 2.0 -> 1.0 ms here


def test_env_step_columns_are_pooled_over_machines_with_the_number_of_seeds_that_reached_them():
    never = _run("pcB", 4, 2.0, rises_at=99)  # never rises: no plateau (no improvement), no target level
    runs = {("ppo", "balanced", "main", 0): _run("pcA", 0, 1.0),
            ("ppo", "balanced", "main", 1): _run("pcA", 1, 1.0, rises_at=12),
            ("ppo", "balanced", "main", 2): _run("pcB", 2, 2.0, rises_at=14),
            ("ppo", "balanced", "main", 3): never}
    across = cost_across_seeds(cost_per_model(runs))
    plateau = across[across["metric"] == "steps_to_plateau"]
    assert list(plateau["machine"]) == [ALL_MACHINES]  # hardware independent: one pooled row
    row = plateau.iloc[0]
    assert row["n_runs"] == 4 and row["n_seeds"] == 3  # the run that never improved has NaN and is not counted
    assert row["mean"] == pytest.approx(np.mean([132_000, 156_000, 180_000]))
    assert row["n_censored"] == 0
    strong = across[across["metric"] == "steps_to_strong"].iloc[0]
    # k = 3: the first of three consecutive points that hold; rises at 10 / 12 / 14 -> 132,000 / 156,000 / 180,000
    assert strong["n_seeds"] == 3 and strong["mean"] == pytest.approx(np.mean([132_000, 156_000, 180_000]))
    assert across[across["metric"] == "wall_s_to_strong"]["machine"].tolist() == ["pcA", "pcB"]


def test_gradient_updates_and_samples_per_env_step_are_hardware_independent_and_algorithm_neutral():
    def with_counts(machine, seed, n_updates, n_samples):
        log, meta = _run(machine, seed, 1.0)
        meta["timing"].update(n_updates=n_updates, n_gradient_samples=n_samples)
        return log, meta

    runs = {  # 300,000 env steps per run
        ("ddqn", "balanced", "main", 0): with_counts("pcA", 0, 300_000, 300_000 * 64),  # 1 step per env step, batch 64
        ("ddqn", "balanced", "main", 1): with_counts("pcA", 1, 150_000, 150_000 * 64),  # train_freq 2
        ("ppo", "balanced", "main", 0): with_counts("pcB", 0, 30_000, 3_000_000),  # 300 rollouts x 100 steps of 100
        ("ppo", "balanced", "main", 1): with_counts("pcB", 1, 27_000, 2_700_000),  # target-KL stopped some epochs
        ("ppo", "balanced", "main", 2): with_counts("pcB", 2, 30_000, None),  # samples not recorded -> NaN, not 0
    }
    per_model = cost_per_model(runs).set_index(["algorithm", "train_seed"])
    assert per_model.loc[("ddqn", 0), "gradient_updates_per_env_step"] == pytest.approx(1.0)
    assert per_model.loc[("ddqn", 0), "samples_per_env_step"] == pytest.approx(64.0)
    assert per_model.loc[("ddqn", 1), "gradient_updates_per_env_step"] == pytest.approx(0.5)
    assert per_model.loc[("ppo", 0), "gradient_updates_per_env_step"] == pytest.approx(0.1)
    assert per_model.loc[("ppo", 0), "samples_per_env_step"] == pytest.approx(10.0)  # = n_epochs
    assert per_model.loc[("ppo", 1), "samples_per_env_step"] == pytest.approx(9.0)
    assert math.isnan(per_model.loc[("ppo", 2), "samples_per_env_step"])
    across = cost_across_seeds(cost_per_model(runs))
    for metric in ("gradient_updates_per_env_step", "samples_per_env_step"):
        rows = across[across["metric"] == metric]
        assert set(rows["machine"]) == {ALL_MACHINES}, metric  # pooled: compared between DDQN and PPO
    ppo_samples = across[(across["metric"] == "samples_per_env_step") & (across["algorithm"] == "ppo")].iloc[0]
    assert ppo_samples["n_runs"] == 3 and ppo_samples["n_seeds"] == 2 and ppo_samples["mean"] == pytest.approx(9.5)
    # a run_meta without the counters (e.g. written before they existed) gives NaN, never a crash
    log, meta = _run("pcA", 0, 1.0)
    old = cost_per_model({("ddqn", "balanced", "main", 0): (log, meta)}).iloc[0]
    assert math.isnan(old["gradient_updates_per_env_step"]) and math.isnan(old["samples_per_env_step"])


def test_runs_without_a_recorded_machine_are_grouped_as_unknown_not_pooled_with_others():
    runs = {("ddqn", "balanced", "main", 0): _run("pcA", 0, 1.0),
            ("ddqn", "balanced", "main", 1): _run("", 1, 2.0, hostless=True)}
    across = cost_across_seeds(cost_per_model(runs))
    assert sorted(set(across[across["metric"] == "train_steps_per_s"]["machine"])) == ["pcA", "unknown"]


# ---------------------------------------------------------------------------------------------
# machine identifier and OneDrive
# ---------------------------------------------------------------------------------------------
# A neutral fake root: the pytest temp dir is named after the test and would itself contain "onedrive".
BASE = Path(os.sep + "fakehome")


def test_is_onedrive_path_by_name_and_by_environment_variable():
    f = run_meta.is_onedrive_path
    assert f(BASE / "OneDrive" / "Desktop" / "project", {})
    assert f(BASE / "OneDrive - Some University" / "project", {})
    assert f(BASE / "onedrive" / "project", {})  # case-insensitive
    assert not f(BASE / "Documents" / "project", {})
    # a synced folder with another name, announced by the OneDrive environment variables
    sync = BASE / "Sync"
    assert f(sync / "project" / "results", {"OneDrive": str(sync)})
    assert f(sync, {"OneDriveCommercial": str(sync) + os.sep})  # the folder itself, with a trailing separator
    assert not f(BASE / "Syncing" / "project", {"OneDrive": str(sync)})  # a longer name is not inside it
    assert not f(BASE / "Elsewhere", {"OneDrive": str(sync), "OneDriveConsumer": ""})


def test_reproducibility_info_records_hostname_cpu_and_onedrive_flag(monkeypatch):
    monkeypatch.setattr(run_meta, "ROOT", BASE / "project")
    for name in run_meta.ONEDRIVE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    info = run_meta.reproducibility_info("cpu", 1, BASE / "results")
    assert info["hostname"] == run_meta.hostname() and info["hostname"] != "unknown"
    assert isinstance(info["cpu_model"], str) and info["cpu_model"]
    assert info["onedrive_path"] is False
    monkeypatch.setenv("OneDrive", str(BASE / "sync"))
    assert run_meta.reproducibility_info("cpu", 1, BASE / "sync" / "results")["onedrive_path"] is True  # output dir
    assert run_meta.reproducibility_info("cpu", 1, BASE / "results")["onedrive_path"] is False
    monkeypatch.setattr(run_meta, "ROOT", BASE / "sync" / "project")
    assert run_meta.reproducibility_info("cpu", 1)["onedrive_path"] is True  # project directory
