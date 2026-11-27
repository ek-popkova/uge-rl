"""Supplementary runs (``supp-<name>``) and supplementary reward presets: registry, roles, base-run check, aggregation.

Fast tier only (no environment is created). The registry and the reward registry are module-level dicts; tests that add an
entry use ``monkeypatch.setitem`` so the change is undone afterwards.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from common import config as config_mod
from common import train_config as tc
from common.aggregate import check_sensitivity_runs, check_supplementary_runs, per_model_table, steps_to_plateau
from common.config import REWARDS, SUPPLEMENTARY_REWARDS, TRAIN_SEEDS, check_supplementary_rewards, reward_weights
from common.metrics import recompute_return
from common.paths import (
    MAIN_VARIANT,
    is_supplementary_variant,
    load_eval_episodes,
    run_meta_path,
    save_eval,
    supplementary_variant,
    variant_role,
)
from common.run_meta import new_meta, write_meta
from common.train_config import (
    CONFIG_CLASSES,
    SUPPLEMENTARY_RUNS,
    DDQNConfig,
    FromBase,
    PPOConfig,
    SupplementarySpec,
    TrainBudget,
    apply_overrides,
    check_run_roles,
    check_supplementary_role,
    check_supplementary_run,
    expand_sensitivity_grid,
    expand_supplementary_runs,
    validate_supplementary_runs,
)
from common.training import TrainingRun, resolve_setup

# A valid supplementary reward for the tests: G_succ = 15 > G_slow = 3 > G_crash = -5.5 (indifference_point is defined).
CAUTIOUS = {"w_speed": 0.4, "w_right": 0.1, "w_collision": 10.0, "w_lane": 0.0}


def _spec(**kw):
    base = dict(algo="ppo", overrides={"ent_coef": 0.05}, rewards=("balanced",), seeds=(0, 1, 2), purpose="test")
    return SupplementarySpec(**{**base, **kw})


def _roles(variant, *, algo="ppo", seed=0, reward="balanced", density="medium", overrides=None, budget=()):
    check_run_roles(variant, seed, reward, density, overrides or {}, list(budget), algo=algo)


# ---- the registry -----------------------------------------------------------------------------------------------------
def test_registry_holds_the_decided_ppo_entries_and_no_ddqn_entry():
    assert set(SUPPLEMENTARY_RUNS) == {"ppo-kl", "a2c"}
    assert all(spec.algo == "ppo" for spec in SUPPLEMENTARY_RUNS.values())  # the registry works for DDQN, but has no entry
    kl, a2c = SUPPLEMENTARY_RUNS["ppo-kl"], SUPPLEMENTARY_RUNS["a2c"]
    assert kl.resolved_overrides() == {"policy_loss": "kl"} and kl.rewards == ("balanced",) and kl.seeds == (0, 1, 2)
    ppo = PPOConfig()
    assert a2c.resolved_overrides() == {"minibatch_size": ppo.rollout_size, "n_epochs": 1}  # one full-batch step per rollout
    assert a2c.rewards == ("balanced",) and a2c.seeds == (0, 1, 2)
    cfg = apply_overrides(ppo, a2c.resolved_overrides())
    assert cfg.n_epochs * (cfg.rollout_size // cfg.minibatch_size) == 1
    assert SUPPLEMENTARY_REWARDS == {}  # experiment (b): no supplementary reward yet
    validate_supplementary_runs(SUPPLEMENTARY_RUNS)


def test_from_base_follows_the_defaults_chosen_after_tuning():
    a2c = SUPPLEMENTARY_RUNS["a2c"]
    tuned = PPOConfig(n_envs=8, n_steps=250, minibatch_size=200)
    assert a2c.resolved_overrides(tuned) == {"minibatch_size": 2000, "n_epochs": 1}
    runs = expand_supplementary_runs(configs={"ppo": tuned})
    assert {r.overrides["minibatch_size"] for r in runs if r.name == "a2c"} == {2000}


def test_expand_supplementary_runs_and_cli_round_trip(tmp_path):
    runs = expand_supplementary_runs()
    assert len(runs) == 6  # 2 entries x 1 reward x 3 seeds
    assert {(r.variant, r.seed) for r in runs} == {(v, s) for v in ("supp-ppo-kl", "supp-a2c") for s in (0, 1, 2)}
    for run in runs:
        setup = resolve_setup(run.algo, [*run.cli_args(), "--out-dir", str(tmp_path)])
        assert setup.variant == run.variant and setup.seed == run.seed and setup.reward == run.reward
        assert setup.budget == TrainBudget() and not setup.budget_overrides  # the frozen budget
        assert {k: getattr(setup.cfg, k) for k in run.overrides} == dict(run.overrides)
        assert setup.identity()["role"] == "supplementary"
    kl = next(r for r in runs if r.name == "ppo-kl")
    assert resolve_setup("ppo", [*kl.cli_args(), "--out-dir", str(tmp_path)]).cfg.policy_loss == "kl"


def test_adding_an_entry_is_one_line_and_works_for_ddqn(monkeypatch, tmp_path):
    monkeypatch.setitem(SUPPLEMENTARY_RUNS, "ddqn-soft", SupplementarySpec("ddqn", {"tau": 0.005}, ("safety",), (3, 4), "soft"))
    validate_supplementary_runs(SUPPLEMENTARY_RUNS)
    runs = [r for r in expand_supplementary_runs() if r.name == "ddqn-soft"]
    assert [(r.reward, r.seed) for r in runs] == [("safety", 3), ("safety", 4)]
    setup = resolve_setup("ddqn", [*runs[0].cli_args(), "--out-dir", str(tmp_path)])
    assert setup.variant == "supp-ddqn-soft" and setup.cfg.tau == 0.005
    with pytest.raises(ValueError, match="algorithm"):  # a DDQN entry is not a PPO run
        _roles("supp-ddqn-soft", algo="ppo", reward="safety", seed=3, overrides={"ent_coef": 0.05})


@pytest.mark.parametrize("kw, match", [
    ({"overrides": {"no_such_field": 1}}, "no field"),
    ({"overrides": {"gamma": 0.995}}, "shared"),
    ({"overrides": {"hidden_sizes": (64, 64)}}, "shared"),
    ({"overrides": {"ent_coef": PPOConfig().ent_coef}}, "base values"),
    ({"overrides": {"minibatch_size": 300}}, "invalid config"),  # does not divide the rollout
    ({"overrides": {"n_steps": 256}}, "rollout size"),  # 4 x 256 = 1,024 divides neither 300,000 nor 12,000
    ({"overrides": {"minibatch_size": FromBase("no_such_attr")}}, "FromBase"),
    ({"overrides": {}}, "main run"),  # no overrides on a REWARDS preset
    ({"rewards": ()}, "rewards"),
    ({"rewards": ("balanced", "balanced")}, "rewards"),
    ({"rewards": ("no_such_reward",)}, "unknown reward"),
    ({"seeds": (0, 7)}, "TRAIN_SEEDS"),
    ({"seeds": (99,)}, "TRAIN_SEEDS"),
    ({"seeds": ()}, "seeds"),
    ({"purpose": " "}, "purpose"),
    ({"algo": "sac"}, "unknown algorithm"),
    ({"overrides": {"clip_range": 0.1}}, "sensitivity point"),  # a sensitivity run under another name
])
def test_registry_validation_refuses_invalid_entries(kw, match):
    with pytest.raises(ValueError, match=match):
        validate_supplementary_runs({"x": _spec(**kw)})


def test_registry_validation_refuses_bad_names():
    for name in ("", "supp-x", "tune-x", "a/b"):
        with pytest.raises(ValueError):
            validate_supplementary_runs({name: _spec()})
    validate_supplementary_runs({"x": _spec(overrides={"clip_range": 0.1}, rewards=("safety",))})  # not on the sensitivity reward


# ---- roles: every result run is registered ------------------------------------------------------------------------------
def test_registered_supplementary_runs_pass_and_every_mismatch_is_refused():
    kl = {"policy_loss": "kl"}
    _roles("supp-ppo-kl", overrides=kl)
    for kw, match in (
        ({"variant": "supp-unknown"}, "not registered"),
        ({"algo": "ddqn"}, "algorithm"),
        ({"overrides": {}}, "exactly the overrides"),
        ({"overrides": {**kl, "ent_coef": 0.05}}, "exactly the overrides"),  # an extra override
        ({"overrides": {"policy_loss": "clip"}}, "exactly the overrides"),
        ({"seed": 3}, "seed"),
        ({"reward": "safety"}, "reward"),
        ({"density": "low"}, "density"),
        ({"budget": ["total_env_steps"]}, "budget"),
    ):
        args = {"variant": "supp-ppo-kl", "overrides": kl, **kw}
        with pytest.raises(ValueError, match=match):
            _roles(args.pop("variant"), **args)
    with pytest.raises(ValueError, match="reserved"):  # the tuning seed is never a supplementary seed
        _roles("supp-ppo-kl", overrides=kl, seed=99)


def test_unregistered_variants_are_refused_and_only_grid_points_are_sensitivity_runs(tmp_path):
    point = next(r for r in expand_sensitivity_grid() if r.algo == "ppo" and r.param == "clip_range")
    _roles(point.variant, overrides=point.overrides, seed=point.seed)  # a grid point
    for kw in ({"overrides": {"ent_coef": 0.05}, "variant": "ent_coef=0.05"},  # not a grid parameter
               {"overrides": {"clip_range": 0.15}, "variant": "clip_range=0.15"},  # not a grid value
               {"overrides": point.overrides, "variant": "my-name"},  # a grid point under a custom name
               {"overrides": {}, "variant": "foo"}):  # a custom name without overrides
        with pytest.raises(ValueError, match="not a registered run"):
            _roles(kw.pop("variant"), **kw)
    for kw, match in (({"seed": 3}, "seed"), ({"reward": "safety"}, "reward"), ({"density": "low"}, "density")):
        with pytest.raises(ValueError, match=match):
            _roles(point.variant, overrides=point.overrides, **kw)
    with pytest.raises(ValueError, match="no hyperparameter overrides"):
        _roles(MAIN_VARIANT, overrides={"ent_coef": 0.05})
    _roles("ent_coef=0.05", overrides={"ent_coef": 0.05}, budget=["total_env_steps"])  # a debug run (budget overrides)
    out = ["--out-dir", str(tmp_path)]
    for argv in (["--set", "ent_coef=0.05"], ["--variant", "my-run"], ["--variant", "supp-x", "--set", "ent_coef=0.05"],
                 ["--set", "policy_loss=kl"]):  # PPO-KL is the registered supp-ppo-kl, not an auto-named variant
        with pytest.raises(ValueError):
            resolve_setup("ppo", [*argv, *out])


def test_check_supplementary_role_with_an_explicit_registry():
    reg = {"x": _spec(overrides={"ent_coef": 0.05})}
    check_supplementary_role("supp-x", "ppo", 1, "balanced", "medium", {"ent_coef": 0.05}, [], registry=reg)
    with pytest.raises(ValueError, match="not registered"):
        check_supplementary_role("supp-y", "ppo", 1, "balanced", "medium", {"ent_coef": 0.05}, [], registry=reg)


def test_variant_role_and_prefix():
    assert supplementary_variant("ppo-kl") == "supp-ppo-kl" and is_supplementary_variant("supp-ppo-kl")
    assert [variant_role(v) for v in ("main", "tune-a", "supp-a2c", "clip_range=0.1")] == [
        "main", "tuning", "supplementary", "sensitivity"]


# ---- supplementary reward presets (experiment (b)) ---------------------------------------------------------------------
def test_supplementary_reward_validation():
    check_supplementary_rewards({"cautious": CAUTIOUS})
    for presets, match in (({"balanced": CAUTIOUS}, "shadows"),
                           ({"x": {"w_speed": 0.4}}, "exactly"),
                           ({"x": {**CAUTIOUS, "w_lane": -0.1}}, ">= 0"),
                           ({"x": {**CAUTIOUS, "w_collision": float("inf")}}, "finite"),
                           ({"x": {**CAUTIOUS, "w_collision": 0.0}}, "ordering"),  # a crash would beat slow driving
                           ({"a/b": CAUTIOUS}, "name")):
        with pytest.raises(ValueError, match=match):
            check_supplementary_rewards(presets)


def test_supplementary_reward_only_in_a_registered_supplementary_run(monkeypatch, tmp_path):
    monkeypatch.setitem(SUPPLEMENTARY_REWARDS, "cautious", CAUTIOUS)
    assert reward_weights("cautious") == CAUTIOUS and "cautious" not in REWARDS
    for variant, kw in (("main", {}), ("tune-a", {"seed": 99}), ("lr=0.0001", {"overrides": {"lr": 0.0001}})):
        with pytest.raises(ValueError, match="reward"):
            _roles(variant, reward="cautious", **kw)
    with pytest.raises(ValueError, match="SUPPLEMENTARY_REWARDS"):  # not even as a debug run
        _roles("main", reward="cautious", budget=["total_env_steps"])
    with pytest.raises(ValueError, match="reward"):  # registered, but the entry does not list the reward
        _roles("supp-ppo-kl", reward="cautious", overrides={"policy_loss": "kl"})
    spec = SupplementarySpec("ppo", {}, ("cautious",), (0, 1), "experiment (b): a cautious reward, default hyperparameters")
    validate_supplementary_runs({"cautious": spec})  # no overrides is allowed for a supplementary reward
    monkeypatch.setitem(SUPPLEMENTARY_RUNS, "cautious", spec)
    run = next(r for r in expand_supplementary_runs() if r.name == "cautious")
    setup = resolve_setup("ppo", [*run.cli_args(), "--out-dir", str(tmp_path)])
    assert setup.reward == "cautious" and setup.run_dir == run_meta_path("ppo", "cautious", 0, tmp_path, "supp-cautious").parent
    with pytest.raises(SystemExit):  # an undeclared reward name is not even a CLI choice
        resolve_setup("ppo", ["--reward", "nope", "--out-dir", str(tmp_path)])


def test_supplementary_reward_weights_reach_the_metrics(monkeypatch):
    monkeypatch.setitem(SUPPLEMENTARY_REWARDS, "cautious", CAUTIOUS)
    df = pd.DataFrame({"return_env": [1.0], "sum_high_speed": [10.0], "sum_right_lane": [20.0], "crashed": [True],
                       "lane_changes": [3]})
    assert recompute_return(df, "cautious").iloc[0] == pytest.approx(0.4 * 10 + 0.1 * 20 - 10.0)
    with pytest.raises(ValueError, match="Unknown reward"):
        recompute_return(df, "nope")


# ---- base-run check ----------------------------------------------------------------------------------------------------
def _meta(algo="ppo", *, variant=MAIN_VARIANT, reward="balanced", seed=0, overrides=None, budget=None, hp_changes=None):
    cfg = apply_overrides(CONFIG_CLASSES[algo](), overrides or {})
    meta = new_meta({"algorithm": algo, "train_reward": reward, "train_density": "medium", "seed": seed,
                     "variant": variant, "overridden_keys": sorted(overrides or {})})
    meta["hyperparameters"] = {**cfg.to_dict(), **(hp_changes or {})}
    meta["budget"] = (budget or TrainBudget()).to_dict()
    return meta


def _supp_meta(run, **kw):
    return _meta(run.algo, variant=run.variant, reward=run.reward, seed=run.seed, overrides=run.overrides, **kw)


def test_every_planned_supplementary_run_passes_against_an_untouched_main_run():
    for run in expand_supplementary_runs():
        check_supplementary_run(_supp_meta(run), _meta(run.algo, reward=run.reward, seed=run.seed))


def test_check_supplementary_run_refuses_any_other_difference():
    a2c = next(r for r in expand_supplementary_runs() if r.name == "a2c" and r.seed == 1)
    main = _meta(seed=1)
    for bad, match in (
        (_supp_meta(a2c, hp_changes={"ent_coef": 0.02}), "ent_coef"),
        (_supp_meta(a2c, hp_changes={"n_epochs": 2}), "n_epochs"),  # declared value not held
        (_supp_meta(a2c, budget=TrainBudget(total_env_steps=60_000)), "budget"),
    ):
        with pytest.raises(ValueError, match=match):
            check_supplementary_run(bad, main)
    with pytest.raises(ValueError, match="seed"):
        check_supplementary_run(_supp_meta(a2c), _meta(seed=2))
    with pytest.raises(ValueError, match="main variant"):
        check_supplementary_run(_supp_meta(a2c), _meta(seed=1, variant="clip_range=0.1"))
    # FromBase is resolved against the MAIN run: a main grid trained with 8 envs makes the a2c minibatch 2,000
    tuned_main = _meta(seed=1, hp_changes={"n_envs": 8})
    with pytest.raises(ValueError, match="minibatch_size"):
        check_supplementary_run(_supp_meta(a2c, hp_changes={"n_envs": 8}), tuned_main)
    check_supplementary_run(_supp_meta(a2c, hp_changes={"n_envs": 8, "minibatch_size": 2000}), tuned_main)
    with pytest.raises(ValueError, match="not a registered"):
        check_supplementary_run(_meta(variant="supp-nope"), main)


def test_supplementary_reward_run_is_compared_with_the_balanced_main_run(monkeypatch):
    monkeypatch.setitem(SUPPLEMENTARY_REWARDS, "cautious", CAUTIOUS)
    monkeypatch.setitem(SUPPLEMENTARY_RUNS, "cautious", SupplementarySpec("ppo", {}, ("cautious",), (0,), "b"))
    supp = _meta(variant="supp-cautious", reward="cautious")
    supp["env"] = {"env_id": "highway-fast-v0", "config": {"a": 1}, "reward_weights": CAUTIOUS}
    main = _meta(reward="balanced")
    main["env"] = {"env_id": "highway-fast-v0", "config": {"a": 1}, "reward_weights": REWARDS["balanced"]}
    check_supplementary_run(supp, main)  # the reward weights differ by design
    with pytest.raises(ValueError, match="reward 'balanced'"):
        check_supplementary_run(supp, _meta(reward="safety"))
    supp["env"]["config"] = {"a": 2}
    with pytest.raises(ValueError, match="env.config"):
        check_supplementary_run(supp, main)


def test_check_supplementary_runs_reads_files_and_sensitivity_check_skips_them(tmp_path):
    kl = next(r for r in expand_supplementary_runs() if r.name == "ppo-kl" and r.seed == 2)
    path = run_meta_path("ppo", "balanced", 2, tmp_path, kl.variant)
    write_meta(path, _supp_meta(kl))
    with pytest.raises(FileNotFoundError):  # the main run that provides the base point is missing
        check_supplementary_runs(tmp_path)
    write_meta(run_meta_path("ppo", "balanced", 2, tmp_path), _meta(seed=2))
    assert check_supplementary_runs(tmp_path) == [path]
    assert check_sensitivity_runs(tmp_path) == []  # a supp-* run under the sensitivity reward is not a sensitivity run
    write_meta(path, _supp_meta(kl, hp_changes={"vf_coef": 1.0}))
    with pytest.raises(ValueError, match="vf_coef"):
        check_supplementary_runs(tmp_path)


def test_training_run_checks_a_supplementary_run_against_an_existing_main_run(tmp_path):
    kl = next(r for r in expand_supplementary_runs() if r.name == "ppo-kl" and r.seed == 0)
    tr = TrainingRun(resolve_setup("ppo", [*kl.cli_args(), "--out-dir", str(tmp_path)]), object, quiet=True)
    tr._check_against_main_run(_supp_meta(kl))  # no main run yet
    write_meta(run_meta_path("ppo", "balanced", 0, tmp_path), _meta())
    tr._check_against_main_run(_supp_meta(kl))
    write_meta(run_meta_path("ppo", "balanced", 0, tmp_path), _meta(hp_changes={"gae_lambda": 0.9}))
    with pytest.raises(ValueError, match="gae_lambda"):
        tr._check_against_main_run(_supp_meta(kl))


# ---- aggregation: never in a DDQN vs PPO table --------------------------------------------------------------------------
def _episodes(n=4):
    return pd.DataFrame([
        dict(seed=10000 + i, episode_return=1.0, return_env=1.0, sum_high_speed=10.0, sum_right_lane=5.0, lane_changes=1,
             crashed=i % 2 == 0, completed=i % 2 == 1, near_miss=False, length=30, mean_speed=25.0, distance_m=750.0,
             min_gap=10.0, min_ttc=3.0)
        for i in range(n)
    ])


def test_supplementary_runs_and_rewards_are_excluded_from_the_comparison_tables(monkeypatch, tmp_path):
    monkeypatch.setitem(SUPPLEMENTARY_REWARDS, "cautious", CAUTIOUS)
    save_eval(_episodes(), "ppo", "balanced", 0, "main", tmp_path)
    save_eval(_episodes(), "ppo", "balanced", 0, "main", tmp_path, variant="supp-ppo-kl")
    save_eval(_episodes(), "ppo", "cautious", 0, "main", tmp_path, variant="supp-cautious")
    default = load_eval_episodes(tmp_path)
    assert set(default["variant"]) == {"main"} and set(default["role"]) == {"main"}
    both = load_eval_episodes(tmp_path, include_supplementary=True)
    assert set(both["role"]) == {"main", "supplementary"} and set(both["train_reward"]) == {"balanced", "cautious"}
    assert set(per_model_table(both)["variant"]) == {"main"}  # dropped when passed in, unless asked for
    table = per_model_table(both, include_supplementary=True)
    assert set(zip(table["variant"], table["role"])) == {("main", "main"), ("supp-ppo-kl", "supplementary"),
                                                         ("supp-cautious", "supplementary")}
    own = table.loc[table["train_reward"] == "cautious", "return_own"].iloc[0]  # recomputed: no return_cautious column
    assert own == pytest.approx(0.4 * 10 + 0.1 * 5 - 10.0 * 0.5)  # half of the 4 episodes crash
    # a supplementary reward under any variant name is dropped too (defence in depth)
    stray = both.assign(variant=MAIN_VARIANT, role="main")
    assert set(per_model_table(stray)["train_reward"]) == {"balanced"}


def test_steps_to_plateau_of_a_supplementary_reward_uses_episode_return(monkeypatch):
    monkeypatch.setitem(SUPPLEMENTARY_REWARDS, "cautious", CAUTIOUS)
    values = [0.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    log = pd.DataFrame({"val_idx": range(10), "env_step": [1000 * (i + 1) for i in range(10)], "episode_return": values})
    res = steps_to_plateau(log, "return_own", reward="cautious")
    # final 10, improvement 10, band +/- 1: the window starting at 2,000 (mean 9) is the first inside it
    assert res["env_step"] == 2000 and res["final_level"] == 10.0 and not math.isnan(res["improvement"])
    with pytest.raises(ValueError, match="return_balanced"):
        steps_to_plateau(log, "return_own", reward="balanced")


def test_main_and_supplementary_seeds_share_the_train_seeds():
    for spec in SUPPLEMENTARY_RUNS.values():
        assert set(spec.seeds) <= set(TRAIN_SEEDS)
    assert config_mod.SUPPLEMENTARY_REWARDS is SUPPLEMENTARY_REWARDS and tc.SUPPLEMENTARY_RUNS is SUPPLEMENTARY_RUNS
    assert DDQNConfig().tau != 0.005  # the DDQN example entry above really is an override


def test_frozen_budget_values_passed_explicitly_do_not_make_a_debug_run(tmp_path):
    frozen = TrainBudget()
    same = ["--total-steps", str(frozen.total_env_steps), "--val-every", str(frozen.val_every_env_steps),
            "--val-episodes", str(frozen.val_n_episodes), "--checkpoint-every", str(frozen.checkpoint_every_env_steps)]
    out = ["--out-dir", str(tmp_path)]
    for argv in (["--set", "ent_coef=0.05"], ["--variant", "my-run"]):  # unregistered: refused with or without them
        with pytest.raises(ValueError, match="not a registered run"):
            resolve_setup("ppo", [*argv, *same, *out])
    with pytest.raises(ValueError, match="not a registered run"):  # the checkpoint interval alone does not affect results
        resolve_setup("ppo", ["--set", "ent_coef=0.05", "--checkpoint-every", "5000", *out])
    kl = next(r for r in expand_supplementary_runs() if r.name == "ppo-kl")
    setup = resolve_setup("ppo", [*kl.cli_args(), *same, *out])  # a registered run with the frozen values is still valid
    assert setup.budget == frozen and setup.budget_overrides == []
    tune = resolve_setup("ppo", ["--variant", "tune-a", "--seed", "99", "--reward", "balanced", *same, *out])
    assert tune.budget_overrides == []
    debug = resolve_setup("ppo", ["--set", "ent_coef=0.05", "--total-steps", "12000", *same[2:], *out])  # really differs
    assert debug.budget_overrides == ["total_env_steps"]
