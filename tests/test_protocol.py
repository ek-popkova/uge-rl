"""Shared training decisions: frozen values, tuning protocol, sensitivity grid (all fast, no simulation)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from common.aggregate import (
    check_sensitivity_runs,
    per_model_table,
    select_tuning_config,
    tuning_table,
)
from common.config import (
    PILOT_RUN_SEED,
    PILOT_SEEDS,
    SENSITIVITY_SEEDS,
    TEST_SEEDS,
    TRAIN_SEEDS,
    TUNING_MAX_CONFIGS,
    VAL_SEEDS,
)
from common.failure_cases import VIDEO_CATEGORIES, video_episode_seeds
from common.paths import (
    MAIN_VARIANT,
    is_tuning_variant,
    load_eval_episodes,
    load_tuning_val_logs,
    run_dir,
    run_meta_path,
    save_eval,
    tuning_variant,
)
from common.run_meta import new_meta, write_meta
from common.seeding import MAX_ENVS_PER_RUN, derived_env_seeds
from common.train_config import (
    CONFIG_CLASSES,
    SENSITIVITY_GRID,
    SHARED_KEYS,
    DDQNConfig,
    PPOConfig,
    TrainBudget,
    apply_overrides,
    check_comparable,
    check_run_roles,
    check_sensitivity_run,
    expand_sensitivity_grid,
    placeholders_in_use,
    resolve_hyperparameters,
)
from common.training import RunSetup, TrainingRun, resolve_setup


# ---- frozen shared decisions --------------------------------------------------------------------------------------------
def test_shared_decisions_are_frozen_and_equal_for_both_algorithms():
    d, p = DDQNConfig(), PPOConfig()
    for cfg in (d, p):
        assert cfg.gamma == 0.99 and cfg.hidden_sizes == (256, 256) and cfg.activation == "relu" and cfg.obs_flatten is True
    assert p.ppo_shared_trunk is False  # separate actor and critic
    assert TrainBudget().milestone_fractions == (0.1, 0.25, 0.5)
    check_comparable(d, p)
    # none of the decided values is a placeholder any more; the rest still is
    for cls in (DDQNConfig, PPOConfig):
        names = {n.split(".")[1] for n in placeholders_in_use(cls())}
        assert not names & {"gamma", "hidden_sizes", "activation", "obs_flatten", "ppo_shared_trunk", "device"}
        assert {"lr", "max_grad_norm"} <= names
    # the budget is frozen after the calibration (tests/test_frozen_values.py); only algorithm-specific values are placeholders
    assert placeholders_in_use(TrainBudget()) == []
    assert TrainBudget().checkpoint_every_env_steps == 10_000


def test_check_comparable_covers_gamma_hidden_sizes_activation():
    d = DDQNConfig()
    for kwargs, match in (({"gamma": 0.995}, "gamma"), ({"hidden_sizes": (128, 128)}, "hidden_sizes"),
                          ({"activation": "tanh"}, "activation")):
        with pytest.raises(ValueError, match=match):
            check_comparable(d, PPOConfig(**kwargs))
        with pytest.raises(ValueError, match=match):
            check_comparable(DDQNConfig(**kwargs), PPOConfig())


def test_network_spec_is_validated():
    with pytest.raises(ValueError, match="activation"):
        DDQNConfig(activation="gelu")
    with pytest.raises(ValueError, match="obs_flatten"):
        PPOConfig(obs_flatten=False)
    assert {"gamma", "hidden_sizes", "activation", "obs_flatten", "ppo_shared_trunk"} <= SHARED_KEYS


# ---- PILOT_RUN_SEED -----------------------------------------------------------------------------------------------------
def test_pilot_run_seed_is_disjoint_from_every_seed_set_and_derived_env_seeds():
    assert PILOT_RUN_SEED == 99
    for name, seeds in {"train": TRAIN_SEEDS, "sensitivity": SENSITIVITY_SEEDS, "val": VAL_SEEDS, "test": TEST_SEEDS,
                        "pilot": PILOT_SEEDS}.items():
        assert PILOT_RUN_SEED not in seeds, name
    own = set(derived_env_seeds(PILOT_RUN_SEED, MAX_ENVS_PER_RUN))  # every env index a run may use
    assert len(own) == MAX_ENVS_PER_RUN
    plain = set(TRAIN_SEEDS) | set(SENSITIVITY_SEEDS) | set(VAL_SEEDS) | set(TEST_SEEDS) | set(PILOT_SEEDS) | {PILOT_RUN_SEED}
    assert not own & plain
    for run_seed in sorted(set(TRAIN_SEEDS) | set(SENSITIVITY_SEEDS)):
        assert not own & set(derived_env_seeds(run_seed, MAX_ENVS_PER_RUN)), run_seed  # no shared env seed with a result run


# ---- tuning: roles and variants -----------------------------------------------------------------------------------------
def test_tuning_variant_names_and_exclusion_flag():
    assert tuning_variant("lr-high") == "tune-lr-high" and is_tuning_variant("tune-lr-high")
    assert not is_tuning_variant(MAIN_VARIANT) and not is_tuning_variant("lr=0.0003")
    for bad in ("", "tune-x", "a/b"):
        with pytest.raises(ValueError):
            tuning_variant(bad)


def test_check_run_roles():
    ok = dict(variant="tune-a", seed=PILOT_RUN_SEED, reward="balanced", density="medium", overridden=["lr"],
              budget_overrides=[])
    check_run_roles(**ok)
    check_run_roles("main", 0, "safety", "low", [], [])  # an ordinary run is untouched
    for change, match in (({"seed": 0}, "seed"), ({"reward": "safety"}, "reward"), ({"density": "low"}, "density"),
                          ({"overridden": ["gamma"]}, "shared"), ({"overridden": ["lr", "hidden_sizes"]}, "shared"),
                          ({"budget_overrides": ["total_env_steps"]}, "budget")):
        with pytest.raises(ValueError, match=match):
            check_run_roles(**{**ok, **change})
    with pytest.raises(ValueError, match="reserved"):  # run seed 99 never writes into main or a sensitivity variant
        check_run_roles("main", PILOT_RUN_SEED, "balanced", "medium", [], [])


def test_resolve_setup_enforces_the_tuning_roles(tmp_path):
    out = ["--out-dir", str(tmp_path)]
    setup = resolve_setup("ppo", ["--variant", "tune-clip", "--seed", "99", "--reward", "balanced", "--set", "clip_range=0.1", *out])
    assert setup.variant == "tune-clip" and setup.run_dir == run_dir("ppo", "balanced", 99, tmp_path, "tune-clip")
    for argv in (["--variant", "tune-a", "--seed", "0", "--reward", "balanced"],  # wrong seed
                 ["--variant", "tune-a", "--seed", "99", "--reward", "safety"],  # wrong reward
                 ["--variant", "tune-a", "--seed", "99", "--reward", "balanced", "--set", "gamma=0.995"],  # shared key
                 ["--variant", "tune-a", "--seed", "99", "--reward", "balanced", "--total-steps", "5000"],  # budget
                 ["--seed", "99", "--reward", "balanced"]):  # seed 99 outside a tuning variant
        with pytest.raises(ValueError):
            resolve_setup("ddqn", [*argv, *out])


# ---- tuning: selection ----------------------------------------------------------------------------------------------------
def _log(returns, crash=None):
    n = len(returns)
    return pd.DataFrame({"env_step": [1000 * (i + 1) for i in range(n)], "return_balanced": returns,
                         "crash_rate": crash if crash is not None else [0.5] * n})


def test_select_tuning_config_uses_the_mean_over_the_last_three_points():
    logs = {
        "default": _log([100.0, 5.0, 10.0, 15.0]),  # early spike ignored: last 3 -> mean 10
        "lr-low": _log([0.0, 8.0, 9.0, 10.0]),  # last 3 -> 9
        "lr-high": _log([-50.0, 9.0, 12.0, 11.0]),  # last 3 -> 32/3 = 10.67
    }
    table = tuning_table(logs)
    assert list(table["config"]) == ["default", "lr-low", "lr-high"]
    assert np.allclose(table["criterion"], [10.0, 9.0, 32 / 3])
    assert select_tuning_config(logs) == "lr-high"
    assert list(table["chosen"]) == [False, False, True] and table["last_env_step"].eq(4000).all()


def test_select_tuning_config_sorts_by_env_step_not_row_order():
    shuffled = _log([1.0, 2.0, 30.0, 40.0, 50.0]).iloc[[4, 0, 3, 1, 2]].reset_index(drop=True)
    assert tuning_table({"a": shuffled})["criterion"].iloc[0] == pytest.approx(40.0)  # points at 3000..5000


def test_select_tuning_config_ties_go_to_the_lower_crash_rate_then_to_the_first_listed():
    tied = {
        "a": _log([10.0, 20.0, 30.0], crash=[0.4, 0.4, 0.4]),
        "b": _log([5.0, 20.0, 35.0], crash=[0.1, 0.2, 0.3]),  # same mean 20, crash mean 0.2 < 0.4
        "c": _log([20.0, 20.0, 20.0], crash=[0.2, 0.2, 0.2]),  # same mean 20, same crash mean as b, listed later
        "d": _log([1.0, 1.0, 1.0], crash=[0.0, 0.0, 0.0]),  # lowest crash rate but a worse criterion: never chosen
    }
    assert select_tuning_config(tied) == "b"
    assert select_tuning_config({"c": tied["c"], "b": tied["b"]}) == "c"  # full tie: the first listed (the defaults)
    assert select_tuning_config({"a": tied["a"], "d": tied["d"]}) == "a"


def test_select_tuning_config_rejects_unusable_input():
    with pytest.raises(ValueError, match="no tuning"):
        select_tuning_config({})
    with pytest.raises(ValueError, match="at most"):
        select_tuning_config({f"c{i}": _log([1.0] * 3) for i in range(TUNING_MAX_CONFIGS + 1)})
    assert TUNING_MAX_CONFIGS == 4
    with pytest.raises(ValueError, match="validation points"):
        select_tuning_config({"a": _log([1.0, 2.0])})
    with pytest.raises(ValueError, match="column"):
        select_tuning_config({"a": _log([1.0] * 3).drop(columns="return_balanced")})
    with pytest.raises(ValueError, match="NaN"):
        select_tuning_config({"a": _log([1.0, float("nan"), 3.0])})


def _episodes(n: int = 4) -> pd.DataFrame:
    """Hand-built per-episode test results with every column the level-1 aggregation needs."""
    return pd.DataFrame([
        dict(seed=10000 + i, episode_return=1.0, return_env=1.0, sum_high_speed=10.0, sum_right_lane=5.0, lane_changes=1,
             crashed=i % 2 == 0, completed=i % 2 == 1, near_miss=False, length=30, mean_speed=25.0, distance_m=750.0,
             min_gap=10.0, min_ttc=3.0)
        for i in range(n)
    ])


def test_tuning_runs_are_excluded_from_results_by_default(tmp_path):
    save_eval(_episodes(), "ppo", "balanced", 0, "main", tmp_path)
    save_eval(_episodes(), "ppo", "balanced", PILOT_RUN_SEED, "main", tmp_path, variant="tune-a")
    loaded = load_eval_episodes(tmp_path)
    assert set(loaded["variant"]) == {"main"} and set(loaded["train_seed"]) == {0}
    both = load_eval_episodes(tmp_path, include_tuning=True)
    assert set(both["variant"]) == {"main", "tune-a"}
    # the aggregation drops them too when they are passed in, unless asked for
    assert set(per_model_table(both)["variant"]) == {"main"}
    assert set(per_model_table(both, include_tuning=True)["variant"]) == {"main", "tune-a"}


def test_load_tuning_val_logs(tmp_path):
    for name, seed in (("b", PILOT_RUN_SEED), ("a", PILOT_RUN_SEED), ("c", 0)):
        d = run_dir("ddqn", "balanced", seed, tmp_path, tuning_variant(name))
        d.mkdir(parents=True)
        _log([1.0, 2.0, 3.0]).to_csv(d / "val_log.csv", index=False)
    run_dir("ddqn", "balanced", 0, tmp_path).mkdir(parents=True)  # a main run is not a tuning run
    logs = load_tuning_val_logs("ddqn", tmp_path)
    assert list(logs) == ["a", "b"]  # "c" has no seed-99 log; sorted by name
    assert select_tuning_config(logs) in ("a", "b")
    assert load_tuning_val_logs("ppo", tmp_path) == {}


# ---- sensitivity grid -----------------------------------------------------------------------------------------------------
def test_sensitivity_grid_expands_to_24_runs_skipping_the_base_values():
    runs = expand_sensitivity_grid()
    assert len(runs) == 24
    for algo in ("ddqn", "ppo"):
        assert sum(r.algo == algo for r in runs) == 12
    assert {r.seed for r in runs} == set(SENSITIVITY_SEEDS)
    by_setting = {}
    for r in runs:
        by_setting.setdefault((r.algo, r.variant), []).append(r.seed)
    assert all(seeds == SENSITIVITY_SEEDS for seeds in by_setting.values())  # 3 seeds per setting
    # absolute values for the CURRENT placeholder bases (lr 5e-4 / 3e-4, eps_decay_fraction 0.3, clip_range 0.2)
    assert sorted(by_setting) == sorted([
        ("ddqn", "lr=0.000166667"), ("ddqn", "lr=0.0015"),
        ("ddqn", "eps_decay_fraction=0.1"), ("ddqn", "eps_decay_fraction=0.5"),
        ("ppo", "lr=0.0001"), ("ppo", "lr=0.0009"),
        ("ppo", "clip_range=0.1"), ("ppo", "clip_range=0.3"),
    ])
    # the base points (lr x1, eps_decay_fraction 0.3, clip 0.2) are the main-grid runs and are not repeated
    assert not any(r.overrides.get("eps_decay_fraction") == 0.3 or r.overrides.get("clip_range") == 0.2 for r in runs)
    assert all(len(r.overrides) == 1 and r.variant != MAIN_VARIANT for r in runs)  # one parameter at a time


def test_every_grid_contains_its_base_by_construction_and_never_varies_shared_keys():
    for algo, params in SENSITIVITY_GRID.items():
        assert set(params) <= set(CONFIG_CLASSES[algo].__dataclass_fields__)
        assert not set(params) & SHARED_KEYS  # gamma and the network spec are never varied
        for param, spec in params.items():
            neutral = 1.0 if spec["mode"] == "factor" else 0.0
            assert neutral in spec["values"], (algo, param)  # the base value is a grid point whatever the base is
    assert "eps_decay_steps" not in CONFIG_CLASSES["ddqn"].__dataclass_fields__  # replaced by the fraction


def test_values_are_resolved_relative_to_the_base_config():
    ddqn, ppo = DDQNConfig(), PPOConfig()
    runs = {(r.algo, r.param, r.grid_value): r for r in expand_sensitivity_grid() if r.seed == 0}
    assert runs[("ddqn", "lr", 1 / 3)].overrides == {"lr": pytest.approx(ddqn.lr / 3, rel=1e-5)}
    assert runs[("ddqn", "lr", 3.0)].overrides == {"lr": pytest.approx(ddqn.lr * 3)}
    assert runs[("ppo", "lr", 1 / 3)].overrides == {"lr": pytest.approx(ppo.lr / 3, rel=1e-5)}
    assert runs[("ppo", "lr", 3.0)].overrides == {"lr": pytest.approx(ppo.lr * 3)}
    assert runs[("ddqn", "eps_decay_fraction", -0.2)].overrides == {"eps_decay_fraction": 0.1}  # 0.3 - 0.2, no float noise
    assert runs[("ddqn", "eps_decay_fraction", 0.2)].overrides == {"eps_decay_fraction": 0.5}
    assert runs[("ppo", "clip_range", -0.1)].overrides == {"clip_range": 0.1}
    assert runs[("ppo", "clip_range", 0.1)].overrides == {"clip_range": 0.3}
    # the resolved value reaches the config (and so run_meta.json "hyperparameters") unchanged
    assert apply_overrides(ddqn, runs[("ddqn", "eps_decay_fraction", 0.2)].overrides).eps_decay_fraction == 0.5


def test_grid_stays_valid_for_several_bases_and_budgets():
    budgets = [TrainBudget(total_env_steps=n) for n in (100_000, 150_000, 200_000, 300_000)]
    for eps_base, (low, high) in {0.3: (0.1, 0.5), 0.5: (0.3, 0.7)}.items():
        runs = expand_sensitivity_grid({"ddqn": DDQNConfig(eps_decay_fraction=eps_base)})
        fractions = sorted({r.overrides["eps_decay_fraction"] for r in runs if r.param == "eps_decay_fraction"})
        assert fractions == [low, high]  # base -0.2 and +0.2
        for budget in budgets:  # the fractions resolve to positive absolute steps for every budget
            for f in (eps_base, low, high):
                cfg = DDQNConfig(eps_decay_fraction=f)
                steps = resolve_hyperparameters(cfg, budget)["eps_decay_steps"]
                assert steps == round(f * budget.total_env_steps) and steps > 0
    for clip_base, (low, high) in {0.15: (0.05, 0.25), 0.2: (0.1, 0.3), 0.25: (0.15, 0.35)}.items():
        runs = expand_sensitivity_grid({"ppo": PPOConfig(clip_range=clip_base)})
        clips = sorted({r.overrides["clip_range"] for r in runs if r.param == "clip_range"})
        assert clips == [low, high] and all(0 < c < 1 for c in clips)
    for lr_base in (1e-4, 5e-4, 2e-3):
        runs = expand_sensitivity_grid({"ddqn": DDQNConfig(lr=lr_base), "ppo": PPOConfig(lr=lr_base)})
        assert len(runs) == 24
        assert sorted({r.overrides["lr"] for r in runs if r.algo == "ddqn" and r.param == "lr"}) == [
            pytest.approx(lr_base / 3, rel=1e-5), pytest.approx(lr_base * 3)]


def test_grid_raises_instead_of_clipping_when_a_value_falls_outside():
    for eps_base in (0.2, 0.1, 0.85, 0.9, 1.0):  # base - 0.2 <= 0 or base + 0.2 > 1
        with pytest.raises(ValueError, match="eps_decay_fraction"):
            expand_sensitivity_grid({"ddqn": DDQNConfig(eps_decay_fraction=eps_base)})
    for clip_base in (0.1, 0.05, 0.9, 0.95):  # base - 0.1 <= 0 or base + 0.1 >= 1
        with pytest.raises(ValueError, match="clip_range"):
            expand_sensitivity_grid({"ppo": PPOConfig(clip_range=clip_base)})
    assert expand_sensitivity_grid({"ddqn": DDQNConfig(eps_decay_fraction=0.8)})  # 0.6 / 1.0: 1.0 is allowed for eps


def test_eps_decay_fraction_is_resolved_to_absolute_steps():
    cfg = DDQNConfig()
    assert cfg.eps_decay_fraction == 0.3 and "eps_decay_fraction" in cfg.placeholder_fields()
    assert [cfg.eps_decay_steps(n) for n in (100_000, 150_000, 200_000, 300_000)] == [30_000, 45_000, 60_000, 90_000]
    assert DDQNConfig(eps_decay_fraction=1e-9).eps_decay_steps(1000) == 1  # never 0
    assert resolve_hyperparameters(cfg, TrainBudget(total_env_steps=200_000)) == {"eps_decay_steps": 60_000}
    assert resolve_hyperparameters(PPOConfig(), TrainBudget()) == {"rollout_size": 1000}
    for bad in (0.0, -0.1, 1.01):
        with pytest.raises(ValueError, match="eps_decay_fraction"):
            DDQNConfig(eps_decay_fraction=bad)
    DDQNConfig(eps_decay_fraction=1.0)


def test_sensitivity_run_cli_args_round_trip_into_the_config():
    run = next(r for r in expand_sensitivity_grid() if r.variant == "lr=0.000166667" and r.seed == 2)
    args = run.cli_args()
    assert args[:6] == ["--reward", "balanced", "--density", "medium", "--seed", "2"]
    setup = resolve_setup("ddqn", args)
    assert setup.variant == run.variant and setup.seed == 2 and setup.overridden_keys == ["lr"]
    assert setup.cfg.lr == run.overrides["lr"] and setup.budget == TrainBudget()  # budget not overridden


# ---- sensitivity runs must match the main run -----------------------------------------------------------------------------
def _meta(algo, *, variant=MAIN_VARIANT, seed=0, overrides=None, budget=None, hp_changes=None):
    cfg = apply_overrides(CONFIG_CLASSES[algo](), overrides or {})
    meta = new_meta({"algorithm": algo, "train_reward": "balanced", "train_density": "medium", "seed": seed,
                     "variant": variant, "overridden_keys": sorted(overrides or {})})
    meta["hyperparameters"] = {**cfg.to_dict(), **(hp_changes or {})}
    meta["budget"] = (budget or TrainBudget()).to_dict()
    return meta


def test_sensitivity_run_must_differ_from_main_only_in_the_varied_parameter():
    for run in expand_sensitivity_grid():
        check_sensitivity_run(_meta(run.algo, variant=run.variant, seed=run.seed, overrides=run.overrides),
                              _meta(run.algo, seed=run.seed))  # every planned run passes against an untouched main run
    run = next(r for r in expand_sensitivity_grid() if r.algo == "ppo" and r.param == "clip_range")
    main = _meta("ppo")
    good = _meta("ppo", variant=run.variant, overrides=run.overrides)
    check_sensitivity_run(good, main)
    cases = {
        "ent_coef": _meta("ppo", variant=run.variant, overrides=run.overrides, hp_changes={"ent_coef": 0.02}),
        "gamma": _meta("ppo", variant=run.variant, overrides=run.overrides, hp_changes={"gamma": 0.995}),
        "hidden_sizes": _meta("ppo", variant=run.variant, overrides=run.overrides, hp_changes={"hidden_sizes": [64, 64]}),
        "budget differs": _meta("ppo", variant=run.variant, overrides=run.overrides, budget=TrainBudget(total_env_steps=50_000)),
    }
    for match, bad in cases.items():
        with pytest.raises(ValueError, match=match):
            check_sensitivity_run(bad, main)
    # the main run was trained with different defaults than the sensitivity run (e.g. a teammate edited PPOConfig in between)
    with pytest.raises(ValueError, match="n_epochs"):
        check_sensitivity_run(good, _meta("ppo", hp_changes={"n_epochs": 4}))


def test_sensitivity_check_enforces_the_role_of_the_run():
    main = _meta("ddqn")
    lr = {"lr": 1e-3}
    with pytest.raises(ValueError, match="one parameter at a time"):
        check_sensitivity_run(_meta("ddqn", variant="x", overrides={"lr": 1e-3, "tau": 0.5}), main)
    with pytest.raises(ValueError, match="not a sensitivity parameter"):
        check_sensitivity_run(_meta("ddqn", variant="x", overrides={"batch_size": 32}), main)
    with pytest.raises(ValueError, match="base point"):  # "varied" to the main-grid value = a re-run of the base point
        check_sensitivity_run(_meta("ddqn", variant="x", overrides={"lr": DDQNConfig().lr}), main)
    with pytest.raises(ValueError, match="seed"):
        check_sensitivity_run(_meta("ddqn", variant="x", seed=3, overrides=lr), _meta("ddqn", seed=3))
    with pytest.raises(ValueError, match="seed"):
        check_sensitivity_run(_meta("ddqn", variant="x", seed=1, overrides=lr), main)  # a different main seed
    with pytest.raises(ValueError, match="main variant"):
        check_sensitivity_run(_meta("ddqn", variant="x", overrides=lr), _meta("ddqn", variant="y"))


def test_check_sensitivity_runs_reads_run_meta_files(tmp_path):
    run = next(r for r in expand_sensitivity_grid() if r.algo == "ddqn" and r.seed == 1)
    write_meta(run_meta_path("ddqn", "balanced", 1, tmp_path), _meta("ddqn", seed=1))
    sens_path = run_meta_path("ddqn", "balanced", 1, tmp_path, run.variant)
    write_meta(sens_path, _meta("ddqn", variant=run.variant, seed=1, overrides=run.overrides))
    write_meta(run_meta_path("ddqn", "balanced", PILOT_RUN_SEED, tmp_path, "tune-a"),
               _meta("ddqn", variant="tune-a", seed=PILOT_RUN_SEED, overrides={"batch_size": 32}))  # not a sensitivity run
    assert check_sensitivity_runs(tmp_path) == [sens_path]
    bad = copy.deepcopy(_meta("ddqn", variant=run.variant, seed=1, overrides=run.overrides))
    bad["hyperparameters"]["gamma"] = 0.995
    write_meta(sens_path, bad)
    with pytest.raises(ValueError, match="gamma"):
        check_sensitivity_runs(tmp_path)
    sens_path.parent.parent.joinpath("seed2").mkdir(parents=True)  # a sensitivity run without its main run
    write_meta(run_meta_path("ddqn", "balanced", 2, tmp_path, run.variant),
               _meta("ddqn", variant=run.variant, seed=2, overrides=run.overrides))
    write_meta(sens_path, _meta("ddqn", variant=run.variant, seed=1, overrides=run.overrides))
    with pytest.raises(FileNotFoundError):
        check_sensitivity_runs(tmp_path)


def test_training_run_checks_a_sensitivity_run_against_an_existing_main_run(tmp_path):
    run = next(r for r in expand_sensitivity_grid() if r.algo == "ppo" and r.param == "clip_range" and r.seed == 0)
    setup = resolve_setup("ppo", [*run.cli_args(), "--out-dir", str(tmp_path)])
    assert isinstance(setup, RunSetup)
    tr = TrainingRun(setup, object, quiet=True)  # the trainer class is not needed for this check
    meta = _meta("ppo", variant=run.variant, overrides=run.overrides)
    tr._check_against_main_run(meta)  # no main run yet: nothing to compare
    write_meta(run_meta_path("ppo", "balanced", 0, tmp_path), _meta("ppo"))
    tr._check_against_main_run(meta)  # identical apart from the varied key
    write_meta(run_meta_path("ppo", "balanced", 0, tmp_path), _meta("ppo", hp_changes={"vf_coef": 1.0}))
    with pytest.raises(ValueError, match="vf_coef"):
        tr._check_against_main_run(meta)
    # runs outside the sensitivity role (here two overridden keys, a debug variant) are not checked
    debug = resolve_setup("ppo", ["--set", "n_envs=2", "--set", "n_steps=10", "--set", "minibatch_size=10",
                                  "--out-dir", str(tmp_path)])
    TrainingRun(debug, object, quiet=True)._check_against_main_run(_meta("ppo"))


# ---- video episodes --------------------------------------------------------------------------------------------------------
def _outcome_frame(crashed: list[bool], first_seed: int = 10000) -> pd.DataFrame:
    return pd.DataFrame({"seed": [first_seed + i for i in range(len(crashed))], "crashed": crashed,
                         "completed": [not c for c in crashed], "episode_return": 0.0})


def test_video_episode_seeds_both_and_disagreement_categories():
    #                  seed: 10000  10001  10002  10003  10004  10005  10006
    ddqn = _outcome_frame([True, True, False, False, True, False, True])
    ppo = _outcome_frame([False, True, False, True, True, False, False])
    # both crashed: 10001, 10004 -> lowest 10001; both completed: 10002, 10005 -> 10002
    # DDQN crashed only: 10000, 10006 -> 10000; PPO crashed only: 10003
    assert video_episode_seeds(ddqn, ppo) == {
        "both_crashed": 10001, "both_completed": 10002, "ddqn_crashed_only": 10000, "ppo_crashed_only": 10003}
    assert set(video_episode_seeds(ddqn, ppo)) == set(VIDEO_CATEGORIES)


def test_video_episode_seeds_missing_categories_and_input_forms(tmp_path):
    same = _outcome_frame([True, False, True])
    out = video_episode_seeds(same, same)  # identical models: no disagreement at all
    assert out == {"both_crashed": 10000, "both_completed": 10001, "ddqn_crashed_only": None, "ppo_crashed_only": None}
    ddqn, ppo = _outcome_frame([True, True]), _outcome_frame([False, False])
    assert video_episode_seeds(ddqn, ppo) == {
        "both_crashed": None, "both_completed": None, "ddqn_crashed_only": 10000, "ppo_crashed_only": None}
    ddqn.to_csv(tmp_path / "d.csv", index=False)
    ppo.to_csv(tmp_path / "p.csv", index=False)
    assert video_episode_seeds(tmp_path / "d.csv", str(tmp_path / "p.csv")) == video_episode_seeds(ddqn, ppo)
    # row order does not matter, different seed sets do
    assert video_episode_seeds(ddqn.iloc[::-1], ppo) == video_episode_seeds(ddqn, ppo)
    with pytest.raises(ValueError, match="different seeds"):
        video_episode_seeds(_outcome_frame([True]), _outcome_frame([True], first_seed=10001))
    with pytest.raises(ValueError, match="columns"):
        video_episode_seeds(ddqn.drop(columns="crashed"), ppo)
