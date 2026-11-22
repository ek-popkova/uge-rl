"""Experiment-tracking infrastructure: training harness end to end (RandomTrainer), logger, configs, curves."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from common.actions import ACTION_INDEX
from common.aggregate import auc_across_seeds, curve_auc, learning_curve_table, steps_to_threshold
from common.baselines import BASELINE_NAMES, make_baseline
from common.benchmark import measure_inference, record_val_observations
from common.config import MIN_GAMMA, REWARDS, TARGET_CONSECUTIVE, TARGET_PERFORMANCE, VAL_SEEDS
from common.failure_cases import list_episodes_by_outcome, seeds_by_outcome
from common.metrics import EPISODE_COLUMNS, NEAR_MISS_TTC, REVERSAL_WINDOW, SUMMARY_COLUMNS, check_target_performance
from common.evaluate import validate
from common.paths import (
    eval_dir,
    load_val_logs,
    milestone_checkpoint,
    run_dir,
    run_meta_path,
)
from common.random_trainer import RandomTrainer
from common.run_meta import check_action_mapping, new_meta
from common.train_config import (
    DDQNConfig,
    PPOConfig,
    TrainBudget,
    apply_overrides,
    check_comparable,
    parse_set_args,
    placeholders_in_use,
    variant_name,
)
from common.train_env import finished_episodes, make_train_env, make_vector_env
from common.train_logging import (
    TRAIN_LOG_COLUMNS,
    UPDATE_AGG,
    UPDATE_ALGO_METRICS,
    UPDATE_COMMON_METRICS,
    UPDATE_DERIVED,
    UPDATE_LOG_COLUMNS,
    VAL_LOG_COLUMNS,
    Timers,
    TrainLogger,
    read_logs,
)
from common.training import SimulatedInterrupt, TrainingRun, resolve_setup
from ddqn.train import DDQNTrainer

TIME_COLUMNS = {
    "wall_s", "train_wall_s", "update_time_s", "cum_env_time_s", "cum_update_time_s", "cum_val_time_s",
    "cum_checkpoint_time_s",
}
TOTAL = 80

DDQN_ARGS = ["--total-steps", str(TOTAL), "--val-every", "40", "--val-episodes", "2", "--checkpoint-every", "30",
             "--update-log-every", "2"]
PPO_ARGS = ["--set", "n_envs=2", "--set", "n_steps=10", "--set", "minibatch_size=10", "--total-steps", str(TOTAL),
            "--val-every", "40", "--val-episodes", "2", "--checkpoint-every", "40", "--update-log-every", "1"]
ALGO_ARGS = {"ddqn": DDQN_ARGS, "ppo": PPO_ARGS}


def run_random(root, algo, *, workers=1, benchmark_steps=0, interrupt=None, resume=False):
    argv = ["--reward", "balanced", "--seed", "0", "--out-dir", str(root), "--workers", str(workers),
            "--benchmark-steps", str(benchmark_steps), *ALGO_ARGS[algo]]
    if resume:
        argv.append("--resume")
    setup = resolve_setup(algo, argv)
    TrainingRun(setup, RandomTrainer, interrupt_at_env_step=interrupt, quiet=True).run()
    return setup.run_dir


def stable(df: pd.DataFrame) -> pd.DataFrame:
    """Drop wall-clock columns, which differ between runs by nature."""
    return df.drop(columns=[c for c in df if c in TIME_COLUMNS])


def assert_same_logs(dir_a, dir_b):
    a, b = read_logs(dir_a), read_logs(dir_b)
    for name in ("train", "update", "val"):
        assert len(a[name]) > 0, name
        pd.testing.assert_frame_equal(stable(a[name]), stable(b[name]), obj=name)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The runs shared by the end-to-end tests (each is a few seconds of real simulation)."""
    root = tmp_path_factory.mktemp("runs")
    out = {
        "ddqn": run_random(root / "ddqn", "ddqn", benchmark_steps=20),
        "ppo": run_random(root / "ppo", "ppo"),
    }
    return out


# ---- end to end -------------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("algo", ["ddqn", "ppo"])
def test_all_files_exist_with_the_right_schema(runs, algo):
    d = runs[algo]
    for name in ("train_log.csv", "update_log.csv", "val_log.csv", "run_meta.json", "final.pt", "resume.pt",
                 "milestone_10pct.pt", "milestone_25pct.pt", "milestone_50pct.pt"):
        assert (d / name).exists(), name
    logs = read_logs(d)
    assert list(logs["train"].columns) == TRAIN_LOG_COLUMNS
    assert list(logs["update"].columns) == UPDATE_LOG_COLUMNS[algo]
    assert list(logs["val"].columns) == VAL_LOG_COLUMNS
    assert TRAIN_LOG_COLUMNS[:3] == ["env_step", "wall_s", "vec_env"] and TRAIN_LOG_COLUMNS[3:] == EPISODE_COLUMNS
    assert set(SUMMARY_COLUMNS) <= set(VAL_LOG_COLUMNS) and {f"return_{r}" for r in REWARDS} <= set(VAL_LOG_COLUMNS)
    # algorithm-specific update metrics belong to one algorithm only
    other = "ppo" if algo == "ddqn" else "ddqn"
    assert not set(UPDATE_ALGO_METRICS[other]) & set(logs["update"].columns)
    assert not logs["update"][UPDATE_ALGO_METRICS[algo]].isna().any().any()  # RandomTrainer reports every metric


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["ddqn", "ppo"])
def test_env_step_is_the_monotonic_x_axis(runs, algo):
    logs = read_logs(runs[algo])
    for name in ("train", "update", "val"):
        steps = logs[name]["env_step"].to_numpy()
        assert (np.diff(steps) >= 0).all() and steps.max() <= TOTAL, name
    assert (np.diff(logs["update"]["update_idx"].to_numpy()) > 0).all()
    val = logs["val"]
    assert list(val["val_idx"]) == list(range(len(val))) and val["env_step"].iloc[-1] == TOTAL
    assert {f"return_{r}" for r in REWARDS} <= set(val.columns) and not val[[f"return_{r}" for r in REWARDS]].isna().any().any()
    assert (val["train_wall_s"] <= val["wall_s"]).all() and (val["cum_val_time_s"] > 0).all()
    assert np.allclose(val["wall_s"] - val["cum_val_time_s"], val["train_wall_s"])
    train = logs["train"]
    assert (train[[c for c in train if c.startswith("act_")]].sum(axis=1) == train["length"]).all()
    if algo == "ppo":
        assert set(train["vec_env"]) <= {0, 1}
        assert all(s % 2 == 0 for s in train["env_step"])  # env steps are summed over the 2 vector envs
        assert list(val["env_step"]) == [40, 80]  # first rollout boundary at or after each multiple of 40
    else:
        assert set(train["vec_env"]) == {0} and list(val["env_step"]) == [40, 80]
        assert len(logs["update"]) == (TOTAL // 10) // 2 and logs["update"]["updates_in_row"].eq(2).all()


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["ddqn", "ppo"])
def test_run_meta_has_every_field(runs, algo):
    meta = json.loads((runs[algo] / "run_meta.json").read_text())
    template = new_meta({})
    assert set(meta) == set(template)
    assert set(meta["timing"]) == set(template["timing"])
    assert meta["status"] == "finished" and meta["error"] is None
    ident = meta["identity"]
    assert ident["algorithm"] == algo and ident["train_reward"] == "balanced" and ident["train_density"] == "medium"
    assert ident["seed"] == 0
    assert ident["variant"] == ("main" if algo == "ddqn" else "minibatch_size=10,n_envs=2,n_steps=10")
    assert ident["overridden_keys"] == ([] if algo == "ddqn" else ["minibatch_size", "n_envs", "n_steps"])
    cfg_cls = DDQNConfig if algo == "ddqn" else PPOConfig
    assert cfg_cls.from_dict(meta["hyperparameters"]).to_dict() == meta["hyperparameters"]
    assert meta["budget"]["total_env_steps"] == TOTAL and set(meta["budget_overrides"]) >= {"total_env_steps"}
    assert any(p.endswith("lr") for p in meta["placeholders_in_use"])
    # relative hyperparameters are stored as absolute values: DDQN eps_decay_fraction 0.3 x 80 steps, PPO rollout 2 x 10
    assert meta["resolved"] == ({"eps_decay_steps": 24} if algo == "ddqn" else {"rollout_size": 20})
    env = meta["env"]
    assert env["env_id"] == "highway-fast-v0" and env["config"]["simulation_frequency"] == 5  # resolved config, not BASE_CONFIG
    assert env["config"]["vehicles_density"] == 0.7 and env["config"]["vehicles_count"] == 20
    assert env["reward_weights"] == REWARDS["balanced"] and env["action_index"] == ACTION_INDEX
    assert env["observation_shape"] == [5, 5]
    assert env["near_miss_ttc"] == NEAR_MISS_TTC and env["reversal_window"] == REVERSAL_WINDOW
    rep = meta["reproducibility"]
    assert set(rep["git"]) == {"commit", "dirty"} and rep["torch_num_threads"] == 1 and rep["device"] == "cpu"
    assert {"python", "torch", "gymnasium", "highway_env", "numpy"} <= set(rep["versions"])
    assert rep["os"] and rep["cpu_model"] and rep["logical_cores"] >= 1 and rep["workers"] == 1
    assert set(rep["thread_env_vars"]) == {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
    t = meta["timing"]
    assert t["final_env_step"] == TOTAL and t["start_time"] and t["end_time"] and t["total_wall_s"] > 0
    assert t["env_time_s"] > 0 and t["val_time_s"] > 0 and t["checkpoint_time_s"] > 0 and t["train_steps_per_s"] > 0
    assert t["n_episodes"] == len(read_logs(runs[algo])["train"]) and t["n_val_points"] == 2 and t["n_updates"] > 0
    if algo == "ddqn":  # the quiet benchmark was requested only for this run
        assert t["quiet_benchmark_steps_per_s"] > 0
    assert meta["model"]["n_parameters"] == 4
    inf = meta["inference"]
    assert inf["median_ms"] > 0 and inf["p90_ms"] >= inf["median_ms"] and inf["batch_size"] == 1 and inf["threads"] == 1
    assert [m["fraction"] for m in meta["milestones"]] == [0.1, 0.25, 0.5] and meta["logging"]["update_log_every"] in (1, 2)


@pytest.mark.slow
def test_milestone_and_final_checkpoints_are_loadable_agents(runs):
    from common.random_trainer import RandomPolicyAgent

    agent = RandomPolicyAgent(99)
    path = milestone_checkpoint("ddqn", "balanced", 0, 0.5, runs["ddqn"].parents[4], "main")
    assert path == runs["ddqn"] / "milestone_50pct.pt"
    agent.load(str(path))
    assert agent.seed == 0
    agent.load(str(runs["ppo"] / "final.pt"))
    assert agent.seed == 0


@pytest.mark.slow
@pytest.mark.parametrize("algo, interrupt", [("ddqn", 50), ("ppo", 60)])
def test_resume_reproduces_the_uninterrupted_run(runs, tmp_path, algo, interrupt):
    with pytest.raises(SimulatedInterrupt):
        run_random(tmp_path, algo, interrupt=interrupt)
    d = run_dir(algo, "balanced", 0, tmp_path, runs[algo].parent.name)
    meta = json.loads((d / "run_meta.json").read_text())
    assert meta["status"] == "failed" and "SimulatedInterrupt" in meta["error"]  # also written on a crash
    first_ckpt = 30 if algo == "ddqn" else 40
    before = read_logs(d)
    assert before["update"]["env_step"].max() > first_ckpt  # rows beyond the checkpoint exist and must be dropped
    resumed = run_random(tmp_path, algo, resume=True)
    assert resumed == d
    assert_same_logs(d, runs[algo])  # no duplicates, same rows as the uninterrupted run
    meta = json.loads((d / "run_meta.json").read_text())
    assert meta["status"] == "finished" and meta["timing"]["final_env_step"] == TOTAL
    assert len(meta["resumes"]) == 1 and meta["resumes"][0]["env_step"] == first_ckpt
    assert [m["fraction"] for m in meta["milestones"]] == [0.1, 0.25, 0.5]  # no duplicates after the resume
    assert meta["timing"]["start_time"]  # kept from the first start
    assert not read_logs(d)["train"]["env_step"].diff().dropna().lt(0).any()
    # a finished run is not touched by --resume
    run_random(tmp_path, algo, resume=True)
    assert_same_logs(d, runs[algo])


@pytest.mark.slow
def test_logs_identical_for_one_and_two_workers(runs, tmp_path):
    two = run_random(tmp_path, "ppo", workers=2)  # AsyncVectorEnv (subprocesses) and a validation pool
    assert_same_logs(two, runs["ppo"])
    assert json.loads((two / "run_meta.json").read_text())["reproducibility"]["workers"] == 2


@pytest.mark.slow
def test_existing_run_is_protected_unless_overwrite_or_resume(runs, tmp_path):
    import shutil

    d = run_dir("ddqn", "balanced", 0, tmp_path)
    shutil.copytree(runs["ddqn"], d)
    with pytest.raises(FileExistsError):
        run_random(tmp_path, "ddqn")


def test_stub_trainers_raise_before_anything_is_written(tmp_path):
    setup = resolve_setup("ddqn", ["--out-dir", str(tmp_path), "--benchmark-steps", "0"])
    with pytest.raises(NotImplementedError):
        TrainingRun(setup, DDQNTrainer, quiet=True).run()
    assert not any(tmp_path.iterdir())


def test_total_steps_must_fit_the_rollout_size(tmp_path):
    with pytest.raises(ValueError, match="multiple"):
        resolve_setup("ppo", ["--set", "n_envs=3", "--set", "n_steps=10", "--set", "minibatch_size=10",
                              "--total-steps", "100", "--val-every", "50", "--out-dir", str(tmp_path)])


def test_main_variant_cannot_carry_overrides():
    with pytest.raises(ValueError, match="overwrite"):
        resolve_setup("ddqn", ["--variant", "main", "--set", "lr=0.001"])


# ---- config -------------------------------------------------------------------------------------------------------------
def test_configs_validate_gamma_and_comparable_network_size():
    DDQNConfig(), PPOConfig(), TrainBudget()  # placeholders are valid
    with pytest.raises(ValueError, match="gamma"):
        DDQNConfig(gamma=MIN_GAMMA - 0.01)
    with pytest.raises(ValueError, match="gamma"):
        PPOConfig(gamma=0.9)
    with pytest.raises(ValueError, match="device"):
        PPOConfig(device="cuda")
    with pytest.raises(ValueError, match="hidden_sizes differ"):
        check_comparable(DDQNConfig(hidden_sizes=(64, 64)), PPOConfig(hidden_sizes=(128, 128)))
    check_comparable(DDQNConfig(), PPOConfig())
    with pytest.raises(ValueError):
        PPOConfig(minibatch_size=10_000)
    with pytest.raises(ValueError):
        TrainBudget(val_n_episodes=len(VAL_SEEDS) + 1)
    with pytest.raises(ValueError):
        TrainBudget(val_every_env_steps=10**9)


def test_config_json_round_trip_and_set_overrides():
    for cfg in (DDQNConfig(), PPOConfig(target_kl=0.02), TrainBudget()):
        data = json.loads(json.dumps(cfg.to_dict()))
        assert type(cfg).from_dict(data) == cfg
    with pytest.raises(ValueError):
        PPOConfig.from_dict({"nope": 1})
    ov = parse_set_args(["lr=3e-4", "hidden_sizes=32,32", "tau=0.5"], DDQNConfig)
    assert ov == {"lr": 3e-4, "hidden_sizes": (32, 32), "tau": 0.5}
    assert apply_overrides(DDQNConfig(), ov).hidden_sizes == (32, 32)
    assert parse_set_args(["target_kl=none", "n_envs=1e1", "clip_range=0.1"], PPOConfig) == {
        "target_kl": None, "n_envs": 10, "clip_range": 0.1}
    for bad in (["lr"], ["nope=1"], ["lr=1", "lr=2"], ["buffer_size=1.5"]):
        with pytest.raises(ValueError):
            parse_set_args(bad, DDQNConfig)
    with pytest.raises(ValueError):
        apply_overrides(DDQNConfig(), {"gamma": 0.5})  # overrides are validated too


def test_variant_names_and_no_collision_with_main(tmp_path):
    assert variant_name({}) == "main"
    assert variant_name({"clip_range": 0.1}) == "clip_range=0.1"
    assert variant_name({"lr": 3e-4, "gamma": 0.995}) == variant_name({"gamma": 0.995, "lr": 3e-4}) == "gamma=0.995,lr=0.0003"
    assert variant_name({"hidden_sizes": (32, 32)}) == "hidden_sizes=32x32"
    main = run_dir("ppo", "balanced", 0, tmp_path)
    for v in ("clip_range=0.1", "lr=0.0003", "hidden_sizes=32x32"):
        other = run_dir("ppo", "balanced", 0, tmp_path, v)
        assert other != main and main not in other.parents and other.parent.name == v
        assert eval_dir("ppo", "balanced", 0, tmp_path, v) != eval_dir("ppo", "balanced", 0, tmp_path)
    assert run_meta_path("ppo", "balanced", 0, tmp_path, "lr=0.0003").parent.parent.name == "lr=0.0003"
    for bad in ("../main", "a/b", "a\\b", "", ".", "x."):
        with pytest.raises(ValueError):
            run_dir("ppo", "balanced", 0, tmp_path, bad)


def test_placeholders_are_listed_and_overrides_are_not():
    names = placeholders_in_use(DDQNConfig(), TrainBudget())
    assert "DDQNConfig.lr" in names and not any(n.startswith("TrainBudget.") for n in names)  # the budget is frozen
    assert "DDQNConfig.device" not in names  # decided by CLAUDE.md rule 8
    # shared decisions are plain defaults now (tests/test_protocol.py)
    for decided in ("gamma", "hidden_sizes", "activation", "obs_flatten"):
        assert f"DDQNConfig.{decided}" not in names and f"PPOConfig.{decided}" not in placeholders_in_use(PPOConfig())
    assert "TrainBudget.milestone_fractions" not in names
    assert "DDQNConfig.lr" not in placeholders_in_use(DDQNConfig(), overridden=["lr"])
    assert set(TARGET_PERFORMANCE) == {"beats_idle", "strong"}  # two pre-registered levels
    assert all(len(conds) >= 2 for conds in TARGET_PERFORMANCE.values())  # joint: safety and efficiency
    assert isinstance(TARGET_CONSECUTIVE, int) and TARGET_CONSECUTIVE >= 2  # one lucky point must not count
    check_target_performance(TARGET_PERFORMANCE, TARGET_CONSECUTIVE)  # also run at import of common.metrics


def test_target_performance_structure_is_validated():
    safe = {"metric": "crash_rate", "operator": "<=", "value": 0.2}
    fast = {"metric": "mean_speed", "operator": ">=", "value": 25.5}
    check_target_performance({"a": [safe, fast], "b": [fast, {**safe, "operator": "<"}]}, 2)
    cases = {
        "unknown metric": {"a": [{**safe, "metric": "crash_rte"}, fast]},
        "unknown operator": {"a": [{**safe, "operator": "=="}, fast]},
        "no efficiency metric": {"a": [safe]},
        "no safety metric": {"a": [fast]},
        "one level lacks a kind": {"a": [safe, fast], "b": [fast]},
        "a bare list (old format)": [safe, fast],
        "empty": {},
        "empty level": {"a": []},
        "string value": {"a": [{**safe, "value": "0.2"}, fast]},
        "extra key": {"a": [{**safe, "note": "x"}, fast]},
    }
    for name, bad in cases.items():
        with pytest.raises(ValueError):
            check_target_performance(bad)
    for bad_k in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError):
            check_target_performance({"a": [safe, fast]}, bad_k)


# ---- logger -------------------------------------------------------------------------------------------------------------
def test_logger_schema_aggregation_and_truncation(tmp_path):
    timers = Timers()
    log = TrainLogger(tmp_path, "ppo", timers, update_log_every=2)
    with pytest.raises(KeyError):
        log.log_update(0, td_loss=1.0)  # a DDQN metric in a PPO run
    with pytest.raises(KeyError):
        log.log_val(0, {"nonsense": 1.0})
    with pytest.raises(KeyError):
        log.log_episode(0, 0, {"nonsense": 1.0})
    log.log_update(10, lr=0.1, loss=1.0, grad_norm=2.0, policy_loss=1.0, update_time_s=0.5, n_epochs_done=3)
    assert log.tables["update"].buffer == []  # K = 2: the window is still open
    log.log_update(20, lr=0.2, loss=3.0, grad_norm=4.0, policy_loss=3.0, update_time_s=0.25, n_epochs_done=4)
    for step in (5, 15, 25):
        log.log_episode(step, 0, {"length": step})
    log.log_val(25, {"episode_return": 1.0})
    log.flush()
    row = pd.read_csv(tmp_path / "update_log.csv").iloc[0]
    assert row["env_step"] == 20 and row["update_idx"] == 2 and row["updates_in_row"] == 2
    assert row["loss"] == 2.0 and row["grad_norm"] == 3.0  # means
    assert row["update_time_s"] == 0.75 and row["lr"] == 0.2 and row["n_epochs_done"] == 3.5  # sum / last / mean
    assert row["grad_norm_max"] == 4.0
    assert math.isnan(row["entropy"])  # missing algorithm-specific key -> NaN
    train = pd.read_csv(tmp_path / "train_log.csv")
    assert list(train["env_step"]) == [5, 15, 25] and train["crashed"].isna().all()
    assert log.truncate_to(15) == {"train": 1, "update": 1, "val": 1}  # rows beyond the checkpoint's env step
    assert list(pd.read_csv(tmp_path / "train_log.csv")["env_step"]) == [5, 15]
    assert len(pd.read_csv(tmp_path / "val_log.csv")) == 0 and len(pd.read_csv(tmp_path / "update_log.csv")) == 0
    # state round trip: an open K-update window survives a resume
    log.log_update(30, lr=0.3, loss=5.0, grad_norm=1.0)
    log2 = TrainLogger(tmp_path, "ppo", Timers(), update_log_every=2)
    log2.load_state_dict(log.state_dict())
    log2.log_update(40, lr=0.3, loss=7.0, grad_norm=1.0)
    log2.flush()
    row = pd.read_csv(tmp_path / "update_log.csv").iloc[0]
    assert (row["env_step"], row["update_idx"], row["updates_in_row"], row["loss"]) == (40, 4, 2, 6.0)


def test_update_aggregation_keeps_spikes_and_every_metric_has_one_rule(tmp_path):
    assert set(UPDATE_AGG) == set(UPDATE_COMMON_METRICS) | {m for ms in UPDATE_ALGO_METRICS.values() for m in ms}
    assert UPDATE_AGG["q_max"] == "max" and UPDATE_DERIVED["grad_norm_max"] == ("grad_norm", "max")
    log = TrainLogger(tmp_path, "ddqn", Timers(), update_log_every=5)
    q_max = [1.0, 1.1, 9.0, 1.2, 1.0]  # one overestimation spike
    grad = [0.5, 0.4, 0.6, 30.0, 0.5]  # one gradient spike
    for i, (q, g) in enumerate(zip(q_max, grad)):
        log.log_update(i + 1, lr=1e-3 / (i + 1), loss=1.0 + i, grad_norm=g, td_loss=2.0, q_mean=q / 2, q_max=q,
                       target_q_mean=0.0, epsilon=1.0 - 0.1 * i, buffer_size=100 + i)
    log.flush()
    row = pd.read_csv(tmp_path / "update_log.csv").iloc[0]
    assert row["q_max"] == 9.0 and row["grad_norm_max"] == 30.0  # the spikes survive
    assert row["grad_norm"] == pytest.approx(np.mean(grad)) and row["q_mean"] == pytest.approx(np.mean(q_max) / 2)  # the means do not
    assert row["loss"] == pytest.approx(3.0) and row["update_time_s"] >= 0
    assert row["lr"] == pytest.approx(1e-3 / 5) and row["epsilon"] == pytest.approx(0.6) and row["buffer_size"] == 104  # last
    assert row["updates_in_row"] == 5 and row["update_idx"] == 5 and row["env_step"] == 5
    with pytest.raises(KeyError):
        log.log_update(6, grad_norm_max=1.0)  # derived by the logger, not reported by algorithms


def test_logger_close_emits_a_partial_window_only_on_normal_end(tmp_path):
    log = TrainLogger(tmp_path / "a", "ddqn", Timers(), update_log_every=5)
    log.log_update(7, lr=0.1, loss=1.0, grad_norm=1.0)
    log.close(emit_partial=True)
    assert len(pd.read_csv(tmp_path / "a" / "update_log.csv")) == 1
    log = TrainLogger(tmp_path / "b", "ddqn", Timers(), update_log_every=5)
    log.log_update(7, lr=0.1, loss=1.0, grad_norm=1.0)
    log.close(emit_partial=False)
    assert len(pd.read_csv(tmp_path / "b" / "update_log.csv")) == 0


def test_timers_accumulate_and_resume():
    t = Timers()
    with t.env():
        pass
    with t.val():
        pass
    state = t.state_dict()
    t2 = Timers()
    t2.load_state_dict(state)
    assert t2.seconds == t.seconds and t2.wall() >= state["wall"]
    assert t2.wall() - t2.train_wall() == pytest.approx(t2.seconds["val"], abs=1e-3)


# ---- training envs --------------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_train_env_records_episodes_and_snapshots_state():
    env = make_train_env(reward="balanced")
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(4):
        env.step(int(rng.integers(5)))
    blob = env.get_state()
    actions = [int(a) for a in rng.integers(0, 5, size=40)]

    def play():
        out, info = [], {}
        for a in actions:
            obs, r, te, tr, info = env.step(a)
            out.append((obs.tolist(), r, te, tr))
            if te or tr:
                break
        return out, info

    first, info = play()
    assert info["episode_metrics"]["length"] == 4 + len(first) and "seed" not in info["episode_metrics"]
    assert list(info["episode_metrics"]) == EPISODE_COLUMNS
    env.set_state([blob])  # back to step 4 of the episode, recorder included
    second, info2 = play()
    assert first == second and info2["episode_metrics"] == info["episode_metrics"]
    # the recorder is re-armed by the next reset
    env.reset()
    _, _, _, _, info3 = env.step(ACTION_INDEX["IDLE"])
    assert "episode_metrics" not in info3
    assert finished_episodes(info) == [(0, info["episode_metrics"])] and finished_episodes(info3) == []


@pytest.mark.slow
def test_vector_env_infos_give_per_env_records():
    venv = make_vector_env(2, workers=1, reward="balanced")
    try:
        venv.reset(seed=[100000, 100001])
        seen = []
        for _ in range(40):
            _, _, term, trunc, infos = venv.step(np.array([ACTION_INDEX["FASTER"]] * 2))
            seen += finished_episodes(infos, 2)
            if len(seen) >= 2:
                break
        assert seen and all(list(rec) == EPISODE_COLUMNS for _, rec in seen)
        assert {i for i, _ in seen} <= {0, 1} and all(isinstance(rec["crashed"], bool) for _, rec in seen)
        assert all(rec["length"] > 0 and not math.isnan(rec["mean_speed"]) for _, rec in seen)
    finally:
        venv.close()


@pytest.mark.slow
def test_action_mapping_check_fails_loudly(monkeypatch):
    env = make_train_env()
    assert check_action_mapping(env) == ACTION_INDEX
    monkeypatch.setattr(env.unwrapped.action_type, "actions_indexes", {"LANE_LEFT": 1, "IDLE": 0, "LANE_RIGHT": 2, "FASTER": 3, "SLOWER": 4})
    with pytest.raises(RuntimeError, match="action mapping"):
        check_action_mapping(env)


# ---- curves, cost metrics, failure cases ---------------------------------------------------------------------------------
def _val_log(values, steps=None, metric="crash_rate"):
    n = len(values)
    steps = steps if steps is not None else [1000 * (i + 1) for i in range(n)]
    return pd.DataFrame({"val_idx": range(n), "env_step": steps, "train_wall_s": [s / 10 for s in steps], metric: values})


def _val_frame(**cols):
    n = len(next(iter(cols.values())))
    steps = [1000 * (i + 1) for i in range(n)]
    return pd.DataFrame({"val_idx": range(n), "env_step": steps, "train_wall_s": [x / 10 for x in steps], **cols})


def test_steps_to_threshold_single_condition_and_operators():
    log = _val_log([0.9, 0.5, 0.2, 0.3, 0.1])
    thr = {"metric": "crash_rate", "operator": "<=", "value": 0.2}  # a single condition dict still works
    assert steps_to_threshold(log, thr, consecutive=1) == {"env_step": 3000.0, "train_wall_s": 300.0, "val_idx": 2.0}
    assert steps_to_threshold(log, [thr], consecutive=1) == steps_to_threshold(log, thr, consecutive=1)
    assert steps_to_threshold(log, {**thr, "operator": "<"}, consecutive=1)["env_step"] == 5000.0
    assert math.isnan(steps_to_threshold(log, thr, consecutive=2)["env_step"])  # never two in a row at <= 0.2
    never = steps_to_threshold(_val_log([0.9, 0.8]), thr, consecutive=1)
    assert all(math.isnan(v) for v in never.values())
    up = {"metric": "episode_return", "operator": ">=", "value": 5.0}
    assert steps_to_threshold(_val_log([1.0, 5.0, 9.0], metric="episode_return"), up, consecutive=1)["env_step"] == 2000.0
    assert steps_to_threshold(_val_log([float("nan"), 0.1, 0.1]), thr, consecutive=2)["env_step"] == 2000.0  # NaN never meets it
    with pytest.raises(ValueError):
        steps_to_threshold(log, {**thr, "operator": "=="})
    with pytest.raises(ValueError):
        steps_to_threshold(log, [])
    with pytest.raises(ValueError):
        steps_to_threshold(log, thr, consecutive=0)


def test_steps_to_threshold_needs_every_condition_at_the_same_point():
    safe = {"metric": "crash_rate", "operator": "<=", "value": 0.2}
    fast = {"metric": "mean_speed", "operator": ">=", "value": 25.5}
    # a braking policy: safe from the start, never fast. A fast but crashing phase comes first.
    log = _val_frame(crash_rate=[0.9, 0.1, 0.1, 0.1, 0.1], mean_speed=[28.0, 20.0, 21.0, 26.0, 27.0])
    assert steps_to_threshold(log, [safe, fast], consecutive=1)["env_step"] == 4000.0  # not 2000 (safe only) nor 1000 (fast only)
    assert steps_to_threshold(log, safe, consecutive=1)["env_step"] == 2000.0  # a single condition would reward braking
    only_slow = _val_frame(crash_rate=[0.0, 0.0, 0.0], mean_speed=[20.0, 20.0, 20.0])
    assert all(math.isnan(v) for v in steps_to_threshold(only_slow, [safe, fast], consecutive=1).values())
    assert steps_to_threshold(log, [safe, fast], consecutive=2)["env_step"] == 4000.0
    assert math.isnan(steps_to_threshold(log, [safe, fast], consecutive=3)["env_step"])


def test_an_isolated_passing_point_does_not_count_and_the_first_point_of_the_run_is_reported():
    assert TARGET_CONSECUTIVE == 3
    strong = TARGET_PERFORMANCE["strong"]
    good = dict(crash_rate=0.1, mean_speed=26.0)
    bad = dict(crash_rate=0.5, mean_speed=24.0)
    # one lucky validation point (20 episodes: crash_rate moves in steps of 0.05) in the middle of bad ones
    lucky = pd.DataFrame([bad, bad, good, bad, bad]).assign(
        env_step=[1000, 2000, 3000, 4000, 5000], train_wall_s=[10, 20, 30, 40, 50], val_idx=range(5))
    assert all(math.isnan(v) for v in steps_to_threshold(lucky, strong).values())  # default: TARGET_CONSECUTIVE points
    assert steps_to_threshold(lucky, strong, consecutive=1)["env_step"] == 3000.0  # the single point would have counted
    # two in a row are not enough with TARGET_CONSECUTIVE = 3
    pair = pd.DataFrame([bad, good, good, bad, bad]).assign(
        env_step=[1000, 2000, 3000, 4000, 5000], train_wall_s=[10, 20, 30, 40, 50], val_idx=range(5))
    assert all(math.isnan(v) for v in steps_to_threshold(pair, strong).values())
    assert steps_to_threshold(pair, strong, consecutive=2)["env_step"] == 2000.0
    # isolated points followed later by three in a row: the reported point is the FIRST of the qualifying run
    run = pd.DataFrame([bad, good, bad, good, good, good]).assign(
        env_step=[1000, 2000, 3000, 4000, 5000, 6000], train_wall_s=[10, 20, 30, 40, 50, 60], val_idx=range(6))
    assert steps_to_threshold(run, strong) == {"env_step": 4000.0, "train_wall_s": 40.0, "val_idx": 3.0}


def test_steps_to_threshold_reports_both_pre_registered_levels():
    idle = dict(crash_rate=0.60, mean_speed=24.76)  # idle on VAL_SEEDS: does not beat itself
    better = dict(crash_rate=0.40, mean_speed=25.0)  # beats idle; not strong (speed below 25.5)
    strong = dict(crash_rate=0.15, mean_speed=26.0)
    log = pd.DataFrame([idle, better, better, strong, strong, strong]).assign(
        env_step=[1000, 2000, 3000, 4000, 5000, 6000], train_wall_s=[10, 20, 30, 40, 50, 60], val_idx=range(6))
    out = steps_to_threshold(log)  # default: TARGET_PERFORMANCE and TARGET_CONSECUTIVE
    assert set(out) == {"beats_idle", "strong"}
    assert out["beats_idle"] == {"env_step": 2000.0, "train_wall_s": 20.0, "val_idx": 1.0}
    assert out["strong"] == {"env_step": 4000.0, "train_wall_s": 40.0, "val_idx": 3.0}
    # a run that only beats idle: the strong level is NaN, the weak level still gives a number
    weak = log.iloc[:4]  # idle, better, better, strong: three points that beat idle, only one strong point
    out = steps_to_threshold(weak)
    assert out["beats_idle"]["env_step"] == 2000.0 and math.isnan(out["strong"]["env_step"])
    assert math.isnan(steps_to_threshold(log.iloc[:1])["beats_idle"]["env_step"])  # idle itself reaches neither


@pytest.mark.slow
def test_no_scripted_baseline_meets_target_performance():
    """Guards TARGET_PERFORMANCE against edits: no trivial policy may reach EITHER level.

    VAL_SEEDS only, never TEST_SEEDS. One validation point is enough here: ``consecutive=1`` is the weakest test
    (the default of 2 would pass trivially on a single row).
    """
    for name in BASELINE_NAMES:
        metrics = validate(make_baseline(name), n=len(VAL_SEEDS))
        val_log = pd.DataFrame([{"env_step": 1000, "train_wall_s": 1.0, "val_idx": 0, **metrics}])
        for level, conditions in TARGET_PERFORMANCE.items():
            reached = steps_to_threshold(val_log, conditions, consecutive=1)["env_step"]
            assert math.isnan(reached), f"scripted baseline {name!r} meets level {level!r}: " + ", ".join(
                f"{c['metric']}={metrics[c['metric']]:.3f}" for c in conditions
            )


def test_learning_curve_table_aligns_by_validation_index():
    # PPO-like: the actual env steps differ slightly between seeds; points are matched by index
    logs = {0: _val_log([1.0, 3.0, 5.0], [1000, 2000, 3000], "episode_return"),
            1: _val_log([3.0, 5.0, 7.0], [1024, 2048, 3072], "episode_return"),
            2: _val_log([2.0, 4.0], [1010, 2010], "episode_return")}  # unfinished run
    table = learning_curve_table(logs, "episode_return")
    assert list(table["val_idx"]) == [0, 1, 2] and list(table["n_seeds"]) == [3, 3, 2]
    assert table["mean"].tolist() == pytest.approx([2.0, 4.0, 6.0])
    assert table["env_step_min"][0] == 1000 and table["env_step_max"][0] == 1024
    half = 4.3026527 * np.std([1.0, 3.0, 2.0], ddof=1) / math.sqrt(3)  # t(0.975, 2) * sd / sqrt(n)
    assert table["ci_low"][0] == pytest.approx(2.0 - half) and table["ci_high"][0] == pytest.approx(2.0 + half)
    assert table["ci_low"][2] < table["mean"][2] < table["ci_high"][2]  # n = 2 still has an interval (df = 1)
    assert learning_curve_table([], "episode_return").empty


def test_learning_curve_auc():
    log = _val_log([0.0, 1.0, 1.0], [0, 100, 300], "episode_return")
    assert curve_auc(log, "episode_return", normalize=False) == pytest.approx(0.5 * 100 + 1.0 * 200)
    assert curve_auc(log, "episode_return") == pytest.approx(250 / 300)
    assert math.isnan(curve_auc(_val_log([1.0], metric="episode_return"), "episode_return"))
    res = auc_across_seeds([log, _val_log([1.0, 1.0, 1.0], [0, 100, 300], "episode_return")], "episode_return")
    assert res["n_seeds"] == 2 and res["mean"] == pytest.approx((250 / 300 + 1.0) / 2)


def test_load_val_logs_per_seed(tmp_path):
    for seed in (2, 0):
        d = run_dir("ppo", "balanced", seed, tmp_path)
        d.mkdir(parents=True)
        _val_log([0.5, 0.4]).to_csv(d / "val_log.csv", index=False)
    run_dir("ppo", "balanced", 1, tmp_path).mkdir(parents=True)  # no val log yet
    assert list(load_val_logs("ppo", "balanced", tmp_path)) == [0, 2]
    assert load_val_logs("ppo", "balanced", tmp_path, "lr=0.001") == {}


def _episodes():
    return pd.DataFrame({
        "seed": [10, 11, 12, 13, 14, 15], "episode_return": [3.0, -9.0, 8.0, 1.0, 5.0, -2.0],
        "length": [30, 12, 30, 7, 30, 30], "crashed": [False, True, False, True, False, False],
        "near_miss": [False, False, True, False, False, True], "completed": [True, False, True, False, True, True],
    })


def test_failure_case_listing_by_outcome(tmp_path):
    df = _episodes()
    assert seeds_by_outcome(df, "crashed") == [11, 13]  # worst return first
    assert seeds_by_outcome(df, "crashed", n=1) == [11]
    assert seeds_by_outcome(df, "near_miss") == [15, 12]  # lowest return first
    assert seeds_by_outcome(df, "completed") == [12, 14, 10, 15]  # best return first
    assert seeds_by_outcome(df, "completed", ascending=True)[0] == 15
    path = tmp_path / "main.csv"
    df.to_csv(path, index=False)
    assert list(list_episodes_by_outcome(path, "crashed")["seed"]) == [11, 13]
    with pytest.raises(ValueError):
        seeds_by_outcome(df, "exploded")


def test_video_script_cli_resolves_seeds_and_registry(tmp_path):
    from scripts.record_video import build_parser, load_agent, resolve_seeds

    csv = tmp_path / "main.csv"
    _episodes().to_csv(csv, index=False)
    p = build_parser()
    args = p.parse_args(["--algo", "ppo", "--checkpoint", "x.pt", "--seeds", "10003", "10017", "--condition", "long"])
    assert resolve_seeds(args) == [10003, 10017] and args.condition == "long"
    args = p.parse_args(["--algo", "ppo", "--from-eval-csv", str(csv), "--outcome", "crashed", "--n", "1"])
    assert resolve_seeds(args) == [11]
    assert resolve_seeds(p.parse_args(["--algo", "ppo", "--episodes", "2"])) == [10000, 10001]
    with pytest.raises(ValueError):
        resolve_seeds(p.parse_args(["--algo", "ppo", "--seeds", "1", "--from-eval-csv", str(csv)]))
    for algo in ("ddqn", "ppo"):
        with pytest.raises(NotImplementedError):
            load_agent(algo, "final.pt")
    assert load_agent("idle", "").act(None) == ACTION_INDEX["IDLE"]


@pytest.mark.slow
def test_inference_benchmark_protocol():
    obs = record_val_observations(make_baseline("slower"), n_episodes=1)
    assert obs.shape == (30, 5, 5)
    res = measure_inference(make_baseline("slower"), obs, n=200, warmup=20)
    assert res["median_ms"] > 0 and res["p90_ms"] >= res["median_ms"] and res["batch_size"] == 1 and res["threads"] == 1
    with pytest.raises(ValueError):
        measure_inference(make_baseline("slower"), [], n=10)


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["ddqn", "ppo"])
def test_run_meta_records_the_machine_and_the_onedrive_flag(runs, algo):
    from common import run_meta as rm

    repro = rm.read_meta(runs[algo] / "run_meta.json")["reproducibility"]
    assert repro["hostname"] == rm.hostname() and repro["cpu_model"]  # wall times are only compared within one machine
    assert isinstance(repro["onedrive_path"], bool)


@pytest.mark.slow
def test_a_run_inside_onedrive_prints_a_prominent_warning_but_runs(tmp_path, monkeypatch, capsys):
    from common import run_meta as rm

    monkeypatch.setattr(rm, "is_onedrive_path", lambda path, environ=None: True)
    run_dir_ = run_random(tmp_path, "ddqn")
    err = capsys.readouterr().err
    assert "WARNING" in err and "OneDrive" in err and "move the project out of OneDrive" in err
    meta = rm.read_meta(run_dir_ / "run_meta.json")
    assert meta["status"] == "finished" and meta["reproducibility"]["onedrive_path"] is True  # not blocked, but recorded
