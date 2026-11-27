"""PPOAgent and PPOTrainer without the simulator (fast tier).

``FakeContext`` stands in for ``common.training.TrainContext``: a scripted vector env with SAME_STEP-style
``final_obs`` infos, so collection, the ``final_obs`` bootstrap, the update and the logged metrics are tested on
hand-built episodes. The end-to-end harness tests with the real env are in ``tests/test_training.py`` (slow).
"""

from __future__ import annotations

import copy
import math
import pickle

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from common.networks import count_parameters, flatten_obs
from common.train_config import DDQNConfig, PPOConfig, check_comparable
from common.train_logging import UPDATE_ALGO_METRICS, UPDATE_COMMON_METRICS, Timers
from ppo.agent import PPOAgent
from ppo.train import PPOTrainer, observation_tensor

FINAL_OBS = np.full((5, 5), 0.5, dtype=np.float32)  # the terminal observation of every scripted episode end
SMALL = dict(n_envs=2, n_steps=8, minibatch_size=4)  # rollout of 16 transitions, 4 minibatches per epoch
LOGGED_KEYS = (set(UPDATE_COMMON_METRICS) - {"update_time_s"}) | set(UPDATE_ALGO_METRICS["ppo"])


def scripted_obs(t: int, n_envs: int) -> np.ndarray:
    return (0.1 * np.random.default_rng(t).standard_normal((n_envs, 5, 5))).astype(np.float32)


class FakeContext:
    """The parts of ``TrainContext`` that ``PPOTrainer`` uses: ``cfg, seed, n_envs, obs, timers, step, log_update``.

    ``episode_ends`` maps a step index ``t`` (0-based, the t-th call of ``step``) to ``(env, terminated)``: that env's
    episode ends at step t by a termination (True) or a time-limit truncation (False). Rewards are always 1.
    """

    def __init__(self, cfg: PPOConfig, seed: int = 0, episode_ends: dict | None = None,
                 drop_final_obs: bool = False) -> None:
        self.cfg, self.seed, self.n_envs = cfg, seed, cfg.n_envs
        self.timers = Timers()
        self.episode_ends = episode_ends or {}
        self.drop_final_obs = drop_final_obs  # simulate a vector env that does not report final_obs (wrong autoreset)
        self.t = 0
        self.obs = scripted_obs(0, self.n_envs)
        self.logged: list[dict] = []
        self.counts: list[tuple] = []

    def step(self, actions):
        assert actions.shape == (self.n_envs,)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        infos: dict = {}
        if self.t in self.episode_ends:
            env, is_terminal = self.episode_ends[self.t]
            terminated[env], truncated[env] = is_terminal, not is_terminal
            final_obs = np.empty(self.n_envs, dtype=object)
            final_obs[env] = FINAL_OBS
            infos = {} if self.drop_final_obs else {"final_obs": final_obs, "_final_obs": terminated | truncated}
        self.t += 1
        self.obs = scripted_obs(self.t, self.n_envs)  # after an episode end: the next episode's first observation
        return self.obs, np.ones(self.n_envs), terminated, truncated, infos

    def log_update(self, n_updates: int = 1, n_samples=None, **metrics) -> None:
        self.logged.append(metrics)
        self.counts.append((n_updates, n_samples))  # (gradient steps, samples) reported with the metrics


def make_trainer(seed: int = 0, episode_ends: dict | None = None, **cfg_kwargs) -> PPOTrainer:
    cfg = PPOConfig(**{**SMALL, **cfg_kwargs})
    return PPOTrainer(FakeContext(cfg, seed, episode_ends))


def obs_batch(n: int = 16) -> np.ndarray:
    return (0.3 * np.random.default_rng(123).standard_normal((n, 5, 5))).astype(np.float32)


# ---- networks and initialization ------------------------------------------------------------------------------------
def test_parameter_counts_and_comparable_network():
    trainer = make_trainer()
    assert count_parameters(trainer.actor) == 73_733 and count_parameters(trainer.critic) == 72_705
    assert trainer.n_parameters() == 146_438
    assert count_parameters(trainer.make_eval_agent().actor) == 73_733  # the agent holds the actor only
    check_comparable(DDQNConfig(), PPOConfig())
    check_comparable(DDQNConfig(), trainer.cfg)
    assert trainer.step_granularity == trainer.cfg.rollout_size == 16


def test_orthogonal_initialization_with_small_policy_head():
    trainer = make_trainer()
    a_in, a_hidden, a_out = [m for m in trainer.actor if isinstance(m, torch.nn.Linear)]
    c_out = [m for m in trainer.critic if isinstance(m, torch.nn.Linear)][-1]
    # orthogonal with gain g: (256, 25) has orthogonal columns, W^T W = g^2 I; square / wide: W W^T = g^2 I
    assert torch.allclose(a_in.weight.T @ a_in.weight, 2.0 * torch.eye(25), atol=1e-4)
    assert torch.allclose(a_hidden.weight @ a_hidden.weight.T, 2.0 * torch.eye(256), atol=1e-4)
    assert torch.allclose(a_out.weight.norm(dim=1), torch.full((5,), 0.01), atol=1e-6)  # gain 0.01
    assert c_out.weight.norm().item() == pytest.approx(1.0, abs=1e-5)  # gain 1
    assert all((m.bias == 0).all() for m in [*trainer.actor, *trainer.critic] if isinstance(m, torch.nn.Linear))
    probs = torch.softmax(trainer.actor(observation_tensor(obs_batch())), dim=-1)
    assert torch.allclose(probs, torch.full_like(probs, 0.2), atol=0.01)  # near-uniform initial policy


def test_initialization_is_reproducible_from_the_run_seed():
    a, b, c = make_trainer(seed=0), make_trainer(seed=0), make_trainer(seed=1)
    for name, tensor in a.actor.state_dict().items():
        assert torch.equal(tensor, b.actor.state_dict()[name])
    assert not torch.equal(a.critic[0].weight, c.critic[0].weight)


# ---- collection -----------------------------------------------------------------------------------------------------
def test_rollout_stores_flags_and_the_value_of_the_final_obs():
    # env 1 is truncated at step 1, env 0 terminates at step 2, env 1 is truncated again at the last step 7
    trainer = make_trainer(episode_ends={1: (1, False), 2: (0, True), 7: (1, False)})
    initial_obs = trainer.ctx.obs.copy()
    last_values = trainer.collect_rollout()
    r = trainer.rollout
    assert r.done.nonzero().tolist() == [[1, 1], [2, 0], [7, 1]]
    assert r.terminated.nonzero().tolist() == [[2, 0]]
    with torch.no_grad():
        v_final = trainer.critic(torch.as_tensor(FINAL_OBS.reshape(1, 25))).item()
        expected_final = torch.zeros(8, 2)
        for t, i in ((1, 1), (2, 0), (7, 1)):
            expected_final[t, i] = v_final
        assert torch.allclose(r.final_values, expected_final)
        assert torch.equal(r.obs[0], torch.as_tensor(flatten_obs(initial_obs)))
        assert torch.allclose(r.values, trainer.critic(r.obs).squeeze(-1))
        assert torch.allclose(r.log_probs, Categorical(logits=r.logits).log_prob(r.actions))
        assert torch.allclose(last_values, trainer.critic(observation_tensor(trainer.ctx.obs)).squeeze(-1))
    assert (r.rewards == 1).all() and r.actions.dtype == torch.int64 and ((0 <= r.actions) & (r.actions < 5)).all()


def test_a_done_env_without_final_obs_raises():
    cfg = PPOConfig(**SMALL)
    trainer = PPOTrainer(FakeContext(cfg, episode_ends={1: (0, False)}, drop_final_obs=True))
    with pytest.raises(RuntimeError, match="final observation"):
        trainer.collect_rollout()


@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 2])
def test_real_vector_env_truncation_reports_final_obs_and_its_value(workers):
    """On the real highway-env vector env (in-process and subprocesses): an episode truncated by the time limit gives
    ``_final_obs`` / ``final_obs`` as ``final_obs_values`` expects, ``final_values`` is non-zero exactly at the done
    steps and equals ``V(final_obs)``, and the final observation differs from the reset observation of that step."""
    from common.actions import ACTION_INDEX
    from common.seeding import derived_env_seeds
    from common.train_env import make_vector_env

    cfg = PPOConfig(n_envs=2, n_steps=40, minibatch_size=40)  # episodes last 30 steps: a rollout can hold a truncation
    env = make_vector_env(cfg.n_envs, workers, reward="balanced")
    try:
        ctx = RealEnvContext(cfg, env)
        trainer = PPOTrainer(ctx)
        with torch.no_grad():  # mostly SLOWER, the safest action, so that some episode reaches the time limit
            trainer.actor[-1].bias[ACTION_INDEX["SLOWER"]] = 5.0
        for _ in range(5):
            ctx.steps = []
            trainer.collect_rollout()
            if (trainer.rollout.done & ~trainer.rollout.terminated).any():  # a time-limit truncation happened
                break
        r = trainer.rollout
        truncated_only = r.done & ~r.terminated
        assert truncated_only.any(), "no time-limit truncation in 5 rollouts"
        assert torch.equal(r.final_values != 0, r.done)  # non-zero exactly at the done steps
        for t, (obs, terminated, truncated, infos) in enumerate(ctx.steps):
            done = terminated | truncated
            if not done.any():
                assert "_final_obs" not in infos or not infos["_final_obs"].any()
                continue
            assert np.array_equal(infos["_final_obs"], done)
            for i in np.flatnonzero(done):
                final = infos["final_obs"][i]
                assert final.shape == (5, 5)
                assert not np.array_equal(final, obs[i])  # obs[i] is already the next episode's reset observation
                with torch.no_grad():
                    expected = trainer.critic(observation_tensor(final[None])).item()
                assert r.final_values[t, i].item() == pytest.approx(expected, rel=1e-6)
        assert derived_env_seeds(0, 2) == ctx.reset_seeds
    finally:
        env.close()


class RealEnvContext:
    """``FakeContext`` interface around a real vector env; records ``(obs, terminated, truncated, infos)`` per step."""

    def __init__(self, cfg: PPOConfig, env, seed: int = 0) -> None:
        from common.seeding import derived_env_seeds

        self.cfg, self.seed, self.n_envs, self.env = cfg, seed, cfg.n_envs, env
        self.timers = Timers()
        self.reset_seeds = derived_env_seeds(seed, cfg.n_envs)
        self.obs, _ = env.reset(seed=self.reset_seeds)
        self.steps: list = []

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        self.steps.append((obs.copy(), terminated, truncated, infos))
        self.obs = obs
        return obs, rewards, terminated, truncated, infos

    def log_update(self, **metrics) -> None:
        pass


def test_collect_and_update_logs_every_ppo_metric_once():
    trainer = make_trainer(episode_ends={3: (0, False)})
    with pytest.raises(ValueError, match="whole rollouts"):
        trainer.collect_and_update(8)
    trainer.collect_and_update(16)
    assert len(trainer.ctx.logged) == 1 and trainer.n_updates == 1
    metrics = trainer.ctx.logged[0]
    assert set(metrics) == LOGGED_KEYS
    assert all(math.isfinite(v) for k, v in metrics.items() if k != "kl_beta")
    assert math.isnan(metrics["kl_beta"])  # PPO-Clip has no penalty coefficient
    assert metrics["n_epochs_done"] == 10 and metrics["lr"] == trainer.cfg.lr
    assert metrics["ratio_max"] >= 1.0 and 0.0 <= metrics["clip_fraction"] <= 1.0 and metrics["approx_kl"] >= 0.0
    assert metrics["kl_after_update"] > 0.0  # the policy moved; logged for PPO-Clip too
    assert trainer.ctx.timers.seconds["update"] > 0
    # n_updates = Adam steps actually taken (10 epochs x 16 / 4 minibatches), n_samples = their summed minibatch sizes
    assert trainer.ctx.counts == [(10 * 4, 10 * 16)]


def test_gradient_step_and_sample_counts_follow_the_target_kl_early_stop():
    trainer = make_trainer(target_kl=1e-9)  # stops after the first epoch
    trainer.collect_and_update(16)
    assert trainer.ctx.logged[0]["n_epochs_done"] == 1 and trainer.ctx.counts == [(4, 16)]
    a2c_like = make_trainer(n_epochs=1, minibatch_size=16)  # one step on the whole rollout
    a2c_like.collect_and_update(16)
    assert a2c_like.ctx.counts == [(1, 16)]


def test_kl_after_update_is_the_exact_rollout_kl_after_all_epochs():
    for loss in ("clip", "kl"):
        trainer = make_trainer(policy_loss=loss)
        trainer.collect_and_update(16)
        logged = trainer.ctx.logged[0]["kl_after_update"]
        assert logged == pytest.approx(trainer.mean_rollout_kl(), rel=1e-6)  # the rollout buffer still holds pi_old
        with torch.no_grad():  # recomputed by hand from the stored old logits and the updated actor
            old = torch.log_softmax(trainer.rollout.logits.reshape(-1, 5), dim=-1)
            new = torch.log_softmax(trainer.actor(trainer.rollout.obs.reshape(-1, 25)), dim=-1)
            by_hand = (old.exp() * (old - new)).sum(-1).mean().item()
        assert logged == pytest.approx(by_hand, rel=1e-5)


def _hand_minibatch(trainer: PPOTrainer, n: int = 6):
    """A hand-built minibatch: random states and actions, log-probs / logits of the current actor (ratio 1)."""
    gen = torch.Generator().manual_seed(5)
    obs = 0.3 * torch.randn(n, 25, generator=gen)
    actions = torch.tensor([0, 1, 2, 3, 4, 1])[:n]
    with torch.no_grad():
        logits = trainer.actor(obs)
        log_probs = Categorical(logits=logits).log_prob(actions)
    advantages = torch.tensor([1.0, -0.5, 2.0, -1.5, 0.3, 0.7])[:n]
    returns = torch.tensor([3.0, -2.0, 5.0, 0.5, 1.0, -4.0])[:n]  # far from V(s) ~ 0, so the value loss is clearly > 0
    return obs, actions, log_probs, logits, advantages, returns


@pytest.mark.parametrize("policy_loss", ["clip", "kl"])
def test_gradient_step_loss_is_policy_minus_entropy_plus_value_with_lambda_return_targets(policy_loss):
    """``gradient_step`` on a hand-built minibatch: every logged term is recomputed independently, so a wrong value target,
    a flipped entropy sign or a reversed KL direction in the PPO-KL penalty fails here."""
    trainer = make_trainer(policy_loss=policy_loss, ent_coef=0.1, vf_coef=0.5)
    obs, actions, old_lp, old_logits, adv, returns = _hand_minibatch(trainer)
    trainer.kl_beta = 3.0
    # move the actor away from the stored policy so that the KL term and the ratios are not trivial
    with torch.no_grad():
        trainer.actor[-1].bias += torch.tensor([0.4, -0.3, 0.2, 0.0, -0.5])
        new_logits = trainer.actor(obs)
        new_lp = Categorical(logits=new_logits).log_prob(actions)
        values = trainer.critic(obs).squeeze(-1)
        entropy = Categorical(logits=new_logits).entropy().mean().item()
    adv_n = (adv - adv.mean()) / (adv.std(correction=0) + 1e-8)
    ratio = torch.exp(new_lp - old_lp)
    if policy_loss == "clip":
        eps = trainer.cfg.clip_range
        expected_policy = -torch.min(ratio * adv_n, ratio.clamp(1 - eps, 1 + eps) * adv_n).mean().item()
    else:  # KL(pi_old || pi_new) = sum_a p_old(a) (log p_old(a) - log p_new(a)): the direction of the PPO paper
        p_old = torch.softmax(old_logits, -1)
        kl = (p_old * (torch.log_softmax(old_logits, -1) - torch.log_softmax(new_logits, -1))).sum(-1).mean().item()
        expected_policy = -(ratio * adv_n).mean().item() + 3.0 * kl
    expected_value = ((values - returns) ** 2).mean().item()  # target = the lambda-returns passed in, not advantages

    stats = trainer.gradient_step(obs, actions, old_lp, old_logits, adv, returns)

    assert stats["policy_loss"] == pytest.approx(expected_policy, rel=1e-5, abs=1e-7)
    assert stats["value_loss"] == pytest.approx(expected_value, rel=1e-5)
    assert stats["entropy"] == pytest.approx(entropy, rel=1e-5)
    assert stats["loss"] == pytest.approx(expected_policy - 0.1 * entropy + 0.5 * expected_value, rel=1e-5)


def test_epochs_pass_the_lambda_returns_as_value_targets(monkeypatch):
    """Every minibatch gets ``returns = A + V(s)`` (eq:gae's lambda-returns, with V from before the update) as value target,
    and the raw advantages of the same samples."""
    trainer = make_trainer(episode_ends={1: (1, False), 2: (0, True)})
    last_values = trainer.collect_rollout()
    critic_before = copy.deepcopy(trainer.critic)
    calls = []
    real_step = trainer.gradient_step

    def spy(obs, actions, old_lp, old_logits, advantages, returns):
        calls.append((obs.clone(), advantages.clone(), returns.clone()))
        return real_step(obs, actions, old_lp, old_logits, advantages, returns)

    monkeypatch.setattr(trainer, "gradient_step", spy)
    trainer.learn_from_rollout(last_values)
    assert len(calls) == trainer.cfg.n_epochs * 4
    for obs, advantages, returns in calls:
        with torch.no_grad():
            values_before = critic_before(obs).squeeze(-1)
        assert torch.allclose(returns - advantages, values_before, atol=1e-5)  # returns = A + V, not A
        assert not torch.allclose(returns, advantages)


def test_gradient_step_entropy_bonus_raises_the_entropy_and_value_step_moves_v_toward_the_returns():
    """With zero advantages the policy term has no gradient: the step is driven by the entropy bonus (which must raise the
    entropy) and by the value loss (which must move V toward the returns)."""
    trainer = make_trainer(ent_coef=1.0, vf_coef=0.5, lr=1e-3)
    obs, actions, old_lp, old_logits, _, returns = _hand_minibatch(trainer)
    with torch.no_grad():  # start from a peaked policy, so the entropy can clearly rise
        trainer.actor[-1].bias += torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0])
        old_logits = trainer.actor(obs)
        old_lp = Categorical(logits=old_logits).log_prob(actions)
        entropy_before = Categorical(logits=old_logits).entropy().mean().item()
        value_error_before = ((trainer.critic(obs).squeeze(-1) - returns) ** 2).mean().item()
    trainer.gradient_step(obs, actions, old_lp, old_logits, torch.zeros(len(actions)), returns)
    with torch.no_grad():
        entropy_after = Categorical(logits=trainer.actor(obs)).entropy().mean().item()
        value_error_after = ((trainer.critic(obs).squeeze(-1) - returns) ** 2).mean().item()
    assert entropy_after > entropy_before
    assert value_error_after < value_error_before


def test_gradient_step_does_not_accumulate_gradients_across_minibatches():
    """Two trainers in the same state: one takes an extra step on another minibatch first. The gradient of the next step
    must not depend on that history (optimizer.zero_grad); the Adam moments do, so the gradients are compared directly."""
    # max_grad_norm huge: clipping would rescale an accumulated gradient g1 + g2 and could hide it when g1 is parallel to g2
    a, b = make_trainer(seed=0, max_grad_norm=1e9), make_trainer(seed=0, max_grad_norm=1e9)
    batch = _hand_minibatch(a)
    obs, actions, old_lp, old_logits, adv, returns = _hand_minibatch(b)
    # a genuinely different minibatch (a permutation of the same samples would give the same mean-loss gradient)
    other = (obs + 0.5, actions, old_lp, old_logits, -2.0 * adv.flip(0), returns + 10.0)
    b.gradient_step(*other)  # b now holds that step's gradients in .grad
    b.actor.load_state_dict(a.actor.state_dict())
    b.critic.load_state_dict(a.critic.state_dict())  # same weights again, different .grad history
    a.gradient_step(*batch)
    b.gradient_step(*batch)
    for pa, pb in zip(a.all_parameters, b.all_parameters):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-7)


def test_same_seed_gives_identical_updates():
    a, b = make_trainer(seed=3), make_trainer(seed=3)
    for trainer in (a, b):
        trainer.collect_and_update(16)
        trainer.collect_and_update(16)
    assert a.ctx.logged == b.ctx.logged
    assert torch.equal(a.rollout.actions, b.rollout.actions)


# ---- update rules ---------------------------------------------------------------------------------------------------
def test_target_kl_early_stop():
    stopped = make_trainer(target_kl=1e-9)  # any policy change exceeds it: stop after the first epoch
    stopped.collect_and_update(16)
    assert stopped.ctx.logged[0]["n_epochs_done"] == 1 < stopped.cfg.n_epochs
    loose = make_trainer(target_kl=1e9)
    loose.collect_and_update(16)
    assert loose.ctx.logged[0]["n_epochs_done"] == 10
    unset = make_trainer()  # target_kl = None: no early stop
    unset.collect_and_update(16)
    assert unset.ctx.logged[0]["n_epochs_done"] == 10


@pytest.mark.parametrize("measured_kl, expected_beta", [(0.0, 0.5), (1.0, 2.0), (0.01, 1.0)])
def test_ppo_kl_adapts_beta_after_the_epochs_and_logs_the_new_value(monkeypatch, measured_kl, expected_beta):
    trainer = make_trainer(policy_loss="kl")  # kl_target 0.01, kl_beta_init 1.0
    monkeypatch.setattr(trainer, "mean_rollout_kl", lambda: measured_kl)
    trainer.collect_and_update(16)
    assert trainer.kl_beta == expected_beta and trainer.ctx.logged[0]["kl_beta"] == expected_beta
    assert trainer.ctx.logged[0]["kl_after_update"] == measured_kl  # the same measured KL drives beta and is logged


def test_ppo_kl_measures_the_exact_kl_on_the_rollout():
    trainer = make_trainer(policy_loss="kl")
    trainer.collect_rollout()
    assert trainer.mean_rollout_kl() == pytest.approx(0.0, abs=1e-7)  # before the update the policies are equal
    trainer.learn_from_rollout(torch.zeros(2))
    assert trainer.mean_rollout_kl() > 0.0


def test_a2c_like_setting_has_ratio_one_and_no_clipping():
    # one epoch, one minibatch = the whole rollout: a single gradient step at ratio 1 (plain advantage actor-critic)
    trainer = make_trainer(n_epochs=1, minibatch_size=16)
    trainer.collect_and_update(16)
    metrics = trainer.ctx.logged[0]
    assert metrics["ratio_max"] == pytest.approx(1.0, abs=1e-6) and metrics["clip_fraction"] == 0.0
    assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-7) and metrics["n_epochs_done"] == 1


def test_state_dict_round_trip_continues_identically():
    a = make_trainer(seed=0, policy_loss="kl", episode_ends={2: (1, False)})
    a.collect_and_update(16)
    ctx_b = copy.deepcopy(a.ctx)  # same env position
    ctx_b.seed = 7  # a different initialization and generator, both overwritten by the state
    b = PPOTrainer(ctx_b)
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    b.ctx.logged = []
    a.ctx.logged = []
    a.collect_and_update(16)
    b.collect_and_update(16)
    assert a.ctx.logged == b.ctx.logged and a.kl_beta == b.kl_beta and a.n_updates == b.n_updates == 2
    for name, tensor in a.actor.state_dict().items():
        assert torch.equal(tensor, b.actor.state_dict()[name])


# ---- agent ----------------------------------------------------------------------------------------------------------
def test_agent_save_load_round_trip_gives_identical_actions(tmp_path):
    trainer = make_trainer()
    trainer.collect_and_update(16)  # trained weights, not the initial ones
    agent = trainer.make_eval_agent()
    obs = obs_batch()
    path = tmp_path / "final.pt"
    agent.save(str(path))
    loaded = PPOAgent()
    loaded.load(str(path))
    assert np.array_equal(agent.act(obs), loaded.act(obs))
    with torch.no_grad():
        assert torch.equal(agent.actor(observation_tensor(obs)), loaded.actor(observation_tensor(obs)))
    from scripts.record_video import load_agent

    from_registry = load_agent("ppo", str(path))
    assert isinstance(from_registry, PPOAgent) and np.array_equal(from_registry.act(obs), agent.act(obs))


def test_agent_rebuilds_the_saved_network_spec_and_rejects_other_files(tmp_path):
    small = PPOAgent(PPOConfig(hidden_sizes=(32, 32)))
    small.save(str(tmp_path / "small.pt"))
    loaded = PPOAgent()  # default 256 x 256
    loaded.load(str(tmp_path / "small.pt"))
    assert loaded.cfg.hidden_sizes == (32, 32) and count_parameters(loaded.actor) == count_parameters(small.actor)
    torch.save({"seed": 0}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="PPOAgent"):
        PPOAgent().load(str(tmp_path / "other.pt"))


def test_agent_deterministic_act_single_and_batch():
    agent = make_trainer().make_eval_agent()
    obs = obs_batch()
    first = agent.act(obs)
    assert first.shape == (16,) and np.array_equal(first, agent.act(obs))
    single = agent.act(obs[0])
    assert isinstance(single, int) and single == first[0]
    with torch.no_grad():
        assert np.array_equal(first, agent.actor(observation_tensor(obs)).argmax(dim=-1).numpy())


def test_agent_sampling_follows_the_softmax_and_is_seeded_per_episode():
    agent = PPOAgent()
    out_layer = agent.actor[-1]
    probs = torch.tensor([0.1, 0.2, 0.3, 0.15, 0.25])
    with torch.no_grad():  # constant logits = log probs, whatever the observation
        out_layer.weight.zero_()
        out_layer.bias.copy_(torch.log(probs))
    obs = np.zeros((20_000, 5, 5), dtype=np.float32)
    agent.start_episode(1)
    samples = agent.act(obs, deterministic=False)
    freq = np.bincount(samples, minlength=5) / len(samples)
    assert np.allclose(freq, probs.numpy(), atol=0.015)
    agent.start_episode(1)
    assert np.array_equal(samples, agent.act(obs, deterministic=False))


def test_agent_is_picklable_for_validation_workers():
    agent = make_trainer().make_eval_agent()
    clone = pickle.loads(pickle.dumps(agent))
    assert np.array_equal(clone.act(obs_batch()), agent.act(obs_batch()))
    assert not hasattr(clone, "critic") and not hasattr(clone, "optimizer")  # light: actor only
