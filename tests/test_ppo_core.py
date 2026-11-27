"""PPO math (ppo/core.py) on hand-computed cases: GAE with truncation / termination, the two policy losses, the KL
coefficient rule and the logged diagnostics. Fast tier: no environment, no network."""

from __future__ import annotations

import math

import pytest
import torch

from ppo.core import (
    approx_kl,
    categorical_kl,
    clip_fraction,
    clipped_surrogate_loss,
    compute_gae,
    explained_variance,
    kl_penalty_loss,
    next_state_values,
    normalize_advantages,
    ratio_max,
    update_kl_beta,
)

GAMMA, LAMBDA = 0.5, 0.5  # gamma * lambda = 0.25: every number below is exact in binary floating point


def col(*xs):
    """A one-env rollout column ``(n_steps, 1)``."""
    return torch.tensor(xs, dtype=torch.float32).reshape(-1, 1)


def flags(*xs):
    return torch.tensor(xs, dtype=torch.bool).reshape(-1, 1)


# ---- GAE ----------------------------------------------------------------------------------------------------------------
# One env, 4 steps, r_t = 1, V(s_t) = [1, 2, 3, 4]. The episode ends after step 1; steps 2-3 belong to the next episode,
# which is still running when the rollout ends. V(final_obs of step 1) = 10, V(obs after the rollout) = 8.
REWARDS, VALUES = col(1, 1, 1, 1), col(1, 2, 3, 4)
FINAL_VALUES = col(0, 10, 0, 0)  # only read where done
LAST_VALUES = torch.tensor([8.0])


def test_next_state_values_shift_and_patch_with_final_obs():
    done = flags(0, 1, 0, 0)
    # t=0: values[1] = 2; t=1: episode end -> V(final_obs) = 10 (values[2] = 3 is the next episode's first state);
    # t=2: values[3] = 4; t=3: last rollout step -> last_values = 8
    assert next_state_values(VALUES, LAST_VALUES, FINAL_VALUES, done).flatten().tolist() == [2, 10, 4, 8]


def test_gae_truncation_bootstraps_from_the_final_value_and_cuts_the_recursion():
    terminated, done = flags(0, 0, 0, 0), flags(0, 1, 0, 0)  # time-limit truncation after step 1
    next_values = next_state_values(VALUES, LAST_VALUES, FINAL_VALUES, done)  # [2, 10, 4, 8]
    adv, ret = compute_gae(REWARDS, VALUES, next_values, terminated, done, GAMMA, LAMBDA)
    # deltas: d0 = 1 + 0.5*2  - 1 = 1;  d1 = 1 + 0.5*10 - 2 = 4 (bootstrap from V(final_obs));
    #         d2 = 1 + 0.5*4  - 3 = 0;  d3 = 1 + 0.5*8  - 4 = 1 (bootstrap from the obs after the rollout)
    # A3 = d3 = 1 (A_4 = 0);  A2 = d2 + 0.25*A3 = 0.25;  A1 = d1 + 0.25*0*A2 = 4 (cut);  A0 = d0 + 0.25*A1 = 2
    assert adv.flatten().tolist() == [2, 4, 0.25, 1]
    assert ret.flatten().tolist() == [3, 6, 3.25, 5]  # A + V


def test_gae_termination_has_no_bootstrap():
    terminated, done = flags(0, 1, 0, 0), flags(0, 1, 0, 0)  # collision after step 1
    next_values = next_state_values(VALUES, LAST_VALUES, FINAL_VALUES, done)
    adv, _ = compute_gae(REWARDS, VALUES, next_values, terminated, done, GAMMA, LAMBDA)
    # d1 = 1 + 0.5*0*10 - 2 = -1;  A1 = -1;  A0 = 1 + 0.25*(-1) = 0.75;  A2, A3 as before
    assert adv.flatten().tolist() == [0.75, -1, 0.25, 1]


def test_gae_terminated_and_truncated_at_the_same_step_counts_as_terminated():
    terminated, truncated = flags(0, 1, 0, 0), flags(0, 1, 0, 0)
    done = terminated | truncated
    next_values = next_state_values(VALUES, LAST_VALUES, FINAL_VALUES, done)
    adv, _ = compute_gae(REWARDS, VALUES, next_values, terminated, done, GAMMA, LAMBDA)
    assert adv.flatten().tolist() == [0.75, -1, 0.25, 1]  # identical to the pure termination


# Episode end at the LAST rollout step (t = n_steps - 1 = 3). V(final_obs of step 3) = 10; LAST_VALUES = 8 is then the
# value of the NEXT episode's first observation and must not be used.
FINAL_AT_LAST = col(0, 0, 0, 10)


def test_truncation_at_the_last_rollout_step_bootstraps_from_the_final_value_not_last_values():
    terminated, done = flags(0, 0, 0, 0), flags(0, 0, 0, 1)
    next_values = next_state_values(VALUES, LAST_VALUES, FINAL_AT_LAST, done)
    assert next_values.flatten().tolist() == [2, 3, 4, 10]  # t=3: V(final_obs) = 10, not last_values = 8
    adv, ret = compute_gae(REWARDS, VALUES, next_values, terminated, done, GAMMA, LAMBDA)
    # deltas: d0 = 1 + 0.5*2 - 1 = 1;  d1 = 1 + 0.5*3 - 2 = 0.5;  d2 = 1 + 0.5*4 - 3 = 0;  d3 = 1 + 0.5*10 - 4 = 2
    # A3 = 2;  A2 = 0 + 0.25*2 = 0.5;  A1 = 0.5 + 0.25*0.5 = 0.625;  A0 = 1 + 0.25*0.625 = 1.15625
    # (with last_values by mistake: d3 = 1 + 0.5*8 - 4 = 1, A3 = 1)
    assert adv.flatten().tolist() == [1.15625, 0.625, 0.5, 2]
    assert ret.flatten().tolist() == [2.15625, 2.625, 3.5, 6]  # A + V


def test_termination_at_the_last_rollout_step_has_no_bootstrap():
    terminated, done = flags(0, 0, 0, 1), flags(0, 0, 0, 1)
    next_values = next_state_values(VALUES, LAST_VALUES, FINAL_AT_LAST, done)
    adv, _ = compute_gae(REWARDS, VALUES, next_values, terminated, done, GAMMA, LAMBDA)
    # d3 = 1 + 0.5*0*10 - 4 = -3;  A3 = -3;  A2 = 0 + 0.25*(-3) = -0.75;  A1 = 0.5 + 0.25*(-0.75) = 0.3125;
    # A0 = 1 + 0.25*0.3125 = 1.078125
    assert adv.flatten().tolist() == [1.078125, 0.3125, -0.75, -3]


def test_gae_is_vectorized_over_envs():
    # env 0 = the truncation case, env 1 = the termination case, in one (4, 2) rollout
    values = torch.cat([VALUES, VALUES], dim=1)
    rewards = torch.cat([REWARDS, REWARDS], dim=1)
    terminated = torch.cat([flags(0, 0, 0, 0), flags(0, 1, 0, 0)], dim=1)
    done = torch.cat([flags(0, 1, 0, 0), flags(0, 1, 0, 0)], dim=1)
    final_values = torch.cat([FINAL_VALUES, FINAL_VALUES], dim=1)
    next_values = next_state_values(values, torch.tensor([8.0, 8.0]), final_values, done)
    adv, ret = compute_gae(rewards, values, next_values, terminated, done, GAMMA, LAMBDA)
    assert adv[:, 0].tolist() == [2, 4, 0.25, 1] and adv[:, 1].tolist() == [0.75, -1, 0.25, 1]
    assert torch.equal(ret, adv + values)


def test_gae_lambda_limits():
    # no episode end, 2 steps, r = [1, 1], V = [0, 0], last value 4, gamma = 0.5
    rewards, values = col(1, 1), col(0, 0)
    no = flags(0, 0)
    next_values = next_state_values(values, torch.tensor([4.0]), col(0, 0), no)  # [0, 4]
    adv0, _ = compute_gae(rewards, values, next_values, no, no, 0.5, 0.0)
    assert adv0.flatten().tolist() == [1, 3]  # lambda = 0: one-step TD errors d0 = 1, d1 = 1 + 0.5*4 = 3
    adv1, _ = compute_gae(rewards, values, next_values, no, no, 0.5, 1.0)
    assert adv1.flatten().tolist() == [2.5, 3]  # lambda = 1: d0 + 0.5*d1 = 1 + 1.5 = 2.5 (= r0 + g r1 + g^2 * 4 - V0)


def test_gae_rejects_terminated_without_done():
    with pytest.raises(ValueError, match="terminated implies done"):
        compute_gae(col(1), col(0), col(0), flags(1), flags(0), GAMMA, LAMBDA)


def test_normalize_advantages():
    out = normalize_advantages(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    # mean 2.5, population std sqrt(1.25)
    expected = (torch.tensor([1.0, 2.0, 3.0, 4.0]) - 2.5) / (math.sqrt(1.25) + 1e-8)
    assert torch.allclose(out, expected)
    assert out.mean().abs() < 1e-6 and out.std(correction=0) == pytest.approx(1.0, abs=1e-6)
    assert normalize_advantages(torch.tensor([3.0])).item() == 0.0  # one sample: no NaN


# ---- policy losses -------------------------------------------------------------------------------------------------------
def test_clipped_surrogate_inside_and_outside_the_clip_range_both_advantage_signs():
    clip = 0.2  # clip(rho) in [0.8, 1.2]
    ratios = torch.tensor([1.1, 1.5, 0.5, 1.5, 0.5, 0.9])
    adv = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    new_log_prob = torch.log(ratios).requires_grad_(True)
    old_log_prob = torch.zeros(6)
    loss, ratio = clipped_surrogate_loss(new_log_prob, old_log_prob, adv, clip)
    # per sample min(rho*A, clip(rho)*A):
    #   1.1, A=+1: min(1.1, 1.1)   =  1.1  inside the range
    #   1.5, A=+1: min(1.5, 1.2)   =  1.2  clipped (no incentive to raise rho further)
    #   0.5, A=+1: min(0.5, 0.8)   =  0.5  unclipped term is the minimum
    #   1.5, A=-1: min(-1.5, -1.2) = -1.5  unclipped term is the minimum
    #   0.5, A=-1: min(-0.5, -0.8) = -0.8  clipped
    #   0.9, A=-1: min(-0.9, -0.9) = -0.9  inside
    # mean = (1.1 + 1.2 + 0.5 - 1.5 - 0.8 - 0.9) / 6 = -0.4 / 6, loss = +0.4 / 6
    assert loss.item() == pytest.approx(0.4 / 6, rel=1e-6)
    assert torch.allclose(ratio.detach(), ratios)
    loss.backward()
    # d loss / d log pi_i = -(1/6) * rho_i * A_i where the unclipped term is active, 0 where the clipped term is
    expected = torch.tensor([-1.1, 0.0, -0.5, 1.5, 0.0, 0.9]) / 6
    assert torch.allclose(new_log_prob.grad, expected, atol=1e-7)
    assert new_log_prob.grad[1] == 0 and new_log_prob.grad[4] == 0


def test_categorical_kl_against_a_hand_computation():
    p_old = torch.tensor([[0.5, 0.25, 0.25]])
    p_new = torch.tensor([[0.25, 0.25, 0.5]])
    # KL = 0.5 ln(0.5/0.25) + 0.25 ln(1) + 0.25 ln(0.25/0.5) = 0.5 ln 2 - 0.25 ln 2 = 0.25 ln 2
    kl = categorical_kl(torch.log(p_old), torch.log(p_new))
    assert kl.shape == (1,) and kl.item() == pytest.approx(0.25 * math.log(2), rel=1e-6)
    assert categorical_kl(torch.log(p_old) + 7.0, torch.log(p_new) - 3.0).item() == pytest.approx(0.25 * math.log(2), rel=1e-5)
    assert categorical_kl(torch.log(p_old), torch.log(p_old)).item() == pytest.approx(0.0, abs=1e-7)
    # 5 actions, batched: KL of uniform vs uniform is 0, and every KL is >= 0
    logits = torch.randn(8, 5, generator=torch.Generator().manual_seed(0))
    assert (categorical_kl(logits, torch.zeros(8, 5)) >= 0).all()


# An ASYMMETRIC pair: KL(old || new) != KL(new || old), so the direction is testable (the pair above is a permutation
# of itself and gives the same KL in both directions).
P_OLD = [0.9, 0.05, 0.05]
P_NEW = [0.5, 0.3, 0.2]
KL_OLD_NEW = sum(p * math.log(p / q) for p, q in zip(P_OLD, P_NEW))  # about 0.37
KL_NEW_OLD = sum(q * math.log(q / p) for p, q in zip(P_OLD, P_NEW))  # about 0.52


def test_categorical_kl_direction_is_old_first():
    assert abs(KL_OLD_NEW - KL_NEW_OLD) > 0.1  # the pair really is asymmetric
    kl = categorical_kl(torch.log(torch.tensor([P_OLD])), torch.log(torch.tensor([P_NEW])))
    assert kl.item() == pytest.approx(KL_OLD_NEW, rel=1e-6)  # KL(pi_old || pi_new), the PPO paper's direction


def test_kl_penalty_loss():
    p_old, p_new = torch.tensor([P_OLD] * 2), torch.tensor([P_NEW] * 2)
    log_prob = torch.log(torch.tensor([0.9, 0.05]))  # ratio = 1 for both samples
    adv = torch.tensor([1.0, 3.0])
    loss, ratio = kl_penalty_loss(log_prob, log_prob, adv, torch.log(p_old), torch.log(p_new), kl_beta=2.0)
    # -mean(1 * A) + 2 * mean(KL(old || new)) = -2 + 2 * KL_OLD_NEW; the reversed direction would give -2 + 2 * KL_NEW_OLD
    assert loss.item() == pytest.approx(-2.0 + 2.0 * KL_OLD_NEW, rel=1e-6)
    assert loss.item() != pytest.approx(-2.0 + 2.0 * KL_NEW_OLD, rel=1e-3)
    assert torch.allclose(ratio, torch.ones(2))


def test_update_kl_beta_three_branches():
    target = 0.01  # thresholds: 0.01 / 1.5 = 0.00667 and 0.01 * 1.5 = 0.015
    assert update_kl_beta(1.0, 0.001, target) == 0.5  # too small a step: halve
    assert update_kl_beta(1.0, 0.02, target) == 2.0  # too large a step: double
    assert update_kl_beta(1.0, 0.01, target) == 1.0  # within the band: keep
    assert update_kl_beta(1.0, 0.015, target) == 1.0  # the band is inclusive


# ---- diagnostics ---------------------------------------------------------------------------------------------------------
def test_approx_kl_clip_fraction_and_ratio_max():
    assert approx_kl(torch.tensor([1.0, 2.0])) == pytest.approx((0.0 + (1.0 - math.log(2))) / 2)  # mean((r-1) - ln r)
    assert approx_kl(torch.ones(4)) == 0.0
    assert approx_kl(torch.tensor([0.5, 0.8, 1.3])) >= 0.0
    ratios = torch.tensor([1.0, 1.1, 1.3, 0.7, 0.85])  # |r - 1| = 0, 0.1, 0.3, 0.3, 0.15
    assert clip_fraction(ratios, 0.2) == pytest.approx(2 / 5)
    assert ratio_max(ratios) == pytest.approx(1.3)


def test_explained_variance():
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])  # population variance 1.25
    assert explained_variance(returns, returns) == pytest.approx(1.0)  # perfect critic
    assert explained_variance(torch.zeros(4), returns) == pytest.approx(0.0)  # a constant is no better than the mean
    residual = torch.tensor([1.0, -1.0, 1.0, -1.0])  # population variance 1
    assert explained_variance(returns + residual, returns) == pytest.approx(1 - 1 / 1.25)  # 0.2
    assert math.isnan(explained_variance(torch.zeros(3), torch.full((3,), 2.0)))  # Var(returns) = 0
