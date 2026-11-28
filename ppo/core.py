"""The pure math of PPO: GAE, the two policy losses, the adaptive KL coefficient and the update diagnostics.

No environment, no network, no optimizer lives here: every function maps tensors to tensors (or to Python floats for
the logged diagnostics) and its docstring states the formula it implements, with the label of the report equation
(``report/chapters/03_methods.tex``, section 3.3) where there is one. ``ppo/train.py`` does the env interaction and the
bookkeeping and calls these functions.

Shapes. Rollout tensors are ``(n_steps, n_envs)``: index ``t`` is the vector-env step, the second index the env.
Minibatch tensors are ``(batch,)`` (one entry per sampled transition) or ``(batch, n_actions)`` for logits.

Episode-end flags (docs/evaluation_protocol.md section 1: a time limit is not a termination). For the transition of
step ``t`` (from ``s_t`` with action ``a_t`` to ``s_{t+1}``):
    terminated_t  the episode ended in a true terminal state (a collision): V(s_{t+1}) is 0, no bootstrap.
    done_t        the episode ended for any reason (terminated or truncated by the time limit): the GAE recursion is
                  cut, so no advantage term of the next episode leaks into this one.
In the notation of eq:gae these are m^term_{t+1} and m_{t+1}.
"""

from __future__ import annotations

import math

import torch

ADV_EPS = 1e-8  # added to the std in the advantage normalization (OpenAI baselines ppo2 uses the same value)
KL_ADAPT_FACTOR = 1.5  # PPO paper (Schulman et al. 2017, section 4): adapt beta when KL < d_targ / 1.5 or > 1.5 d_targ
KL_BETA_MULTIPLIER = 2.0  # PPO paper: beta is halved or doubled


# ---------------------------------------------------------------------------------------------
# Advantages (eq:gae)
# ---------------------------------------------------------------------------------------------
def next_state_values(
    values: torch.Tensor, last_values: torch.Tensor, final_values: torch.Tensor, done: torch.Tensor
) -> torch.Tensor:
    """``V(s_{t+1})`` for every step of a rollout, the bootstrap value used in delta_t of eq:gae.

    ``values[t] = V(s_t)`` was computed during collection, so for a running episode ``V(s_{t+1}) = values[t+1]``, and
    after the last step of the rollout ``V(s_{n_steps}) = last_values`` (value of the observation after the rollout).
    Where ``done_t`` the vector env has already reset (autoreset mode SAME_STEP), so ``values[t+1]`` belongs to the NEXT
    episode; the value of the true last state, ``V(final_obs)``, is taken from ``final_values`` instead. (It only
    matters after a truncation: ``compute_gae`` multiplies it by 0 after a termination.)

    Args:
        values: ``(n_steps, n_envs)``, ``V(s_t)``.
        last_values: ``(n_envs,)``, ``V`` of the observation after the last rollout step.
        final_values: ``(n_steps, n_envs)``, ``V(final_obs)`` where ``done``, any value elsewhere (ignored).
        done: ``(n_steps, n_envs)`` bool, the episode ended at step t (terminated or truncated).

    Returns:
        ``(n_steps, n_envs)`` tensor of ``V(s_{t+1})``.
    """
    shifted = torch.empty_like(values)  # (n_steps, n_envs)
    shifted[:-1] = values[1:]  # running episode: the value of the next stored state
    shifted[-1] = last_values  # last step of the rollout: the state after the rollout
    return torch.where(done, final_values, shifted)  # episode end: the value of the episode's own last state


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation with the two episode-end masks. Implements eq:gae in report section 3.3.

        delta_t = r_t + gamma * (1 - terminated_t) * V(s_{t+1}) - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1},      A_{n_steps} = 0
        returns_t = A_t + V(s_t)                                           (the lambda-returns, critic targets)

    A truncation (done, not terminated) keeps the bootstrap ``V(s_{t+1})`` but cuts the recursion; a termination zeroes
    both. An env that is terminated and truncated at the same step is treated as terminated (value 0). After the last
    step of the rollout ``A_{n_steps} = 0``: the advantage of a running episode is bootstrapped through ``next_values``.

    Args:
        rewards, values, next_values: ``(n_steps, n_envs)`` float; ``next_values`` from ``next_state_values``.
        terminated, done: ``(n_steps, n_envs)`` bool; ``done = terminated | truncated``.
        gamma, gae_lambda: discount and GAE lambda.

    Returns:
        ``(advantages, returns)``, both ``(n_steps, n_envs)``.
    """
    if (terminated & ~done).any():
        raise ValueError("terminated implies done: pass done = terminated | truncated")
    not_terminated = 1.0 - terminated.float()  # (n_steps, n_envs)
    not_done = 1.0 - done.float()  # (n_steps, n_envs)
    deltas = rewards + gamma * not_terminated * next_values - values  # (n_steps, n_envs)

    n_steps = rewards.shape[0]
    advantages = torch.zeros_like(values)  # (n_steps, n_envs)
    next_advantage = torch.zeros_like(values[0])  # A_{t+1}, starts as A_{n_steps} = 0, (n_envs,)
    for t in reversed(range(n_steps)):  # backward in time, vectorized over envs
        next_advantage = deltas[t] + gamma * gae_lambda * not_done[t] * next_advantage
        advantages[t] = next_advantage
    returns = advantages + values  # (n_steps, n_envs)
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """``(A - mean(A)) / (std(A) + 1e-8)`` over one minibatch.

    A code-level choice of the reference PPO implementation (OpenAI baselines ppo2); Andrychowicz et al. 2021
    (section 3.3) found per-minibatch advantage normalization to have little effect on performance. The population std
    (``correction=0``) is used as in baselines; a minibatch of one sample gives 0.
    """
    return (advantages - advantages.mean()) / (advantages.std(correction=0) + ADV_EPS)


# ---------------------------------------------------------------------------------------------
# Policy losses (eq:ppo-clip and the adaptive KL penalty)
# ---------------------------------------------------------------------------------------------
def probability_ratio(new_log_prob: torch.Tensor, old_log_prob: torch.Tensor) -> torch.Tensor:
    """``rho_t = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t) = exp(log pi_theta - log pi_theta_old)`` (eq:ppo-clip)."""
    return torch.exp(new_log_prob - old_log_prob)


def clipped_surrogate_loss(
    new_log_prob: torch.Tensor, old_log_prob: torch.Tensor, advantages: torch.Tensor, clip_range: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPO-Clip policy loss, the negative of L^CLIP. Implements eq:ppo-clip in report section 3.3.

        loss = -mean_t( min( rho_t * A_t,  clip(rho_t, 1 - eps, 1 + eps) * A_t ) )

    Args:
        new_log_prob, old_log_prob, advantages: ``(batch,)``; ``old_log_prob`` from the rollout (no gradient).
        clip_range: eps_clip.

    Returns:
        ``(loss, ratio)``: the scalar loss to minimize and the ``(batch,)`` ratios (for the diagnostics).
    """
    ratio = probability_ratio(new_log_prob, old_log_prob)  # (batch,)
    surrogate_unclipped = ratio * advantages  # (batch,)
    surrogate_clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages  # (batch,)
    surrogate = torch.min(surrogate_unclipped, surrogate_clipped)  # (batch,) pessimistic bound
    loss = -surrogate.mean()
    return loss, ratio


def categorical_kl(old_logits: torch.Tensor, new_logits: torch.Tensor) -> torch.Tensor:
    """Exact ``KL(pi_old(.|s) || pi_new(.|s)) = sum_a p_old(a) * (log p_old(a) - log p_new(a))`` per state.

    Args:
        old_logits, new_logits: ``(batch, n_actions)`` unnormalized log-probabilities.

    Returns:
        ``(batch,)`` KL divergences (>= 0).
    """
    old_log_probs = torch.log_softmax(old_logits, dim=-1)  # (batch, n_actions)
    new_log_probs = torch.log_softmax(new_logits, dim=-1)  # (batch, n_actions)
    old_probs = torch.exp(old_log_probs)  # (batch, n_actions)
    return (old_probs * (old_log_probs - new_log_probs)).sum(dim=-1)  # (batch,)


def kl_penalty_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
    kl_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPO-KL policy loss: the unclipped surrogate with an adaptive KL penalty (report section 3.3, PPO-KL).

        loss = -mean_t( rho_t * A_t ) + beta * mean_t( KL(pi_old(.|s_t) || pi_theta(.|s_t)) )

    The KL is exact over the actions (``categorical_kl``), the direction of the PPO paper (old policy first).

    Args:
        new_log_prob, old_log_prob, advantages: ``(batch,)``.
        old_logits, new_logits: ``(batch, n_actions)``; ``old_logits`` from the rollout (no gradient).
        kl_beta: the penalty coefficient beta.

    Returns:
        ``(loss, ratio)`` as in ``clipped_surrogate_loss``.
    """
    ratio = probability_ratio(new_log_prob, old_log_prob)  # (batch,)
    surrogate = ratio * advantages  # (batch,)
    kl = categorical_kl(old_logits, new_logits)  # (batch,)
    loss = -surrogate.mean() + kl_beta * kl.mean()
    return loss, ratio


def update_kl_beta(kl_beta: float, measured_kl: float, kl_target: float) -> float:
    """Adaptive KL coefficient of the PPO paper (Schulman et al. 2017, section 4), applied once per update:

        KL < d_targ / 1.5  ->  beta / 2      (the policy moved too little: weaken the penalty)
        KL > d_targ * 1.5  ->  beta * 2      (the policy moved too much: strengthen the penalty)
        otherwise          ->  beta
    """
    if measured_kl < kl_target / KL_ADAPT_FACTOR:
        return kl_beta / KL_BETA_MULTIPLIER
    if measured_kl > kl_target * KL_ADAPT_FACTOR:
        return kl_beta * KL_BETA_MULTIPLIER
    return kl_beta


# ---------------------------------------------------------------------------------------------
# Diagnostics (logged in update_log.csv, never used in a gradient)
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def approx_kl(ratio: torch.Tensor) -> float:
    """Estimate of ``KL(pi_old || pi_new)`` on the sampled actions: ``mean((rho - 1) - log rho)``.

    The "k3" estimator (unbiased and always >= 0); the same formula as Stable-Baselines3 logs, so the values are
    comparable with the SB3 calibration runs (docs/calibration.md). Used by the target-KL early stop.
    """
    return ((ratio - 1.0) - torch.log(ratio)).mean().item()


@torch.no_grad()
def clip_fraction(ratio: torch.Tensor, clip_range: float) -> float:
    """Fraction of samples whose ratio lies outside the clip range: ``mean(|rho - 1| > eps_clip)``."""
    return ((ratio - 1.0).abs() > clip_range).float().mean().item()


@torch.no_grad()
def ratio_max(ratio: torch.Tensor) -> float:
    """``max(rho)``: clipping does not bound an individual step (Engstrom et al. 2020), this shows how far it went."""
    return ratio.max().item()


@torch.no_grad()
def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """``1 - Var(returns - values) / Var(returns)`` (population variances); NaN if ``Var(returns) = 0``.

    1 = the critic predicts the returns perfectly, 0 = no better than a constant, < 0 = worse. Computed once per rollout
    with the values from before the update.
    """
    var_returns = returns.var(correction=0).item()
    if var_returns == 0.0:
        return math.nan
    return 1.0 - (returns - values).var(correction=0).item() / var_returns
