"""CLI entry point and ``PPOTrainer``: PPO-Clip and PPO-KL (report section 3.3), from scratch in PyTorch.

All plumbing (CLI, seeding, vector envs, logs, validation, checkpoints, resume, run_meta.json) is
``common.training.TrainingRun``; this file supplies only the algorithm:

    python -m ppo.train --reward balanced --seed 0 [--set clip_range=0.1] [--workers 4] [--resume]
    python -m ppo.train --variant supp-ppo-kl --reward balanced --seed 0 --set policy_loss=kl   # registered supplementary run

One call of ``PPOTrainer.collect_and_update`` is one PPO iteration:
  1. ``collect_rollout``: ``n_steps`` vector-env steps with the current policy (``n_envs * n_steps`` transitions), via
     ``ctx.step`` (the harness counts env steps, times the simulator and logs finished episodes);
  2. ``learn_from_rollout``: GAE advantages and lambda-returns, ``n_epochs`` passes over shuffled minibatches
     (``gradient_step``), the target-KL early stop and, for PPO-KL, the adaptive ``kl_beta``;
  3. one ``ctx.log_update`` call with every metric of ``common.train_logging.UPDATE_ALGO_METRICS["ppo"]``, reporting
     ``n_updates`` = the Adam steps of the rollout and ``n_samples`` = their summed minibatch sizes (the same units as
     DDQN, so ``update_idx`` and ``run_meta.json`` ``timing.n_updates`` count optimizer steps for both algorithms).
The formulas are in ``ppo/core.py``; ``ppo/README.md`` maps each one to the report and lists the code-level choices.

Contract with the harness (``common.training.Trainer``): ``step_granularity = n_envs * n_steps``; the vector env has
autoreset mode SAME_STEP (a finished episode returns the next episode's first observation, the terminal observation is
in ``infos["final_obs"]``); ``--workers`` only decides whether the envs run in this process or in subprocesses, the data
and therefore the result are identical.
"""

from __future__ import annotations

from common.parallel import limit_threads

limit_threads()  # before numpy / torch are imported (CLAUDE.md rule 7)

import copy  # noqa: E402
import math  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any, Mapping, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.distributions import Categorical  # noqa: E402

from common.actions import N_ACTIONS  # noqa: E402
from common.networks import OBS_DIM, build_action_network, build_value_network, count_parameters, flatten_obs  # noqa: E402
from common.training import TrainContext, Trainer, TrainingRun  # noqa: E402
from ppo import core  # noqa: E402
from ppo.agent import PPOAgent  # noqa: E402

ADAM_EPS = 1e-5  # Adam epsilon of the reference PPO implementation (OpenAI baselines ppo2)
HIDDEN_GAIN = math.sqrt(2.0)  # orthogonal init gain of the ReLU hidden layers (OpenAI baselines ppo2)
ACTOR_OUTPUT_GAIN = 0.01  # small final policy layer -> near-uniform initial policy (Andrychowicz et al. 2021)
CRITIC_OUTPUT_GAIN = 1.0  # value head (baselines ppo2)
# Purposes of the seeds derived from the run seed (common/seeding.py rule 5: derive from (run seed, purpose)).
INIT_STREAM = 1  # network initialization
SAMPLING_STREAM = 2  # action sampling during collection and minibatch shuffling


def derived_generator(run_seed: int, stream: int) -> torch.Generator:
    """A ``torch.Generator`` seeded from ``(run_seed, stream)`` (``numpy.random.SeedSequence`` mixes the two)."""
    seed = int(np.random.SeedSequence([run_seed, stream]).generate_state(1)[0])
    return torch.Generator().manual_seed(seed)


def init_weights(network: nn.Sequential, output_gain: float, generator: torch.Generator) -> None:
    """Orthogonal weights (gain sqrt(2) for the hidden layers, ``output_gain`` for the last layer), zero biases."""
    linear_layers = [layer for layer in network if isinstance(layer, nn.Linear)]
    for layer in linear_layers[:-1]:
        nn.init.orthogonal_(layer.weight, gain=HIDDEN_GAIN, generator=generator)
        nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(linear_layers[-1].weight, gain=output_gain, generator=generator)
    nn.init.zeros_(linear_layers[-1].bias)


@dataclass
class Rollout:
    """Preallocated rollout buffer; index ``[t, i]`` = vector-env step ``t``, env ``i``.

    The transition of step ``t`` goes from ``obs[t]`` with ``actions[t]`` to the next state. ``final_values[t, i]`` is
    ``V(final_obs)`` where ``done[t, i]`` (the true last state of the episode), 0 elsewhere.
    """

    obs: torch.Tensor  # (n_steps, n_envs, OBS_DIM) float32
    actions: torch.Tensor  # (n_steps, n_envs) int64
    log_probs: torch.Tensor  # (n_steps, n_envs)  log pi_old(a_t | s_t)
    logits: torch.Tensor  # (n_steps, n_envs, N_ACTIONS)  pi_old, for the exact KL of PPO-KL
    values: torch.Tensor  # (n_steps, n_envs)  V(s_t) before the update
    rewards: torch.Tensor  # (n_steps, n_envs)
    terminated: torch.Tensor  # (n_steps, n_envs) bool
    done: torch.Tensor  # (n_steps, n_envs) bool, terminated or truncated
    final_values: torch.Tensor  # (n_steps, n_envs)

    @classmethod
    def empty(cls, n_steps: int, n_envs: int) -> "Rollout":
        return cls(
            obs=torch.zeros(n_steps, n_envs, OBS_DIM),
            actions=torch.zeros(n_steps, n_envs, dtype=torch.int64),
            log_probs=torch.zeros(n_steps, n_envs),
            logits=torch.zeros(n_steps, n_envs, N_ACTIONS),
            values=torch.zeros(n_steps, n_envs),
            rewards=torch.zeros(n_steps, n_envs),
            terminated=torch.zeros(n_steps, n_envs, dtype=torch.bool),
            done=torch.zeros(n_steps, n_envs, dtype=torch.bool),
            final_values=torch.zeros(n_steps, n_envs),
        )


def observation_tensor(obs: Any) -> torch.Tensor:
    """``(batch, 5, 5)`` numpy observations -> ``(batch, OBS_DIM)`` float32 tensor (one conversion for the batch)."""
    return torch.as_tensor(flatten_obs(np.asarray(obs, dtype=np.float32)))


class PPOTrainer(Trainer):
    """PPO with separate actor and critic (``ppo_shared_trunk = False``), one Adam optimizer over both."""

    def __init__(self, ctx: TrainContext) -> None:
        super().__init__(ctx)
        cfg = ctx.cfg
        self.cfg = cfg
        self.step_granularity = cfg.rollout_size

        self.actor = build_action_network(cfg)  # 25 -> 256 -> 256 -> 5 logits
        self.critic = build_value_network(cfg)  # 25 -> 256 -> 256 -> 1 value
        init_generator = derived_generator(ctx.seed, INIT_STREAM)
        init_weights(self.actor, ACTOR_OUTPUT_GAIN, init_generator)
        init_weights(self.critic, CRITIC_OUTPUT_GAIN, init_generator)
        self.all_parameters = [*self.actor.parameters(), *self.critic.parameters()]
        # Constant learning rate (no annealing): keeps the lr axis of the sensitivity analysis easy to interpret.
        self.optimizer = torch.optim.Adam(self.all_parameters, lr=cfg.lr, eps=ADAM_EPS)

        self.generator = derived_generator(ctx.seed, SAMPLING_STREAM)  # action sampling and minibatch shuffling
        self.kl_beta = cfg.kl_beta_init  # PPO-KL penalty coefficient (unused by PPO-Clip)
        self.n_updates = 0  # PPO iterations done (one per rollout)
        self.rollout = Rollout.empty(cfg.n_steps, cfg.n_envs)

    # -- one PPO iteration ---------------------------------------------------------------------------------------------
    def collect_and_update(self, n_env_steps: int) -> None:
        if n_env_steps != self.cfg.rollout_size:
            raise ValueError(f"PPO advances by whole rollouts of {self.cfg.rollout_size} env steps, got {n_env_steps}")
        last_values = self.collect_rollout()
        with self.ctx.timers.update():  # GAE + epochs: everything from the rollout to the new weights
            metrics, n_gradient_steps, n_samples = self.learn_from_rollout(last_values)
        # n_updates = the Adam steps of this rollout (the unit DDQN reports too), n_samples = their summed minibatch sizes
        self.ctx.log_update(n_updates=n_gradient_steps, n_samples=n_samples, **metrics)

    @torch.no_grad()
    def collect_rollout(self) -> torch.Tensor:
        """Fill ``self.rollout`` with ``n_steps`` vector steps; return ``V`` of the observation after the rollout."""
        rollout = self.rollout
        for t in range(self.cfg.n_steps):
            obs = observation_tensor(self.ctx.obs)  # (n_envs, OBS_DIM)
            logits = self.actor(obs)  # (n_envs, N_ACTIONS)
            policy = Categorical(logits=logits)
            actions = torch.multinomial(policy.probs, num_samples=1, generator=self.generator).squeeze(-1)  # (n_envs,)
            values = self.critic(obs).squeeze(-1)  # (n_envs,)

            _, rewards, terminated, truncated, infos = self.ctx.step(actions.numpy())

            rollout.obs[t] = obs
            rollout.actions[t] = actions
            rollout.log_probs[t] = policy.log_prob(actions)
            rollout.logits[t] = logits
            rollout.values[t] = values
            rollout.rewards[t] = torch.as_tensor(rewards, dtype=torch.float32)
            rollout.terminated[t] = torch.as_tensor(terminated)
            rollout.done[t] = torch.as_tensor(terminated | truncated)
            rollout.final_values[t] = self.final_obs_values(infos, terminated | truncated)
        return self.critic(observation_tensor(self.ctx.obs)).squeeze(-1)  # (n_envs,)

    @torch.no_grad()
    def final_obs_values(self, infos: Mapping[str, Any], done: np.ndarray) -> torch.Tensor:
        """``V(final_obs)`` for the envs whose episode ended in this step, 0 for the others; shape ``(n_envs,)``.

        SAME_STEP autoreset: ``infos["_final_obs"]`` masks the envs that ended, ``infos["final_obs"]`` holds their last
        observation (an object array, ``None`` for the other envs). Only the few ended envs are stacked. ``done``
        (``(n_envs,)`` bool, terminated or truncated) must match that mask: a done env without a final observation
        would silently bootstrap from the next episode, so it raises instead.
        """
        final_values = torch.zeros(self.cfg.n_envs)  # (n_envs,)
        if not done.any():
            return final_values
        ended = infos.get("_final_obs")
        if ended is None or not np.array_equal(ended, done):
            raise RuntimeError(
                f"episode end without a matching final observation: done = {done.tolist()}, "
                f"infos['_final_obs'] = {None if ended is None else ended.tolist()} (the vector env must use SAME_STEP "
                "autoreset)"
            )
        ended_envs = np.flatnonzero(ended)  # (n_ended,)
        final_obs = np.stack([infos["final_obs"][i] for i in ended_envs])  # (n_ended, 5, 5)
        final_values[torch.as_tensor(ended_envs)] = self.critic(observation_tensor(final_obs)).squeeze(-1)
        return final_values

    def learn_from_rollout(self, last_values: torch.Tensor) -> tuple[dict[str, float], int, int]:
        """GAE, the epochs over minibatches and the KL bookkeeping.

        Returns ``(metrics, n_gradient_steps, n_samples)``: the metrics of ``ctx.log_update``, the number of Adam steps
        taken and their summed minibatch sizes (``n_epochs_done * rollout_size``).
        """
        cfg, rollout = self.cfg, self.rollout
        next_values = core.next_state_values(rollout.values, last_values, rollout.final_values, rollout.done)
        advantages, returns = core.compute_gae(
            rollout.rewards, rollout.values, next_values, rollout.terminated, rollout.done, cfg.gamma, cfg.gae_lambda
        )  # (n_steps, n_envs) each
        explained_variance = core.explained_variance(rollout.values, returns)  # values from before the update

        metrics, n_gradient_steps, n_samples = self.run_epochs(advantages, returns)

        # Exact mean KL(pi_old || pi_new) over the whole rollout after ALL epochs, for both losses: the policy change the
        # pre-step diagnostics (approx_kl, ratio_max) miss, and PPO-KL's measured KL d of the beta rule.
        kl_after_update = self.mean_rollout_kl()
        if cfg.policy_loss == "kl":  # adapt beta once per update, on the KL of the whole rollout
            self.kl_beta = core.update_kl_beta(self.kl_beta, kl_after_update, cfg.kl_target)
        self.n_updates += 1
        metrics = {
            "lr": cfg.lr,
            **metrics,
            "explained_variance": explained_variance,
            "kl_beta": self.kl_beta if cfg.policy_loss == "kl" else math.nan,  # the value after update_kl_beta
            "kl_after_update": kl_after_update,
        }
        return metrics, n_gradient_steps, n_samples

    def run_epochs(self, advantages: torch.Tensor, returns: torch.Tensor) -> tuple[dict[str, float], int, int]:
        """``n_epochs`` passes over the rollout in shuffled minibatches, with the optional target-KL early stop.

        Early stop (only if ``target_kl`` is set): after an epoch whose mean ``approx_kl`` exceeds ``target_kl`` the
        remaining epochs are skipped. Returns ``(metrics, n_gradient_steps, n_samples)``: the per-minibatch metrics
        averaged over all minibatches that ran (``ratio_max``: the maximum) plus ``n_epochs_done``, the number of
        ``gradient_step`` calls and the number of samples they processed.
        """
        cfg, rollout = self.cfg, self.rollout
        # (n_steps, n_envs, ...) -> (rollout_size, ...): every transition is one sample
        obs = rollout.obs.reshape(-1, OBS_DIM)  # (rollout_size, OBS_DIM)
        actions = rollout.actions.reshape(-1)  # (rollout_size,)
        old_log_probs = rollout.log_probs.reshape(-1)  # (rollout_size,)
        old_logits = rollout.logits.reshape(-1, N_ACTIONS)  # (rollout_size, N_ACTIONS)
        advantages = advantages.reshape(-1)  # (rollout_size,)
        returns = returns.reshape(-1)  # (rollout_size,)

        history: dict[str, list[float]] = {}  # metric -> one value per minibatch
        n_epochs_done = n_gradient_steps = n_samples = 0
        for _ in range(cfg.n_epochs):
            permutation = torch.randperm(cfg.rollout_size, generator=self.generator)  # (rollout_size,)
            epoch_kls = []
            for start in range(0, cfg.rollout_size, cfg.minibatch_size):
                idx = permutation[start : start + cfg.minibatch_size]  # (minibatch_size,); PPOConfig: it divides the rollout
                stats = self.gradient_step(obs[idx], actions[idx], old_log_probs[idx], old_logits[idx],
                                           advantages[idx], returns[idx])
                n_gradient_steps += 1
                n_samples += len(idx)
                for name, value in stats.items():
                    history.setdefault(name, []).append(value)
                epoch_kls.append(stats["approx_kl"])
            n_epochs_done += 1
            if cfg.target_kl is not None and float(np.mean(epoch_kls)) > cfg.target_kl:
                break

        metrics = {name: float(np.mean(values)) for name, values in history.items()}
        metrics["ratio_max"] = max(history["ratio_max"])
        metrics["n_epochs_done"] = n_epochs_done
        return metrics, n_gradient_steps, n_samples

    def gradient_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_logits: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        """One Adam step on one minibatch: ``loss = policy_loss - ent_coef * entropy + vf_coef * value_loss``.

        Policy loss: eq:ppo-clip (``policy_loss = "clip"``) or the KL-penalized surrogate (``"kl"``). Value loss:
        ``mean((V(s) - returns)^2)`` with the lambda-returns as targets, no value clipping (Andrychowicz et al. 2021
        found it hurts). The gradient norm over both networks is clipped to ``max_grad_norm``; the logged ``grad_norm``
        is the norm before clipping.
        """
        cfg = self.cfg
        advantages = core.normalize_advantages(advantages)  # (batch,)

        logits = self.actor(obs)  # (batch, N_ACTIONS)
        policy = Categorical(logits=logits)
        new_log_probs = policy.log_prob(actions)  # (batch,)
        if cfg.policy_loss == "clip":
            policy_loss, ratio = core.clipped_surrogate_loss(new_log_probs, old_log_probs, advantages, cfg.clip_range)
        else:
            policy_loss, ratio = core.kl_penalty_loss(new_log_probs, old_log_probs, advantages, old_logits, logits,
                                                      self.kl_beta)
        values = self.critic(obs).squeeze(-1)  # (batch,)
        value_loss = ((values - returns) ** 2).mean()
        entropy = policy.entropy().mean()
        loss = policy_loss - cfg.ent_coef * entropy + cfg.vf_coef * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.all_parameters, cfg.max_grad_norm)  # norm before clipping
        self.optimizer.step()

        ratio = ratio.detach()  # diagnostics of the ratios this step was computed with
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "grad_norm": grad_norm.item(),
            "approx_kl": core.approx_kl(ratio),
            "clip_fraction": core.clip_fraction(ratio, cfg.clip_range),  # a diagnostic for PPO-KL too
            "ratio_max": core.ratio_max(ratio),
        }

    @torch.no_grad()
    def mean_rollout_kl(self) -> float:
        """Mean exact ``KL(pi_old || pi_new)`` over every state of the rollout (PPO-KL's measured KL)."""
        new_logits = self.actor(self.rollout.obs.reshape(-1, OBS_DIM))  # (rollout_size, N_ACTIONS)
        old_logits = self.rollout.logits.reshape(-1, N_ACTIONS)  # (rollout_size, N_ACTIONS)
        return core.categorical_kl(old_logits, new_logits).mean().item()

    # -- checkpoints ---------------------------------------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        # The rollout buffer is not saved: the harness checkpoints only between collect_and_update calls, where the
        # buffer holds nothing that is still needed. ctx.obs (the observation to continue from) is saved by the harness.
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "kl_beta": self.kl_beta,
            "n_updates": self.n_updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])
        self.kl_beta = float(state["kl_beta"])
        self.n_updates = int(state["n_updates"])

    def make_eval_agent(self) -> PPOAgent:
        return PPOAgent(self.cfg, copy.deepcopy(self.actor), seed=self.ctx.seed)

    def n_parameters(self) -> int:
        return count_parameters(self.actor) + count_parameters(self.critic)  # 73,733 + 72,705 = 146,438


def main(argv: Optional[Sequence[str]] = None) -> None:
    TrainingRun.from_cli("ppo", PPOTrainer, argv).run()


if __name__ == "__main__":
    main()
