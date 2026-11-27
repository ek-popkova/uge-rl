"""``RandomTrainer``: a test double that exercises the whole training harness without any learning.

It plays uniformly random actions and fakes an "update" (a random walk of a 4-parameter vector) that
logs dummy update metrics in the schema of the chosen algorithm. It is the proof that
``common.training.TrainingRun`` writes every file with the right schema, resumes exactly, and behaves
identically for ``--workers 1`` and ``--workers N``. NO ALGORITHM LIVES HERE.

    python -m common.random_trainer --algo ddqn --total-steps 200 --val-every 100 --val-episodes 2 \
        --checkpoint-every 100 --benchmark-steps 0 --out-dir results/_smoke
    python -m common.random_trainer --algo ppo --set n_envs=2 --set n_steps=20 --set minibatch_size=20 ...

``--algo ddqn`` drives a single env (granularity 1, one fake update every ``FAKE_UPDATE_EVERY`` steps),
``--algo ppo`` a vector env of ``n_envs`` envs (granularity ``n_envs * n_steps``, one fake update per
rollout).
"""

from __future__ import annotations

import sys
from typing import Any, Optional, Sequence

import numpy as np
import torch

from common.actions import N_ACTIONS
from common.baselines import RandomAgent
from common.training import TrainContext, Trainer, TrainingRun

FAKE_UPDATE_EVERY = 10  # env steps between fake updates in single-env mode


class RandomPolicyAgent(RandomAgent):
    """Per-episode seeded random policy that can be saved and loaded (a stand-in for a trained agent)."""

    def save(self, path: str) -> None:
        torch.save({"seed": self.seed}, path)

    def load(self, path: str) -> None:
        self.seed = int(torch.load(path, weights_only=False)["seed"])


class RandomTrainer(Trainer):
    """Random actions + fake updates; see the module docstring."""

    def __init__(self, ctx: TrainContext) -> None:
        super().__init__(ctx)
        self.rng = np.random.default_rng([ctx.seed, 99])
        self.params = torch.zeros(4)
        self.n_fake_updates = 0
        self.step_granularity = ctx.cfg.rollout_size if ctx.vector else 1

    def collect_and_update(self, n_env_steps: int) -> None:
        ctx = self.ctx
        if ctx.vector:
            for _ in range(n_env_steps // ctx.n_envs):
                ctx.step(self.rng.integers(N_ACTIONS, size=ctx.n_envs))
            self._fake_update()
            return
        for _ in range(n_env_steps):
            _, _, terminated, truncated, _ = ctx.step(int(self.rng.integers(N_ACTIONS)))
            if terminated or truncated:
                ctx.reset_episode()
            if ctx.env_step % FAKE_UPDATE_EVERY == 0:
                self._fake_update()

    def _fake_update(self) -> None:
        ctx = self.ctx
        with ctx.timers.update():
            self.params += torch.as_tensor(self.rng.standard_normal(4), dtype=torch.float32)
            loss = float(self.params.abs().mean())
            u = self.rng.random(8)
        self.n_fake_updates += 1
        common = {"lr": ctx.cfg.lr, "loss": loss, "grad_norm": float(u[0])}
        cfg = ctx.cfg
        if ctx.algo == "ddqn":  # one fake gradient step on one batch
            ctx.log_update(n_updates=1, n_samples=cfg.batch_size, **common, td_loss=loss, q_mean=float(u[1]),
                           q_max=float(u[2]) + 1.0, target_q_mean=float(u[3]),
                           epsilon=max(0.05, 1.0 - ctx.env_step / 1000), buffer_size=ctx.env_step)
        else:  # one fake rollout update: n_epochs passes over rollout_size samples in minibatches, as PPOTrainer counts it
            ctx.log_update(n_updates=cfg.n_epochs * (cfg.rollout_size // cfg.minibatch_size),
                           n_samples=cfg.n_epochs * cfg.rollout_size, **common, policy_loss=float(u[1]),
                           value_loss=float(u[2]), entropy=float(u[3]), approx_kl=float(u[4]) * 0.01,
                           clip_fraction=float(u[5]), explained_variance=float(u[6]), n_epochs_done=cfg.n_epochs,
                           ratio_max=1.0 + float(u[7]), kl_beta=cfg.kl_beta_init, kl_after_update=float(u[4]) * 0.01)

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self.rng.bit_generator.state, "params": self.params.clone(), "n_fake_updates": self.n_fake_updates}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
        self.params = state["params"].clone()
        self.n_fake_updates = int(state["n_fake_updates"])

    def make_eval_agent(self) -> RandomPolicyAgent:
        return RandomPolicyAgent(self.ctx.seed)

    def n_parameters(self) -> int:
        return int(self.params.numel())


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    algo = "ddqn"
    if "--algo" in argv:
        i = argv.index("--algo")
        algo = argv[i + 1]
        del argv[i : i + 2]
    TrainingRun.from_cli(algo, RandomTrainer, argv).run()


if __name__ == "__main__":
    main()
