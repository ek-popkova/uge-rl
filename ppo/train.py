"""CLI entry point for training PPO (the algorithm is still a stub).

All plumbing (CLI, seeding, vector envs, logs, validation, checkpoints, resume, run_meta.json) is
``common.training.TrainingRun``; this file only supplies the algorithm. To implement PPO, fill in
``PPOTrainer`` below and nothing else; every number needed for the comparisons is then logged by the
harness:

    python -m ppo.train --reward balanced --seed 0 [--set clip_range=0.1] [--workers 4] [--resume]

Contract of ``PPOTrainer`` (see ``common.training.Trainer``):
  * ``step_granularity = cfg.rollout_size`` (``n_envs * n_steps``, set it in ``__init__``);
    ``collect_and_update`` collects one rollout with ``ctx.step(actions)`` on the vector env
    (``ctx.env`` has autoreset mode SAME_STEP: a finished episode returns the next episode's first
    observation, the terminal observation is in ``infos["final_obs"]``; bootstrap from it when the
    episode was only truncated) and runs the epochs inside ``with ctx.timers.update():``.
  * ``ctx.log_update(lr=..., loss=..., grad_norm=..., policy_loss=..., value_loss=..., entropy=...,
    approx_kl=..., clip_fraction=..., explained_variance=..., n_epochs_done=...)`` once per update
    (grad_norm before clipping, other metrics averaged over the minibatches).
  * ``--workers 1`` runs the ``n_envs`` envs in this process, ``--workers > 1`` in subprocesses; the data
    is identical, so the algorithm must not depend on it.
  * ``state_dict`` must contain network, optimizer, rollout state and the sampling RNG state.
"""

from __future__ import annotations

from common.parallel import limit_threads

limit_threads()  # before numpy / torch are imported (CLAUDE.md rule 7)

from typing import Any, Optional, Sequence  # noqa: E402

from common.agent_base import Agent  # noqa: E402
from common.training import Trainer, TrainingRun  # noqa: E402


class PPOTrainer(Trainer):
    """PPO training loop (TODO: not implemented)."""

    @classmethod
    def check_available(cls) -> None:
        raise NotImplementedError("TODO: implement PPOTrainer (the training harness in common.training is ready)")

    def collect_and_update(self, n_env_steps: int) -> None:
        raise NotImplementedError("TODO: rollout on the vector env, GAE, clipped PPO update")

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError("TODO: network, optimizer, rollout state, RNG state, counters")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError("TODO: inverse of state_dict")

    def make_eval_agent(self) -> Agent:
        raise NotImplementedError("TODO: return a light PPOAgent snapshot (argmax / mean action, no rollout buffer)")

    def n_parameters(self) -> int:
        raise NotImplementedError("TODO: number of trainable parameters of the actor-critic network")


def main(argv: Optional[Sequence[str]] = None) -> None:
    TrainingRun.from_cli("ppo", PPOTrainer, argv).run()


if __name__ == "__main__":
    main()
