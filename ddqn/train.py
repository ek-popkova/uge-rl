"""CLI entry point for training Double DQN (the algorithm is still a stub).

All plumbing (CLI, seeding, envs, logs, validation, checkpoints, resume, run_meta.json) is
``common.training.TrainingRun``; this file only supplies the algorithm. To implement DDQN, fill in
``DDQNTrainer`` below and nothing else; every number needed for the comparisons is then logged by the
harness:

    python -m ddqn.train --reward balanced --seed 0 [--set lr=0.001] [--workers 1] [--resume]

Contract of ``DDQNTrainer`` (see ``common.training.Trainer``):
  * ``step_granularity = 1``; ``collect_and_update(1)`` plays one env step with
    ``ctx.step(action)`` (``ctx.reset_episode()`` after a finished episode, ``ctx.obs`` is the current
    observation) and does the gradient updates that are due inside ``with ctx.timers.update():``.
  * after each update call ``ctx.log_update(lr=..., loss=..., grad_norm=..., td_loss=..., q_mean=...,
    q_max=..., target_q_mean=..., epsilon=..., buffer_size=...)`` (grad_norm before clipping).
  * time-limit truncation bootstraps, only a crash (``terminated``) is terminal (evaluation_protocol.md).
  * ``state_dict`` must contain networks, optimizer, replay buffer, the exploration / sampling RNG state
    and counters, so that ``--resume`` continues exactly.
"""

from __future__ import annotations

from common.parallel import limit_threads

limit_threads()  # before numpy / torch are imported (CLAUDE.md rule 7)

from typing import Any, Optional, Sequence  # noqa: E402

from common.agent_base import Agent  # noqa: E402
from common.training import Trainer, TrainingRun  # noqa: E402


class DDQNTrainer(Trainer):
    """Double DQN training loop (TODO: not implemented)."""

    step_granularity = 1

    @classmethod
    def check_available(cls) -> None:
        raise NotImplementedError("TODO: implement DDQNTrainer (the training harness in common.training is ready)")

    def collect_and_update(self, n_env_steps: int) -> None:
        raise NotImplementedError("TODO: epsilon-greedy step, replay buffer, Double DQN update")

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError("TODO: networks, optimizer, replay buffer, RNG state, counters")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError("TODO: inverse of state_dict")

    def make_eval_agent(self) -> Agent:
        raise NotImplementedError("TODO: return a light DDQNAgent snapshot (greedy policy, no replay buffer)")

    def n_parameters(self) -> int:
        raise NotImplementedError("TODO: number of trainable parameters of the Q-network")


def main(argv: Optional[Sequence[str]] = None) -> None:
    TrainingRun.from_cli("ddqn", DDQNTrainer, argv).run()


if __name__ == "__main__":
    main()
