"""Training environments: the shared ``EpisodeRecorder`` inside the env, vector envs, env state for resume.

``TrainEnvWrapper`` wraps one env (inside the worker process for vector envs, where the simulator
state cannot be reached from the parent) and runs a ``common.metrics.EpisodeRecorder`` over every
training episode, so a ``train_log.csv`` row is computed by exactly the code that computes a test row.
At the last step of an episode it adds ``info["episode_metrics"]`` (the finished record without
``seed``). ``finished_episodes`` extracts these records from a single-env ``info`` or from the
gymnasium vector ``infos`` (autoreset mode SAME_STEP: the finished episode sits in ``final_info``).

Vector envs: ``workers <= 1`` gives a ``SyncVectorEnv`` (this process), otherwise an
``AsyncVectorEnv`` (one subprocess per env). Both produce identical data; ``--workers`` is a pure
speed switch and ``n_envs`` stays an algorithmic hyperparameter (``PPOConfig.n_envs``).
In SAME_STEP mode a step that ends an episode returns the first observation of the next episode;
the terminal observation is in ``infos["final_obs"]`` and the terminal info in ``infos["final_info"]``.

Resume: the simulator state (road, vehicles, RNG) of an env is picklable (about 10 kB), and
``get_state`` / ``set_state`` snapshot the env together with its recorder. A resumed run therefore
continues the in-flight episodes exactly where the checkpoint was taken.
"""

from __future__ import annotations

import functools
import pickle
from typing import Any, Mapping, Optional

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

from common.config import TRAIN_DENSITY
from common.envs import make_env
from common.metrics import EpisodeRecorder


class TrainEnvWrapper(gym.Wrapper):
    """Feeds an ``EpisodeRecorder`` and publishes finished episodes in ``info["episode_metrics"]``.

    Put it OUTSIDE ``common.envs.make_env`` (it needs ``info["env_reward"]`` and the lane-change flag
    of ``CustomRewardWrapper``). ``env_index`` is this env's position in a vector env (used by
    ``set_state`` to pick its own snapshot).
    """

    def __init__(self, env: gym.Env, with_leader: bool = True, env_index: int = 0) -> None:
        super().__init__(env)
        self.with_leader = with_leader
        self.env_index = env_index
        self.recorder: Optional[EpisodeRecorder] = None

    def reset(self, **kwargs: Any):
        out = self.env.reset(**kwargs)
        if self.recorder is None:
            self.recorder = EpisodeRecorder(self.env, with_leader=self.with_leader)
        else:
            self.recorder.start()
        return out

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.recorder.step(reward, info, action)
        if terminated or truncated:
            record = self.recorder.finish(-1, truncated, info)
            record.pop("seed")  # training episodes use unseeded resets
            info["episode_metrics"] = record
        return obs, reward, terminated, truncated, info

    # -- resume ------------------------------------------------------------------------------
    def get_state(self) -> bytes:
        """Snapshot of the simulator and the recorder of the running episode (one pickle, shared references)."""
        return pickle.dumps((self.env, self.recorder), protocol=pickle.HIGHEST_PROTOCOL)

    def set_state(self, blobs: Any) -> None:
        """Restore the snapshot of env ``env_index`` from the list ``blobs`` of ``get_state`` results."""
        self.env, self.recorder = pickle.loads(blobs[self.env_index])


def make_train_env(
    density: str = TRAIN_DENSITY, reward: str = "default", with_leader: bool = True, env_index: int = 0
) -> TrainEnvWrapper:
    """One training env (created once per run; the caller resets it, see ``common.seeding``)."""
    return TrainEnvWrapper(make_env(density=density, reward=reward), with_leader=with_leader, env_index=env_index)


def make_vector_env(
    n_envs: int,
    workers: int = 1,
    density: str = TRAIN_DENSITY,
    reward: str = "default",
    with_leader: bool = True,
):
    """Vector env of ``n_envs`` training envs, created once per run (CLAUDE.md rules 4 and 6).

    ``workers <= 1``: ``SyncVectorEnv`` in this process; otherwise ``AsyncVectorEnv`` with one
    subprocess per env (spawned, so no torch state is forked). Autoreset mode is SAME_STEP.
    """
    fns = [
        functools.partial(make_train_env, density=density, reward=reward, with_leader=with_leader, env_index=i)
        for i in range(n_envs)
    ]
    if workers <= 1:
        return SyncVectorEnv(fns, autoreset_mode=AutoresetMode.SAME_STEP)
    return AsyncVectorEnv(fns, context="spawn", autoreset_mode=AutoresetMode.SAME_STEP)


def finished_episodes(info: Mapping[str, Any], n_envs: Optional[int] = None) -> list[tuple[int, dict[str, Any]]]:
    """Episode records that ended in this step: ``[(env_index, record), ...]``.

    ``n_envs=None`` for a single env (``info`` of ``env.step``), else ``infos`` of a vector env step
    (SAME_STEP autoreset). Values are plain Python scalars.
    """
    if n_envs is None:
        record = info.get("episode_metrics")
        return [] if record is None else [(0, dict(record))]
    final_info = info.get("final_info")
    if final_info is None or "episode_metrics" not in final_info:
        return []
    metrics = final_info["episode_metrics"]
    done = metrics["_length"]  # mask: which envs finished an episode in this step
    return [(i, {k: v[i].item() for k, v in metrics.items() if not k.startswith("_")}) for i in range(n_envs) if done[i]]
