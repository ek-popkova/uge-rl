"""The shared MLP builder: shapes, parameter counts, equal size of the DDQN Q-network and the PPO actor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from common.actions import N_ACTIONS
from common.networks import (
    OBS_DIM,
    OBS_SHAPE,
    build_action_network,
    build_mlp,
    build_value_network,
    count_parameters,
    expected_parameters,
    flatten_obs,
)
from common.train_config import DDQNConfig, PPOConfig


def test_output_shapes_are_batch_by_five_actions():
    for cfg in (DDQNConfig(), PPOConfig()):
        net = build_action_network(cfg)
        assert net(torch.zeros(7, OBS_DIM)).shape == (7, N_ACTIONS) == (7, 5)
    assert build_value_network(PPOConfig())(torch.zeros(7, OBS_DIM)).shape == (7, 1)


def test_parameter_count_is_identical_for_the_q_network_and_the_ppo_actor():
    q_net, actor = build_action_network(DDQNConfig()), build_action_network(PPOConfig())
    # hand count for 25 -> 256 -> 256 -> 5: (25*256 + 256) + (256*256 + 256) + (256*5 + 5)
    assert count_parameters(q_net) == count_parameters(actor) == 6_656 + 65_792 + 1_285 == 73_733
    assert [tuple(p.shape) for p in q_net.parameters()] == [tuple(p.shape) for p in actor.parameters()]
    # the PPO critic is an extra network of the same size with one output: 25 -> 256 -> 256 -> 1
    assert count_parameters(build_value_network(PPOConfig())) == 6_656 + 65_792 + 257 == 72_705


def test_expected_parameters_matches_the_builder():
    for hidden in ((256, 256), (64,), (32, 32, 32)):
        for out in (1, 5):
            assert count_parameters(build_mlp(OBS_DIM, out, hidden)) == expected_parameters(OBS_DIM, out, hidden)


def test_layer_structure_and_activation():
    net = build_mlp(OBS_DIM, N_ACTIONS, (256, 256), "relu")
    kinds = [type(m).__name__ for m in net]
    assert kinds == ["Linear", "ReLU", "Linear", "ReLU", "Linear"]  # no activation after the output layer
    assert [type(m).__name__ for m in build_mlp(4, 2, (8,), "tanh")] == ["Linear", "Tanh", "Linear"]
    with pytest.raises(ValueError, match="activation"):
        build_mlp(4, 2, (8,), "gelu")
    for bad in ((0, 2, (8,)), (4, 0, (8,)), (4, 2, ()), (4, 2, (8, 0))):
        with pytest.raises(ValueError):
            build_mlp(*bad)


def test_networks_follow_the_config():
    net = build_action_network(DDQNConfig(hidden_sizes=(32, 16)))
    assert [tuple(p.shape) for p in net.parameters() if p.ndim == 2] == [(32, 25), (16, 32), (5, 16)]


def test_flatten_obs_single_and_batch_numpy_and_torch():
    single = np.arange(25, dtype=np.float32).reshape(5, 5)
    assert flatten_obs(single).shape == (25,) and (flatten_obs(single) == np.arange(25)).all()
    batch = np.stack([single, single + 100])
    flat = flatten_obs(batch)
    assert flat.shape == (2, 25) and (flat[1] == np.arange(25) + 100).all()
    assert flatten_obs(torch.as_tensor(batch)).shape == (2, 25)
    with pytest.raises(ValueError, match="shape"):
        flatten_obs(np.zeros((4, 5)))


@pytest.mark.slow
def test_obs_dim_matches_the_env_observation():
    from common.envs import make_env

    env = make_env()
    assert tuple(env.observation_space.shape) == OBS_SHAPE and OBS_DIM == 25
    obs, _ = env.reset(seed=0)
    assert flatten_obs(obs).shape == (OBS_DIM,)
    env.close()
