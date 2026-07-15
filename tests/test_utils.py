"""Tests for rnd_convergence.utils."""

import gymnasium as gym
import numpy as np
import torch

from rnd_convergence.utils import make_env, set_seed


def test_make_env_returns_env():
    env = make_env("CartPole-v1", seed=0)
    assert isinstance(env, gym.Env)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.close()


def test_make_env_seeded_action_space_is_reproducible():
    env_a = make_env("CartPole-v1", seed=42)
    env_b = make_env("CartPole-v1", seed=42)
    actions_a = [env_a.action_space.sample() for _ in range(10)]
    actions_b = [env_b.action_space.sample() for _ in range(10)]
    assert actions_a == actions_b
    env_a.close()
    env_b.close()


def test_set_seed_is_reproducible():
    set_seed(123)
    np_first = np.random.rand(3)
    torch_first = torch.rand(3)

    set_seed(123)
    assert np.allclose(np_first, np.random.rand(3))
    assert torch.allclose(torch_first, torch.rand(3))
