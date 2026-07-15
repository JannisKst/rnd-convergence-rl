"""General utilities: seeding and environment construction."""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all relevant random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id: str, seed: int | None = None, **env_kwargs) -> gym.Env:
    """Create a Gymnasium environment with seeded spaces.

    The environment itself is seeded on the first ``reset(seed=...)`` call,
    which callers are expected to do themselves.
    """
    env = gym.make(env_id, **env_kwargs)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env
