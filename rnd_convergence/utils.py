"""General utilities: seeding.

Environment construction lives in :mod:`rnd_convergence.envs`. This module used to carry
its own ``make_env`` that only called ``gym.make``; two functions of the same name with
different behaviour is a trap, and the plain one silently could not build the MarsRover or
MiniGrid rungs.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all relevant random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
