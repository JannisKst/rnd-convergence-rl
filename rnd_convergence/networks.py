"""Actor and critic networks for the PPO agent.

The two-network layout (a softmax policy over a discrete action space plus a scalar
value head) follows the ``week_6/networks.py`` scaffold of the course exercise repo
(automl-edu/RL-exercises); the implementation is our own.

Two deliberate departures from that scaffold:

* the hidden depth is configurable rather than fixed at one layer, since the ladder spans
  environments from a 1-D discrete state up to 8-D continuous ones;
* layers use orthogonal initialisation with the gains recommended by Schulman et al.
  (2017), which materially improves PPO's reliability and matters here because the
  experiment schedule leaves no room for per-environment tuning.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def _init_layer(layer: nn.Linear, gain: float) -> nn.Linear:
    """Orthogonal weight init with zero bias, the standard PPO choice."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _build_trunk(input_dim: int, hidden_size: int, n_layers: int) -> tuple[nn.Sequential, int]:
    """Shared ``[Linear, Tanh] * n_layers`` body; returns the stack and its output width."""
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(n_layers):
        layers.append(_init_layer(nn.Linear(in_dim, hidden_size), np.sqrt(2)))
        layers.append(nn.Tanh())
        in_dim = hidden_size
    return nn.Sequential(*layers), in_dim


class Policy(nn.Module):
    """Categorical policy over a discrete action space.

    Discrete actions only, by design: every rung of the environment ladder uses one, so a
    single agent covers the whole study and no algorithmic difference confounds the
    comparison between rungs.
    """

    def __init__(
        self, obs_dim: int, n_actions: int, hidden_size: int = 128, n_layers: int = 2
    ) -> None:
        super().__init__()
        self.trunk, trunk_out = _build_trunk(obs_dim, hidden_size, n_layers)
        # Small gain on the head keeps the initial policy close to uniform.
        self.head = _init_layer(nn.Linear(trunk_out, n_actions), 0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Action logits, shape ``(B, n_actions)``."""
        return self.head(self.trunk(obs))

    def distribution(self, obs: torch.Tensor) -> Categorical:
        """Action distribution induced by ``obs``."""
        return Categorical(logits=self.forward(obs))


class ValueNetwork(nn.Module):
    """State-value critic."""

    def __init__(self, obs_dim: int, hidden_size: int = 128, n_layers: int = 2) -> None:
        super().__init__()
        self.trunk, trunk_out = _build_trunk(obs_dim, hidden_size, n_layers)
        self.head = _init_layer(nn.Linear(trunk_out, 1), 1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """State values, shape ``(B,)``."""
        return self.head(self.trunk(obs)).squeeze(-1)
