"""MarsRover — the 1-D discrete anchor rung of the environment ladder.

The environment *specification* (a rover on a 1-D strip of 5 cells, starting in the
middle, two actions, per-cell rewards ``[1, 0, 0, 0, 10]``, a fixed horizon, and a
per-step probability of the action being flipped) is the one used in ``week_2`` of the
course exercise repo (automl-edu/RL-exercises), where it is provided as a complete
environment rather than a stub. This implementation is written for this project; the
course version is not imported or copied, and the exercise repo is not a dependency.

Why it is worth having at all: it is the only rung where the state space is small enough
that state-visitation entropy is not merely "exact" in principle but computed over a
handful of states, so both signals saturate almost immediately. That makes it the control
against which the degradation further down the ladder is read. There is no Gymnasium id
for it, so :func:`~rnd_convergence.envs.make_env` special-cases the name ``MarsRover``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

DEFAULT_REWARDS = (1.0, 0.0, 0.0, 0.0, 10.0)
START_POSITION = 2


class MarsRover(gym.Env):
    """A rover moving left or right along a 1-D strip of discrete cells.

    Observations are the integer position, ``Discrete(n_cells)``. Actions are
    ``Discrete(2)``: 0 moves left, 1 moves right, both clamped at the ends of the strip.
    The reward is the value of the cell the rover *lands* in, so the optimal policy is to
    run to the far end and stay there — reachable well inside the horizon, which is what
    makes this rung converge almost immediately.

    ``transition_probability`` is the chance the chosen action is actually executed; with
    the remaining probability the opposite action is taken instead. It defaults to 1.0
    (deterministic). Episodes never terminate; they are truncated at ``horizon`` steps.

    Parameters
    ----------
    rewards
        Per-cell reward, one entry per cell. Its length sets the size of the strip.
    horizon
        Steps before truncation.
    transition_probability
        Probability that the chosen action is executed rather than flipped.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rewards: tuple[float, ...] = DEFAULT_REWARDS,
        horizon: int = 10,
        transition_probability: float = 1.0,
    ) -> None:
        if len(rewards) < 2:
            raise ValueError(f"need at least 2 cells, got {len(rewards)}")
        if not 0.0 <= transition_probability <= 1.0:
            raise ValueError(
                f"transition_probability must be in [0, 1], got {transition_probability}"
            )
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        self.rewards = np.asarray(rewards, dtype=np.float64)
        self.horizon = int(horizon)
        self.transition_probability = float(transition_probability)

        self.observation_space = gym.spaces.Discrete(self.rewards.size)
        self.action_space = gym.spaces.Discrete(2)

        self.position = START_POSITION
        self.current_step = 0
        self._rng = np.random.default_rng()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        """Return the rover to the middle cell."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.position = min(START_POSITION, self.observation_space.n - 1)
        self.current_step = 0
        return self.position, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Move one cell, with the action possibly flipped, and collect the cell reward."""
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}, expected 0 (left) or 1 (right)")

        self.current_step += 1
        if self.transition_probability < 1.0 and self._rng.random() >= self.transition_probability:
            action = 1 - action

        step = 1 if action == 1 else -1
        self.position = int(np.clip(self.position + step, 0, self.observation_space.n - 1))

        # Episodes end only on the time limit: there is no goal state to reach, so the
        # rover keeps collecting the reward of whatever cell it sits in. That makes
        # `terminated` always False and the truncation bootstrap in compute_gae the one
        # that matters on this rung.
        return (
            self.position,
            float(self.rewards[self.position]),
            False,
            self.current_step >= self.horizon,
            {},
        )
