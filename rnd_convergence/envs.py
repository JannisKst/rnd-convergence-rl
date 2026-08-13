"""Environment construction for the ladder, and the state representation each rung logs.

The ladder is an axis of *state-space continuity and dimensionality*, so both the policy
input and the logged state must be the compact state on every rung — otherwise the axis
is not what the table claims it is. MiniGrid is the case where this needs work: its
default observation is a ``Dict`` of a 7x7x3 egocentric image, a direction and a mission
string. Feeding that to the policy would make the "3-D" rung a 147-D partially-observed
one, out of order with CartPole below it, and ``gymnasium.spaces.utils.flatdim`` raises
``NotImplementedError`` on the ``MissionSpace`` anyway.

:class:`MiniGridStateWrapper` therefore replaces the observation with the underlying
``(x, y, dir)`` state, which is exactly the quantity the entropy baseline is defined on
for those rungs.

:func:`make_env` is the single place that knows which rung needs which treatment;
:func:`state_fn_for` returns the matching ``state_fn`` for :class:`~rnd_convergence.ppo.PPOAgent`.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401  -- imported for its side effect of registering MiniGrid-* ids
import numpy as np

from rnd_convergence.mars_rover import MarsRover
from rnd_convergence.streams import StateFn, compact_state_fn

MINIGRID_PREFIX = "MiniGrid-"
MARS_ROVER_ID = "MarsRover"


class MiniGridStateWrapper(gym.ObservationWrapper):
    """Expose a MiniGrid environment's ``(x, y, dir)`` state as the observation.

    Fully observable by construction. That is deliberate: the study compares coverage
    signals across state spaces of growing dimensionality, and a partially-observed rung
    would confound "SVE degrades with dimensionality" with "the agent cannot see where it
    is". It also makes the logged stream and the policy input the same object, so the
    entropy estimate is over the distribution the policy actually induces.

    The action space is left at MiniGrid's full ``Discrete(7)``. Trimming it to the three
    navigation actions would speed up the Empty rung but would make the action space
    differ between the two MiniGrid rungs, and DoorKey needs pickup and toggle.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        inner = env.unwrapped
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([inner.width - 1, inner.height - 1, 3.0], dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation: Any) -> np.ndarray:
        """Read ``(x, y, dir)`` off the underlying grid world, ignoring the image obs."""
        inner = self.env.unwrapped
        x, y = inner.agent_pos
        return np.array([x, y, inner.agent_dir], dtype=np.float32)


def make_env(env_id: str, seed: int | None = None, **kwargs: Any) -> gym.Env:
    """Build a ladder environment with the wrappers that rung needs.

    ``"MarsRover"`` builds :class:`~rnd_convergence.mars_rover.MarsRover`, which has no
    Gymnasium id. MiniGrid ids get :class:`MiniGridStateWrapper`. Everything else is
    returned as ``gymnasium`` builds it.

    ``seed`` seeds the action and observation spaces only. The environment itself is
    seeded on the first ``reset(seed=...)``, which callers do themselves.
    """
    env = MarsRover(**kwargs) if env_id == MARS_ROVER_ID else gym.make(env_id, **kwargs)
    if env_id.startswith(MINIGRID_PREFIX):
        env = MiniGridStateWrapper(env)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def state_fn_for(env: gym.Env) -> StateFn:
    """The ``state_fn`` to log for ``env``.

    A thin wrapper over :func:`~rnd_convergence.streams.compact_state_fn`, which is also
    :class:`~rnd_convergence.ppo.PPOAgent`'s default — so passing this explicitly changes
    nothing and exists only to make the choice visible at a call site.
    """
    return compact_state_fn(env.observation_space)
