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

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401  -- imported for its side effect of registering MiniGrid-* ids
import numpy as np
from gymnasium.spaces.utils import flatten

MINIGRID_PREFIX = "MiniGrid-"


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


def make_env(env_id: str, **kwargs: Any) -> gym.Env:
    """Build a ladder environment with the wrappers that rung needs.

    MiniGrid ids get :class:`MiniGridStateWrapper`; everything else is returned as
    ``gymnasium`` builds it.
    """
    env = gym.make(env_id, **kwargs)
    if env_id.startswith(MINIGRID_PREFIX):
        return MiniGridStateWrapper(env)
    return env


def state_fn_for(env: gym.Env) -> Callable[[Any], np.ndarray]:
    """The ``state_fn`` to log for ``env``: the observation itself, flattened.

    Because :func:`make_env` already reduces every rung to its compact state, the logged
    state and the policy input coincide and this is just a flatten. It exists as a named
    function so that a rung which later needs them to differ has one place to say so.
    """
    space = env.observation_space

    def state_fn(obs: Any) -> np.ndarray:
        return np.asarray(flatten(space, obs), dtype=np.float32)

    return state_fn
