"""On-disk schema for training runs.

RND is used here as a monitoring signal rather than an intrinsic reward, so no signal
has to be computed inside the training loop. Training writes two artefacts per
``(env_id, seed)`` run:

``StateStream``
    Every visited state, in visit order. Replaying it offline reproduces the online
    RND prediction error exactly and supports every entropy variant, bin count and
    detector threshold without retraining.

``EvalCurve``
    Periodic greedy-policy evaluation returns, plus the random-policy baseline needed
    to normalise the convergence criterion.

This module is the contract between the training half of the project and the offline
analysis half. It also owns :func:`compact_state_fn`, the rule for what the logged state
representation actually *is* — the two halves have to agree on that, and the ladder's
dimensionality axis is only meaningful if they do.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import flatten

STATE_STREAM_SUFFIX = ".states.npz"
EVAL_CURVE_SUFFIX = ".evals.npz"

StateFn = Callable[[Any], np.ndarray]


def compact_state_fn(space: gym.Space) -> StateFn:
    """The state representation to log for an environment with observation ``space``.

    A ``Discrete`` space is logged as the raw integer, *not* as the one-hot vector a
    network wants. MarsRover is why: the ladder calls it the 1-D anchor rung, and
    one-hotting its 5 positions would make the entropy baseline see 5 dimensions and
    quietly stop the control from being a control. Every other space is flattened, which
    for the rungs :func:`~rnd_convergence.envs.make_env` builds is already the compact
    state.

    This lives here, next to :class:`StateStream`, because it defines what goes *into*
    the stream — and because both the agent that writes it and the analysis that reads it
    have to agree on it. It is :class:`~rnd_convergence.ppo.PPOAgent`'s default, so the
    correct representation is what you get without opting in.
    """
    if isinstance(space, gym.spaces.Discrete):

        def discrete_state_fn(obs: Any) -> np.ndarray:
            return np.asarray([obs], dtype=np.float32)

        return discrete_state_fn

    def flat_state_fn(obs: Any) -> np.ndarray:
        return np.asarray(flatten(space, obs), dtype=np.float32)

    return flat_state_fn


@dataclass(frozen=True)
class StateStream:
    """States visited during a single training run, in visit order.

    Parameters
    ----------
    steps
        Global environment-step index of each observation, shape ``(T,)``, int64.
        Monotonically non-decreasing.
    observations
        The state representation fed to the RND networks and entropy estimators,
        shape ``(T, D)``, float32. For discrete-observation environments this is the
        compact encoding (e.g. ``(x, y, dir)`` for MiniGrid) rather than a rendered
        observation, which keeps files small and matches how the entropy baseline is
        defined on those rungs.
    episode_ends
        ``True`` at the final transition of an episode, shape ``(T,)``, bool.
    env_id
        Gymnasium id (or ``MarsRover``) the run was collected on.
    seed
        Seed the run was trained with.
    """

    steps: np.ndarray
    observations: np.ndarray
    episode_ends: np.ndarray
    env_id: str
    seed: int

    def __post_init__(self) -> None:
        if self.observations.ndim != 2:
            raise ValueError(
                f"observations must be 2-D (T, D), got shape {self.observations.shape}"
            )
        n = self.observations.shape[0]
        if self.steps.shape != (n,):
            raise ValueError(f"steps must have shape ({n},), got {self.steps.shape}")
        if self.episode_ends.shape != (n,):
            raise ValueError(f"episode_ends must have shape ({n},), got {self.episode_ends.shape}")
        if n and np.any(np.diff(self.steps) < 0):
            raise ValueError("steps must be monotonically non-decreasing (visit order)")

    @property
    def n_steps(self) -> int:
        """Number of logged states."""
        return int(self.observations.shape[0])

    @property
    def obs_dim(self) -> int:
        """Dimensionality of the state representation."""
        return int(self.observations.shape[1])

    @property
    def n_episodes(self) -> int:
        """Number of completed episodes in the stream."""
        return int(self.episode_ends.sum())


@dataclass(frozen=True)
class EvalCurve:
    """Periodic evaluation returns for a single training run.

    Parameters
    ----------
    steps
        Environment-step index at which each evaluation was run, shape ``(E,)``, int64.
    returns
        Per-episode returns, shape ``(E, n_eval_episodes)``, float32. Stored
        un-aggregated so that seed-level uncertainty can be reported later.
    random_return
        Mean return of a random policy in this environment. Anchors the convergence
        criterion so it stays well defined for negative returns and for 0-1
        sparse-reward ranges alike.
    env_id
        Gymnasium id (or ``MarsRover``) the run was collected on.
    seed
        Seed the run was trained with.
    """

    steps: np.ndarray
    returns: np.ndarray
    random_return: float
    env_id: str
    seed: int

    def __post_init__(self) -> None:
        if self.returns.ndim != 2:
            raise ValueError(f"returns must be 2-D (E, n_episodes), got shape {self.returns.shape}")
        if self.steps.shape != (self.returns.shape[0],):
            raise ValueError(
                f"steps must have shape ({self.returns.shape[0]},), got {self.steps.shape}"
            )
        if self.steps.size and np.any(np.diff(self.steps) <= 0):
            raise ValueError("evaluation steps must be strictly increasing")

    @property
    def mean_returns(self) -> np.ndarray:
        """Mean return per evaluation point, shape ``(E,)``."""
        return self.returns.mean(axis=1)


def save_state_stream(path: str | Path, stream: StateStream) -> Path:
    """Write a :class:`StateStream` to a compressed ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        steps=stream.steps.astype(np.int64),
        observations=stream.observations.astype(np.float32),
        episode_ends=stream.episode_ends.astype(bool),
        env_id=np.array(stream.env_id),
        seed=np.array(stream.seed, dtype=np.int64),
    )
    return path


def load_state_stream(path: str | Path) -> StateStream:
    """Read a :class:`StateStream` written by :func:`save_state_stream`."""
    with np.load(Path(path), allow_pickle=False) as data:
        return StateStream(
            steps=data["steps"],
            observations=data["observations"],
            episode_ends=data["episode_ends"],
            env_id=str(data["env_id"]),
            seed=int(data["seed"]),
        )


def save_eval_curve(path: str | Path, curve: EvalCurve) -> Path:
    """Write an :class:`EvalCurve` to a compressed ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        steps=curve.steps.astype(np.int64),
        returns=curve.returns.astype(np.float32),
        random_return=np.array(curve.random_return, dtype=np.float64),
        env_id=np.array(curve.env_id),
        seed=np.array(curve.seed, dtype=np.int64),
    )
    return path


def load_eval_curve(path: str | Path) -> EvalCurve:
    """Read an :class:`EvalCurve` written by :func:`save_eval_curve`."""
    with np.load(Path(path), allow_pickle=False) as data:
        return EvalCurve(
            steps=data["steps"],
            returns=data["returns"],
            random_return=float(data["random_return"]),
            env_id=str(data["env_id"]),
            seed=int(data["seed"]),
        )


def run_stem(env_id: str, seed: int) -> str:
    """Canonical file stem for a run, e.g. ``CartPole-v1__seed0``.

    Environment ids may contain characters awkward in filenames (MiniGrid ids contain
    ``-`` only, but this keeps the convention explicit and in one place).
    """
    return f"{env_id}__seed{seed}"
