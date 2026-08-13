"""Tests for rnd_convergence.envs.

The point of the wrapper is that the MiniGrid rungs are constructible at all: gymnasium's
``flatdim`` raises ``NotImplementedError`` on MiniGrid's ``MissionSpace``, so without it
two of the five ladder rungs cannot be trained.
"""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces.utils import flatdim

from rnd_convergence.envs import MiniGridStateWrapper, make_env, state_fn_for
from rnd_convergence.ppo import PPOAgent

MINIGRID_ID = "MiniGrid-Empty-5x5-v0"


class TestMiniGridStateWrapper:
    def test_raw_minigrid_cannot_be_flattened(self):
        # The failure this wrapper exists to fix.
        with pytest.raises(NotImplementedError):
            flatdim(gym.make(MINIGRID_ID).observation_space)

    def test_observation_is_the_three_dimensional_state(self):
        env = make_env(MINIGRID_ID)
        obs, _ = env.reset(seed=0)

        assert isinstance(env, MiniGridStateWrapper)
        assert flatdim(env.observation_space) == 3
        assert obs.shape == (3,)
        assert obs.dtype == np.float32

    def test_observation_tracks_the_underlying_agent_state(self):
        env = make_env(MINIGRID_ID)
        env.reset(seed=0)
        inner = env.unwrapped

        for action in (0, 2, 1, 2):
            obs, *_ = env.step(action)
            assert np.array_equal(obs, [*inner.agent_pos, inner.agent_dir])

    def test_observation_stays_inside_the_declared_space(self):
        env = make_env(MINIGRID_ID)
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        for action in range(7):
            obs, *_ = env.step(action)
            assert env.observation_space.contains(obs)

    def test_non_minigrid_envs_are_left_alone(self):
        env = make_env("CartPole-v1")
        assert not isinstance(env, MiniGridStateWrapper)
        assert flatdim(env.observation_space) == 4


class TestStateFnFor:
    def test_logs_the_compact_state_on_minigrid(self):
        env = make_env(MINIGRID_ID)
        obs, _ = env.reset(seed=0)
        assert state_fn_for(env)(obs).shape == (3,)


class TestPPOOnMiniGrid:
    def test_agent_constructs_and_trains(self):
        agent = PPOAgent(lambda: make_env(MINIGRID_ID), rollout_steps=64, seed=0)
        assert agent.obs_dim == 3
        assert agent.n_actions == 7

        stream, curve = agent.train(128, eval_interval=128, eval_episodes=1, random_episodes=1)
        assert stream.n_steps == 128
        assert stream.obs_dim == 3  # the entropy baseline's exact state
        assert curve.returns.shape[0] == 1

    def test_logged_state_matches_the_policy_input(self):
        # The ladder is an axis of state dimensionality, so the rung must be 3-D for both.
        agent = PPOAgent(lambda: make_env(MINIGRID_ID), rollout_steps=32, seed=0)
        rollout = agent.collect_rollout(32)
        assert np.array_equal(rollout.states, rollout.observations)
