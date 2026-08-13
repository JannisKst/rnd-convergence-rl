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
from rnd_convergence.mars_rover import MarsRover
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


class TestMarsRoverRung:
    def test_make_env_builds_the_anchor_rung(self):
        # README lists MarsRover as the anchor rung; it has no Gymnasium id, so make_env
        # has to special-case it or the documented ladder does not run.
        env = make_env("MarsRover")
        assert isinstance(env, MarsRover)
        assert env.observation_space.n == 5

    def test_logged_state_is_one_dimensional_not_one_hot(self):
        # The ladder calls this the 1-D rung. flatten() of a Discrete space gives a
        # 5-wide one-hot, which would make the entropy baseline see 5 dimensions.
        env = make_env("MarsRover")
        obs, _ = env.reset(seed=0)
        assert state_fn_for(env)(obs).shape == (1,)
        assert state_fn_for(env)(obs)[0] == obs

    def test_policy_sees_the_one_hot_while_the_stream_logs_the_integer(self):
        # No state_fn passed: the DEFAULT has to be the compact state. Logging the
        # one-hot here would silently make the 1-D anchor rung 5-D, and every other rung
        # is read against it.
        agent = PPOAgent(lambda: make_env("MarsRover"), rollout_steps=64, seed=0)
        assert agent.obs_dim == 5  # one-hot, the right input for a network
        assert agent.state_dim == 1  # the 1-D state the ladder claims

        stream, _ = agent.train(256, eval_interval=256, eval_episodes=2, random_episodes=2)
        assert stream.obs_dim == 1
        assert set(np.unique(stream.observations).tolist()) <= {0.0, 1.0, 2.0, 3.0, 4.0}

    def test_explicit_state_fn_matches_the_default(self):
        # state_fn_for is a convenience for making the choice visible at a call site; it
        # must not be the only way to get the right one.
        env = make_env("MarsRover")
        explicit = PPOAgent(
            lambda: make_env("MarsRover"), state_fn=state_fn_for(env), rollout_steps=32, seed=0
        )
        default = PPOAgent(lambda: make_env("MarsRover"), rollout_steps=32, seed=0)

        assert explicit.state_dim == default.state_dim == 1
        for observation in range(env.observation_space.n):
            assert np.array_equal(explicit.state_fn(observation), default.state_fn(observation))

    def test_ladder_dimensions_match_the_readme_table(self):
        expected = {
            "MarsRover": 1,
            "MiniGrid-Empty-5x5-v0": 3,
            "MiniGrid-DoorKey-5x5-v0": 3,
            "MiniGrid-DoorKey-8x8-v0": 3,
            "CartPole-v1": 4,
            "LunarLander-v3": 8,
        }
        for env_id, dim in expected.items():
            env = make_env(env_id)
            obs, _ = env.reset(seed=0)
            assert state_fn_for(env)(obs).shape == (dim,), env_id
            # And what an agent logs by default, which is what actually reaches disk.
            assert PPOAgent(lambda i=env_id: make_env(i), seed=0).state_dim == dim, env_id

    def test_make_env_seeds_spaces_when_asked(self):
        # Folded in from the deleted utils.make_env, which could not build these rungs.
        first = make_env("MarsRover", seed=3).action_space.sample()
        assert first == make_env("MarsRover", seed=3).action_space.sample()
