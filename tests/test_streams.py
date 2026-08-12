"""Tests for rnd_convergence.streams — the on-disk contract between training and analysis."""

import numpy as np
import pytest

from rnd_convergence.streams import (
    EvalCurve,
    StateStream,
    load_eval_curve,
    load_state_stream,
    run_stem,
    save_eval_curve,
    save_state_stream,
)


def make_stream(n=10, dim=3):
    return StateStream(
        steps=np.arange(n, dtype=np.int64),
        observations=np.arange(n * dim, dtype=np.float32).reshape(n, dim),
        episode_ends=np.zeros(n, dtype=bool),
        env_id="CartPole-v1",
        seed=7,
    )


class TestStateStream:
    def test_roundtrip_preserves_contents(self, tmp_path):
        stream = make_stream()
        loaded = load_state_stream(save_state_stream(tmp_path / "run.states.npz", stream))

        assert np.array_equal(loaded.steps, stream.steps)
        assert np.allclose(loaded.observations, stream.observations)
        assert np.array_equal(loaded.episode_ends, stream.episode_ends)
        assert loaded.env_id == "CartPole-v1"
        assert loaded.seed == 7

    def test_creates_missing_parent_directories(self, tmp_path):
        path = save_state_stream(tmp_path / "nested" / "deeper" / "run.states.npz", make_stream())
        assert path.exists()

    def test_derived_properties(self):
        stream = StateStream(
            steps=np.arange(6, dtype=np.int64),
            observations=np.zeros((6, 4), dtype=np.float32),
            episode_ends=np.array([False, True, False, False, True, False]),
            env_id="MiniGrid-Empty-8x8-v0",
            seed=0,
        )
        assert stream.n_steps == 6
        assert stream.obs_dim == 4
        assert stream.n_episodes == 2

    def test_rejects_one_dimensional_observations(self):
        with pytest.raises(ValueError, match="2-D"):
            StateStream(
                steps=np.arange(3, dtype=np.int64),
                observations=np.zeros(3, dtype=np.float32),
                episode_ends=np.zeros(3, dtype=bool),
                env_id="x",
                seed=0,
            )

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="steps must have shape"):
            StateStream(
                steps=np.arange(2, dtype=np.int64),
                observations=np.zeros((3, 2), dtype=np.float32),
                episode_ends=np.zeros(3, dtype=bool),
                env_id="x",
                seed=0,
            )

    def test_rejects_out_of_order_steps(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            StateStream(
                steps=np.array([0, 5, 2], dtype=np.int64),
                observations=np.zeros((3, 2), dtype=np.float32),
                episode_ends=np.zeros(3, dtype=bool),
                env_id="x",
                seed=0,
            )

    def test_empty_stream_is_allowed(self):
        stream = StateStream(
            steps=np.empty(0, dtype=np.int64),
            observations=np.empty((0, 3), dtype=np.float32),
            episode_ends=np.empty(0, dtype=bool),
            env_id="x",
            seed=0,
        )
        assert stream.n_steps == 0


class TestEvalCurve:
    def test_roundtrip_preserves_contents(self, tmp_path):
        curve = EvalCurve(
            steps=np.array([5_000, 10_000], dtype=np.int64),
            returns=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            random_return=-0.5,
            env_id="LunarLander-v3",
            seed=3,
        )
        loaded = load_eval_curve(save_eval_curve(tmp_path / "run.evals.npz", curve))

        assert np.array_equal(loaded.steps, curve.steps)
        assert np.allclose(loaded.returns, curve.returns)
        assert loaded.random_return == pytest.approx(-0.5)
        assert loaded.env_id == "LunarLander-v3"
        assert loaded.seed == 3

    def test_mean_returns_averages_over_episodes(self):
        curve = EvalCurve(
            steps=np.array([1, 2], dtype=np.int64),
            returns=np.array([[0.0, 10.0], [4.0, 6.0]], dtype=np.float32),
            random_return=0.0,
            env_id="x",
            seed=0,
        )
        assert np.allclose(curve.mean_returns, [5.0, 5.0])

    def test_rejects_non_increasing_steps(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            EvalCurve(
                steps=np.array([10, 10], dtype=np.int64),
                returns=np.zeros((2, 1), dtype=np.float32),
                random_return=0.0,
                env_id="x",
                seed=0,
            )

    def test_rejects_one_dimensional_returns(self):
        with pytest.raises(ValueError, match="2-D"):
            EvalCurve(
                steps=np.array([1, 2], dtype=np.int64),
                returns=np.zeros(2, dtype=np.float32),
                random_return=0.0,
                env_id="x",
                seed=0,
            )


def test_run_stem_is_stable():
    assert run_stem("MiniGrid-DoorKey-8x8-v0", 3) == "MiniGrid-DoorKey-8x8-v0__seed3"
