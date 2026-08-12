"""Tests for rnd_convergence.rnd."""

import numpy as np
import pytest
import torch

from rnd_convergence.rnd import (
    PredictorNetwork,
    RunningNormalizer,
    TargetNetwork,
    streaming_rnd_error,
)
from rnd_convergence.streams import StateStream


def stream_from(observations, step_scale=1):
    observations = np.asarray(observations, dtype=np.float32)
    n = observations.shape[0]
    return StateStream(
        steps=np.arange(n, dtype=np.int64) * step_scale,
        observations=observations,
        episode_ends=np.zeros(n, dtype=bool),
        env_id="Synthetic-v0",
        seed=0,
    )


class TestRunningNormalizer:
    def test_matches_batch_statistics(self):
        rng = np.random.default_rng(0)
        data = rng.normal(3.0, 2.0, size=(1_000, 4))

        normalizer = RunningNormalizer(4)
        for start in range(0, 1_000, 100):
            normalizer.update(data[start : start + 100])

        assert np.allclose(normalizer.mean, data.mean(axis=0), atol=1e-8)
        assert np.allclose(normalizer.var, data.var(axis=0), atol=1e-8)
        assert normalizer.count == 1_000

    def test_normalized_output_is_standardised(self):
        rng = np.random.default_rng(1)
        data = rng.normal(-5.0, 0.5, size=(500, 2))
        normalizer = RunningNormalizer(2)
        normalizer.update(data)

        normalized = normalizer.normalize(data)
        assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
        assert np.allclose(normalized.std(axis=0), 1.0, atol=1e-3)

    def test_clipping_bounds_the_output(self):
        normalizer = RunningNormalizer(1)
        normalizer.update(np.array([[0.0], [1.0]]))
        assert normalizer.normalize(np.array([[1e6]]), clip=5.0).max() == pytest.approx(5.0)

    def test_empty_update_is_a_no_op(self):
        normalizer = RunningNormalizer(3)
        normalizer.update(np.empty((0, 3)))
        assert normalizer.count == 0


class TestNetworks:
    def test_target_parameters_are_frozen(self):
        target = TargetNetwork(input_dim=4)
        assert all(not p.requires_grad for p in target.parameters())

    def test_target_is_unchanged_by_predictor_training(self):
        torch.manual_seed(0)
        target = TargetNetwork(input_dim=4)
        predictor = PredictorNetwork(input_dim=4)
        before = [p.clone() for p in target.parameters()]

        optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-2)
        batch = torch.randn(16, 4)
        loss = (predictor(batch) - target(batch)).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert all(torch.equal(a, b) for a, b in zip(before, target.parameters()))

    def test_both_networks_share_output_shape(self):
        batch = torch.randn(8, 6)
        target = TargetNetwork(input_dim=6, output_dim=32)
        predictor = PredictorNetwork(input_dim=6, output_dim=32)
        assert target(batch).shape == predictor(batch).shape == (8, 32)

    def test_target_is_deterministic_given_a_seed(self):
        batch = torch.randn(4, 3)
        torch.manual_seed(11)
        first = TargetNetwork(input_dim=3)(batch)
        torch.manual_seed(11)
        second = TargetNetwork(input_dim=3)(batch)
        assert torch.allclose(first, second)

    def test_rejects_zero_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            TargetNetwork(input_dim=3, n_layers=0)


class TestStreamingRndError:
    def test_error_falls_on_a_repeated_state(self):
        # The same state over and over: the predictor should fit it and the error decay.
        observations = np.tile(np.array([[0.5, -0.2, 1.0]]), (10_000, 1))
        steps, errors = streaming_rnd_error(stream_from(observations), lr=1e-3, window=1_000)

        assert steps.size == errors.size == 10
        assert errors[-1] < errors[0] / 2

    def test_novel_states_keep_the_error_higher_than_repeated_ones(self):
        rng = np.random.default_rng(0)
        n = 10_000
        repeated = np.tile(np.array([[0.5, -0.2, 1.0]]), (n, 1))
        # A distribution that keeps drifting into unseen regions of the state space.
        novel = np.cumsum(rng.normal(0.0, 1.0, size=(n, 3)), axis=0)

        _, repeated_errors = streaming_rnd_error(stream_from(repeated), lr=1e-3, window=1_000)
        _, novel_errors = streaming_rnd_error(stream_from(novel), lr=1e-3, window=1_000)

        assert novel_errors[-1] > repeated_errors[-1]

    def test_replay_is_deterministic(self):
        rng = np.random.default_rng(2)
        observations = rng.normal(size=(2_000, 3))
        first = streaming_rnd_error(stream_from(observations), seed=5, window=500)
        second = streaming_rnd_error(stream_from(observations), seed=5, window=500)

        assert np.array_equal(first[0], second[0])
        assert np.allclose(first[1], second[1])

    def test_different_seeds_give_different_curves(self):
        rng = np.random.default_rng(3)
        observations = rng.normal(size=(2_000, 3))
        _, first = streaming_rnd_error(stream_from(observations), seed=0, window=500)
        _, second = streaming_rnd_error(stream_from(observations), seed=1, window=500)
        assert not np.allclose(first, second)

    def test_windows_are_labelled_by_their_upper_edge(self):
        observations = np.zeros((3_000, 2), dtype=np.float32)
        steps, _ = streaming_rnd_error(stream_from(observations), window=1_000)
        assert np.array_equal(steps, [1_000, 2_000, 3_000])

    def test_empty_stream_returns_empty_curve(self):
        empty = StateStream(
            steps=np.empty(0, dtype=np.int64),
            observations=np.empty((0, 3), dtype=np.float32),
            episode_ends=np.empty(0, dtype=bool),
            env_id="x",
            seed=0,
        )
        steps, errors = streaming_rnd_error(empty)
        assert steps.size == 0 and errors.size == 0

    def test_errors_are_non_negative(self):
        rng = np.random.default_rng(4)
        observations = rng.normal(size=(1_000, 5))
        _, errors = streaming_rnd_error(stream_from(observations), window=250)
        assert np.all(errors >= 0.0)
