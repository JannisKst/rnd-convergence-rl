"""Tests for rnd_convergence.entropy.

Where a closed form exists it is asserted against: a uniform distribution over K grid
cells has entropy log K, and a standard normal has differential entropy
``0.5 * log(2 * pi * e)``.
"""

import numpy as np
import pytest

from rnd_convergence.entropy import (
    entropy_curve,
    grid_entropy,
    grid_occupancy,
    knn_entropy,
    occupancy_curve,
)
from rnd_convergence.rnd import RunningNormalizer
from rnd_convergence.streams import StateStream

GAUSSIAN_ENTROPY_1D = 0.5 * np.log(2 * np.pi * np.e)


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


class TestGridEntropy:
    def test_uniform_over_four_cells_is_log_four(self):
        # Cell centres for bins_per_dim=4 on [-1, 1].
        points = np.array([[-0.75], [-0.25], [0.25], [0.75]])
        assert grid_entropy(points, bins_per_dim=4, clip=1.0) == pytest.approx(np.log(4))

    def test_single_occupied_cell_has_zero_entropy(self):
        points = np.full((100, 3), 0.1)
        assert grid_entropy(points, bins_per_dim=10) == pytest.approx(0.0)

    def test_concentrated_is_lower_than_spread(self):
        rng = np.random.default_rng(0)
        spread = rng.uniform(-3.0, 3.0, size=(2_000, 2))
        concentrated = rng.normal(0.0, 0.05, size=(2_000, 2))
        assert grid_entropy(concentrated, 10) < grid_entropy(spread, 10)

    def test_never_exceeds_log_sample_count(self):
        rng = np.random.default_rng(1)
        points = rng.normal(size=(500, 3))
        assert grid_entropy(points, bins_per_dim=20) <= np.log(500) + 1e-9

    def test_high_dimensional_grid_does_not_allocate_all_cells(self):
        # 20**8 cells is 2.6e10; only the occupied ones may be materialised.
        rng = np.random.default_rng(2)
        points = rng.normal(size=(1_000, 8))
        assert 0.0 < grid_entropy(points, bins_per_dim=20) <= np.log(1_000) + 1e-9

    def test_out_of_range_values_fold_into_edge_bins(self):
        inside = np.array([[0.0], [0.5]])
        outside = np.array([[0.0], [1e6]])
        assert grid_entropy(outside, 4, clip=1.0) == pytest.approx(np.log(2))
        assert grid_entropy(inside, 4, clip=1.0) == pytest.approx(np.log(2))

    def test_empty_input_is_zero(self):
        assert grid_entropy(np.empty((0, 2)), bins_per_dim=5) == 0.0

    def test_rejects_degenerate_bin_count(self):
        with pytest.raises(ValueError, match="bins_per_dim"):
            grid_entropy(np.zeros((5, 2)), bins_per_dim=1)


class TestKnnEntropy:
    def test_matches_the_gaussian_closed_form(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(size=(3_000, 1))
        estimate = knn_entropy(samples, k=4, max_samples=4_000)
        assert estimate == pytest.approx(GAUSSIAN_ENTROPY_1D, abs=0.1)

    def test_scaling_shifts_entropy_by_log_scale(self):
        rng = np.random.default_rng(1)
        samples = rng.normal(size=(3_000, 1))
        base = knn_entropy(samples, k=4, max_samples=4_000)
        scaled = knn_entropy(samples * 2.0, k=4, max_samples=4_000)
        assert scaled - base == pytest.approx(np.log(2.0), abs=0.05)

    def test_concentrated_is_lower_than_spread(self):
        rng = np.random.default_rng(2)
        spread = rng.normal(0.0, 1.0, size=(2_000, 2))
        concentrated = rng.normal(0.0, 0.01, size=(2_000, 2))
        assert knn_entropy(concentrated) < knn_entropy(spread)

    def test_can_be_negative_for_concentrated_distributions(self):
        rng = np.random.default_rng(3)
        assert knn_entropy(rng.normal(0.0, 0.001, size=(1_000, 1))) < 0.0

    def test_duplicate_states_stay_finite(self):
        # Discrete state spaces produce exact duplicates; the estimate is meaningless
        # there but must not be inf or nan.
        duplicated = np.zeros((500, 2))
        assert np.isfinite(knn_entropy(duplicated))

    def test_too_few_samples_is_nan(self):
        assert np.isnan(knn_entropy(np.zeros((3, 2)), k=4))

    def test_subsampling_is_deterministic(self):
        rng = np.random.default_rng(4)
        samples = rng.normal(size=(5_000, 2))
        assert knn_entropy(samples, max_samples=1_000, seed=0) == knn_entropy(
            samples, max_samples=1_000, seed=0
        )


class TestEntropyCurve:
    def test_returns_aligned_increasing_steps(self):
        rng = np.random.default_rng(0)
        stream = stream_from(rng.normal(size=(5_000, 3)))
        steps, values = entropy_curve(stream, window=1_000)

        assert steps.size == values.size > 0
        assert np.all(np.diff(steps) > 0)

    def test_shrinking_coverage_lowers_the_sliding_entropy(self):
        rng = np.random.default_rng(1)
        # The agent explores widely, then collapses onto a single region.
        exploring = rng.uniform(-3.0, 3.0, size=(5_000, 2))
        collapsed = rng.normal(0.0, 0.01, size=(5_000, 2))
        stream = stream_from(np.vstack([exploring, collapsed]))

        _, values = entropy_curve(stream, window=1_000, bins_per_dim=10)
        assert values[-1] < values[0]

    def test_cumulative_mode_is_non_decreasing_for_growing_coverage(self):
        rng = np.random.default_rng(2)
        stream = stream_from(rng.uniform(-3.0, 3.0, size=(6_000, 2)))
        _, values = entropy_curve(stream, mode="cumulative", window=1_000, bins_per_dim=5)
        # Coverage only accumulates, so the estimate should not fall materially.
        assert np.all(np.diff(values) > -0.05)

    def test_knn_estimator_runs_end_to_end(self):
        rng = np.random.default_rng(3)
        stream = stream_from(rng.normal(size=(4_000, 2)))
        steps, values = entropy_curve(stream, estimator="knn", window=1_000, max_samples=500)
        assert steps.size == values.size > 0
        assert np.all(np.isfinite(values))

    def test_bin_count_changes_the_estimate(self):
        # The point of the study: the grid estimator's answer depends on a knob with no
        # principled setting.
        rng = np.random.default_rng(4)
        stream = stream_from(rng.normal(size=(4_000, 4)))
        _, coarse = entropy_curve(stream, window=1_000, bins_per_dim=5)
        _, fine = entropy_curve(stream, window=1_000, bins_per_dim=20)
        assert not np.allclose(coarse, fine)

    def test_empty_stream_returns_empty_curve(self):
        empty = StateStream(
            steps=np.empty(0, dtype=np.int64),
            observations=np.empty((0, 2), dtype=np.float32),
            episode_ends=np.empty(0, dtype=bool),
            env_id="x",
            seed=0,
        )
        steps, values = entropy_curve(empty)
        assert steps.size == 0 and values.size == 0

    def test_rejects_unknown_estimator(self):
        stream = stream_from(np.zeros((100, 2)))
        with pytest.raises(ValueError, match="unknown estimator"):
            entropy_curve(stream, estimator="histogram", window=50)


class TestGridOccupancy:
    """The saturation diagnostic for the grid estimator.

    Once cells outnumber samples badly enough that every sample owns a cell, the
    histogram is uniform over N cells and grid_entropy returns log N whatever the policy
    does. Without this diagnostic a flat SVE curve on the bottom rung reads as evidence
    about exploration when it is arithmetic about sample counts.
    """

    def test_low_dimensional_grid_is_not_saturated(self):
        rng = np.random.default_rng(0)
        assert grid_occupancy(rng.normal(size=(5_000, 2)), bins_per_dim=10) < 0.05

    def test_high_dimensional_grid_saturates(self):
        rng = np.random.default_rng(0)
        observations = rng.normal(size=(5_000, 8))
        assert grid_occupancy(observations, bins_per_dim=20) > 0.99

    def test_saturated_entropy_is_log_n(self):
        rng = np.random.default_rng(0)
        observations = rng.normal(size=(5_000, 8))
        assert grid_entropy(observations, bins_per_dim=20) == pytest.approx(np.log(5_000), abs=1e-2)

    def test_saturated_entropy_ignores_the_shape_of_the_distribution(self):
        # A Gaussian and a uniform of comparable spread have genuinely different
        # entropies, but once both saturate the estimator returns log N for each and the
        # difference is invisible. entropy_curve standardises every window to unit
        # variance, so "comparable spread" is the normal case, not a contrived one.
        rng = np.random.default_rng(0)
        gaussian = rng.normal(size=(5_000, 8))
        uniform = rng.uniform(-1.7, 1.7, size=(5_000, 8))

        assert grid_occupancy(gaussian, bins_per_dim=20) > 0.99
        assert grid_occupancy(uniform, bins_per_dim=20) > 0.99
        assert grid_entropy(gaussian, bins_per_dim=20) == pytest.approx(
            grid_entropy(uniform, bins_per_dim=20), abs=1e-3
        )

    def test_unsaturated_grid_still_discriminates(self):
        # The complement: where occupancy is low the estimator is doing real work, which
        # is what makes occupancy the right thing to report next to every SVE curve.
        rng = np.random.default_rng(0)
        broad = rng.normal(size=(5_000, 8))
        narrow = rng.normal(size=(5_000, 8)) * 0.05

        assert grid_occupancy(narrow, bins_per_dim=20) < 0.1
        assert grid_entropy(narrow, bins_per_dim=20) < grid_entropy(broad, bins_per_dim=20) - 1.0

    def test_occupancy_curve_tracks_the_entropy_curve_grid(self):
        rng = np.random.default_rng(0)
        stream = stream_from(rng.normal(size=(4_000, 3)))
        entropy_steps, _ = entropy_curve(stream, window=1_000, bins_per_dim=10)
        occupancy_steps, values = occupancy_curve(stream, window=1_000, bins_per_dim=10)

        assert np.array_equal(entropy_steps, occupancy_steps)
        assert np.all((values >= 0) & (values <= 1))


class TestMillerMadow:
    def test_correction_raises_the_estimate(self):
        rng = np.random.default_rng(0)
        observations = rng.normal(size=(500, 3))
        plain = grid_entropy(observations, bins_per_dim=10)
        corrected = grid_entropy(observations, bins_per_dim=10, correction="miller_madow")
        assert corrected > plain

    def test_correction_is_the_occupied_cell_formula(self):
        rng = np.random.default_rng(1)
        observations = rng.normal(size=(400, 2))
        occupied = grid_occupancy(observations, bins_per_dim=10) * 400
        plain = grid_entropy(observations, bins_per_dim=10)
        corrected = grid_entropy(observations, bins_per_dim=10, correction="miller_madow")
        assert corrected - plain == pytest.approx((occupied - 1) / (2 * 400))

    def test_rejects_unknown_correction(self):
        with pytest.raises(ValueError, match="unknown correction"):
            grid_entropy(np.zeros((10, 2)), bins_per_dim=5, correction="jackknife")


class TestSharedNormaliser:
    def test_entropy_uses_the_same_normaliser_as_the_rnd_replay(self):
        # Preprocessing is part of the "equal footing" claim, so both signals must
        # standardise with the same running statistics, not merely similar ones.
        rng = np.random.default_rng(0)
        stream = stream_from(rng.normal(loc=50.0, scale=10.0, size=(2_000, 3)))
        # Read the observations back off the stream: it stores float32, and the curve is
        # computed from that, not from the float64 array it was built with.
        observations = stream.observations.astype(np.float64)
        # Steps are 0-based and the evaluation point is inclusive, so the first window of
        # a run with stride 1000 covers steps 1..1000, i.e. rows 1:1001.
        end = 1_001

        normalizer = RunningNormalizer(3)
        normalizer.update(observations[:end])
        expected = grid_entropy(
            normalizer.normalize(observations[1:end], clip=None), bins_per_dim=10
        )

        steps, values = entropy_curve(stream, window=1_000, bins_per_dim=10)
        assert steps[0] == 1_000
        assert values[0] == pytest.approx(expected)

    def test_curve_is_invariant_to_a_constant_offset(self):
        # Running standardisation is what makes the signal track novelty rather than
        # observation scale; shifting every observation must not move the curve.
        rng = np.random.default_rng(0)
        observations = rng.normal(size=(2_000, 3))
        _, base = entropy_curve(stream_from(observations), window=500, bins_per_dim=10)
        _, shifted = entropy_curve(stream_from(observations + 100.0), window=500, bins_per_dim=10)
        assert np.allclose(base, shifted)
