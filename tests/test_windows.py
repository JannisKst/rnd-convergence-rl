"""Tests for the evaluation grid both monitoring signals are measured on.

The test that matters most is at the bottom: RND error and every SVE variant must land on
an identical step axis. `Delta = t_plateau - t_conv` is compared between the two signals in
the same table, and plateau_time reads a plateau off the differences between consecutive
points -- so a curve built on overlapping windows and one built on disjoint buckets yield
two t_plateau values whose difference is partly an artefact of the grids. The two curves
are produced by completely different machinery (a streaming replay against a re-run
set-valued estimator), so nothing keeps them aligned except deriving both from
iter_windows, and nothing keeps *that* true except this test.
"""

import numpy as np
import pytest

from rnd_convergence.entropy import entropy_curve, occupancy_curve
from rnd_convergence.rnd import streaming_rnd_error
from rnd_convergence.streams import StateStream
from rnd_convergence.windows import iter_windows


def stream_of(n_steps, n_dims=3, seed=0):
    rng = np.random.default_rng(seed)
    return StateStream(
        steps=np.arange(n_steps, dtype=np.int64),
        observations=rng.normal(size=(n_steps, n_dims)).astype(np.float32),
        episode_ends=np.zeros(n_steps, dtype=bool),
        env_id="synthetic",
        seed=0,
    )


class TestIterWindows:
    def test_points_are_labelled_by_the_upper_edge_of_their_window(self):
        steps = np.arange(3_000, dtype=np.int64)
        points = [p for p, _, _ in iter_windows(steps, window=1_000)]
        assert points == [1_000, 2_000, 3_000]

    def test_stride_defaults_to_the_window_giving_adjacent_windows(self):
        steps = np.arange(3_000, dtype=np.int64)
        assert list(iter_windows(steps, window=1_000)) == list(
            iter_windows(steps, window=1_000, stride=1_000)
        )

    def test_stride_alone_sets_the_number_of_points(self):
        # The property check_resolution counts on: widening the window changes what each
        # point averages over, not how many points there are.
        steps = np.arange(10_000, dtype=np.int64)
        for window in (1_000, 2_000, 5_000):
            assert len(list(iter_windows(steps, window=window, stride=500))) == 20

    def test_a_window_never_reaches_past_its_own_point(self):
        # Causality: a curve point at step t must not see states logged after t.
        steps = np.arange(5_000, dtype=np.int64)
        for point, _, end in iter_windows(steps, window=1_000, stride=500):
            assert steps[end - 1] <= point

    def test_sliding_windows_look_back_exactly_one_window(self):
        steps = np.arange(5_000, dtype=np.int64)
        for point, start, _ in iter_windows(steps, window=1_000, stride=500):
            assert steps[start] > point - 1_000

    def test_cumulative_windows_start_at_the_beginning_of_the_run(self):
        steps = np.arange(5_000, dtype=np.int64)
        assert all(
            start == 0 for _, start, _ in iter_windows(steps, mode="cumulative", window=1_000)
        )

    def test_windows_holding_fewer_than_two_states_are_skipped(self):
        # Both curves must drop the same points, or they end up on different axes.
        steps = np.array([0, 1, 2, 900, 5_000], dtype=np.int64)
        assert [p for p, _, _ in iter_windows(steps, window=1_000, stride=1_000)] == [1_000]

    def test_empty_stream_yields_nothing(self):
        assert list(iter_windows(np.empty(0, dtype=np.int64), window=100)) == []

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"window": 0}, "window must be >= 1"),
            ({"window": 100, "stride": 0}, "stride must be >= 1"),
            ({"window": 100, "mode": "rolling"}, "unknown mode"),
        ],
    )
    def test_rejects_nonsensical_settings(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            list(iter_windows(np.arange(10, dtype=np.int64), **kwargs))


class TestSignalsShareOneGrid:
    """RND error and SVE must be measured at the same points, or Delta is not comparable."""

    @pytest.mark.parametrize("window, stride", [(1_000, 500), (2_000, 1_000), (4_000, 2_000)])
    def test_rnd_and_entropy_curves_land_on_identical_steps(self, window, stride):
        stream = stream_of(10_000)
        rnd_steps, _ = streaming_rnd_error(stream, window=window, stride=stride)
        entropy_steps, _ = entropy_curve(stream, window=window, stride=stride)
        occupancy_steps, _ = occupancy_curve(stream, window=window, stride=stride)

        assert rnd_steps.size == 10_000 // stride
        assert np.array_equal(rnd_steps, entropy_steps)
        assert np.array_equal(rnd_steps, occupancy_steps)

    def test_both_signals_get_the_point_count_the_stride_implies(self):
        # The regression this file exists for. streaming_rnd_error used to bucket by
        # window with no stride at all, so at the shipped window = 2 x stride it produced
        # half the points check_resolution had verified -- on the project's primary signal.
        stream = stream_of(20_000)
        window, stride = 2_000, 1_000
        rnd_steps, _ = streaming_rnd_error(stream, window=window, stride=stride)
        entropy_steps, _ = entropy_curve(stream, window=window, stride=stride)
        assert rnd_steps.size == entropy_steps.size == 20_000 // stride

    def test_the_rnd_window_still_averages_over_the_full_window(self):
        # Sharing the grid must not have quietly narrowed what each RND point covers:
        # overlapping windows mean a state contributes to more than one point.
        stream = stream_of(4_000)
        wide, _ = streaming_rnd_error(stream, window=2_000, stride=1_000, seed=0)
        narrow, _ = streaming_rnd_error(stream, window=1_000, stride=1_000, seed=0)
        assert wide.size > narrow.size // 2
        assert not np.allclose(
            streaming_rnd_error(stream, window=2_000, stride=1_000, seed=0)[1][:3],
            streaming_rnd_error(stream, window=1_000, stride=1_000, seed=0)[1][:3],
        )
