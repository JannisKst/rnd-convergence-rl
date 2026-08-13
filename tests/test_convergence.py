"""Tests for rnd_convergence.convergence.

The detectors are the component most able to invalidate the whole results table while
looking healthy, so they are exercised on synthetic curves whose correct answer is known
in advance rather than on recorded runs.
"""

import numpy as np
import pytest

from rnd_convergence.convergence import (
    convergence_report,
    convergence_time,
    plateau_time,
    retained_performance,
    signal_lag,
    trailing_mean,
)
from rnd_convergence.streams import EvalCurve


def make_curve(mean_returns, random_return=0.0, step=5_000):
    """EvalCurve with one evaluation episode per point and evenly spaced steps."""
    mean_returns = np.asarray(mean_returns, dtype=np.float32)
    steps = np.arange(1, mean_returns.size + 1, dtype=np.int64) * step
    return EvalCurve(
        steps=steps,
        returns=mean_returns[:, None],
        random_return=random_return,
        env_id="Synthetic-v0",
        seed=0,
    )


class TestTrailingMean:
    def test_window_one_is_identity(self):
        values = np.array([3.0, 1.0, 4.0])
        assert np.allclose(trailing_mean(values, 1), values)

    def test_averages_only_over_the_past(self):
        values = np.array([0.0, 0.0, 3.0, 3.0])
        # Index 2 sees [0, 0, 3]; a centred window would have pulled in the later 3.
        assert np.allclose(trailing_mean(values, 3), [0.0, 0.0, 1.0, 2.0])

    def test_partial_windows_at_the_start(self):
        values = np.array([2.0, 4.0])
        assert np.allclose(trailing_mean(values, 5), [2.0, 3.0])


class TestConvergenceTime:
    def test_step_function_converges_where_the_step_is(self):
        # Five evaluations at 0, then five at 100: convergence is the sixth point.
        curve = make_curve([0.0] * 5 + [100.0] * 5)
        assert convergence_time(curve, patience=5) == 30_000

    def test_no_learning_returns_none(self):
        curve = make_curve([0.0] * 10, random_return=0.0)
        assert convergence_time(curve) is None

    def test_performance_below_random_returns_none(self):
        curve = make_curve([-50.0] * 10, random_return=0.0)
        assert convergence_time(curve) is None

    def test_single_spike_does_not_trigger(self):
        # One lucky evaluation reaching the final level must not count as converged.
        returns = [0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0]
        assert convergence_time(make_curve(returns), patience=3) == 40_000

    def test_never_sustained_returns_none(self):
        returns = [0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0]
        assert convergence_time(make_curve(returns), patience=3) is None

    def test_threshold_is_anchored_on_the_random_baseline(self):
        # Random policy already scores 90; reaching 95 is 50% of the available headroom,
        # not 95% of the final value, so it must not count as converged.
        curve = make_curve([95.0] * 5 + [100.0] * 5, random_return=90.0)
        assert convergence_time(curve, frac=0.95, patience=5) == 30_000

    def test_negative_returns_are_handled(self):
        curve = make_curve([-1000.0] * 5 + [-100.0] * 5, random_return=-1000.0)
        assert convergence_time(curve, patience=5) == 30_000

    def test_empty_curve_returns_none(self):
        assert convergence_time(make_curve([])) is None

    def test_rejects_invalid_frac(self):
        with pytest.raises(ValueError, match="frac"):
            convergence_time(make_curve([0.0, 1.0]), frac=1.5)


class TestPlateauTime:
    @staticmethod
    def steps_for(n, step=1_000):
        return np.arange(1, n + 1, dtype=np.int64) * step

    def test_decay_to_a_floor_plateaus_once_flat(self):
        n = 60
        steps = self.steps_for(n)
        values = 10.0 * np.exp(-np.arange(n) / 5.0) + 1.0
        detected = plateau_time(steps, values, tau=0.02, patience=5)

        assert detected is not None
        # It must fire in the tail: under 3% of the initial excess above the floor left.
        index = int(np.searchsorted(steps, detected))
        assert values[index] - 1.0 < 0.03 * (values[0] - 1.0)
        assert detected > steps[5]

    def test_linear_ramp_never_plateaus(self):
        # A constant rate of change is never slow relative to itself, whatever tau is.
        n = 200
        values = np.arange(n, dtype=np.float64)
        for tau in (0.01, 0.1, 0.5):
            assert plateau_time(self.steps_for(n), values, tau=tau, patience=5) is None

    def test_constant_signal_raises_rather_than_returning_none(self):
        # A signal that never varied gives no scale to be flat relative to. Returning
        # None would file it under "never plateaued" alongside genuinely still-moving
        # signals; a saturated grid-SVE curve pinned to log N lands here, and that is a
        # fact about the estimator running out of samples, not about exploration.
        n = 30
        values = np.full(n, 7.0)
        with pytest.raises(ValueError, match="constant"):
            plateau_time(self.steps_for(n), values, tau=0.02, patience=5)

    def test_noise_around_a_flat_level_still_plateaus(self):
        rng = np.random.default_rng(0)
        n = 80
        values = np.concatenate(
            [np.linspace(10.0, 1.0, 30), 1.0 + rng.normal(0.0, 0.01, size=n - 30)]
        )
        detected = plateau_time(self.steps_for(n), values, tau=0.05, patience=5)
        assert detected is not None
        assert detected >= self.steps_for(n)[30]

    def test_handles_negative_signals(self):
        # Differential entropy is routinely negative; the detector must not require logs.
        n = 60
        values = -5.0 + 4.0 * np.exp(-np.arange(n) / 5.0)
        assert plateau_time(self.steps_for(n), values, tau=0.02, patience=5) is not None

    def test_log_mode_rejects_non_positive_signals(self):
        n = 20
        values = np.linspace(-1.0, 1.0, n)
        with pytest.raises(ValueError, match="strictly positive"):
            plateau_time(self.steps_for(n), values, tau=0.02, log=True)

    def test_larger_patience_never_fires_earlier(self):
        n = 80
        steps = self.steps_for(n)
        values = 10.0 * np.exp(-np.arange(n) / 5.0) + 1.0
        early = plateau_time(steps, values, tau=0.02, patience=3)
        late = plateau_time(steps, values, tau=0.02, patience=10)
        assert early is not None and late is not None
        assert late >= early

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="equal shape"):
            plateau_time(np.arange(5), np.arange(4, dtype=np.float64))


class TestLagAndRetainedPerformance:
    def test_lag_is_signed(self):
        assert signal_lag(30_000, 50_000) == -20_000
        assert signal_lag(50_000, 30_000) == 20_000

    def test_lag_is_none_when_either_side_is_undefined(self):
        assert signal_lag(None, 10) is None
        assert signal_lag(10, None) is None

    def test_retained_performance_at_convergence_is_one(self):
        curve = make_curve([0.0] * 5 + [100.0] * 5)
        assert retained_performance(curve, 30_000) == pytest.approx(1.0)

    def test_stopping_early_retains_less(self):
        curve = make_curve([0.0, 25.0, 50.0, 75.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        assert retained_performance(curve, 15_000) == pytest.approx(0.5)

    def test_no_learning_gives_none(self):
        assert retained_performance(make_curve([0.0] * 5), 10_000) is None


class TestPlateauReferenceRobustness:
    """A transient spike must not decide where the plateau is.

    The reference rate was originally the running maximum, so the single largest change
    anywhere in the run set the threshold for every point after it. RND error spikes
    exactly when the agent reaches a new region, which made the detector fire far too
    early on precisely the runs the study cares about.
    """

    ramp_then_flat = np.concatenate([np.linspace(10.0, 1.0, 20), np.full(20, 1.0)])
    steps = np.arange(1, 41, dtype=np.int64) * 5_000

    def test_spike_barely_moves_the_quantile_reference(self):
        spiked = self.ramp_then_flat.copy()
        spiked[1] = 200.0

        clean = plateau_time(self.steps, self.ramp_then_flat)
        assert plateau_time(self.steps, spiked) == clean

    def test_running_maximum_is_the_fragile_case_it_replaced(self):
        spiked = self.ramp_then_flat.copy()
        spiked[1] = 200.0

        clean = plateau_time(self.steps, self.ramp_then_flat, reference_quantile=1.0)
        spiked_max = plateau_time(self.steps, spiked, reference_quantile=1.0)
        assert spiked_max is not None and clean is not None
        assert spiked_max < clean  # the behaviour the default now avoids

    def test_quantile_one_reproduces_the_running_maximum(self):
        values = np.concatenate([np.linspace(5.0, 1.0, 15), np.full(15, 1.0)])
        steps = np.arange(1, 31, dtype=np.int64) * 5_000
        assert plateau_time(steps, values, reference_quantile=1.0) is not None

    def test_rejects_out_of_range_quantile(self):
        with pytest.raises(ValueError, match="reference_quantile"):
            plateau_time(self.steps, self.ramp_then_flat, reference_quantile=0.0)


class TestPlateauRejectsNonFinite:
    """``None`` means "never plateaued" and must not also mean "there was a NaN"."""

    def test_nan_raises_instead_of_silently_returning_none(self):
        steps = np.arange(1, 41, dtype=np.int64) * 5_000
        values = np.concatenate([np.linspace(10.0, 1.0, 20), np.full(20, 1.0)])
        values[3] = np.nan

        with pytest.raises(ValueError, match="finite"):
            plateau_time(steps, values)

    def test_inf_raises_too(self):
        steps = np.arange(1, 11, dtype=np.int64) * 5_000
        values = np.ones(10)
        values[0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            plateau_time(steps, values)


class TestConvergenceReport:
    def test_converged_run_is_labelled_and_timed(self):
        result = convergence_report(make_curve([0.0] * 3 + [100.0] * 7))
        assert result.status == "converged"
        assert result.t_conv == 20_000

    def test_run_that_never_beat_random_is_not_learned(self):
        result = convergence_report(make_curve([0.0] * 10, random_return=0.0))
        assert result.status == "not_learned"
        assert result.t_conv is None

    def test_still_rising_run_is_distinguished_from_not_learning(self):
        # A curve improving right up to the budget: t_conv is None, but for the opposite
        # reason. Pooling this with "not_learned" and dropping both would bias the
        # per-cell mean of Delta towards the fastest-converging seeds.
        result = convergence_report(make_curve(np.linspace(10.0, 500.0, 20), random_return=10.0))
        assert result.status == "still_improving"
        assert result.t_conv is None
        assert result.improvement > 0

    def test_convergence_time_agrees_with_the_report(self):
        for returns in ([0.0] * 3 + [100.0] * 7, [0.0] * 10, list(np.linspace(1.0, 50.0, 20))):
            curve = make_curve(returns)
            assert convergence_time(curve) == convergence_report(curve).t_conv


class TestRetainedPerformanceConsistency:
    def test_smoothing_matches_convergence_time(self):
        # Both numbers land in the same table row, so they must be read off the same
        # smoothed curve; the default of no smoothing matches convergence_time's.
        curve = make_curve([0.0, 60.0, 20.0, 80.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        smoothed = retained_performance(curve, 15_000, smooth_window=3)
        raw = retained_performance(curve, 15_000, smooth_window=1)
        assert smoothed is not None and raw is not None
        assert smoothed != raw

    def test_n_final_is_keyword_only(self):
        with pytest.raises(TypeError):
            retained_performance(make_curve([0.0] * 5 + [100.0] * 5), 30_000, 5)


class TestMinimumCurveLength:
    """The detector has a structural minimum of ``min_points + patience`` points.

    changes has n-1 entries and the first min_points-1 are suppressed, so below that
    there is no window position where the criterion could ever be met. Returning None
    would file "the curve is too coarse to answer" under "the signal never plateaued".
    """

    def test_too_few_points_raises(self):
        values = np.linspace(10.0, 1.0, 9)
        steps = np.arange(1, 10, dtype=np.int64) * 10_000
        with pytest.raises(ValueError, match="at least 10 curve points"):
            plateau_time(steps, values)

    def test_exactly_two_patience_points_is_allowed(self):
        values = np.concatenate([np.linspace(10.0, 1.0, 5), np.full(5, 1.0)])
        steps = np.arange(1, 11, dtype=np.int64) * 10_000
        plateau_time(steps, values)  # must not raise

    def test_the_bound_follows_patience(self):
        steps = np.arange(1, 7, dtype=np.int64) * 10_000
        values = np.linspace(10.0, 1.0, 6)
        with pytest.raises(ValueError, match="at least 8 curve points"):
            plateau_time(steps, values, patience=4)

    def test_min_points_shifts_the_bound(self):
        # min_points=2 + patience=5 requires 7 points: 7 is accepted, 6 is not.
        steps = np.arange(1, 8, dtype=np.int64) * 10_000
        values = np.linspace(10.0, 1.0, 7)
        plateau_time(steps, values, min_points=2)  # must not raise

        with pytest.raises(ValueError, match=r"min_points=2 \+ patience=5"):
            plateau_time(steps[:6], values[:6], min_points=2)

    def test_min_points_lets_the_detector_fire_earlier(self):
        # Documented behaviour of the parameter: it controls how many leading points are
        # suppressed before the detector may fire.
        values = np.concatenate([np.linspace(10.0, 9.0, 3), np.full(20, 9.0)])
        steps = np.arange(1, 24, dtype=np.int64) * 5_000
        assert plateau_time(steps, values, min_points=1) <= plateau_time(steps, values)

    def test_a_100k_run_at_window_10k_is_at_the_boundary(self):
        # The concrete case: 100k steps / 10k window = 10 points, exactly the minimum,
        # leaving a single valid window position. The README protocol has to state a
        # budget that clears this comfortably.
        steps = np.arange(1, 11, dtype=np.int64) * 10_000
        values = 10.0 * np.exp(-np.arange(10) / 3.0) + 1.0
        plateau_time(steps, values)  # must not raise
        with pytest.raises(ValueError, match="at least 10"):
            plateau_time(steps[:9], values[:9])
