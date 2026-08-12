"""Tests for rnd_convergence.convergence.

The detectors are the component most able to invalidate the whole results table while
looking healthy, so they are exercised on synthetic curves whose correct answer is known
in advance rather than on recorded runs.
"""

import numpy as np
import pytest

from rnd_convergence.convergence import (
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

    def test_constant_signal_returns_none(self):
        # A signal that never varied gives no scale to be flat relative to, so it is
        # reported as undetected rather than as plateaued from step one.
        n = 30
        values = np.full(n, 7.0)
        assert plateau_time(self.steps_for(n), values, tau=0.02, patience=5) is None

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
