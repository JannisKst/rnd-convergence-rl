"""Convergence ground truth, signal plateau detection, and the lag between them.

Three quantities, all computed offline over a completed run:

``convergence_time``
    ``t_conv`` — when the policy actually converged, read off the evaluation-return
    curve. This is the oracle the candidate signals are judged against; it deliberately
    uses the whole training curve, since its job is to say what the right answer was.

``plateau_time``
    ``t_plateau`` — when a monitoring signal (RND error, or any entropy variant)
    stopped changing. Applied unchanged to every signal so the comparison is fair.

``signal_lag``
    ``Delta = t_plateau - t_conv``. Negative means the signal fires early and stopping on
    it costs performance; positive means it fires late and saves no compute.
"""

from __future__ import annotations

import numpy as np

from rnd_convergence.streams import EvalCurve


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average, same length as the input.

    Trailing rather than centred so that smoothing never mixes in information from
    later in the run. The first ``window - 1`` entries average over however many points
    are available.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window == 1 or values.size == 0:
        return values.astype(np.float64)
    cumulative = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    counts = np.minimum(np.arange(1, values.size + 1), window)
    starts = np.maximum(np.arange(values.size) + 1 - window, 0)
    return (cumulative[1:] - cumulative[starts]) / counts


def _first_sustained(flags: np.ndarray, patience: int) -> int | None:
    """Start index of the first run of ``patience`` consecutive True values."""
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    if flags.size < patience:
        return None
    # Element j of the valid-mode convolution is the number of Trues in flags[j : j + patience].
    window_sums = np.convolve(flags.astype(np.int64), np.ones(patience, dtype=np.int64), "valid")
    (candidates,) = np.nonzero(window_sums == patience)
    if candidates.size == 0:
        return None
    return int(candidates[0])


def convergence_time(
    curve: EvalCurve,
    *,
    frac: float = 0.95,
    patience: int = 5,
    n_final: int = 5,
    smooth_window: int = 1,
    min_improvement: float = 1e-8,
) -> int | None:
    """Environment step at which the policy converged, or ``None`` if it never did.

    ``t_conv`` is the first evaluation point at which the smoothed mean return reaches
    ``R_0 + frac * (R_ref - R_0)`` and stays there for ``patience`` consecutive
    evaluations, where ``R_ref`` is the mean of the final ``n_final`` evaluations and
    ``R_0`` the random-policy return recorded with the run.

    Anchoring on the random-policy baseline rather than on a percentage of the final
    value keeps the criterion meaningful for negative returns (Pendulum-like) and for
    0-1 sparse-reward ranges alike.

    Returns ``None`` when the run never improved on a random policy — the agent did not
    learn, so there is no convergence point to detect. That is a real outcome on the
    hard sparse-reward rung, not an error.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    returns = trailing_mean(curve.mean_returns, smooth_window)
    if returns.size == 0:
        return None

    reference = float(np.mean(returns[-n_final:]))
    improvement = reference - curve.random_return
    if improvement <= min_improvement:
        return None

    threshold = curve.random_return + frac * improvement
    index = _first_sustained(returns >= threshold, patience)
    return None if index is None else int(curve.steps[index])


def plateau_time(
    steps: np.ndarray,
    values: np.ndarray,
    *,
    tau: float = 0.02,
    patience: int = 5,
    smooth_window: int = 5,
    log: bool = False,
    min_points: int | None = None,
) -> int | None:
    """Environment step at which a monitoring signal flattened, or ``None``.

    The signal is smoothed, then the change between consecutive points is divided by the
    largest change seen so far. ``t_plateau`` is the first point at which that ratio
    stays below ``tau`` for ``patience`` consecutive points, i.e. the signal is now
    moving at less than ``tau`` times its own fastest observed rate.

    Three properties motivate this choice. It is dimensionless, so one ``tau`` applies to
    RND error and to entropy without retuning, which is what makes the cross-signal
    comparison fair. It is defined for signals that are negative or zero, which
    differential entropy routinely is, unlike a log-derivative. And a signal descending
    at a constant rate never satisfies it — normalising by the *accumulated range*
    instead would eventually declare a steady ramp plateaued merely because it had
    already travelled a long way, which is precisely the false early stop the study is
    trying to measure rather than commit.

    The reference rate is a running maximum, so a single early spike would depress every
    later ratio. Smoothing before differencing is what keeps that from biting; raise
    ``smooth_window`` for noisy signals.

    Set ``log=True`` to measure the change in log-space instead (positive signals only);
    an exponentially decaying signal has constant slope there and will never be declared
    plateaued, so this is a sensitivity check rather than the default.
    """
    if steps.shape != values.shape:
        raise ValueError(
            f"steps and values must have equal shape, got {steps.shape} {values.shape}"
        )
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    smoothed = trailing_mean(values, smooth_window)
    if log:
        if np.any(smoothed <= 0):
            raise ValueError("log=True requires a strictly positive signal")
        smoothed = np.log(smoothed)

    if smoothed.size < 2:
        return None

    changes = np.abs(np.diff(smoothed))
    fastest_so_far = np.maximum.accumulate(changes)

    # Before the signal has moved at all there is nothing to be slow relative to.
    flags = np.zeros(changes.shape, dtype=bool)
    established = fastest_so_far > 0
    flags[established] = changes[established] / fastest_so_far[established] < tau

    floor = patience if min_points is None else min_points
    flags[: max(floor - 1, 0)] = False

    index = _first_sustained(flags, patience)
    return None if index is None else int(steps[index + 1])


def signal_lag(t_plateau: int | None, t_conv: int | None) -> int | None:
    """``Delta = t_plateau - t_conv``, or ``None`` if either is undefined."""
    if t_plateau is None or t_conv is None:
        return None
    return int(t_plateau - t_conv)


def retained_performance(curve: EvalCurve, t_stop: int | None, n_final: int = 5) -> float | None:
    """Fraction of final performance retained by stopping at ``t_stop``.

    Measured on the random-policy-anchored scale, so 0 means "no better than random"
    and 1 means "as good as the fully trained policy". Values above 1 are possible when
    the return curve is non-monotonic.
    """
    if t_stop is None or curve.steps.size == 0:
        return None
    returns = curve.mean_returns
    reference = float(np.mean(returns[-n_final:]))
    improvement = reference - curve.random_return
    if improvement <= 0:
        return None
    index = int(np.searchsorted(curve.steps, t_stop, side="right")) - 1
    if index < 0:
        return None
    return float((returns[index] - curve.random_return) / improvement)
