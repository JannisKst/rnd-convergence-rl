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

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rnd_convergence.streams import EvalCurve

ConvergenceStatus = Literal["converged", "not_learned", "still_improving"]


@dataclass(frozen=True)
class ConvergenceResult:
    """``t_conv`` together with *why* it is what it is.

    ``t_conv is None`` has two very different causes and they must not be pooled. The
    agent may never have beaten a random policy (``not_learned``) — a real finding on the
    sparse-reward rung. Or it may still have been improving when the step budget ran out
    (``still_improving``) — a statement about the budget, not the agent. Dropping both
    from a per-cell mean of ``Delta`` biases the table towards the seeds that happened to
    converge fastest, so the aggregation layer has to count them separately and report
    the rates alongside the means.
    """

    t_conv: int | None
    status: ConvergenceStatus
    reference: float
    improvement: float


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


def _running_quantile(values: np.ndarray, q: float) -> np.ndarray:
    """Element ``i`` is the ``q``-quantile of ``values[: i + 1]``.

    Naive O(n^2); ``n`` is the number of evaluation points (order 100), so this is
    nowhere near the cost of anything else in the pipeline.
    """
    if q >= 1.0:
        return np.maximum.accumulate(values)
    return np.array([np.quantile(values[: i + 1], q) for i in range(values.size)])


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

    Returns ``None`` when the run never improved on a random policy and when it was still
    improving at the end of the budget. Use :func:`convergence_report` to tell those two
    apart — they mean opposite things and must not be pooled when aggregating.
    """
    return convergence_report(
        curve,
        frac=frac,
        patience=patience,
        n_final=n_final,
        smooth_window=smooth_window,
        min_improvement=min_improvement,
    ).t_conv


def convergence_report(
    curve: EvalCurve,
    *,
    frac: float = 0.95,
    patience: int = 5,
    n_final: int = 5,
    smooth_window: int = 1,
    min_improvement: float = 1e-8,
) -> ConvergenceResult:
    """:func:`convergence_time` plus the diagnosis of why it did or did not fire.

    See :class:`ConvergenceResult` for why the diagnosis is not optional.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    returns = trailing_mean(curve.mean_returns, smooth_window)
    if returns.size == 0:
        return ConvergenceResult(None, "not_learned", float("nan"), float("nan"))

    reference = float(np.mean(returns[-n_final:]))
    improvement = reference - curve.random_return
    if improvement <= min_improvement:
        return ConvergenceResult(None, "not_learned", reference, improvement)

    threshold = curve.random_return + frac * improvement
    index = _first_sustained(returns >= threshold, patience)
    if index is None:
        # The agent learned, but never held above the threshold for `patience`
        # evaluations. On a curve still rising at the end of the run the threshold sits
        # in the tail, so this is the signature of an exhausted step budget.
        return ConvergenceResult(None, "still_improving", reference, improvement)
    return ConvergenceResult(int(curve.steps[index]), "converged", reference, improvement)


def plateau_time(
    steps: np.ndarray,
    values: np.ndarray,
    *,
    tau: float = 0.1,
    patience: int = 5,
    smooth_window: int = 5,
    reference_quantile: float = 0.5,
    log: bool = False,
    min_points: int | None = None,
) -> int | None:
    """Environment step at which a monitoring signal flattened, or ``None``.

    The signal is smoothed, then the change between consecutive points is divided by a
    reference rate: the ``reference_quantile`` quantile of all changes seen so far.
    ``t_plateau`` is the first point at which that ratio stays below ``tau`` for
    ``patience`` consecutive points, i.e. the signal is now moving at less than ``tau``
    times its own characteristic rate of change.

    Three properties motivate normalising by a rate. It is dimensionless, so one ``tau``
    applies to RND error and to entropy without retuning, which is what makes the
    cross-signal comparison fair. It is defined for signals that are negative or zero,
    which differential entropy routinely is, unlike a log-derivative. And a signal
    descending at a constant rate never satisfies it — normalising by the *accumulated
    range* instead would eventually declare a steady ramp plateaued merely because it had
    already travelled a long way, which is precisely the false early stop the study is
    trying to measure rather than commit.

    The reference is the running *median* rather than the running maximum because the
    maximum is set by the single worst point in the run: one transient spike — exactly
    what RND error does when the agent reaches a new region — relaxes the threshold for
    every later point and can pull ``t_plateau`` tens of thousands of steps earlier.
    Smoothing attenuates a spike but does not remove it, and an expanding-window high
    quantile is no better while the window is still short, which is when early spikes
    land. Pass ``reference_quantile=1.0`` to recover the running maximum as a sensitivity
    check.

    ``tau`` is bounded below by the signal's noise floor: the criterion can only be met
    if the residual point-to-point wobble on the flat part is smaller than ``tau`` times
    the typical rate of change while the signal was still moving. On synthetic ramps the
    detector is stable for ``tau`` in roughly ``[0.1, 0.3]`` at noise levels up to ~2% of
    the signal range and returns ``None`` below that band. This is why the default is not
    smaller; the planned ``tau`` sweep should stay inside the band and report where the
    edge is, since falling off it looks identical to "the signal never plateaued".

    Set ``log=True`` to measure the change in log-space instead (positive signals only);
    an exponentially decaying signal has constant slope there and will never be declared
    plateaued, so this is a sensitivity check rather than the default.

    ``min_points`` overrides how many leading points are suppressed before the detector
    is allowed to fire, which defaults to ``patience``. Lower it only to probe the very
    start of a curve; the suppression exists because the first few smoothed points sit on
    a short trailing window and move erratically.

    Raises on a signal that never changes at all, rather than returning ``None``: a
    constant signal has no rate to be slow relative to, and reporting it as "never
    plateaued" would hide the saturated-grid case that
    :func:`~rnd_convergence.entropy.grid_occupancy` exists to expose.

    Raises, too, on a curve too short to resolve a plateau. The detector needs
    ``min_points + patience`` points — ``2 * patience``, i.e. 10, at the defaults —
    because differencing costs one point and the first ``min_points - 1`` are suppressed.
    Below that there is no window position where the criterion could ever be met, so the
    curve resolution has to be checked against the step budget: a 100k-step run measured
    with ``window=10_000`` yields exactly 10 points and can only ever answer at the very
    last one.
    """
    if steps.shape != values.shape:
        raise ValueError(
            f"steps and values must have equal shape, got {steps.shape} {values.shape}"
        )
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if not 0.0 < reference_quantile <= 1.0:
        raise ValueError(f"reference_quantile must be in (0, 1], got {reference_quantile}")
    floor = patience if min_points is None else min_points
    required = floor + patience
    if values.size < required:
        raise ValueError(
            f"need at least {required} curve points to resolve a plateau "
            f"(min_points={floor} + patience={patience}), got {values.size}; "
            "use a smaller measurement window or a longer run"
        )
    if values.size and not np.all(np.isfinite(values)):
        # NaN comparisons are False, so a single non-finite point would silently make the
        # detector return None — which in this codebase is the meaningful "the signal
        # never plateaued" outcome. Those two must not be confusable.
        raise ValueError(
            f"values must be finite, got {int(np.sum(~np.isfinite(values)))} non-finite of "
            f"{values.size} (knn_entropy returns NaN for windows with <= k samples)"
        )

    smoothed = trailing_mean(values, smooth_window)
    if log:
        if np.any(smoothed <= 0):
            raise ValueError("log=True requires a strictly positive signal")
        smoothed = np.log(smoothed)

    changes = np.abs(np.diff(smoothed))
    if not np.any(changes > 0):
        raise ValueError(
            "signal is constant, so it has no rate of change to be measured against "
            "(a grid-SVE curve pinned to log N does this: check grid_occupancy)"
        )
    reference = _running_quantile(changes, reference_quantile)

    # Before the signal has moved at all there is nothing to be slow relative to.
    flags = np.zeros(changes.shape, dtype=bool)
    established = reference > 0
    flags[established] = changes[established] / reference[established] < tau
    flags[: max(floor - 1, 0)] = False

    index = _first_sustained(flags, patience)
    return None if index is None else int(steps[index + 1])


def signal_lag(t_plateau: int | None, t_conv: int | None) -> int | None:
    """``Delta = t_plateau - t_conv``, or ``None`` if either is undefined."""
    if t_plateau is None or t_conv is None:
        return None
    return int(t_plateau - t_conv)


def retained_performance(
    curve: EvalCurve,
    t_stop: int | None,
    *,
    n_final: int = 5,
    smooth_window: int = 1,
) -> float | None:
    """Fraction of final performance retained by stopping at ``t_stop``.

    Measured on the random-policy-anchored scale, so 0 means "no better than random"
    and 1 means "as good as the fully trained policy". Values above 1 are possible when
    the return curve is non-monotonic.

    ``smooth_window`` must match the one passed to :func:`convergence_time`: this number
    and ``t_conv`` are reported in the same table row, and reading them off differently
    smoothed versions of the same curve makes them quietly incomparable.
    """
    if t_stop is None or curve.steps.size == 0:
        return None
    returns = trailing_mean(curve.mean_returns, smooth_window)
    reference = float(np.mean(returns[-n_final:]))
    improvement = reference - curve.random_return
    if improvement <= 0:
        return None
    index = int(np.searchsorted(curve.steps, t_stop, side="right")) - 1
    if index < 0:
        return None
    return float((returns[index] - curve.random_return) / improvement)
