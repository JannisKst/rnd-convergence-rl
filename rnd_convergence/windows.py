"""The evaluation grid both monitoring signals are measured on.

This is one function, in its own module, for a reason that is central to what the study
claims. `Delta = t_plateau - t_conv` is compared *between* RND error and every SVE variant
in the same table, and :func:`~rnd_convergence.convergence.plateau_time` reads a plateau off
the differences between consecutive curve points. So the spacing of those points and the
overlap between the windows behind them are not presentation details --- they set the time
scale the detector is sensitive to. Two signals measured on different grids produce two
``t_plateau`` values whose difference is partly an artefact of how each curve was built.

RND error and SVE are estimated in completely different ways: the RND curve averages a
per-state error recorded during a single streaming replay, while an entropy curve has to
re-run a set-valued estimator over each window. Sharing the grid is therefore not something
that falls out of the implementations --- it has to be deliberate, and the only way to keep
it true under later edits is to let both signals derive their points from one function.

What the shared grid does *not* make interchangeable is ``mode``. The step axis comes from
``stride`` alone, so a cumulative curve and a sliding one land on exactly the same points ---
but a cumulative point summarises the whole run so far and a sliding one only the last
``window`` steps, which is a different time scale behind an identical axis.
:func:`~rnd_convergence.convergence.plateau_time` reads the difference between consecutive
points, so it responds to that: on synthetic decay-then-flat curves the sliding version
plateaus and the cumulative transform of the same signal does not fire at all, the same
asymmetry the pilot reported for cumulative SVE. An identical step axis is therefore
necessary for two ``t_plateau`` values to be comparable, not sufficient --- the mode has to
match too. RND error is measured in sliding mode only, for the reason given in
:func:`~rnd_convergence.rnd.streaming_rnd_error`.

The windows are half-open on the left, ``(point - window, point]``, and labelled by their
upper edge: a curve point at step ``t`` summarises the ``window`` steps of training that
ended at ``t``, and never contains information from after it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np

CurveMode = Literal["sliding", "cumulative"]


def iter_windows(
    steps: np.ndarray,
    *,
    mode: CurveMode = "sliding",
    window: int,
    stride: int | None = None,
) -> Iterator[tuple[int, int, int]]:
    """Yield ``(step, start, end)`` index slices for each evaluation point of a curve.

    Parameters
    ----------
    steps
        The environment-step index of each logged state, monotonically non-decreasing.
    mode
        ``"sliding"`` yields the last ``window`` steps at each point --- the current
        policy's behaviour. ``"cumulative"`` yields everything since the start of the run,
        the closer analogue of "has the agent stopped finding new regions".
    window
        Width of the sliding window in environment steps. Sets how much data each curve
        point summarises. ``"cumulative"`` ignores it, since every window there starts at
        the beginning of the run; the grid comes from ``stride`` in both modes, so the two
        modes yield the same points regardless.
    stride
        Spacing of curve points in environment steps; defaults to ``window``, which makes
        consecutive windows adjacent and non-overlapping. Note that ``stride`` alone
        decides how many points a run yields, which is what
        :func:`~rnd_convergence.runs.check_resolution` verifies before a run is launched.

    Windows holding fewer than two states are skipped rather than yielded, so that an
    estimator is never handed a sample it cannot say anything about. That rule is part of
    the shared grid: a curve that dropped different points than its counterpart would put
    the two signals back onto different step axes, which is what this module exists to
    prevent.
    """
    if mode not in ("sliding", "cumulative"):
        raise ValueError(f"unknown mode {mode!r}, expected 'sliding' or 'cumulative'")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    stride = window if stride is None else stride
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if steps.size == 0:
        return

    last_step = int(steps[-1])
    for point in range(stride, last_step + stride, stride):
        end = int(np.searchsorted(steps, point, side="right"))
        if end == 0:
            continue
        start = (
            int(np.searchsorted(steps, point - window, side="right")) if mode == "sliding" else 0
        )
        if end - start < 2:
            continue
        yield point, start, end
