"""Offline analysis: from logged runs to the rows of the results table.

Training writes two artefacts and a sidecar per run; this module turns a directory of them
into one tidy frame in which every row is a single ``(run, signal, detector setting)``
measurement. The table the study reports, the sensitivity sweeps and the figures are all
selections from that frame, which is why the sweeps are part of the frame rather than
separate passes: ``Delta`` moves with ``tau`` and ``smooth_window``, so a frame that fixed
them would be an answer with its uncertainty already discarded.

Two things here carry most of the weight.

**Nothing this module calls is allowed to kill a batch.**
:func:`~rnd_convergence.convergence.plateau_time` deliberately *raises* on a curve too
short to resolve a plateau and on a signal that never changes --- correct for a library,
where returning ``None`` would make "no plateau" and "unmeasurable" indistinguishable, and
fatal for a sweep, where one unmeasurable SVE cell would take out the other 269 rows of the
run. :func:`detect_plateau` maps each of those conditions to a *recorded status* instead.

**Every failure mode has to survive into the frame.** ``t_conv`` is undefined for opposite
reasons on different rungs (``not_learned`` on DoorKey-8x8, ``still_improving`` on an
exhausted budget), and a signal can fail to plateau for three more. Dropping any of them
biases each cell towards the seeds that happened to converge fastest, so they are counted
and reported next to the means rather than filtered out here.

One consequence of the first point is worth stating separately, because it is easy to read
the frame the wrong way round. ``plateau_status == "constant_signal"`` fires only on a curve
that is constant to within floating-point arithmetic. A *saturated* grid-SVE curve is not
that: it is pinned to ``log N`` where ``N`` is the number of states per window, which wobbles
by a few as episode boundaries move, so it varies by around ``1e-4`` relative and comes back
as an ordinary ``plateau`` or ``no_plateau``. Saturation is read off ``occupancy_at_plateau``
sitting at 1.0, which is why every grid-SVE row carries one; it is not read off
``plateau_status``, and not off ``occupancy_max`` either --- see :func:`_row`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from rnd_convergence.convergence import (
    ConvergenceResult,
    convergence_report,
    plateau_time,
    retained_report,
    signal_lag,
)
from rnd_convergence.entropy import Correction, entropy_curve, occupancy_curve
from rnd_convergence.rnd import streaming_rnd_error
from rnd_convergence.runs import RUN_META_SUFFIX, RunMeta, load_run_meta
from rnd_convergence.streams import (
    EVAL_CURVE_SUFFIX,
    STATE_STREAM_SUFFIX,
    EvalCurve,
    StateStream,
    load_eval_curve,
    load_state_stream,
)
from rnd_convergence.windows import CurveMode, iter_windows

PlateauStatus = Literal[
    "plateau",
    "no_plateau",
    "too_short",
    "constant_signal",
    "non_finite",
    "empty",
    "detector_error",
]

# Substrings of the ValueError messages plateau_time raises for conditions that are facts
# about the data rather than bugs in the caller. Matching on message text is fragile in
# general; it is acceptable here because both sides live in this repo, and because
# TestPlateauStatuses provokes each condition for real and asserts the resulting status ---
# so a reworded message fails the suite loudly instead of silently degrading every affected
# cell to "detector_error".
_DATA_CONDITIONS: tuple[tuple[str, PlateauStatus], ...] = (
    ("curve points to resolve a plateau", "too_short"),
    ("signal is constant", "constant_signal"),
    ("must be finite", "non_finite"),
)

# The detector settings the study sweeps. tau below ~0.1 stops firing at all on noisy
# signals, and "never fired" is indistinguishable from "never plateaued", so the sweep
# stays inside the band where the detector is known to be stable. reference_quantile 1.0
# is the running maximum, the sensitivity check the README pins the median against.
DEFAULT_TAUS: tuple[float, ...] = (0.1, 0.2, 0.3)
DEFAULT_SMOOTH_WINDOWS: tuple[int, ...] = (1, 5, 9)
DEFAULT_REFERENCE_QUANTILES: tuple[float, ...] = (0.5, 1.0)

# Smoothing applied to the *evaluation* curve before t_conv is read off it. Swept because
# the MiniGrid rungs produce an all-or-nothing greedy return --- 0.955 or 0.0, depending on
# which start state the episode drew --- on which an unsmoothed t_conv lands wherever the
# final few coin flips happened to fall. Smoothing turns that curve into a success rate,
# which is the thing worth thresholding.
DEFAULT_CONV_SMOOTH_WINDOWS: tuple[int, ...] = (1, 3, 5)

DEFAULT_BIN_COUNTS: tuple[int, ...] = (5, 10, 20)

# Standardised observations are clipped to [-clip, clip] before the grid is laid over them.
# A sequence rather than a scalar because grid_entropy's docstring is explicit that this is
# a sweep parameter and not a constant: the share of samples folded into the edge bins grows
# with dimensionality, so a fixed clip lets clipping masquerade as the binning degradation
# the study is trying to measure. It defaults to one value because on a ladder topping out
# at 4-D nothing is near the edge --- the sweep is what LunarLander will need.
DEFAULT_CLIPS: tuple[float, ...] = (5.0,)

# Seeds for the two estimators that are not deterministic given the stream: the RND replay
# (target and predictor initialisation) and the kNN subsample. Fixed at one seed everywhere
# in the current results, which is a real limitation --- t_plateau for RND depends on which
# random target was drawn, and with a single seed the frame cannot say how much of Delta is
# that draw. Pass several to find out; the grid estimators are deterministic and are built
# once regardless.
DEFAULT_ANALYSIS_SEEDS: tuple[int, ...] = (0,)

# Bias corrections for the grid estimator. Miller-Madow compensates the plug-in estimator's
# downward bias when cells are undersampled, which is the regime the top of the ladder lives
# in, so the README offers it as a sensitivity check --- and a sensitivity check that no
# entry point can reach is not one. It is a sweep axis rather than a flag because the
# comparison being made is plug-in against corrected, on the same curve.
DEFAULT_CORRECTIONS: tuple[Correction, ...] = ("none",)

# Key the curve settings are stored under inside a saved curve archive.
CURVE_MANIFEST = "__params__"

# Columns that are step counts, i.e. integers that are legitimately absent. Held as pandas
# nullable integers so that a missing t_conv does not silently turn the whole column to
# float and write every step count to CSV as "20480.0".
NULLABLE_INT_COLUMNS: tuple[str, ...] = ("t_conv", "t_plateau", "delta")


# eq=False because the fields include numpy arrays, for which the generated __eq__ raises
# on an ambiguous truth value and the generated __hash__ raises outright. Identity semantics
# are what the call sites actually want; nothing here compares two curves for equality.
@dataclass(frozen=True, eq=False)
class Curve:
    """One monitoring signal measured over a run, with the settings that produced it.

    ``signal`` is the estimator family (``rnd``, ``sve_grid``, ``sve_knn``, ``occupancy``)
    and the remaining fields are what distinguishes curves within it. They travel with the
    values because they become columns of the frame: the bin count in particular is an
    independent variable of the study rather than a hyperparameter, since the sweep over it
    is how discretization sensitivity is demonstrated.

    The rule for what belongs here is that every setting which *moves the number* has to be
    recoverable from the frame. That is why ``analysis_seed`` and ``rnd_overrides`` are
    fields and not just arguments: two frames built with different RND learning rates, or
    with different target-network draws, would otherwise be indistinguishable once written
    to CSV. ``None`` for a setting means it does not apply to this estimator --- the grid
    estimators are deterministic given the stream, so they carry no ``analysis_seed``.

    ``rnd_overrides`` is JSON of the ``streaming_rnd_error`` arguments that were overridden,
    empty when none were. It records the overrides rather than the effective values so that
    the defaults live in that function's signature and nowhere else.
    """

    signal: str
    steps: np.ndarray
    values: np.ndarray
    mode: CurveMode = "sliding"
    bins_per_dim: int | None = None
    clip: float | None = None
    correction: Correction | None = None
    k: int | None = None
    max_samples: int | None = None
    analysis_seed: int | None = None
    rnd_overrides: str = ""

    @property
    def label(self) -> str:
        """Stable short name, unique within one :func:`build_curves` call.

        Carries exactly the settings that are swept, so that two curves differing in any of
        them cannot collide: bin count and clip for the grid estimators, ``k`` for kNN, and
        the analysis seed for the two estimators whose value depends on it. Uniqueness is
        scoped to one call because that is the set that lands in one archive and one legend;
        concatenating the output of two calls that differ in something not swept here is not
        a case this name is trying to survive.
        """
        parts = [self.signal]
        if self.bins_per_dim is not None:
            parts.append(f"b{self.bins_per_dim}")
        if self.clip is not None:
            parts.append(f"c{self.clip:g}")
        if self.correction not in (None, "none"):
            parts.append("mm")
        if self.k is not None:
            parts.append(f"k{self.k}")
        if self.analysis_seed is not None:
            parts.append(f"s{self.analysis_seed}")
        parts.append(self.mode)
        return "_".join(parts)

    @property
    def is_diagnostic(self) -> bool:
        """True for curves that describe an estimator rather than the agent.

        Occupancy is the grid estimator's saturation diagnostic, not a convergence signal:
        asking when it plateaus would be asking when the ratio of occupied cells to samples
        stopped moving, which answers nothing the study poses. It is built alongside the
        grid-SVE curves and attached to their rows instead.
        """
        return self.signal == "occupancy"

    @property
    def params(self) -> dict[str, Any]:
        """The identifying settings, as frame columns.

        Also the round-trip format for :func:`save_curves`, so every key has to be a
        constructor argument and JSON-representable.
        """
        return {
            "signal": self.signal,
            "mode": self.mode,
            "bins_per_dim": self.bins_per_dim,
            "clip": self.clip,
            "correction": self.correction,
            "k": self.k,
            "max_samples": self.max_samples,
            "analysis_seed": self.analysis_seed,
            "rnd_overrides": self.rnd_overrides,
        }

    @property
    def grid_key(self) -> tuple[int | None, float | None, str]:
        """What identifies the grid a discretizing curve was measured on.

        Used to pair each grid-SVE curve with the occupancy curve that diagnoses it. It has
        to include ``clip`` as well as the bin count: both are swept, and two curves at the
        same bin count but different clips sit on genuinely different grids. ``correction``
        is deliberately *not* part of it --- a bias correction changes the entropy estimate,
        not which cells were occupied, so both corrections share one occupancy curve.
        """
        return (self.bins_per_dim, self.clip, self.mode)


@dataclass(frozen=True)
class DetectorSpec:
    """One configuration of :func:`~rnd_convergence.convergence.plateau_time`.

    ``conv_smooth_window`` rides along because it is not free either: it smooths the
    evaluation curve that ``t_conv`` is read from, and ``Delta`` is the difference of the
    two. It is applied identically to :func:`~rnd_convergence.convergence.retained_performance`,
    which that function's docstring requires --- reading the two numbers of one table row
    off differently smoothed versions of the same curve makes them quietly incomparable.
    """

    tau: float = 0.1
    smooth_window: int = 5
    reference_quantile: float = 0.5
    conv_smooth_window: int = 1

    def __post_init__(self) -> None:
        # Fail on a malformed sweep now rather than turning every row of it into a
        # detector_error status, which would look like a finding about the data.
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if not 0.0 < self.reference_quantile <= 1.0:
            raise ValueError(f"reference_quantile must be in (0, 1], got {self.reference_quantile}")
        if self.smooth_window < 1 or self.conv_smooth_window < 1:
            raise ValueError("smoothing windows must be >= 1")

    @property
    def params(self) -> dict[str, Any]:
        """The settings, as frame columns."""
        return {
            "tau": self.tau,
            "smooth_window": self.smooth_window,
            "reference_quantile": self.reference_quantile,
            "conv_smooth_window": self.conv_smooth_window,
        }

    @property
    def plateau_key(self) -> tuple[float, int, float]:
        """The settings ``plateau_time`` actually reads.

        ``conv_smooth_window`` is not among them --- it belongs to the evaluation curve, not
        to the signal --- so every ``t_plateau`` is shared by the specs that differ only in
        it. :func:`analyse_run` caches on this, the same way it caches
        ``convergence_report`` on ``conv_smooth_window`` alone.
        """
        return (self.tau, self.smooth_window, self.reference_quantile)


def detector_grid(
    taus: Sequence[float] = DEFAULT_TAUS,
    smooth_windows: Sequence[int] = DEFAULT_SMOOTH_WINDOWS,
    reference_quantiles: Sequence[float] = DEFAULT_REFERENCE_QUANTILES,
    conv_smooth_windows: Sequence[int] = DEFAULT_CONV_SMOOTH_WINDOWS,
) -> list[DetectorSpec]:
    """The full cross-product of detector settings to sweep."""
    return [
        DetectorSpec(tau, smooth_window, quantile, conv_smooth)
        for tau in taus
        for smooth_window in smooth_windows
        for quantile in reference_quantiles
        for conv_smooth in conv_smooth_windows
    ]


def build_curves(
    stream: StateStream,
    *,
    window: int,
    stride: int,
    bin_counts: Sequence[int] = DEFAULT_BIN_COUNTS,
    clips: Sequence[float] = DEFAULT_CLIPS,
    corrections: Sequence[Correction] = DEFAULT_CORRECTIONS,
    modes: Sequence[CurveMode] = ("sliding",),
    knn_k: int = 4,
    knn_max_samples: int = 4_000,
    seeds: Sequence[int] = DEFAULT_ANALYSIS_SEEDS,
    rnd_kwargs: dict[str, Any] | None = None,
) -> list[Curve]:
    """Measure every monitoring signal over one logged run.

    All curves share the grid :func:`~rnd_convergence.windows.iter_windows` lays out from
    ``window`` and ``stride``, which is what makes their ``t_plateau`` values comparable.
    Both should come from the run's :class:`~rnd_convergence.runs.RunMeta`: they are a
    property of how the run was *sized*, and choosing a different stride afterwards moves
    every plateau.

    The RND curve is built once per seed and only in sliding mode --- see
    :func:`~rnd_convergence.rnd.streaming_rnd_error` for why there is no cumulative variant
    --- while the SVE curves are built per requested mode. Note that a cumulative SVE curve
    lands on the same step axis as the sliding RND curve without being comparable to it;
    ``mode`` is carried into the frame so that comparison is never made by accident.

    ``seeds`` is swept over the estimators whose output depends on a random draw --- the RND
    replay always, and the kNN estimate only where it actually subsamples --- and never over
    the grid estimators, which are deterministic given the stream and would otherwise be
    recomputed identically. Sweeping it costs one full replay per seed, which is the
    expensive part of this function, so the default is a single seed; several are what it
    takes to say how much of ``Delta`` is the target-network draw rather than the agent.

    The kNN qualification matters more than it sounds. :func:`~rnd_convergence.entropy.
    knn_entropy` only draws a subsample when a window holds more than ``knn_max_samples``
    states, so on a rung whose ``signal_window`` is at or below that --- MarsRover measures
    at 1000 --- the curve is identical for every seed. Emitting one row per seed there would
    write exact duplicates, and any spread-over-seeds statistic computed from them would
    report a variance of zero: "this estimator is stable" when the truth is "the sweep never
    applied". Those curves are marked ``analysis_seed=None`` instead, the same way the grid
    estimators are.
    """
    overrides = json.dumps(rnd_kwargs, sort_keys=True) if rnd_kwargs else ""
    curves = [
        Curve(
            signal="rnd",
            analysis_seed=seed,
            rnd_overrides=overrides,
            **_named_curve(
                streaming_rnd_error(
                    stream, window=window, stride=stride, seed=seed, **(rnd_kwargs or {})
                )
            ),
        )
        for seed in seeds
    ]
    for mode in modes:
        for bins in bin_counts:
            for clip in clips:
                for correction in corrections:
                    curves.append(
                        Curve(
                            signal="sve_grid",
                            mode=mode,
                            bins_per_dim=bins,
                            clip=clip,
                            correction=correction,
                            **_named_curve(
                                entropy_curve(
                                    stream,
                                    estimator="grid",
                                    mode=mode,
                                    window=window,
                                    stride=stride,
                                    bins_per_dim=bins,
                                    clip=clip,
                                    correction=correction,
                                )
                            ),
                        )
                    )
                # One occupancy curve per grid, not per correction: a bias correction moves
                # the entropy estimate, not which cells the states landed in.
                curves.append(
                    Curve(
                        signal="occupancy",
                        mode=mode,
                        bins_per_dim=bins,
                        clip=clip,
                        **_named_curve(
                            occupancy_curve(
                                stream,
                                mode=mode,
                                window=window,
                                stride=stride,
                                bins_per_dim=bins,
                                clip=clip,
                            )
                        ),
                    )
                )
        knn_seeds = _knn_seeds(
            stream,
            mode=mode,
            window=window,
            stride=stride,
            max_samples=knn_max_samples,
            seeds=seeds,
        )
        for seed in knn_seeds:
            curves.append(
                Curve(
                    signal="sve_knn",
                    mode=mode,
                    k=knn_k,
                    max_samples=knn_max_samples,
                    analysis_seed=seed,
                    **_named_curve(
                        entropy_curve(
                            stream,
                            estimator="knn",
                            mode=mode,
                            window=window,
                            stride=stride,
                            k=knn_k,
                            max_samples=knn_max_samples,
                            seed=seeds[0] if seed is None else seed,
                        )
                    ),
                )
            )
    return curves


def _knn_seeds(
    stream: StateStream,
    *,
    mode: CurveMode,
    window: int,
    stride: int,
    max_samples: int,
    seeds: Sequence[int],
) -> tuple[int | None, ...]:
    """The seeds worth building a kNN curve at, or ``(None,)`` when the estimate is exact.

    ``knn_entropy`` subsamples only when a window holds more than ``max_samples`` states, so
    the question is whether any window on this grid does. Answered from the window layout
    rather than from ``window <= max_samples``, because the two differ: cumulative windows
    grow past any fixed width, and a stream logged at less than one state per environment
    step holds fewer states than its width suggests.
    """
    subsamples = any(
        end - start > max_samples
        for _, start, end in iter_windows(stream.steps, mode=mode, window=window, stride=stride)
    )
    return tuple(seeds) if subsamples else (None,)


def _named_curve(curve: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    steps, values = curve
    return {"steps": steps, "values": values}


def detect_plateau(curve: Curve, spec: DetectorSpec) -> tuple[int | None, PlateauStatus]:
    """``t_plateau`` for one curve under one detector setting, and why it is what it is.

    Never raises on a data condition. ``plateau_time`` distinguishes "this signal never
    flattened" (``None``) from "this curve cannot answer the question" (raises), and both
    distinctions matter to the result --- a grid-SVE curve pinned to ``log N`` by cell
    saturation is arithmetic about sample counts, not evidence that exploration never
    settled. So each is mapped to its own status rather than collapsed or propagated.
    """
    if curve.values.size == 0:
        return None, "empty"
    try:
        t_plateau = plateau_time(
            curve.steps,
            curve.values,
            tau=spec.tau,
            smooth_window=spec.smooth_window,
            reference_quantile=spec.reference_quantile,
        )
    except ValueError as error:
        message = str(error)
        for needle, status in _DATA_CONDITIONS:
            if needle in message:
                return None, status
        # Not a known data condition. Recorded rather than raised so the rest of the sweep
        # survives, but it means something unanticipated happened and should be read as a
        # bug report, not as a property of this run.
        return None, "detector_error"
    return (None, "no_plateau") if t_plateau is None else (t_plateau, "plateau")


def _reject_resolution_overrides(curve_kwargs: dict[str, Any]) -> None:
    """Refuse ``window``/``stride`` from a caller that has a ``RunMeta`` to hand.

    Shared by :func:`analyse_run` and :func:`analyse_directory` so that the same mistake
    gets the same explanation. Without it the directory-level call reaches ``build_curves``
    and fails with "got multiple values for keyword argument 'window'", which says nothing
    about why passing one was wrong.
    """
    overrides = set(curve_kwargs) & {"window", "stride"}
    if overrides:
        raise TypeError(
            f"{', '.join(sorted(overrides))} come from the run's RunMeta, not from the "
            "caller: the run was sized for them and check_resolution verified them before "
            "it was launched. To measure at another resolution, call build_curves directly "
            "and pass curves="
        )


def analyse_run(
    stream: StateStream,
    eval_curve: EvalCurve,
    meta: RunMeta,
    *,
    curves: Sequence[Curve] | None = None,
    detectors: Sequence[DetectorSpec] | None = None,
    **curve_kwargs: Any,
) -> list[dict[str, Any]]:
    """One row per ``(signal, detector setting)`` for a single run.

    Pass ``curves`` to reuse a set already built for this run; otherwise they are built
    here at the resolution ``meta`` records. Building is the expensive half and the sweep
    the cheap one, so the two are separable on purpose --- re-sweeping detector settings
    over cached curves costs nothing.

    ``window`` and ``stride`` are not accepted here: they come from ``meta``, because they
    are a property of how the run was *sized* and measuring one run at a different
    resolution from the rest of the table moves its ``t_plateau`` for reasons that have
    nothing to do with its signal. Build the curves yourself and pass them in if that is
    genuinely what you want.
    """
    _reject_resolution_overrides(curve_kwargs)
    if curves is not None and curve_kwargs:
        # Silently ignoring these would produce a frame whose columns say the sweep ran at
        # settings it never saw.
        raise TypeError(
            f"curves= and build settings are mutually exclusive, got {sorted(curve_kwargs)} "
            "alongside curves=. The passed curves were already built at their own settings; "
            "drop one or the other"
        )
    if curves is None:
        curves = build_curves(
            stream, window=meta.signal_window, stride=meta.signal_stride, **curve_kwargs
        )
    detectors = detector_grid() if detectors is None else detectors

    occupancy = {curve.grid_key: curve for curve in curves if curve.is_diagnostic}
    reports: dict[int, ConvergenceResult] = {}
    # Both detectors are cached on the axis they actually depend on. t_conv varies only with
    # conv_smooth_window, and t_plateau only with the other three, so the full cross-product
    # of specs asks each question three times over.
    plateaus: dict[tuple[int, tuple[float, int, float]], tuple[int | None, PlateauStatus]] = {}
    rows: list[dict[str, Any]] = []

    for spec in detectors:
        if spec.conv_smooth_window not in reports:
            reports[spec.conv_smooth_window] = convergence_report(
                eval_curve, smooth_window=spec.conv_smooth_window
            )
        report = reports[spec.conv_smooth_window]

        for index, curve in enumerate(curves):
            if curve.is_diagnostic:
                continue
            key = (index, spec.plateau_key)
            if key not in plateaus:
                plateaus[key] = detect_plateau(curve, spec)
            t_plateau, status = plateaus[key]
            rows.append(_row(meta, curve, spec, report, t_plateau, status, eval_curve, occupancy))
    return rows


def _row(
    meta: RunMeta,
    curve: Curve,
    spec: DetectorSpec,
    report: ConvergenceResult,
    t_plateau: int | None,
    status: PlateauStatus,
    eval_curve: EvalCurve,
    occupancy: dict[tuple[int | None, float | None, str], Curve],
) -> dict[str, Any]:
    """Assemble one frame row."""
    # Delta is excluded for the control by the `frozen` *flag*, never by its convergence
    # status. An untrained network is still a consistent policy: the frozen CartPole pilot
    # scored 168.7 against a random baseline of 20.7 on a flat curve, which satisfies the
    # convergence criterion and reports `converged`. Trusting that status would put a
    # meaningless Delta into the table on the one run whose job is to be a credibility
    # anchor.
    comparable = not meta.frozen
    diagnostic = occupancy.get(curve.grid_key) if curve.bins_per_dim is not None else None
    retained = (
        retained_report(eval_curve, t_plateau, smooth_window=spec.conv_smooth_window)
        if comparable
        else None
    )

    return {
        "env_id": meta.env_id,
        "seed": meta.seed,
        "tag": meta.tag,
        "frozen": meta.frozen,
        "total_steps": meta.total_steps,
        "signal_window": meta.signal_window,
        "signal_stride": meta.signal_stride,
        "wall_clock_seconds": meta.wall_clock_seconds,
        **curve.params,
        "label": curve.label,
        **spec.params,
        "n_points": int(curve.values.size),
        "value_first": float(curve.values[0]) if curve.values.size else float("nan"),
        "value_last": float(curve.values[-1]) if curve.values.size else float("nan"),
        # occupancy_at_plateau is the one to read: the claim it defends is "the plateau you
        # just detected is arithmetic about sample counts, not exploration", and that is a
        # statement about the grid *where the plateau was found*. occupancy_max is over the
        # whole curve, whose first point is a half-width window -- occupancy is cells over
        # samples, so it is structurally inflated there and does peak at point 0 on most
        # runs. Taking the max would let one transient early window condemn a curve whose
        # plateau region was perfectly well sampled.
        "occupancy_max": _summarise(diagnostic, np.max),
        "occupancy_last": _summarise(diagnostic, lambda values: values[-1]),
        "occupancy_at_plateau": _at_step(diagnostic, t_plateau),
        "t_conv": report.t_conv if comparable else None,
        "conv_status": report.status if comparable else "control",
        "random_return": eval_curve.random_return,
        "reference_return": report.reference,
        "t_plateau": t_plateau,
        "plateau_status": status,
        "delta": signal_lag(t_plateau, report.t_conv) if comparable else None,
        "retained": retained.retained if retained is not None else None,
        "retained_status": retained.status if retained is not None else "control",
    }


def _summarise(curve: Curve | None, reduce: Callable[[np.ndarray], Any]) -> float | None:
    if curve is None or curve.values.size == 0:
        return None
    return float(reduce(curve.values))


def _at_step(curve: Curve | None, step: int | None) -> float | None:
    """The curve's value at the last point at or before ``step``.

    ``searchsorted`` rather than an exact match: a diagnostic curve and the signal it
    diagnoses share a step axis today, but reading a value off a grid by equality would
    break silently the first time that stopped being true.
    """
    if curve is None or step is None or curve.values.size == 0:
        return None
    index = int(np.searchsorted(curve.steps, step, side="right")) - 1
    return float(curve.values[index]) if index >= 0 else None


def find_runs(directory: str | Path) -> list[tuple[Path, RunMeta]]:
    """Every completed run in ``directory``, as ``(stem path, metadata)``.

    Discovery is by sidecar rather than by ``.npz``, so a run whose training died between
    writing the stream and writing the sidecar is skipped instead of being analysed as if
    it were complete.
    """
    directory = Path(directory)
    found = []
    for path in sorted(directory.glob(f"*{RUN_META_SUFFIX}")):
        stem = directory / path.name[: -len(RUN_META_SUFFIX)]
        if (stem.with_name(stem.name + STATE_STREAM_SUFFIX)).exists():
            found.append((stem, load_run_meta(path)))
    return found


def load_run(stem: Path) -> tuple[StateStream, EvalCurve, RunMeta]:
    """Load the three artefacts of the run at ``stem``."""
    return (
        load_state_stream(stem.with_name(stem.name + STATE_STREAM_SUFFIX)),
        load_eval_curve(stem.with_name(stem.name + EVAL_CURVE_SUFFIX)),
        load_run_meta(stem.with_name(stem.name + RUN_META_SUFFIX)),
    )


RunStarted = Callable[[Path, RunMeta], None]
RunFinished = Callable[[Path, RunMeta, list[dict[str, Any]], list[Curve]], None]


def analyse_directory(
    directory: str | Path,
    *,
    detectors: Sequence[DetectorSpec] | None = None,
    on_run_start: RunStarted | None = None,
    on_run: RunFinished | None = None,
    **curve_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, list[Curve]]]:
    """Analyse every run in ``directory``.

    Returns the tidy frame and the built curves keyed by run stem name --- the curves are
    what the figures are drawn from, and rebuilding them for plotting would repeat the
    expensive half of this function.

    Two callbacks rather than one, because the wait that needs explaining is *inside* a run,
    not between runs: a single 500k-step DoorKey replay is minutes on its own, so a callback
    that only fires on completion leaves exactly the silence it was meant to fill.
    ``on_run_start`` is called with ``(stem, meta)`` before the curves are built and
    ``on_run`` with ``(stem, meta, rows, curves)`` after.

    ``on_run`` receives the curves as well as the rows so that a caller can persist each
    run's archive as it lands. Everything is also accumulated and returned, but a batch that
    dies on the last run should not throw away the replays of the ones before it.
    """
    _reject_resolution_overrides(curve_kwargs)
    rows: list[dict[str, Any]] = []
    curves_by_run: dict[str, list[Curve]] = {}

    for stem, meta in find_runs(directory):
        if on_run_start is not None:
            on_run_start(stem, meta)
        stream, eval_curve, meta = load_run(stem)
        curves = build_curves(
            stream, window=meta.signal_window, stride=meta.signal_stride, **curve_kwargs
        )
        run_rows = analyse_run(stream, eval_curve, meta, curves=curves, detectors=detectors)
        curves_by_run[stem.name] = curves
        rows.extend(run_rows)
        if on_run is not None:
            on_run(stem, meta, run_rows, curves)

    return as_frame(rows), curves_by_run


def as_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Rows to frame, with the step-count columns held as nullable integers.

    Without this, one ``None`` from a frozen control or a run that never converged makes
    pandas widen the whole column to float, and ``t_conv`` reaches the report as
    ``20480.0``. The values are step counts; they should read as step counts.
    """
    frame = pd.DataFrame(rows)
    for column in NULLABLE_INT_COLUMNS:
        if column in frame:
            frame[column] = pd.array(frame[column], dtype="Int64")
    return frame


def save_curves(path: str | Path, curves: Sequence[Curve]) -> Path:
    """Write a run's curves to a single ``.npz`` so figures need no rebuild.

    The identifying settings are stored as a JSON manifest inside the archive rather than
    encoded in the array names. Labels are for humans reading a legend; recovering
    ``bins_per_dim`` by parsing one back would make the round trip depend on the label
    format never changing, and would lose ``clip``, which no label carries.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        CURVE_MANIFEST: np.array(json.dumps([curve.params for curve in curves]))
    }
    for index, curve in enumerate(curves):
        arrays[f"{index}__steps"] = curve.steps
        arrays[f"{index}__values"] = curve.values
    np.savez_compressed(path, **arrays)
    return path


def load_curves(path: str | Path) -> list[Curve]:
    """Read curves written by :func:`save_curves`."""
    with np.load(Path(path), allow_pickle=False) as data:
        manifest = json.loads(str(data[CURVE_MANIFEST]))
        return [
            Curve(steps=data[f"{index}__steps"], values=data[f"{index}__values"], **params)
            for index, params in enumerate(manifest)
        ]


def iter_status_counts(frame: pd.DataFrame) -> Iterator[tuple[str, str, int]]:
    """Yield ``(column, status, count)`` for each status column.

    The rates belong with any mean of ``Delta`` or ``retained`` computed from this frame:
    the statuses are exactly the rows such a mean silently drops, and they are not missing
    at random.
    """
    for column in ("conv_status", "plateau_status", "retained_status"):
        if column in frame:
            for status, count in frame[column].value_counts().items():
                yield column, str(status), int(count)
