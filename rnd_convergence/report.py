"""Aggregation: from the tidy signal frame to the numbers the report prints.

:mod:`~rnd_convergence.analysis` produces one row per ``(run, signal, detector setting)``
and deliberately keeps every sweep axis and every failure mode in the frame. This module is
the other half of that bargain: the frame is only honest if the layer that reduces it says
which slice it took and what it dropped. So three things are structural here rather than
incidental.

**One pinned slice, named in the caption.** ``Delta`` moves with ``tau``, with both smoothing
windows and with the bin count, so "the" Delta does not exist --- a table is a reading at one
setting. :class:`Pin` is that setting, it is a single object shared by the table and the
figures, and :meth:`Pin.caption` is what gets printed underneath them. The same pin is used
for every signal: the README pins detector settings per signal, and using one setting for
all of them is the strictest form of that, since a per-signal pin invites the reader to
wonder whether the winner was chosen by its tuning.

**Nothing is pooled across a sweep axis by accident.** :func:`select` filters the estimator
axes it pins and then *asserts* that one row per ``(run, signal)`` survives. Averaging two
bin counts, two clips or two analysis seeds into one "mean Delta" is the failure this guards:
it would not raise, it would not look wrong, and it would silently answer a different
question. Note the wrinkle that makes the filter non-obvious --- ``bins_per_dim`` is null on
an RND row, ``correction`` is null off the grid, ``analysis_seed`` is null wherever the
estimator is deterministic --- so each axis matches "this row's value, or null because the
setting does not apply to this estimator". Filtering on equality alone empties the RND column.

**Every mean carries the rate of the rows it is not computed over.** ``t_conv`` is undefined
on the sparse rung by design and ``t_plateau`` is undefined wherever a signal never
flattened; both are exactly the runs a mean drops, and they are not missing at random. Each
summary row therefore carries ``n_seeds`` beside ``n_delta``, and the rung's ``conv_status``
rates travel with the table rather than in a footnote.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from rnd_convergence.analysis import DetectorSpec
from rnd_convergence.runs import FROZEN_TAG

# The ladder, in the order the table's rows run --- the study's independent variable, so the
# row order is part of the result and not a presentation choice. ``dim`` is the state
# dimensionality the rung contributes to the axis; it is printed with the row because the
# hypothesis is a trend in it.
LADDER: tuple[tuple[str, str, int], ...] = (
    ("MarsRover", "MarsRover", 1),
    ("MiniGrid-Empty-5x5-v0", "MiniGrid-Empty", 3),
    ("MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-5x5", 3),
    ("MiniGrid-DoorKey-8x8-v0", "MiniGrid-DoorKey-8x8", 3),
    ("CartPole-v1", "CartPole", 4),
    ("LunarLander-v3", "LunarLander", 8),
)

# Signal columns, in the order they appear across the table. The three grid entries are one
# estimator at three bin counts rather than three estimators: the bin count is an independent
# variable of the study, so it is spread across columns instead of being averaged over.
RND = "RND"
SVE_GRID = "SVE grid b{bins}"
SVE_KNN = "SVE kNN"

# Statuses that mean "this run has a t_conv", i.e. the rows a Delta can exist on at all.
CONVERGED = "converged"
PLATEAU = "plateau"


@dataclass(frozen=True)
class Pin:
    """The one ``(detector, estimator)`` setting a table or figure is read at.

    Two halves. ``detector`` is the plateau/convergence configuration --- swept in the frame,
    fixed here. The rest pins the estimator axes that are also swept: curve ``mode``, the
    grid ``clip`` and bias ``correction``, and the ``analysis_seed`` of the estimators that
    depend on a random draw.

    ``analysis_seed`` deserves the note. It is null on every deterministic curve, so pinning
    it means "this seed where the estimator has one, and the curve itself where it does not"
    --- see :func:`select`. Pinning it matters because ``t_plateau`` for RND depends on which
    random target was drawn: a frame built with several analysis seeds holds several RND rows
    per run, and a mean over them would report the spread of the target draw as if it were
    the spread over agents.
    """

    detector: DetectorSpec = field(default_factory=DetectorSpec)
    mode: str = "sliding"
    clip: float = 5.0
    correction: str = "none"
    analysis_seed: int = 0

    @property
    def caption(self) -> str:
        """The settings, spelled out for the figure or table caption.

        Printed rather than implied: a reader who takes a number off this table is entitled
        to know it is one cell of a sweep, and which one.
        """
        detector = self.detector
        return (
            f"tau={detector.tau:g}, smooth_window={detector.smooth_window}, "
            f"reference_quantile={detector.reference_quantile:g}, "
            f"conv_smooth_window={detector.conv_smooth_window}, "
            f"mode={self.mode}, clip={self.clip:g}, correction={self.correction}, "
            f"analysis_seed={self.analysis_seed}"
        )


def signal_label(signal: str, bins_per_dim: float | None) -> str:
    """Display name for a signal column, from the estimator and its bin count.

    Built from the frame's own columns rather than from ``label``, which also carries the
    clip, the analysis seed and the mode --- settings that are pinned by :func:`select`
    rather than spread across columns, and that would otherwise split one column into
    several near-identical ones.
    """
    if signal == "sve_grid":
        if bins_per_dim is None or (isinstance(bins_per_dim, float) and np.isnan(bins_per_dim)):
            raise ValueError("a grid-SVE row must carry bins_per_dim")
        return SVE_GRID.format(bins=int(bins_per_dim))
    return {"rnd": RND, "sve_knn": SVE_KNN}.get(signal, signal)


def signal_order(frame: pd.DataFrame) -> list[str]:
    """The signal columns present, ordered RND, grid by ascending bin count, then kNN."""
    present = set(frame["signal_label"]) if "signal_label" in frame else set()
    grid = sorted(
        (label for label in present if label.startswith("SVE grid")),
        key=lambda label: int(label.rsplit("b", 1)[1]),
    )
    return [label for label in (RND, *grid, SVE_KNN) if label in present]


def ladder_order(frame: pd.DataFrame) -> list[str]:
    """Environment ids present, ordered down the ladder.

    Anything not on the ladder is appended rather than dropped: an unexpected ``env_id`` in
    the results directory is a thing to notice, and silently omitting its row is how a table
    comes to describe fewer runs than were made.
    """
    present = set(frame["env_id"])
    known = [env for env, _, _ in LADDER if env in present]
    return known + sorted(present - set(known))


def rung_display(env_id: str) -> tuple[str, int | None]:
    """``(short name, state dimensionality)`` for a rung, for axis and row labels."""
    for known, display, dim in LADDER:
        if known == env_id:
            return display, dim
    return env_id, None


def with_signal_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``frame`` carrying the display column the summaries group by."""
    labelled = frame.copy()
    labelled["signal_label"] = [
        signal_label(signal, bins)
        for signal, bins in zip(frame["signal"], frame["bins_per_dim"], strict=True)
    ]
    return labelled


def _matches(column: pd.Series, value: Any) -> pd.Series:
    """Rows whose ``column`` equals ``value`` *or* is null because it does not apply.

    The estimator axes are sparse by construction: ``bins_per_dim`` and ``clip`` exist only
    on grid rows, ``correction`` only there too, ``analysis_seed`` only where the estimator
    draws random numbers. A plain equality filter on any of them silently deletes every
    signal the setting is not defined for --- which is how a Delta table loses its RND
    column and looks merely sparse rather than broken.
    """
    return column.isna() | (column == value)


def _tag_filter(frame: pd.DataFrame, tag: str | None) -> pd.Series:
    """Rows belonging to run condition ``tag``, where ``None`` means the primary runs.

    ``tag`` holds two different kinds of thing, because ``scripts/train.py`` writes both
    through one field: an *experimental condition* set by ``run_tag``, and the marker the
    frozen-policy control is filed under so it cannot overwrite the trained run of the same
    seed. Only the first is a condition to select on --- the control is already identified by
    the ``frozen`` column, and treating its marker as a condition would make
    ``select(..., frozen=True)`` return nothing at all, which is a silent empty control in
    every figure that draws one.
    """
    return (
        frame["tag"].isna() | (frame["tag"] == FROZEN_TAG) if tag is None else frame["tag"] == tag
    )


def select(
    frame: pd.DataFrame,
    pin: Pin,
    *,
    frozen: bool | None = False,
    tag: str | None = None,
    require_unique: bool = True,
) -> pd.DataFrame:
    """The rows of one pinned slice: one per ``(run, signal)``.

    ``frozen`` selects the trained runs by default, ``True`` the control and ``None`` both
    --- the confound figure is the one caller that wants both. ``tag`` selects a run
    condition: ``None`` means the untagged primary runs, so a learning-rate sweep written
    alongside them under ``run_tag`` never leaks into the headline table.

    With ``require_unique`` the result is checked to hold exactly one row per
    ``(env, seed, tag, frozen, signal column)``. That check is the point of this function:
    every duplicate it would catch is a sweep axis that did not get pinned, and the
    consequence of missing one is a mean over two settings reported as a mean over seeds.
    """
    labelled = with_signal_labels(frame)
    detector = pin.detector
    keep = (
        (labelled["tau"] == detector.tau)
        & (labelled["smooth_window"] == detector.smooth_window)
        & (labelled["reference_quantile"] == detector.reference_quantile)
        & (labelled["conv_smooth_window"] == detector.conv_smooth_window)
        & (labelled["mode"] == pin.mode)
        & _matches(labelled["clip"], pin.clip)
        & _matches(labelled["correction"], pin.correction)
        & _matches(labelled["analysis_seed"], pin.analysis_seed)
        & _tag_filter(labelled, tag)
    )
    if frozen is not None:
        keep &= labelled["frozen"] == frozen
    selected = labelled[keep]

    if require_unique:
        key = ["env_id", "seed", "tag", "frozen", "signal_label"]
        duplicated = selected.duplicated(subset=key, keep=False)
        if duplicated.any():
            offending = selected.loc[duplicated, key].drop_duplicates()
            raise ValueError(
                f"{int(duplicated.sum())} rows share a (run, signal) key after pinning "
                f"{pin.caption}: {offending.to_dict('records')[:3]}. Some swept axis is not "
                "pinned, and aggregating this slice would average over it as if it were seeds"
            )
    return selected


def bootstrap_ci(
    values: Sequence[float],
    *,
    reps: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap CI of the mean over runs, or ``(nan, nan)`` below two values.

    The percentile bootstrap over runs --- resample the cell's runs with replacement, take
    the mean of each resample, read off the quantiles. With one task per cell this is exactly
    what ``rliable``'s stratified bootstrap computes, and it is computed here rather than
    called there for one reason: ``rliable`` forwards its ``random_state`` to ``arch``, which
    has deprecated that argument and ignores it, falling back to the global legacy RNG. The
    interval then moves between two runs of the report over the same frame --- silently, and
    by a few thousand steps on a four-seed cell. A committed table cannot have that, and
    seeding a global RNG from inside a library function to work around it is worse than
    drawing the resamples here from a generator this function owns. (That is also why
    ``rliable`` is not a dependency of this project: nothing else here would have used it.)

    A single seed gets ``nan`` rather than the zero-width interval a bootstrap over one
    value returns. That interval is not narrow, it is empty --- printing ``[21116, 21116]``
    next to a mean of one run states a precision the run cannot support, and the pilot table
    is mostly single-seed rungs.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, values.size, size=(reps, values.size))
    means = values[resamples].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def summarise_cells(
    frame: pd.DataFrame,
    *,
    by: Sequence[str],
    value: str = "delta",
    reps: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean of ``value`` per cell with its bootstrap CI, and what the mean is over.

    One row per group in ``by``. ``n_runs`` counts the runs in the cell and ``n_defined``
    the ones the mean is actually computed over; they differ exactly when a signal did not
    plateau or a run has no ``t_conv``, which is the information that keeps a cell of three
    surviving seeds from reading like a cell of ten.
    """
    rows = []
    for key, group in frame.groupby(list(by), dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        defined = group[value].dropna().to_numpy(dtype=np.float64)
        low, high = bootstrap_ci(defined, reps=reps, seed=seed)
        rows.append(
            {
                **dict(zip(by, key, strict=True)),
                "n_runs": int(len(group)),
                "n_defined": int(defined.size),
                "mean": float(np.mean(defined)) if defined.size else float("nan"),
                "ci_low": low,
                "ci_high": high,
                "plateau_rate": float((group["plateau_status"] == PLATEAU).mean()),
            }
        )
    return pd.DataFrame(rows)


# At or below this many distinct evaluation returns, `retained` stops being a noisy estimate
# and becomes arithmetic on a lattice: its numerator is one point of a curve with a handful of
# levels and its denominator a difference of two more. The threshold does no delicate work,
# because the measured distribution is bimodal --- 1 on MarsRover and DoorKey-8x8 (a constant
# greedy return, and a rung that never scores), up to 3 on MiniGrid-Empty, against 94 on
# CartPole and 100 on LunarLander. Any cut inside that gap gives the same answer. See
# `eval_distinct_returns` in analysis._row.
QUANTISED_EVAL_LEVELS = 3

# Occupancy at which the grid counts as saturated: essentially every sample owning its own
# cell, where the plug-in estimator returns log N identically and the curve stops being about
# exploration at all. Not exactly 1.0, because occupancy is a ratio of two counts that can sit
# a hair below it while the estimator is already pinned.
SATURATED_OCCUPANCY = 0.999


def delta_summary(frame: pd.DataFrame, pin: Pin, *, reps: int = 10_000) -> pd.DataFrame:
    """The Delta table as a tidy frame: one row per ``(rung, signal)``, ladder-ordered.

    Carries three quantities per cell, not one, because the README promises all three next to
    each other and each is meaningless alone. ``Delta`` is the headline. ``occupancy`` is what
    licenses it on the discretizing signals --- a grid-SVE plateau found where occupancy sits
    at 1.0 is arithmetic about sample counts, so the table making a claim about 8-D has to
    carry the number that says the estimator was not pinned at its ceiling. ``retained`` is
    the practical cost of stopping on the signal, and it comes with the flag that says where
    it is not interpretable at all.
    """
    selected = select(frame, pin)
    summary = summarise_cells(selected, by=("env_id", "signal_label"), value="delta", reps=reps)
    if summary.empty:
        # A pin the frame was never swept over. Returned as the empty frame the caller checks
        # for, rather than merged against two more empty frames that have no columns to join.
        return summary
    retained = summarise_cells(
        selected, by=("env_id", "signal_label"), value="retained", reps=reps
    ).rename(
        columns={
            "mean": "retained_mean",
            "ci_low": "retained_ci_low",
            "ci_high": "retained_ci_high",
            "n_defined": "n_retained",
        }
    )
    merged = summary.merge(
        retained[
            [
                "env_id",
                "signal_label",
                "retained_mean",
                "retained_ci_low",
                "retained_ci_high",
                "n_retained",
            ]
        ],
        on=["env_id", "signal_label"],
        how="left",
    )
    return _ordered(
        merged.merge(_cell_diagnostics(selected), on=["env_id", "signal_label"]), frame=selected
    )


def _cell_diagnostics(selected: pd.DataFrame) -> pd.DataFrame:
    """Per cell: occupancy where the plateau was found, and whether `retained` can mean anything.

    Occupancy is averaged only over the rows that have one --- it exists on the grid signals
    and only where a plateau was detected, so a cell's occupancy describes the seeds that fired
    rather than the whole cell, and ``n_occupancy`` says how many those were.
    """
    rows = []
    for (env, signal), group in selected.groupby(["env_id", "signal_label"], sort=False):
        occupancy = group["occupancy_at_plateau"].dropna().astype(float)
        # Where no seed plateaued there is no occupancy *at* a plateau, and those are exactly
        # the cells the saturation question is sharpest for: "grid SVE never fires at 8-D" and
        # "the 8-D grid is pinned at log N" are different claims, and only the second is a
        # defect of the estimator. The last window's occupancy is carried as the fallback so
        # that no discretizing cell is silent about the grid it was measured on.
        last = group["occupancy_last"].dropna().astype(float) if "occupancy_last" in group else []
        distinct = group.get("eval_distinct_returns")
        rows.append(
            {
                "env_id": env,
                "signal_label": signal,
                "occupancy_mean": float(occupancy.mean()) if len(occupancy) else float("nan"),
                "occupancy_max": float(occupancy.max()) if len(occupancy) else float("nan"),
                "n_occupancy": int(len(occupancy)),
                "occupancy_last_mean": float(np.mean(last)) if len(last) else float("nan"),
                # A rung, not a cell, property --- it describes the evaluation curve --- but it
                # travels per cell so the formatter never has to look a rung up somewhere else.
                "quantised_eval": (
                    bool(distinct.max() <= QUANTISED_EVAL_LEVELS)
                    if distinct is not None and distinct.notna().any()
                    else False
                ),
            }
        )
    return pd.DataFrame(rows)


def _ordered(summary: pd.DataFrame, *, frame: pd.DataFrame) -> pd.DataFrame:
    """Sort a summary down the ladder and across the signal columns."""
    if summary.empty:
        return summary
    envs = ladder_order(frame)
    signals = signal_order(frame)
    summary = summary.copy()
    summary["env_id"] = pd.Categorical(summary["env_id"], categories=envs, ordered=True)
    summary["signal_label"] = pd.Categorical(
        summary["signal_label"], categories=signals, ordered=True
    )
    return summary.sort_values(["env_id", "signal_label"]).reset_index(drop=True)


def conv_status_rates(frame: pd.DataFrame, pin: Pin) -> pd.DataFrame:
    """Per rung: how many of its runs have a ``t_conv``, and why the others do not.

    ``conv_status`` is a property of the run, not of the signal, so it is counted over runs
    --- the pinned slice repeats it once per signal column, and counting rows would multiply
    every rate by the number of signals.
    """
    selected = select(frame, pin)
    runs = selected.drop_duplicates(subset=["env_id", "seed", "tag", "frozen"])
    rows = []
    for env, group in runs.groupby("env_id", sort=False):
        counts = group["conv_status"].value_counts()
        rows.append(
            {
                "env_id": env,
                "n_runs": int(len(group)),
                "n_converged": int(counts.get(CONVERGED, 0)),
                "statuses": ", ".join(f"{status} {count}" for status, count in counts.items()),
            }
        )
    order = ladder_order(selected)
    frame_out = pd.DataFrame(rows)
    frame_out["env_id"] = pd.Categorical(frame_out["env_id"], categories=order, ordered=True)
    return frame_out.sort_values("env_id").reset_index(drop=True)


# The detector axes t_plateau actually depends on. conv_smooth_window is not among them: it
# smooths the *evaluation* curve, so the frame holds one copy of each plateau result per
# value of it. Counting rows without collapsing that would triple every denominator --- the
# fraction would come out the same, which is exactly why it is worth being explicit about.
PLATEAU_AXES: tuple[str, ...] = ("tau", "smooth_window", "reference_quantile")


def robustness(
    frame: pd.DataFrame,
    pin: Pin,
    *,
    tag: str | None = None,
    split_reference_quantile: bool = False,
) -> pd.DataFrame:
    """Fraction of detector configurations in which each signal plateaus at all, per rung.

    A result in its own right rather than a diagnostic: a signal that only fires under one
    hand-picked ``tau`` is not a usable stopping criterion however good its ``Delta`` looks
    at that ``tau``. The denominator is the detector sweep in the frame, deduplicated onto
    the axes ``plateau_time`` reads, times the seeds of the rung --- which is why ``n_seeds``
    is returned beside ``fired`` and belongs next to it wherever the fraction is shown. A
    rung measured at one seed and a rung measured at five are not the same claim.

    ``split_reference_quantile`` splits the denominator by that axis instead of pooling it,
    and the split is worth looking at: half the configurations use ``reference_quantile=1.0``,
    the running *maximum*, which the README keeps as a sensitivity check precisely because one
    transient spike relaxes the threshold for every later point and moves ``t_plateau``
    earlier. A pooled rate therefore blends a candidate detector with one the project has
    argued against, and the pooled number should not be quoted without that split to hand.

    The estimator axes stay pinned by ``pin`` --- this counts over detector settings, not
    over bin counts, since the bin count is a column of the table and its own axis of the
    hypothesis.
    """
    labelled = with_signal_labels(frame)
    keep = (
        (labelled["mode"] == pin.mode)
        & _matches(labelled["clip"], pin.clip)
        & _matches(labelled["correction"], pin.correction)
        & _matches(labelled["analysis_seed"], pin.analysis_seed)
        & (~labelled["frozen"])
        & _tag_filter(labelled, tag)
    )
    configs = labelled[keep].drop_duplicates(
        subset=["env_id", "seed", "signal_label", *PLATEAU_AXES]
    )
    group_keys = ["env_id", "signal_label"]
    if split_reference_quantile:
        group_keys.append("reference_quantile")
    rows = []
    for key, group in configs.groupby(group_keys, sort=False):
        rows.append(
            {
                **dict(zip(group_keys, key if isinstance(key, tuple) else (key,), strict=True)),
                "fired": float((group["plateau_status"] == PLATEAU).mean()),
                "n_configs": int(group.groupby(list(PLATEAU_AXES), sort=False).ngroups),
                "n_seeds": int(group["seed"].nunique()),
            }
        )
    return _ordered(pd.DataFrame(rows), frame=configs)


def confound_pin(
    frame: pd.DataFrame, pin: Pin, *, env_id: str, seed: int, signal: str = RND
) -> tuple[Pin, str]:
    """The setting to draw the trained-vs-frozen figure at, and why it is that one.

    The confound figure's whole claim is a *comparison*: the plateau lands in nearly the same
    place whether the policy is learning or frozen, so what the detector fires on is largely
    the predictor converging. A setting at which the control does not plateau cannot carry
    that claim --- it draws one ring and leaves the reader to assume the second detection was
    simply omitted. On CartPole the pinned setting is exactly such a case.

    So: keep the table's pin when both conditions plateau there, and otherwise take the
    nearest configuration in the sweep that does, "nearest" being the pinned
    ``reference_quantile`` first (the running maximum is a sensitivity check, not a candidate
    detector), then the smallest change in ``tau``, then in ``smooth_window``. The chosen
    setting is returned with a sentence naming it, which the figure prints: a figure drawn at
    a different setting from the table beside it has to say so, or it is the same silent
    mismatch in a new place.
    """
    labelled = with_signal_labels(frame)
    rows = labelled[
        (labelled["env_id"] == env_id)
        & (labelled["seed"] == seed)
        & (labelled["signal_label"] == signal)
        & (labelled["conv_smooth_window"] == pin.detector.conv_smooth_window)
        & (labelled["mode"] == pin.mode)
        & _matches(labelled["analysis_seed"], pin.analysis_seed)
    ]
    detector = pin.detector
    both = [
        (tau, smooth, quantile)
        for (tau, smooth, quantile), group in rows.groupby(list(PLATEAU_AXES), sort=False)
        if set(group.loc[group["plateau_status"] == PLATEAU, "frozen"]) >= {False, True}
    ]
    if (detector.tau, detector.smooth_window, detector.reference_quantile) in both:
        return pin, "the same setting as the table"
    if not both:
        return pin, (
            "the table's setting; no configuration in the sweep detects a plateau in both "
            "conditions, so the control's non-detection is marked on the panel"
        )
    tau, smooth, quantile = min(
        both,
        key=lambda candidate: (
            candidate[2] != detector.reference_quantile,
            abs(candidate[0] - detector.tau),
            abs(candidate[1] - detector.smooth_window),
            candidate,
        ),
    )
    chosen = Pin(
        detector=DetectorSpec(
            tau=tau,
            smooth_window=int(smooth),
            reference_quantile=quantile,
            conv_smooth_window=detector.conv_smooth_window,
        ),
        mode=pin.mode,
        clip=pin.clip,
        correction=pin.correction,
        analysis_seed=pin.analysis_seed,
    )
    return chosen, (
        f"not the table's setting (tau={detector.tau:g}, "
        f"smooth_window={detector.smooth_window}) but the nearest one in the sweep at which "
        "the detector fires on the control as well as on the trained run, which is what makes "
        "the two comparable"
    )


def delta_band(frame: pd.DataFrame, pin: Pin, *, tag: str | None = None) -> pd.DataFrame:
    """Every ``Delta`` a rung/signal cell takes across the detector sweep.

    The Delta table reports one cell per ``(rung, signal)`` at one pinned detector setting;
    this is the same cell measured at all of them, which is what the sensitivity figure
    draws. ``conv_smooth_window`` stays pinned so that the spread shown is the plateau
    detector's --- moving both at once would mix a sweep over when the *signal* flattened
    with a sweep over when the *policy* is judged to have converged.
    """
    labelled = with_signal_labels(frame)
    keep = (
        (labelled["mode"] == pin.mode)
        & _matches(labelled["clip"], pin.clip)
        & _matches(labelled["correction"], pin.correction)
        & _matches(labelled["analysis_seed"], pin.analysis_seed)
        & (labelled["conv_smooth_window"] == pin.detector.conv_smooth_window)
        & (~labelled["frozen"])
        & _tag_filter(labelled, tag)
        & labelled["delta"].notna()
    )
    columns = ["env_id", "seed", "signal_label", "delta", *PLATEAU_AXES]
    return _ordered(labelled.loc[keep, columns].copy(), frame=labelled[keep])


def tracking(frame: pd.DataFrame, pin: Pin, tags: Sequence[str]) -> pd.DataFrame:
    """Per signal: how strongly ``t_plateau`` moves with ``t_conv`` across run conditions.

    The condition means are what a table shows, but three means cannot separate "tracks
    convergence" from "sits at a fixed step" when the conditions themselves are only a few
    thousand steps apart. This pairs the two times *per run* and reports the rank correlation
    between them together with the slope of ``t_plateau`` on ``t_conv``: a signal that tracks
    convergence has a slope near 1, and a signal that fires at a fixed step regardless has a
    slope near 0 however robust its detector looks.

    Rank correlation rather than Pearson: the question is whether the plateau moves in the
    same *order* as convergence, and with a handful of runs one slow seed would otherwise set
    the answer. The slope is still least squares, because its units --- steps of plateau per
    step of convergence --- are what makes "near 1" or "near 0" mean anything.
    """
    labelled = pd.concat([select(frame, pin, tag=tag) for tag in tags], ignore_index=True)
    if labelled.empty:
        return pd.DataFrame()
    rows = [
        {"signal_label": signal, **_paired_stats(group)}
        for signal, group in labelled.groupby("signal_label", sort=False)
    ]
    return pd.DataFrame(rows)


def tracking_sweep(frame: pd.DataFrame, pin: Pin, tags: Sequence[str]) -> pd.DataFrame:
    """:func:`tracking`, repeated at every detector configuration in the frame.

    The reason this exists rather than one pinned answer: ``t_plateau`` is itself a function
    of the detector setting, so "does the plateau move with convergence" can be answered
    differently by different cells of the same sweep --- and a headline resting on the single
    most favourable one is the failure mode the whole study is built to avoid. Reading the
    slope across all of them separates "this signal tracks convergence" from "this signal
    tracks convergence at one setting out of eighteen".

    ``conv_smooth_window`` stays pinned, for the reason given in :func:`delta_band`: it moves
    ``t_conv``, which is the axis being correlated against.
    """
    labelled = pd.concat([select(frame, pin, tag=tag) for tag in tags], ignore_index=True)
    if labelled.empty:
        return pd.DataFrame()
    swept = with_signal_labels(frame)
    swept = swept[
        swept["tag"].isin(list(tags))
        & (~swept["frozen"])
        & (swept["mode"] == pin.mode)
        & _matches(swept["clip"], pin.clip)
        & _matches(swept["correction"], pin.correction)
        & _matches(swept["analysis_seed"], pin.analysis_seed)
        & (swept["conv_smooth_window"] == pin.detector.conv_smooth_window)
    ]
    rows = []
    for key, group in swept.groupby(["signal_label", *PLATEAU_AXES], sort=False):
        signal, tau, smooth_window, reference_quantile = key
        rows.append(
            {
                "signal_label": signal,
                "tau": tau,
                "smooth_window": smooth_window,
                "reference_quantile": reference_quantile,
                **_paired_stats(group),
            }
        )
    swept = pd.DataFrame(rows)
    # Ordered by bin count rather than lexically, so the summary line under the table reads
    # b5, b10, b20 like every other table in the document.
    swept["signal_label"] = pd.Categorical(
        swept["signal_label"], categories=signal_order(swept), ordered=True
    )
    return swept.sort_values(["signal_label", *PLATEAU_AXES]).reset_index(drop=True)


def _paired_stats(group: pd.DataFrame) -> dict[str, Any]:
    """Rank correlation and slope of ``t_plateau`` on ``t_conv`` over the runs of ``group``.

    Rank correlation rather than Pearson: the question is whether the plateau moves in the
    same *order* as convergence, and with a handful of runs one slow seed would otherwise set
    the answer. The slope is still least squares, because its units --- steps of plateau per
    step of convergence --- are what makes "near 1" or "near 0" mean anything.
    """
    paired = group[["t_conv", "t_plateau"]].dropna().astype(float)
    conv, plateau = paired["t_conv"].to_numpy(), paired["t_plateau"].to_numpy()
    # A constant t_conv and a constant t_plateau both leave the correlation undefined, and
    # they mean opposite things: the first is a lever that never moved, the second a signal
    # that never followed. n_pairs and the two ranges are what separate them.
    usable = paired.shape[0] >= 3 and np.ptp(conv) > 0 and np.ptp(plateau) > 0
    rho, p_value = stats.spearmanr(conv, plateau) if usable else (np.nan, np.nan)
    return {
        "n_pairs": int(paired.shape[0]),
        "n_runs": int(len(group)),
        "t_conv_range": float(np.ptp(conv)) if conv.size else float("nan"),
        "t_plateau_range": float(np.ptp(plateau)) if plateau.size else float("nan"),
        "spearman": float(rho) if usable else float("nan"),
        "p_value": float(p_value) if usable else float("nan"),
        "slope": float(np.polyfit(conv, plateau, 1)[0]) if usable else float("nan"),
    }


def discrimination_summary(frame: pd.DataFrame, pin: Pin, tags: Sequence[str]) -> pd.DataFrame:
    """``t_conv`` against ``t_plateau`` per run condition, for each signal.

    The experiment this serves: train one rung at learning rates that make ``t_conv``
    genuinely differ, and ask whether each signal's plateau *moves with* it. A signal that
    tracks convergence has a ``t_plateau`` that varies across the conditions the way
    ``t_conv`` does; a signal that is stably wrong reports the same step every time and gets
    its apparent robustness for free. Only ``Delta`` seen against both columns separates
    those two, which is why this table carries all three.
    """
    rows = []
    for tag in tags:
        selected = select(frame, pin, tag=tag)
        if selected.empty:
            continue
        runs = selected.drop_duplicates(subset=["env_id", "seed"])
        conv = runs["t_conv"].dropna().astype(float)
        for signal, group in selected.groupby("signal_label", sort=False):
            plateau = group["t_plateau"].dropna().astype(float)
            delta = group["delta"].dropna().astype(float)
            rows.append(
                {
                    "tag": tag,
                    "signal_label": signal,
                    "n_runs": int(len(group)),
                    "t_conv_mean": float(conv.mean()) if len(conv) else float("nan"),
                    "t_conv_std": float(conv.std(ddof=1)) if len(conv) > 1 else float("nan"),
                    "n_conv": int(len(conv)),
                    "t_plateau_mean": float(plateau.mean()) if len(plateau) else float("nan"),
                    "t_plateau_std": (
                        float(plateau.std(ddof=1)) if len(plateau) > 1 else float("nan")
                    ),
                    "n_plateau": int(len(plateau)),
                    "delta_mean": float(delta.mean()) if len(delta) else float("nan"),
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    # Ordered by bin count like every other table, not lexically: b10 before b5 reads as a
    # sort nobody chose, and the bin count is the axis the reader is scanning down.
    summary["signal_label"] = pd.Categorical(
        summary["signal_label"],
        categories=signal_order(summary),
        ordered=True,
    )
    return summary.sort_values(["tag", "signal_label"]).reset_index(drop=True)


def _format_cell(row: pd.Series) -> str:
    """One Delta cell: the mean, its interval, and everything needed to read it.

    Four facts, because each disarms a way of misreading the others. The mean and its
    interval; how many seeds plateaued at this setting (a mean over two of five seeds is a
    different claim from one over five); the occupancy of the grid where those plateaus were
    found, without which a grid-SVE number at 8-D cannot be told from ``log N``; and
    ``retained``, the cost of actually stopping there, suppressed where the evaluation curve
    is quantised and the ratio is arithmetic rather than measurement.

    A cell with no ``Delta`` still prints the last three. The cells that carry no Delta at 8-D
    are precisely the ones the saturation question is asked of, so a bare dash there would
    withhold the evidence exactly where it is load-bearing.
    """
    lines = []
    if row["n_defined"] == 0:
        lines.append("—")
    else:
        interval = (
            ""
            if np.isnan(row["ci_low"])
            else f" [{_steps(row['ci_low'])}, {_steps(row['ci_high'])}]"
        )
        lines.append(f"{_steps(row['mean'])}{interval} ({row['n_defined']}/{row['n_runs']})")
    lines.append(
        f"plateau in {int(round(row['plateau_rate'] * row['n_runs']))}/{row['n_runs']} seeds"
    )
    if row["n_occupancy"] > 0:
        occupancy = f"occ {row['occupancy_mean']:.3f}"
        if row["occupancy_max"] >= SATURATED_OCCUPANCY:
            # Saturated: the estimator is pinned at log N and the plateau is arithmetic.
            occupancy += " — SATURATED"
        lines.append(occupancy)
    elif not np.isnan(row["occupancy_last_mean"]):
        occupancy = f"occ {row['occupancy_last_mean']:.3f} (last window)"
        if row["occupancy_last_mean"] >= SATURATED_OCCUPANCY:
            occupancy += " — SATURATED"
        lines.append(occupancy)
    if row["n_retained"] > 0:
        retained = f"retained {row['retained_mean']:.2f} ({row['n_retained']}/{row['n_runs']})"
        lines.append("retained n/a" if row["quantised_eval"] else retained)
    return "<br>".join(lines)


def _steps(value: float) -> str:
    """A signed step count with thin thousands separators.

    Thin spaces rather than commas because these numbers sit inside comma-separated
    intervals; ``[+30,834, +36,797]`` has to be parsed before it can be read.
    """
    return f"{value:+,.0f}".replace(",", " ")


def format_delta_table(
    summary: pd.DataFrame,
    conv: pd.DataFrame,
    pin: Pin,
    *,
    title: str = "Signal lag Delta = t_plateau - t_conv, in environment steps",
) -> str:
    """The headline table as Markdown, with its caption and its status rates.

    ``t_conv``'s statuses sit in a row-level column rather than in every cell because they
    are a property of the run: on a rung where no seed converged, no cell in the row can
    hold a ``Delta``, and repeating that fact five times across the row would read as five
    independent failures instead of one.
    """
    if summary.empty:
        return f"### {title}\n\n(no rows at {pin.caption})\n"

    signals = [str(label) for label in summary["signal_label"].cat.categories]
    cells = summary.copy()
    cells["cell"] = cells.apply(_format_cell, axis=1)
    pivot = cells.pivot(index="env_id", columns="signal_label", values="cell")
    seeds = cells.groupby("env_id", observed=True)["n_runs"].max()
    conv = conv.set_index("env_id")

    header = ["Rung", "Dim", "Seeds", "t_conv"] + signals
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for env in pivot.index:
        display, dim = rung_display(str(env))
        statuses = conv.loc[env] if env in conv.index else None
        conv_cell = (
            f"{statuses['n_converged']}/{statuses['n_runs']}" if statuses is not None else "—"
        )
        row = [
            display,
            str(dim) if dim is not None else "—",
            str(int(seeds.get(env, 0))),
            conv_cell,
        ]
        for signal in signals:
            value = pivot.loc[env, signal]
            row.append("—" if not isinstance(value, str) else value)
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        f"Detector and estimator settings, identical for every signal: {pin.caption}.",
        "",
        "Each cell reads `mean Delta [95% CI] (seeds with a Delta / seeds run)`, then how many "
        "of the rung's seeds plateaued **at this setting** --- not to be confused with the "
        "robustness table below, which counts detector configurations across the sweep --- "
        "then, on the discretizing signals, the mean occupancy of the grid where those "
        "plateaus were found, and last `retained`, the fraction of final performance kept by "
        "stopping there. The interval is a percentile bootstrap of the mean over seeds "
        "(10 000 resamples, seeded) and is omitted below two seeds, since a bootstrap over one "
        "value returns a zero-width interval rather than a narrow one.",
        "",
        "`Seeds` is the number of runs behind the row and belongs with every number in it. "
        "`t_conv` counts the runs that reached the convergence criterion --- where none did, "
        "no cell in the row can hold a Delta, and the reason is in the status table below. "
        "Delta < 0 means the signal fires before the policy converged, so stopping on it "
        "costs return; Delta > 0 means it fires late and saves no compute. `occ` is "
        "`occupancy_at_plateau`: at 1.0 the grid is saturated, the estimator is pinned to "
        "`log N` whatever the policy does, and the plateau above it is arithmetic about "
        "sample counts rather than evidence about exploration --- such cells are marked "
        "SATURATED --- and where a cell has no plateau at all, the last window's occupancy "
        'is shown instead, since "the grid was saturated" and "the detector never fired" '
        "are different claims and only the first is a defect of the estimator. "
        "`retained n/a` marks a rung whose evaluation curve takes at most "
        f"{QUANTISED_EVAL_LEVELS} distinct values (`eval_distinct_returns`): there the ratio "
        "divides one point of a handful-of-levels curve by a difference of two more, which is "
        "an arithmetic artefact rather than a noisy estimate. The rungs split cleanly, 1-3 "
        "levels against 94-100.",
    ]
    return "\n".join(lines) + "\n"


def format_status_table(conv: pd.DataFrame) -> str:
    """Why ``t_conv`` is undefined where it is, per rung."""
    lines = [
        "### Convergence status per rung",
        "",
        "| Rung | Runs | Statuses |",
        "| --- | --- | --- |",
    ]
    for _, row in conv.iterrows():
        display, _ = rung_display(str(row["env_id"]))
        lines.append(f"| {display} | {row['n_runs']} | {row['statuses']} |")
    return "\n".join(lines) + "\n"


def format_robustness_table(fired: pd.DataFrame, by_quantile: pd.DataFrame | None = None) -> str:
    """Detector robustness as a table, so the figure has a readable companion."""
    if fired.empty:
        return "### Detector robustness\n\n(no rows)\n"
    signals = [str(label) for label in fired["signal_label"].cat.categories]
    pivot = fired.pivot(index="env_id", columns="signal_label", values="fired")

    lines = [
        "### Detector robustness: fraction of configurations in which the signal plateaus",
        "",
        "| Rung | Dim | Seeds | Configs | " + " | ".join(signals) + " |",
        "| " + " | ".join("---" for _ in range(4 + len(signals))) + " |",
    ]
    for env in pivot.index:
        display, dim = rung_display(str(env))
        cells = fired[fired["env_id"] == env]
        row = [
            display,
            str(dim) if dim is not None else "—",
            _one_or_range(cells["n_seeds"]),
            # Per rung rather than off the first signal's row: they should agree, and where
            # they do not the reader is looking at cells with different denominators.
            _one_or_range(cells["n_configs"]),
        ]
        row += [
            "—" if pd.isna(pivot.loc[env, signal]) else f"{pivot.loc[env, signal]:.2f}"
            for signal in signals
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "Fraction over the detector sweep (tau x smooth_window x reference_quantile) times "
        "the seeds of the rung; estimator settings pinned. 1.00 means the signal plateaued "
        "under every configuration tried. `Seeds` is the denominator's other half and is the "
        "reason these numbers are a trend rather than a measurement: a rung at one seed says "
        "much less than a rung at five.",
        "",
        "Note what half the denominator is. Nine of the eighteen configurations use "
        "`reference_quantile = 1.0`, the running maximum, which this project keeps as a "
        "sensitivity check rather than as a candidate detector --- one transient spike "
        "relaxes its threshold for every later point, and it fires systematically early. The "
        "split below separates the two so the pooled rate is never read as the rate of a "
        "detector anyone proposes using.",
    ]
    if by_quantile is not None and not by_quantile.empty:
        lines += ["", _quantile_split(by_quantile, signals)]
    return "\n".join(lines) + "\n"


def _quantile_split(by_quantile: pd.DataFrame, signals: Sequence[str]) -> str:
    """The robustness rates split by ``reference_quantile``, as a nested Markdown table."""
    lines = [
        "| Rung | reference_quantile | " + " | ".join(signals) + " |",
        "| " + " | ".join("---" for _ in range(2 + len(signals))) + " |",
    ]
    for env in by_quantile["env_id"].drop_duplicates():
        display, _ = rung_display(str(env))
        for quantile in sorted(by_quantile["reference_quantile"].unique()):
            cells = by_quantile[
                (by_quantile["env_id"] == env) & (by_quantile["reference_quantile"] == quantile)
            ].set_index("signal_label")["fired"]
            row = [display, f"{quantile:g} ({'median' if quantile < 1 else 'maximum'})"]
            row += [
                "—" if signal not in cells.index else f"{cells[signal]:.2f}" for signal in signals
            ]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _one_or_range(values: pd.Series) -> str:
    """``"18"`` where every cell agrees, ``"12-18"`` where they do not."""
    unique = sorted(set(int(value) for value in values))
    return str(unique[0]) if len(unique) == 1 else f"{unique[0]}-{unique[-1]}"


def format_discrimination_table(
    summary: pd.DataFrame,
    pin: Pin,
    tracked: pd.DataFrame | None = None,
    swept: pd.DataFrame | None = None,
) -> str:
    """``t_conv`` vs ``t_plateau`` across run conditions, as Markdown."""
    if summary.empty:
        return "### Discrimination\n\n(no tagged runs)\n"
    lines = [
        "### Does the plateau move with convergence?",
        "",
        "| Condition | Signal | t_conv (mean +/- sd) | t_plateau (mean +/- sd) | Delta | n |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary.iterrows():
        conv = _mean_sd(row["t_conv_mean"], row["t_conv_std"], row["n_conv"], row["n_runs"])
        plateau = _mean_sd(
            row["t_plateau_mean"], row["t_plateau_std"], row["n_plateau"], row["n_runs"]
        )
        delta = "—" if np.isnan(row["delta_mean"]) else _steps(row["delta_mean"])
        lines.append(
            f"| {row['tag']} | {row['signal_label']} | {conv} | {plateau} | {delta} "
            f"| {row['n_runs']} |"
        )
    if tracked is not None and not tracked.empty:
        lines += [
            "",
            "| Signal | pairs | t_conv range | t_plateau range | Spearman rho | slope |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for _, row in tracked.iterrows():
            rho = (
                "—"
                if np.isnan(row["spearman"])
                else f"{row['spearman']:+.2f} (p={row['p_value']:.2f})"
            )
            slope = "—" if np.isnan(row["slope"]) else f"{row['slope']:+.2f}"
            lines.append(
                f"| {row['signal_label']} | {row['n_pairs']}/{row['n_runs']} "
                f"| {_span(row['t_conv_range'])} | {_span(row['t_plateau_range'])} "
                f"| {rho} | {slope} |"
            )
        lines += [
            "",
            "The second table pairs the two times within each run rather than comparing "
            "condition means, because the conditions are only a few thousand steps apart and "
            "three means cannot tell a signal that tracks convergence from one that fires at "
            "a fixed step. Slope is steps of t_plateau per step of t_conv: near 1 the signal "
            "tracks, near 0 it is stably wrong. Both are undefined where fewer than three "
            "runs have both times, or where either time never moved.",
        ]
        if swept is not None and not swept.empty:
            lines += ["", _sweep_sentence(swept)]
    lines += ["", f"Settings: {pin.caption}."]
    return "\n".join(lines) + "\n"


def _span(value: float) -> str:
    """A range of step counts, or a dash where there were none to take a range over."""
    return "—" if np.isnan(value) else f"{value:,.0f}".replace(",", " ")


def _sweep_sentence(swept: pd.DataFrame) -> str:
    """One line summarising :func:`tracking_sweep`, which is the answer the table implies.

    The pinned row is one cell of eighteen, and picking the cell that answers most favourably
    is exactly the failure this study is built to avoid. So the sweep's own distribution ---
    how many configurations agree in sign, how many reach significance, and what the typical
    slope is --- is printed under the table rather than left in a CSV.
    """
    usable = swept[swept["slope"].notna()]
    if usable.empty:
        return (
            "Across the detector sweep no configuration has three runs with both times, so "
            "the pinned row is the only reading available."
        )
    parts = []
    for signal, group in usable.groupby("signal_label", sort=False, observed=True):
        parts.append(
            f"{signal}: rho > 0 in {int((group['spearman'] > 0).sum())}/{len(group)} "
            f"configurations, p < 0.05 in {int((group['p_value'] < 0.05).sum())}, median slope "
            f"{group['slope'].median():+.2f}"
        )
    return (
        "Across the whole detector sweep rather than at the pinned setting --- "
        + "; ".join(parts)
        + ". A signal whose slope is near zero at most settings is not tracking convergence "
        "at the one where it looks like it is. Full table in tracking_sweep.csv."
    )


def _mean_sd(mean: float, sd: float, defined: int, total: int) -> str:
    if np.isnan(mean):
        return f"— (0/{total})"
    text = f"{mean:,.0f}".replace(",", " ")
    if not np.isnan(sd):
        text += f" +/- {sd:,.0f}".replace(",", " ")
    return text if defined == total else f"{text} ({defined}/{total})"
