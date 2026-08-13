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
    """Stratified-bootstrap CI of the mean, or ``(nan, nan)`` below two values.

    The percentile bootstrap over runs --- resample the cell's runs with replacement, take
    the mean of each resample, read off the quantiles. With one task per cell this is exactly
    what ``rliable``'s stratified bootstrap computes, and it is computed here rather than
    called there for one reason: ``rliable`` forwards its ``random_state`` to ``arch``, which
    has deprecated that argument and ignores it, falling back to the global legacy RNG. The
    interval then moves between two runs of the report over the same frame --- silently, and
    by a few thousand steps on a four-seed cell. A committed table cannot have that, and
    seeding a global RNG from inside a library function to work around it is worse than
    drawing the resamples here from a generator this function owns.

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


def delta_summary(frame: pd.DataFrame, pin: Pin, *, reps: int = 10_000) -> pd.DataFrame:
    """The Delta table as a tidy frame: one row per ``(rung, signal)``, ladder-ordered."""
    selected = select(frame, pin)
    summary = summarise_cells(selected, by=("env_id", "signal_label"), value="delta", reps=reps)
    return _ordered(summary, frame=selected)


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


def robustness(frame: pd.DataFrame, pin: Pin, *, tag: str | None = None) -> pd.DataFrame:
    """Fraction of detector configurations in which each signal plateaus at all, per rung.

    A result in its own right rather than a diagnostic: a signal that only fires under one
    hand-picked ``tau`` is not a usable stopping criterion however good its ``Delta`` looks
    at that ``tau``. The denominator is the detector sweep in the frame, deduplicated onto
    the axes ``plateau_time`` reads, times the seeds of the rung.

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
    rows = []
    for (env, signal), group in configs.groupby(["env_id", "signal_label"], sort=False):
        rows.append(
            {
                "env_id": env,
                "signal_label": signal,
                "fired": float((group["plateau_status"] == PLATEAU).mean()),
                "n_configs": int(group.groupby(list(PLATEAU_AXES), sort=False).ngroups),
                "n_seeds": int(group["seed"].nunique()),
            }
        )
    return _ordered(pd.DataFrame(rows), frame=configs)


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
    return pd.DataFrame(rows).sort_values(["signal_label", *PLATEAU_AXES]).reset_index(drop=True)


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
    order = [label for label in (RND, *sorted(set(summary["signal_label"]) - {RND, SVE_KNN}))]
    order += [SVE_KNN] if SVE_KNN in set(summary["signal_label"]) else []
    summary["signal_label"] = pd.Categorical(summary["signal_label"], categories=order)
    return summary.sort_values(["tag", "signal_label"]).reset_index(drop=True)


def _format_cell(row: pd.Series) -> str:
    """One Delta cell: the mean, its interval, and what it was computed over."""
    if row["n_defined"] == 0:
        return "—"
    mean = _steps(row["mean"])
    interval = (
        "" if np.isnan(row["ci_low"]) else f" [{_steps(row['ci_low'])}, {_steps(row['ci_high'])}]"
    )
    return f"{mean}{interval} ({row['n_defined']}/{row['n_runs']})"


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
    plateau = cells.pivot(index="env_id", columns="signal_label", values="plateau_rate")
    conv = conv.set_index("env_id")

    header = ["Rung", "Dim", "t_conv"] + signals
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for env in pivot.index:
        display, dim = rung_display(str(env))
        statuses = conv.loc[env] if env in conv.index else None
        conv_cell = (
            f"{statuses['n_converged']}/{statuses['n_runs']}" if statuses is not None else "—"
        )
        row = [display, str(dim) if dim is not None else "—", conv_cell]
        for signal in signals:
            value = pivot.loc[env, signal]
            rate = plateau.loc[env, signal]
            row.append(
                "—" if not isinstance(value, str) else f"{value}<br>fired {rate:.0%} of seeds"
            )
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        f"Detector and estimator settings, identical for every signal: {pin.caption}.",
        "",
        "Cells are `mean Delta [95% bootstrap CI] (seeds with a Delta / seeds run)`; the CI is "
        "a stratified bootstrap over seeds (rliable) and is omitted below two seeds. `t_conv` "
        "counts the runs of the rung that reached the convergence criterion --- where none "
        "did, no cell in the row can hold a Delta, and the reason is in the status table "
        "below. Delta < 0 means the signal fires before the policy converged, so stopping on "
        "it costs return; Delta > 0 means it fires late and saves no compute.",
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


def format_robustness_table(fired: pd.DataFrame) -> str:
    """Detector robustness as a table, so the figure has a readable companion."""
    if fired.empty:
        return "### Detector robustness\n\n(no rows)\n"
    signals = [str(label) for label in fired["signal_label"].cat.categories]
    pivot = fired.pivot(index="env_id", columns="signal_label", values="fired")
    counts = fired.set_index(["env_id", "signal_label"])[["n_configs", "n_seeds"]]

    lines = [
        "### Detector robustness: fraction of configurations in which the signal plateaus",
        "",
        "| Rung | Dim | Configs | " + " | ".join(signals) + " |",
        "| " + " | ".join("---" for _ in range(3 + len(signals))) + " |",
    ]
    for env in pivot.index:
        display, dim = rung_display(str(env))
        first = counts.loc[env].iloc[0]
        row = [display, str(dim) if dim is not None else "—", f"{first['n_configs']}"]
        row += [
            "—" if pd.isna(pivot.loc[env, signal]) else f"{pivot.loc[env, signal]:.2f}"
            for signal in signals
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "Fraction over the detector sweep (tau x smooth_window x reference_quantile) times "
        "the seeds of the rung; estimator settings pinned. 1.00 means the signal plateaued "
        "under every configuration tried.",
    ]
    return "\n".join(lines) + "\n"


def format_discrimination_table(
    summary: pd.DataFrame, pin: Pin, tracked: pd.DataFrame | None = None
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
            conv_range = _mean_sd(row["t_conv_range"], float("nan"), 1, 1)
            plateau_range = _mean_sd(row["t_plateau_range"], float("nan"), 1, 1)
            lines.append(
                f"| {row['signal_label']} | {row['n_pairs']}/{row['n_runs']} | {conv_range} "
                f"| {plateau_range} | {rho} | {slope} |"
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
    lines += ["", f"Settings: {pin.caption}."]
    return "\n".join(lines) + "\n"


def _mean_sd(mean: float, sd: float, defined: int, total: int) -> str:
    if np.isnan(mean):
        return f"— (0/{total})"
    text = f"{mean:,.0f}".replace(",", " ")
    if not np.isnan(sd):
        text += f" +/- {sd:,.0f}".replace(",", " ")
    return text if defined == total else f"{text} ({defined}/{total})"
