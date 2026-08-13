"""The figures the report shows, drawn from the committed frame and curve archives.

Nothing here re-measures a run. The curves come from the archives
:mod:`~rnd_convergence.analysis` wrote and the annotations from the frame, so a figure and
the table beside it are guaranteed to be the same measurement --- redrawing from a fresh
replay would eventually disagree with the table for reasons no reader could see.

Three conventions hold across every figure, because the figures are read together.

**Colour identifies the estimator, never the condition.** RND is one hue, grid SVE is one
hue in three ordered steps (the bin count is ordered, so the encoding is too: lighter is
coarser) and kNN is a third. Trained-versus-frozen is line style, so the confound figure can
sit next to the Delta table without the RND curve changing colour between them.

**A signal that did not fire is drawn as absent, not as zero.** Where a signal never
plateaued there is no Delta, and a bar of height zero would read as "fired at exactly
t_conv" --- the opposite of the finding. Those cells are left empty and their count is on
the axis.

**Delta is drawn on a symmetric log axis.** It spans -480 to +218 000 across the sweep, and
on a linear axis every rung except the widest collapses onto the zero line; the sign is the
thing being read, so the axis has to keep zero and both signs legible at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # figures are written to files; there is never a display here.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rnd_convergence.analysis import Curve, load_curves  # noqa: E402
from rnd_convergence.report import (  # noqa: E402
    RND,
    Pin,
    ladder_order,
    rung_display,
    select,
    signal_label,
    signal_order,
)
from rnd_convergence.streams import run_stem  # noqa: E402

# Categorical hues for the estimator families, plus an ordered three-step ramp of the second
# hue for the grid bin counts. Validated as a set for colour-vision deficiency: the three
# families clear the all-pairs separation floor, and the ramp is monotone in lightness with
# every step clear of the surface. Bin count is also encoded by marker and by the label, so
# no reading depends on telling two steps of one hue apart.
COLORS: dict[str, str] = {
    RND: "#2a78d6",
    "SVE grid b5": "#f08a5c",
    "SVE grid b10": "#e35a22",
    "SVE grid b20": "#96350f",
    "SVE kNN": "#1baf7a",
}
FALLBACK_COLOR = "#6b6a66"
MARKERS: dict[str, str] = {
    RND: "o",
    "SVE grid b5": "^",
    "SVE grid b10": "s",
    "SVE grid b20": "D",
    "SVE kNN": "v",
}

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dcdbd6"

# Below this many steps Delta is drawn linearly, above it logarithmically. One evaluation
# interval on the fastest rung: differences smaller than that are not resolvable by the
# measurement, so compressing them is not hiding anything.
SYMLOG_THRESHOLD = 1_000


def _style() -> dict[str, Any]:
    """Recessive axes, thin marks, text in ink rather than in the series colour."""
    return {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
    }


def color_for(label: str) -> str:
    return COLORS.get(label, FALLBACK_COLOR)


def marker_for(label: str) -> str:
    return MARKERS.get(label, "o")


def _thousands(value: float, _pos: int = 0) -> str:
    """Axis tick text with thin thousands separators, and a sign where one is meaningful."""
    return f"{value:+,.0f}".replace(",", " ") if value else "0"


def figure_robustness(fired: pd.DataFrame, path: Path, *, pin: Pin) -> Path:
    """Grouped bars: in what fraction of detector configurations does each signal plateau?

    The headline robustness result. Ordered down the ladder along x so the trend the study
    predicts --- SVE degrading with dimensionality and with bin count, RND not --- is read
    left to right, and each bar is labelled with its own value so the comparison never
    depends on judging bar heights across groups.
    """
    signals = [str(label) for label in fired["signal_label"].cat.categories]
    envs = [str(env) for env in fired["env_id"].cat.categories if env in set(fired["env_id"])]
    values = fired.set_index(["env_id", "signal_label"])["fired"]

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(1.55 * len(envs) + 2.4, 4.0))
        width = 0.8 / max(len(signals), 1)
        for index, signal in enumerate(signals):
            offsets = np.arange(len(envs)) + (index - (len(signals) - 1) / 2) * width
            heights = [float(values.get((env, signal), np.nan)) for env in envs]
            ax.bar(
                offsets,
                heights,
                width=width * 0.86,
                color=color_for(signal),
                label=signal,
                zorder=3,
            )
            for offset, height in zip(offsets, heights, strict=True):
                if not np.isnan(height):
                    ax.text(
                        offset,
                        height + 0.02,
                        f"{height:.2f}".lstrip("0"),
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        color=INK,
                    )

        ax.set_xticks(np.arange(len(envs)))
        ax.set_xticklabels([_rung_tick(env) for env in envs])
        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("fraction of detector configurations\nin which the signal plateaus")
        ax.set_xlabel("rung of the ladder (state dimensionality)")
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
        ax.legend(ncols=len(signals), loc="upper center", bbox_to_anchor=(0.5, 1.13))
        ax.set_title("Robustness of the plateau detector, by signal and rung", loc="left", pad=28)
        _caption(
            fig,
            ax,
            "Fraction over the detector sweep (tau x smooth_window x reference_quantile) and "
            f"the seeds of each rung. Estimator settings: mode={pin.mode}, clip={pin.clip:g}, "
            f"correction={pin.correction}, analysis_seed={pin.analysis_seed}.",
        )
        return _save(fig, path)


def _rung_tick(env_id: str) -> str:
    display, dim = rung_display(env_id)
    return display if dim is None else f"{display}\n({dim}-D)"


def figure_delta_sensitivity(
    band: pd.DataFrame, path: Path, *, pin: Pin, omitted: Sequence[str] = ()
) -> Path:
    """The spread of Delta across the detector sweep, per rung and signal.

    One row per ``(rung, signal)``: the full range as a rule, the median as a marker, and one
    faint dot per ``(seed, detector configuration)``. A single number per cell would not
    survive contact with this spread --- on CartPole the RND cell alone runs from a few
    hundred steps early to tens of thousands late --- so the table's pinned value is drawn on
    top of the band it was taken from rather than instead of it.

    ``omitted`` names rungs that produced no ``Delta`` at all --- a rung whose runs never
    converged has nothing to plot here, and a figure that simply lacked its row would read as
    a rung that was never run.
    """
    envs = [str(env) for env in ladder_order(band)]
    rows: list[tuple[str, str]] = []
    for env in envs:
        for signal in signal_order(band[band["env_id"] == env]):
            rows.append((env, signal))

    jitter = np.random.default_rng(0)
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(rows) + 2.2))
        seen: set[str] = set()
        for position, (env, signal) in enumerate(rows):
            values = band.loc[
                (band["env_id"] == env) & (band["signal_label"] == signal), "delta"
            ].to_numpy(dtype=float)
            if values.size == 0:
                continue
            color = color_for(signal)
            y = len(rows) - 1 - position
            ax.hlines(y, values.min(), values.max(), color=color, linewidth=2.0, zorder=3)
            ax.scatter(
                values,
                y + jitter.uniform(-0.16, 0.16, size=values.size),
                s=7,
                color=color,
                alpha=0.35,
                linewidths=0,
                zorder=4,
            )
            ax.scatter(
                np.median(values),
                y,
                s=42,
                color=color,
                marker=marker_for(signal),
                edgecolors="white",
                linewidths=0.9,
                zorder=5,
                label=signal if signal not in seen else None,
            )
            seen.add(signal)

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([signal for _, signal in reversed(rows)], fontsize=8)
        ax.axvline(0, color=MUTED, linewidth=1.0, zorder=2)
        ax.set_xscale("symlog", linthresh=SYMLOG_THRESHOLD)
        # Ticks at the decades and at zero only. Symlog's default puts a tick at each end of
        # the linear region as well, which lands +/-1000 on top of the zero label.
        widest = float(np.abs(band["delta"]).max())
        top = max(4, int(np.ceil(np.log10(widest))) if widest > 0 else 4)
        decades = [10**power for power in range(3, top + 1)]
        ax.set_xticks([-value for value in reversed(decades)] + [0] + decades)
        ax.xaxis.set_major_formatter(_thousands)
        ax.set_xlabel("Delta = t_plateau - t_conv  (environment steps, symmetric log)")
        ax.set_ylim(-0.8, len(rows) - 0.2)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)

        # Rung labels down the right-hand side, with a rule between rungs: the y axis already
        # carries the signal names, and repeating the rung on every row would triple the text.
        boundaries = [
            index for index in range(1, len(rows)) if rows[index][0] != rows[index - 1][0]
        ]
        for index in boundaries:
            ax.axhline(len(rows) - 0.5 - index, color=GRID, linewidth=0.8, zorder=1)
        for env in envs:
            positions = [i for i, (candidate, _) in enumerate(rows) if candidate == env]
            if not positions:
                continue
            middle = len(rows) - 1 - (positions[0] + positions[-1]) / 2
            display, dim = rung_display(env)
            ax.text(
                1.01,
                middle,
                display if dim is None else f"{display}\n{dim}-D",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=8,
                color=INK,
            )

        ax.set_title("How far Delta moves when the detector setting moves", loc="left", pad=8)
        missing = (
            ""
            if not omitted
            else " No Delta is defined on "
            + ", ".join(rung_display(env)[0] for env in omitted)
            + (", so it has no row here" if len(omitted) == 1 else ", so they have no row here")
            + "; see the status table."
        )
        _caption(
            fig,
            ax,
            "Rule: full range over the detector sweep and the seeds of the rung. Marker: "
            "median. Faint dots: one per (seed, detector configuration). Configurations in "
            "which the signal never plateaued contribute no point, so a short rule can mean a "
            "stable Delta or a signal that fired only once --- read it with the robustness "
            f"figure.{missing} conv_smooth_window={pin.detector.conv_smooth_window}, "
            f"mode={pin.mode}, clip={pin.clip:g}, correction={pin.correction}.",
        )
        return _save(fig, path)


def find_curve(
    curves: list[Curve], *, signal: str, pin: Pin, bins_per_dim: int | None = None
) -> Curve | None:
    """The one curve in an archive matching a signal and the pinned estimator settings.

    Returns ``None`` rather than raising when the archive has no such curve: a rung analysed
    without a bin count the figure asks for is a reason to draw a smaller figure, not to fail
    the report. Raises if it matches more than one, since that means the pin is incomplete
    and picking the first would silently pick an arbitrary setting.
    """
    matched = [
        curve
        for curve in curves
        if curve.signal == signal
        and curve.mode == pin.mode
        and (bins_per_dim is None or curve.bins_per_dim == bins_per_dim)
        and (curve.clip is None or curve.clip == pin.clip)
        and (curve.correction in (None, pin.correction))
        and (curve.analysis_seed in (None, pin.analysis_seed))
    ]
    if len(matched) > 1:
        raise ValueError(
            f"{len(matched)} curves match signal={signal!r} at {pin.caption}: "
            f"{[curve.label for curve in matched]}. The pin does not identify one curve"
        )
    return matched[0] if matched else None


def figure_confound(
    frame: pd.DataFrame,
    curves_dir: Path,
    path: Path,
    *,
    pin: Pin,
    env_id: str,
    seed: int = 0,
    bins_per_dim: int = 10,
) -> Path | None:
    """Trained against frozen, for both signals, on one rung.

    The control's whole purpose in one picture. The RND predictor takes a gradient step for
    every state it sees whether or not the agent found anything, so its error decays under a
    policy that never learns --- and if the decay looks the same in both conditions, the
    plateau the study detects is the predictor converging rather than novelty running out.
    Grid SVE is the contrast: it moves only when the policy moves, so its frozen curve is
    flat. Drawn as two panels because the two signals differ by orders of magnitude and
    sharing one axis would make the smaller of them a flat line by construction.

    Returns ``None`` when either condition is missing from the archives, since the figure's
    claim is a comparison and half of it is not worth drawing.
    """
    trained = _load_archive(curves_dir, env_id, seed, tag=None)
    frozen = _load_archive(curves_dir, env_id, seed, tag="frozen")
    if trained is None or frozen is None:
        return None

    panels = [
        (RND, "rnd", None),
        (signal_label("sve_grid", bins_per_dim), "sve_grid", bins_per_dim),
    ]
    marks = _plateau_marks(frame, pin, env_id=env_id, seed=seed)
    display, dim = rung_display(env_id)

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(1, len(panels), figsize=(9.4, 3.9))
        drew = False
        for ax, (label, signal, bins) in zip(np.atleast_1d(axes), panels, strict=True):
            color = color_for(label)
            for condition, archive, style in (
                ("trained", trained, "-"),
                ("frozen", frozen, (0, (5, 2))),
            ):
                curve = find_curve(archive, signal=signal, pin=pin, bins_per_dim=bins)
                if curve is None:
                    continue
                drew = True
                ax.plot(
                    curve.steps,
                    curve.values,
                    linestyle=style,
                    color=color,
                    label=condition,
                    zorder=3,
                )
                step = marks.get((condition, label))
                if step is not None:
                    index = int(np.searchsorted(curve.steps, step, side="right")) - 1
                    if index >= 0:
                        ax.scatter(
                            curve.steps[index],
                            curve.values[index],
                            s=52,
                            facecolor="white",
                            edgecolor=color,
                            linewidths=1.6,
                            zorder=5,
                        )
            t_conv = marks.get(("t_conv", None))
            if t_conv is not None:
                ax.axvline(t_conv, color=MUTED, linewidth=1.0, linestyle=(0, (1, 2)), zorder=2)
                # Labelled at the foot of the line rather than at its head: both signals are
                # near their maximum this early in the run, so the top of the panel is where
                # the curves are and the bottom is empty.
                ax.text(
                    t_conv,
                    0.02,
                    " t_conv",
                    transform=ax.get_xaxis_transform(),
                    fontsize=7,
                    color=MUTED,
                    va="bottom",
                )
            if signal == "rnd":
                ax.set_yscale("log")
            ax.set_title(label, loc="left", color=INK)
            ax.set_xlabel("environment steps")
            ax.set_ylabel("prediction error" if signal == "rnd" else "entropy (nats)")
            ax.xaxis.set_major_formatter(lambda value, _: f"{value / 1000:g}k")
            ax.set_axisbelow(True)
            ax.legend(loc="best")

        if not drew:
            plt.close(fig)
            return None
        fig.suptitle(
            f"Trained against frozen policy on {display}"
            + (f" ({dim}-D)" if dim is not None else ""),
            x=0.0,
            ha="left",
            fontsize=11,
            color=INK,
        )
        _caption(
            fig,
            axes,
            "Solid: the trained agent. Dashed: the frozen-policy control, identical in every "
            "respect except that no update is applied. Rings mark where the plateau detector "
            f"fires ({pin.caption}). RND error is on a log axis. Where the two RND curves "
            "decay alike, the detected plateau is the predictor converging rather than "
            "novelty being exhausted; grid SVE moves only when the policy does.",
        )
        return _save(fig, path)


def _load_archive(
    curves_dir: Path, env_id: str, seed: int, *, tag: str | None
) -> list[Curve] | None:
    path = Path(curves_dir) / f"{run_stem(env_id, seed, tag)}.npz"
    return load_curves(path) if path.exists() else None


def _plateau_marks(
    frame: pd.DataFrame, pin: Pin, *, env_id: str, seed: int
) -> dict[tuple[str, str | None], float]:
    """``t_plateau`` per (condition, signal) and the trained run's ``t_conv``, from the frame.

    Read from the frame rather than recomputed so that the rings on the curves are the same
    detections the table reports; a figure that ran its own detector would eventually
    disagree with the table by a setting nobody could see.
    """
    marks: dict[tuple[str, str | None], float] = {}
    for condition, is_frozen in (("trained", False), ("frozen", True)):
        rows = select(frame, pin, frozen=is_frozen)
        rows = rows[(rows["env_id"] == env_id) & (rows["seed"] == seed)]
        for _, row in rows.iterrows():
            if pd.notna(row["t_plateau"]):
                marks[(condition, row["signal_label"])] = float(row["t_plateau"])
            if not is_frozen and pd.notna(row["t_conv"]):
                marks[("t_conv", None)] = float(row["t_conv"])
    return marks


def _caption(fig: plt.Figure, axes: Any, text: str, *, gap: float = 0.05) -> None:
    """Put the caption below everything the axes already occupy.

    Placed by measuring rather than by a guessed offset: the tick labels and the axis label
    are laid out at draw time and their height depends on the figure size, so a fixed y in
    figure coordinates collides with the x label on exactly the figures that need the longest
    caption. ``get_tightbbox`` is what ``bbox_inches="tight"`` itself uses, so this asks the
    same question the save does.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottom = min(ax.get_tightbbox(renderer).y0 for ax in np.atleast_1d(axes))
    y = fig.transFigure.inverted().transform((0.0, bottom))[1] - gap
    fig.text(0.0, y, text, fontsize=7, color=MUTED, va="top", wrap=True)


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
