"""Analysis entry point: a directory of runs in, the tidy results frame out.

    python scripts/analyze.py                          # results/ -> docs/results/
    python scripts/analyze.py --results /scratch/runs --out docs/results
    python scripts/analyze.py --modes sliding cumulative --quick
    python scripts/analyze.py --seeds 0 1 2            # how much of Delta is the RND draw

Writes two things to ``--out``:

    signals.csv              one row per (run, signal, detector setting)
    curves/<stem>.npz        every measured curve, for the figures to draw from

Curves are saved because building them is the expensive half --- an RND replay plus a kNN
estimate per window --- while sweeping detector settings over them is nearly free. Plotting
later should read these rather than re-measure the streams. Each archive is written as its
run finishes, so a batch that dies partway leaves the runs before it on disk.

This is argparse rather than Hydra, unlike ``scripts/train.py``. The Hydra configs describe
an *experiment to run*, one composition per launch, and ``chdir: true`` moves the working
directory out from under relative paths. This script reads whatever runs happen to exist
and writes one artefact set for all of them, so there is nothing to compose.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rnd_convergence.analysis import (
    DEFAULT_ANALYSIS_SEEDS,
    DEFAULT_BIN_COUNTS,
    DEFAULT_CLIPS,
    DEFAULT_CORRECTIONS,
    Curve,
    DetectorSpec,
    analyse_directory,
    detector_grid,
    find_runs,
    iter_status_counts,
    save_curves,
)
from rnd_convergence.runs import RunMeta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results", type=Path, default=Path("results"), help="directory of training runs"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/results"), help="where to write the frame"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["sliding"],
        choices=["sliding", "cumulative"],
        help="SVE curve modes to measure (RND is sliding-only by construction)",
    )
    parser.add_argument(
        "--bins", nargs="+", type=int, default=list(DEFAULT_BIN_COUNTS), help="grid bin counts"
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        type=float,
        default=list(DEFAULT_CLIPS),
        help="standardised-observation clip bounds to lay the grid over",
    )
    parser.add_argument(
        "--corrections",
        nargs="+",
        default=list(DEFAULT_CORRECTIONS),
        choices=["none", "miller_madow"],
        help="grid-entropy bias corrections to measure (the README's sensitivity check)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_ANALYSIS_SEEDS),
        help="analysis seeds; sweeps the RND target draw and the kNN subsample, one replay each",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="a single detector setting instead of the full sweep, for a fast look at a pilot",
    )
    return parser.parse_args(argv)


# The conv_smooth_window the digest prefers, and the one --quick pins. Not DetectorSpec's
# default of 1: the MiniGrid rungs return an all-or-nothing greedy score, so an unsmoothed
# eval curve puts t_conv wherever the last few coin flips landed and makes `retained` a
# single-evaluation reading. The pilot frame shows what that costs -- retained ranges over
# [-1.26, 4.39] at conv_smooth_window=1, which is noise, not non-monotonicity. The full
# sweep still carries 1; it just is not what the console offers as the headline.
DIGEST_CONV_SMOOTH = 3


def digest_setting(frame: pd.DataFrame) -> dict[str, float | int]:
    """The one detector setting the console digest is read at.

    Three of the four come from :class:`DetectorSpec`'s own defaults rather than from the
    data, so that the digest keeps meaning "the default setting" under a custom sweep
    instead of silently re-defining itself as whatever happened to be swept.
    """
    default = DetectorSpec()
    smoothed = sorted(w for w in frame.conv_smooth_window.unique() if w > 1)
    return {
        "tau": default.tau,
        "smooth_window": default.smooth_window,
        "reference_quantile": default.reference_quantile,
        "conv_smooth_window": smoothed[0] if smoothed else frame.conv_smooth_window.min(),
    }


def summarise(frame: pd.DataFrame) -> str:
    """A per-run, per-signal digest of the frame, at one detector setting.

    Printed because the first question asked of a pilot is whether each signal plateaued at
    all, and scrolling a 400-row CSV to find out defeats the purpose of running one. The
    setting is named in the header: this is one slice of a sweep, and a reader who takes the
    printed `Delta` for *the* answer should at least be told which cell it came from.
    """
    if frame.empty:
        return "no rows"
    setting = digest_setting(frame)
    selected = frame
    for column, value in setting.items():
        selected = selected[selected[column] == value]

    header = ", ".join(f"{column}={value}" for column, value in setting.items())
    if selected.empty:
        return f"no rows at {header}; showing the whole frame\n" + _table(frame)
    return f"at {header}:\n" + _table(selected)


def _table(frame: pd.DataFrame) -> str:
    columns = ["env_id", "seed", "frozen", "label", "n_points", "t_conv", "conv_status"]
    columns += ["t_plateau", "plateau_status", "delta", "retained", "occupancy_max"]
    return frame[columns].to_string(index=False, max_colwidth=24)


def report_orphan_archives(out: Path, curves_by_run: dict[str, list[Curve]]) -> list[str]:
    """Name any curve archive in ``out`` that this run did not write, and return them.

    Archives are replaced per stem and never cleaned, so re-running against a smaller
    ``--results`` leaves the previous batch's archives sitting next to a frame that does not
    describe them. They are not deleted: the state streams they were measured from are not
    in the repository, so a stale archive may be the only surviving copy of that curve. It
    is said out loud instead, because the failure it causes -- figures drawn from curves the
    frame has no rows for -- is otherwise silent.
    """
    orphans = sorted(p.name for p in (out / "curves").glob("*.npz") if p.stem not in curves_by_run)
    if orphans:
        print(
            f"\nwarning: {len(orphans)} archive(s) in {out / 'curves'} are not in this frame "
            f"and were left untouched: {', '.join(orphans)}"
        )
    return orphans


def quick_grid() -> list[DetectorSpec]:
    """One detector setting: the digest's, so --quick prints the frame it just built."""
    default = DetectorSpec()
    return detector_grid(
        taus=(default.tau,),
        smooth_windows=(default.smooth_window,),
        reference_quantiles=(default.reference_quantile,),
        conv_smooth_windows=(DIGEST_CONV_SMOOTH,),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = find_runs(args.results)
    if not runs:
        print(f"no runs found in {args.results} (looking for *.run.json with a state stream)")
        return 1

    print(f"analysing {len(runs)} run(s) from {args.results}")
    detectors = quick_grid() if args.quick else detector_grid()
    args.out.mkdir(parents=True, exist_ok=True)

    def starting(stem: Path, meta: RunMeta) -> None:
        # Before the build, not after: the replay is where the minutes go, and a line that
        # only appears once it is over is a line that appears once the waiting is done.
        print(f"  {stem.name}: replaying {meta.n_states} states ...", flush=True)

    def finished(stem: Path, meta: RunMeta, rows: list[dict], curves: list[Curve]) -> None:
        # Written per run rather than after the batch so that a failure on the last run
        # does not discard the replays of every run before it.
        path = save_curves(args.out / "curves" / f"{stem.name}.npz", curves)
        print(f"    {len(rows)} rows, {len(curves)} curves -> {path.name}", flush=True)

    frame, curves_by_run = analyse_directory(
        args.results,
        detectors=detectors,
        modes=args.modes,
        bin_counts=args.bins,
        clips=args.clips,
        corrections=args.corrections,
        seeds=args.seeds,
        on_run_start=starting,
        on_run=finished,
    )

    frame_path = args.out / "signals.csv"
    frame.to_csv(frame_path, index=False)
    report_orphan_archives(args.out, curves_by_run)

    print(f"\n{summarise(frame)}\n")
    # The statuses are exactly the rows a mean of Delta silently drops, and they are not
    # missing at random -- so they are printed with the frame, not left to be discovered.
    for column, status, count in iter_status_counts(frame):
        print(f"  {column:16s} {status:16s} {count}")
    print(f"\nwrote {frame_path} ({len(frame)} rows) and {len(curves_by_run)} curve archives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
