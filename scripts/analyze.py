"""Analysis entry point: a directory of runs in, the tidy results frame out.

    python scripts/analyze.py                          # results/ -> docs/results/
    python scripts/analyze.py --results /scratch/runs --out docs/results
    python scripts/analyze.py --modes sliding cumulative --quick

Writes two things to ``--out``:

    signals.csv              one row per (run, signal, detector setting)
    curves/<stem>.npz        every measured curve, for the figures to draw from

Curves are saved because building them is the expensive half --- an RND replay plus a kNN
estimate per window --- while sweeping detector settings over them is nearly free. Plotting
later should read these rather than re-measure the streams.

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
    DEFAULT_BIN_COUNTS,
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
        "--quick",
        action="store_true",
        help="one detector setting instead of the full sweep, for a fast look at a pilot",
    )
    return parser.parse_args(argv)


def summarise(frame: pd.DataFrame) -> str:
    """A per-run, per-signal digest of the frame, at the default detector setting.

    Printed because the first question asked of a pilot is whether each signal plateaued at
    all, and scrolling a 400-row CSV to find out defeats the purpose of running one.
    """
    if frame.empty:
        return "no rows"
    defaults = frame[
        (frame.tau == frame.tau.min())
        & (frame.smooth_window == 5)
        & (frame.reference_quantile == 0.5)
        & (frame.conv_smooth_window == frame.conv_smooth_window.min())
    ]
    view = defaults if not defaults.empty else frame
    columns = ["env_id", "seed", "frozen", "label", "n_points", "t_conv", "conv_status"]
    columns += ["t_plateau", "plateau_status", "delta", "retained", "occupancy_max"]
    return view[columns].to_string(index=False, max_colwidth=24)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = find_runs(args.results)
    if not runs:
        print(f"no runs found in {args.results} (looking for *.run.json with a state stream)")
        return 1

    print(f"analysing {len(runs)} run(s) from {args.results}")
    detectors = detector_grid(taus=(0.1,), smooth_windows=(5,)) if args.quick else detector_grid()

    def report(stem: Path, meta: RunMeta, rows: list[dict]) -> None:
        print(f"  {stem.name}: {meta.n_states} states -> {len(rows)} rows")

    frame, curves_by_run = analyse_directory(
        args.results,
        detectors=detectors,
        modes=args.modes,
        bin_counts=args.bins,
        on_run=report,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    frame_path = args.out / "signals.csv"
    frame.to_csv(frame_path, index=False)
    for name, curves in curves_by_run.items():
        save_curves(args.out / "curves" / f"{name}.npz", curves)

    print(f"\n{summarise(frame)}\n")
    # The statuses are exactly the rows a mean of Delta silently drops, and they are not
    # missing at random -- so they are printed with the frame, not left to be discovered.
    for column, status, count in iter_status_counts(frame):
        print(f"  {column:16s} {status:16s} {count}")
    print(f"\nwrote {frame_path} ({len(frame)} rows) and {len(curves_by_run)} curve archives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
