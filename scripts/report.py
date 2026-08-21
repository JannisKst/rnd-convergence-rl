"""Reporting entry point: the committed frame in, the report's table and figures out.

    python scripts/report.py                                  # docs/results -> docs/figures
    python scripts/report.py --tau 0.2 --smooth-window 9      # the same table, elsewhere in the sweep
    python scripts/report.py --conditions lr1e-3 lr3e-4 lr1e-4  # the discrimination table

Reads ``docs/results/signals.csv`` and ``docs/results/curves/`` and nothing else. That
restriction is the point of the split: the state streams those artefacts were measured from
are hundreds of megabytes and are not in the repository, so anything that had to replay one
would be unrunnable for whoever picks the report up. It also keeps the figures and the table
the *same* measurement --- both are selections from one frame, so they cannot drift apart.

Writes to ``--out`` (``docs/figures`` by default):

    delta_table.md            the headline table, its caption and the status rates
    delta_summary.csv         the same numbers tidy, one row per (rung, signal)
    detector_robustness.*     how often each signal plateaus at all
    robustness_by_quantile.csv  the same rates split by reference_quantile
    delta_sensitivity.*       how far Delta moves across the detector sweep
    confound_<rung>.*         trained against frozen, both signals, one rung

and, when --conditions names run tags, the experiment that asks whether a plateau tracks
convergence at all:

    discrimination.csv        t_conv and t_plateau per condition and signal
    tracking.csv              their rank correlation and slope, paired within runs
    tracking_sweep.csv        the same, at every detector setting rather than the pinned one

Every artefact names the detector and estimator settings it was read at, because ``Delta``
moves with them: a table without its pin is a number without its units.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from rnd_convergence.analysis import DetectorSpec
from rnd_convergence.figures import figure_confound, figure_delta_sensitivity, figure_robustness, figure_detector_sensitivity
from rnd_convergence.report import (
    Pin,
    confound_pin,
    conv_status_rates,
    delta_band,
    delta_summary,
    discrimination_summary,
    format_delta_table,
    format_discrimination_table,
    format_robustness_table,
    format_status_table,
    robustness,
    tracking,
    tracking_sweep,
)

# The rung the confound figure is drawn for by default. CartPole is the one where the
# comparison is sharpest: it converges cleanly, so the trained curve has a plateau worth
# comparing against, and its frozen control still produces a decaying RND curve.
DEFAULT_CONFOUND_ENV = "CartPole-v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("docs/results"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])

    detector = DetectorSpec()
    parser.add_argument("--tau", type=float, default=detector.tau)
    parser.add_argument("--smooth-window", type=int, default=detector.smooth_window)
    parser.add_argument("--reference-quantile", type=float, default=detector.reference_quantile)
    parser.add_argument(
        "--conv-smooth-window",
        type=int,
        default=detector.conv_smooth_window,
        help="smoothing of the evaluation curve t_conv is read from; 1 is the settled primary",
    )
    parser.add_argument("--mode", default="sliding", choices=["sliding", "cumulative"])
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--correction", default="none", choices=["none", "miller_madow"])
    parser.add_argument("--analysis-seed", type=int, default=0)

    parser.add_argument(
        "--confound-env",
        default=DEFAULT_CONFOUND_ENV,
        help="rung the trained-vs-frozen figure is drawn for",
    )
    parser.add_argument("--confound-seed", type=int, default=0)
    parser.add_argument(
        "--confound-bins",
        type=int,
        default=10,
        help="bin count of the grid-SVE panel in the confound figure",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[],
        metavar="RUN_TAG",
        help="run_tags to compare in the discrimination table, e.g. a learning-rate sweep",
    )
    parser.add_argument(
        "--reps", type=int, default=10_000, help="bootstrap replications for the CIs"
    )
    return parser.parse_args(argv)


def pin_from(args: argparse.Namespace) -> Pin:
    return Pin(
        detector=DetectorSpec(
            tau=args.tau,
            smooth_window=args.smooth_window,
            reference_quantile=args.reference_quantile,
            conv_smooth_window=args.conv_smooth_window,
        ),
        mode=args.mode,
        clip=args.clip,
        correction=args.correction,
        analysis_seed=args.analysis_seed,
    )


def invocation(argv: list[str] | None) -> str:
    """The command that produced this output, for the generated file's header.

    Stamped because the artefacts are committed and the defaults do not reproduce them: the
    discrimination section only exists when ``--conditions`` names run tags, so a reader who
    runs the documented bare command gets a different document and no way to tell why.
    """
    arguments = sys.argv[1:] if argv is None else list(argv)
    return " ".join(["python", "scripts/report.py", *arguments]).rstrip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frame_path = args.results / "signals.csv"
    if not frame_path.exists():
        print(f"no frame at {frame_path}; run scripts/analyze.py first")
        return 1

    frame = pd.read_csv(frame_path)
    pin = pin_from(args)
    args.out.mkdir(parents=True, exist_ok=True)

    summary = delta_summary(frame, pin, reps=args.reps)
    if summary.empty:
        # An empty slice means the pin names a setting the frame was not swept over. Said
        # here rather than left to produce four empty figures.
        print(f"the frame has no rows at {pin.caption}")
        return 1
    conv = conv_status_rates(frame, pin)
    fired = robustness(frame, pin)
    by_quantile = robustness(frame, pin, split_reference_quantile=True)
    band = delta_band(frame, pin)

    document = "\n".join(
        [
            "<!-- Generated by scripts/report.py; edit that, not this. -->",
            f"<!-- Reproduce with: {invocation(argv)} -->",
            "# Results",
            "",
            format_delta_table(summary, conv, pin),
            format_status_table(conv),
            format_robustness_table(fired, by_quantile),
        ]
    )
    if args.conditions:
        discrimination = discrimination_summary(frame, pin, args.conditions)
        tracked = tracking(frame, pin, args.conditions)
        # The same question asked at every detector setting rather than at the pinned one.
        # t_plateau is itself a function of that setting, so an answer read off one cell of
        # the sweep is the most favourable cell until it is shown not to be.
        swept = tracking_sweep(frame, pin, args.conditions)
        document += "\n" + format_discrimination_table(discrimination, pin, tracked, swept)
        discrimination.to_csv(args.out / "discrimination.csv", index=False)
        tracked.to_csv(args.out / "tracking.csv", index=False)
        swept.to_csv(args.out / "tracking_sweep.csv", index=False)

    (args.out / "delta_table.md").write_text(document)
    summary.to_csv(args.out / "delta_summary.csv", index=False)
    fired.to_csv(args.out / "robustness.csv", index=False)
    by_quantile.to_csv(args.out / "robustness_by_quantile.csv", index=False)

    written = [args.out / "delta_table.md", args.out / "delta_summary.csv"]
    written.append(
        figure_robustness(fired, args.out / f"detector_robustness.{args.format}", pin=pin)
    )

    # A rung whose runs never converged has no Delta on any signal, so it has no row in the
    # sensitivity figure. Named in its caption rather than left to look like a missing run.
    omitted = [env for env in summary["env_id"].cat.categories if env not in set(band["env_id"])]
    written.append(
        figure_delta_sensitivity(
            band, args.out / f"delta_sensitivity.{args.format}", pin=pin, omitted=omitted
        )
    )

    written.append(
        figure_detector_sensitivity(
            band, args.out / f"detector_sensitivity.{args.format}", pin=pin
        )
    )

    poster_band = band[band["signal_label"].isin(["RND", "SVE grid b10"])].copy()
    written.append(
        figure_delta_sensitivity(
            poster_band, args.out / f"delta_sensitivity_poster.{args.format}", pin=pin, omitted=omitted
        ) 
    )

    # The confound figure is a comparison, so it is drawn where the detector fires on both
    # conditions; that is not always the table's pinned setting, and the figure says which.
    drawn_at, note = confound_pin(frame, pin, env_id=args.confound_env, seed=args.confound_seed)
    confound = figure_confound(
        frame,
        args.results / "curves",
        args.out / f"confound_{args.confound_env}.{args.format}",
        pin=drawn_at,
        env_id=args.confound_env,
        seed=args.confound_seed,
        bins_per_dim=args.confound_bins,
        setting_note=note,
    )
    if confound is None:
        # The figure needs a trained *and* a frozen archive for the rung; without both there
        # is no comparison to draw, and a missing control is worth a line rather than a
        # silently absent figure.
        print(
            f"no confound figure: {args.confound_env} seed {args.confound_seed} needs both a "
            f"trained and a frozen curve archive in {args.results / 'curves'}"
        )
    else:
        written.append(confound)

    print(format_delta_table(summary, conv, pin))
    print(format_robustness_table(fired))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
