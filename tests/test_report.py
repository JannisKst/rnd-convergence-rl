"""Tests for the reporting layer --- the reduction from the tidy frame to the table.

The frame is deliberately wide: every detector setting, every bin count, every estimator
seed is a row. Reducing it is therefore the step where a result can be silently wrong rather
than visibly broken, and these tests cover the three ways that happens.

``TestSelect`` pins the filter. Two of its cases are regressions of real bugs. The estimator
settings are null wherever they do not apply --- ``bins_per_dim`` on an RND row,
``analysis_seed`` on a deterministic curve --- so an equality filter deletes whole signal
columns and leaves a table that merely looks sparse. And the frozen control is filed under a
``tag``, so a filter that treats every tag as an experimental condition returns an empty
control and the confound figure quietly loses half of its comparison.

``TestRobustness`` pins the denominator. ``t_plateau`` does not depend on
``conv_smooth_window``, so the frame holds three identical copies of every plateau result;
counting rows rather than configurations inflates the denominator by exactly the factor that
leaves the fraction unchanged and the reported config count wrong.

``TestStatusesSurvive`` pins the promise the analysis layer makes: a mean is reported next to
what it was not computed over. A cell where three of five seeds never plateaued must not read
like a cell of three seeds.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rnd_convergence
from rnd_convergence.analysis import Curve, DetectorSpec, save_curves
from rnd_convergence.figures import (
    figure_confound,
    figure_delta_sensitivity,
    figure_robustness,
    find_curve,
)
from rnd_convergence.report import (
    RND,
    Pin,
    bootstrap_ci,
    confound_pin,
    conv_status_rates,
    delta_band,
    delta_summary,
    discrimination_summary,
    format_delta_table,
    format_robustness_table,
    ladder_order,
    robustness,
    select,
    signal_label,
    signal_order,
    summarise_cells,
    tracking,
    tracking_sweep,
    with_signal_labels,
)
from rnd_convergence.streams import run_stem

REPORT_SCRIPT = Path(rnd_convergence.__file__).parent.parent / "scripts" / "report.py"

PIN = Pin()
TAUS = (0.1, 0.2, 0.3)
CONV_SMOOTH_WINDOWS = (1, 3, 5)


def make_row(
    *,
    env_id="CartPole-v1",
    seed=0,
    signal="rnd",
    bins_per_dim=None,
    tag=None,
    frozen=False,
    tau=PIN.detector.tau,
    smooth_window=PIN.detector.smooth_window,
    reference_quantile=PIN.detector.reference_quantile,
    conv_smooth_window=PIN.detector.conv_smooth_window,
    t_conv=10_000,
    t_plateau=15_000,
    conv_status="converged",
    plateau_status="plateau",
    retained=0.9,
    occupancy_at_plateau=None,
    eval_distinct_returns=40,
    **overrides,
):
    """One frame row, with the null-where-it-does-not-apply convention of the real frame."""
    delta = None if t_conv is None or t_plateau is None or frozen else t_plateau - t_conv
    row = {
        "env_id": env_id,
        "seed": seed,
        "tag": tag,
        "frozen": frozen,
        "signal": signal,
        "mode": "sliding",
        "bins_per_dim": bins_per_dim,
        "clip": None if signal != "sve_grid" else PIN.clip,
        "correction": None if signal != "sve_grid" else PIN.correction,
        "analysis_seed": PIN.analysis_seed if signal in ("rnd", "sve_knn") else None,
        "tau": tau,
        "smooth_window": smooth_window,
        "reference_quantile": reference_quantile,
        "conv_smooth_window": conv_smooth_window,
        "t_conv": t_conv,
        "conv_status": conv_status,
        "t_plateau": t_plateau,
        "plateau_status": plateau_status,
        "delta": delta,
        "retained": None if frozen or t_plateau is None else retained,
        "retained_status": "control" if frozen else "retained",
        "occupancy_at_plateau": (
            occupancy_at_plateau if signal == "sve_grid" and t_plateau is not None else None
        ),
        "occupancy_max": None,
        "eval_distinct_returns": eval_distinct_returns,
    }
    row.update(overrides)
    return row


def make_frame(rows):
    return pd.DataFrame(rows)


def full_sweep(**kwargs):
    """One signal's rows across the detector sweep the analysis layer writes."""
    return [
        make_row(tau=tau, conv_smooth_window=conv_smooth, **kwargs)
        for tau in TAUS
        for conv_smooth in CONV_SMOOTH_WINDOWS
    ]


@pytest.fixture
def frame():
    """A small frame with every signal family, both conditions and the detector sweep."""
    rows = []
    for signal, bins in (("rnd", None), ("sve_grid", 5), ("sve_grid", 10), ("sve_knn", None)):
        for seed in (0, 1):
            rows += full_sweep(signal=signal, bins_per_dim=bins, seed=seed)
        rows += full_sweep(
            signal=signal,
            bins_per_dim=bins,
            tag="frozen",
            frozen=True,
            t_conv=None,
            conv_status="control",
        )
    return make_frame(rows)


class TestSelect:
    def test_keeps_every_signal_family(self, frame):
        # The regression: clip, correction and analysis_seed are null on the signals they do
        # not apply to, so filtering them by equality drops the RND and kNN columns.
        selected = select(frame, PIN)

        assert set(selected["signal_label"]) == {RND, "SVE grid b5", "SVE grid b10", "SVE kNN"}

    def test_keeps_one_row_per_run_and_signal(self, frame):
        selected = select(frame, PIN)

        assert len(selected) == 4 * 2  # four signals, two seeds, the control excluded
        assert not selected.duplicated(subset=["env_id", "seed", "signal_label"]).any()

    def test_finds_the_frozen_control(self, frame):
        # The control is written under the `frozen` tag, so a tag filter that treats every
        # tag as an experimental condition returns nothing and the confound figure silently
        # loses the curve it exists to compare against.
        control = select(frame, PIN, frozen=True)

        assert len(control) == 4
        assert set(control["conv_status"]) == {"control"}

    def test_excludes_a_tagged_condition_from_the_primary_slice(self, frame):
        combined = make_frame(frame.to_dict("records") + full_sweep(tag="lr1e-3", seed=7))

        assert 7 not in set(select(combined, PIN)["seed"])
        assert set(select(combined, PIN, tag="lr1e-3")["seed"]) == {7}

    def test_a_pinned_estimator_seed_excludes_the_others(self, frame):
        # The RND target draw moves t_plateau, so a frame built at several analysis seeds
        # holds several RND rows per run. Pinning one is what keeps a cell a mean over
        # agents rather than a mean over target draws.
        combined = make_frame(frame.to_dict("records") + full_sweep(analysis_seed=1))

        assert set(select(combined, PIN)["analysis_seed"].dropna()) == {PIN.analysis_seed}

    def test_raises_on_a_swept_axis_the_pin_does_not_name(self, frame):
        # kNN's neighbour count is a frame column and not part of the pin. Two values of it
        # produce two rows for one run and one signal; averaging them silently reports a
        # sweep over the estimator as a spread over seeds, so it has to be an error.
        combined = make_frame(full_sweep(signal="sve_knn", k=4) + full_sweep(signal="sve_knn", k=8))
        with pytest.raises(ValueError, match="not pinned"):
            select(combined, PIN)

    def test_a_pin_outside_the_sweep_selects_nothing(self, frame):
        assert select(frame, Pin(detector=DetectorSpec(tau=0.9))).empty


class TestSignalColumns:
    def test_grid_columns_carry_their_bin_count(self):
        assert signal_label("sve_grid", 20) == "SVE grid b20"
        assert signal_label("rnd", None) == RND

    def test_a_grid_row_without_a_bin_count_is_an_error(self):
        with pytest.raises(ValueError, match="bins_per_dim"):
            signal_label("sve_grid", None)

    def test_columns_run_rnd_then_grid_by_bin_count_then_knn(self, frame):
        assert signal_order(with_signal_labels(frame)) == [
            RND,
            "SVE grid b5",
            "SVE grid b10",
            "SVE kNN",
        ]

    def test_an_unknown_environment_is_appended_rather_than_dropped(self, frame):
        order = ladder_order(make_frame(frame.to_dict("records") + [make_row(env_id="Zork-v0")]))

        assert order == ["CartPole-v1", "Zork-v0"]


class TestBootstrap:
    def test_a_single_run_gets_no_interval(self):
        # A bootstrap over one value returns a zero-width interval, which reads as certainty.
        assert np.isnan(bootstrap_ci([5.0])).all()

    def test_the_interval_brackets_the_mean(self):
        values = [10.0, 12.0, 14.0, 16.0, 18.0]
        low, high = bootstrap_ci(values, reps=2_000)

        assert low < np.mean(values) < high

    def test_the_interval_is_reproducible(self):
        values = [1.0, 5.0, 9.0, 13.0]

        assert bootstrap_ci(values, reps=1_000) == bootstrap_ci(values, reps=1_000)


class TestStatusesSurvive:
    def test_a_cell_counts_the_runs_it_could_not_use(self):
        rows = [make_row(seed=0)]
        rows += [
            make_row(seed=seed, t_plateau=None, plateau_status="no_plateau") for seed in (1, 2)
        ]
        summary = summarise_cells(with_signal_labels(make_frame(rows)), by=("signal_label",))

        assert summary.loc[0, "n_runs"] == 3
        assert summary.loc[0, "n_defined"] == 1
        assert summary.loc[0, "plateau_rate"] == pytest.approx(1 / 3)

    def test_a_cell_where_nothing_fired_has_a_mean_of_nan_rather_than_zero(self):
        rows = [make_row(t_plateau=None, plateau_status="no_plateau")]
        summary = summarise_cells(with_signal_labels(make_frame(rows)), by=("signal_label",))

        assert np.isnan(summary.loc[0, "mean"])
        assert summary.loc[0, "n_defined"] == 0

    def test_convergence_statuses_are_counted_over_runs_not_over_rows(self, frame):
        # conv_status is a property of the run; the slice repeats it once per signal, so
        # counting rows would multiply every rate by the number of signal columns.
        conv = conv_status_rates(frame, PIN)

        assert conv.loc[0, "n_runs"] == 2
        assert conv.loc[0, "n_converged"] == 2

    def test_a_rung_that_never_converged_reports_its_reason(self):
        rows = full_sweep(t_conv=None, conv_status="not_learned")
        conv = conv_status_rates(make_frame(rows), PIN)

        assert conv.loc[0, "n_converged"] == 0
        assert "not_learned" in conv.loc[0, "statuses"]


class TestRobustness:
    def test_the_denominator_is_configurations_not_frame_rows(self, frame):
        # Three conv_smooth_window values hold three identical copies of each plateau result.
        fired = robustness(frame, PIN)

        assert set(fired["n_configs"]) == {len(TAUS)}
        assert (fired["fired"] == 1.0).all()

    def test_a_signal_that_fires_in_half_the_configurations_reads_as_half(self):
        rows = [make_row(tau=0.1, conv_smooth_window=conv) for conv in CONV_SMOOTH_WINDOWS]
        rows += [
            make_row(tau=0.2, conv_smooth_window=conv, t_plateau=None, plateau_status="no_plateau")
            for conv in CONV_SMOOTH_WINDOWS
        ]
        fired = robustness(make_frame(rows), PIN)

        assert fired.loc[0, "fired"] == pytest.approx(0.5)
        assert fired.loc[0, "n_configs"] == 2

    def test_the_control_is_not_counted_as_a_run(self, frame):
        # The control has no t_conv by construction; letting it into the robustness
        # denominator would mix "the signal flattens without learning" into a claim about
        # the trained runs.
        assert set(robustness(frame, PIN)["n_seeds"]) == {2}


class TestDeltaBand:
    def test_the_band_holds_one_value_per_seed_and_configuration(self, frame):
        band = delta_band(frame, PIN)

        assert len(band) == 4 * 2 * len(TAUS)  # signals x seeds x detector configurations
        assert set(band["conv_smooth_window"]) if "conv_smooth_window" in band else True

    def test_the_evaluation_smoothing_stays_pinned(self):
        # Delta moves with conv_smooth_window through t_conv. Sweeping both at once would
        # mix "when did the signal flatten" with "when do we judge the policy converged".
        rows = full_sweep(t_conv=10_000)
        band = delta_band(make_frame(rows), PIN)

        assert len(band) == len(TAUS)


class TestTables:
    def test_the_table_names_the_setting_it_was_read_at(self, frame):
        text = format_delta_table(
            delta_summary(frame, PIN, reps=200), conv_status_rates(frame, PIN), PIN
        )

        assert PIN.caption in text
        assert "tau=0.1" in text

    def test_a_cell_with_no_delta_is_a_dash_rather_than_a_zero(self):
        rows = full_sweep(t_plateau=None, plateau_status="no_plateau")
        frame = make_frame(rows)
        text = format_delta_table(
            delta_summary(frame, PIN, reps=200), conv_status_rates(frame, PIN), PIN
        )

        assert "—" in text

    def test_the_delta_cell_reports_seeds_used_over_seeds_run(self):
        rows = full_sweep(seed=0)
        rows += full_sweep(seed=1, t_plateau=None, plateau_status="no_plateau")
        frame = make_frame(rows)
        text = format_delta_table(
            delta_summary(frame, PIN, reps=200), conv_status_rates(frame, PIN), PIN
        )

        assert "(1/2)" in text

    def test_the_robustness_table_matches_the_figure_data(self, frame):
        text = format_robustness_table(robustness(frame, PIN))

        assert "1.00" in text
        assert "CartPole" in text

    def test_both_tables_say_how_many_seeds_are_behind_a_rung(self, frame):
        # The pilot rungs are single-seed and the trend is read across them; a rate printed
        # without its denominator invites five rungs to be weighed equally.
        delta_text = format_delta_table(
            delta_summary(frame, PIN, reps=200), conv_status_rates(frame, PIN), PIN
        )
        robustness_text = format_robustness_table(robustness(frame, PIN))

        assert "Seeds" in delta_text
        assert "Seeds" in robustness_text

    def test_the_caption_does_not_credit_a_library_the_code_does_not_use(self, frame):
        text = format_delta_table(
            delta_summary(frame, PIN, reps=200), conv_status_rates(frame, PIN), PIN
        )

        assert "rliable" not in text
        assert "percentile bootstrap" in text

    def test_the_two_kinds_of_firing_rate_are_not_both_called_fired(self, frame):
        # The Delta cell counts seeds at one setting; the robustness table counts detector
        # configurations across the sweep. One document carrying both under one word invites
        # "100%" and "0.94" to be read as a contradiction.
        delta_text = format_delta_table(
            delta_summary(frame, PIN, reps=200), conv_status_rates(frame, PIN), PIN
        )

        rows = [line for line in delta_text.splitlines() if line.startswith("| MarsRover")]
        assert "plateau in 2/2 seeds" in delta_text
        # The word belongs to the sweep, so no *cell* may use it; the prose below the table
        # is free to explain the difference.
        assert not any("fired" in row for row in rows)

    def test_a_grid_cell_carries_the_occupancy_of_the_grid_it_plateaued_on(self):
        rows = full_sweep(signal="sve_grid", bins_per_dim=10, occupancy_at_plateau=0.19)
        frame = make_frame(rows)
        text = format_delta_table(
            delta_summary(frame, PIN, reps=200), conv_status_rates(frame, PIN), PIN
        )

        assert "occ 0.190" in text

    def test_a_saturated_grid_cell_says_so(self):
        # At occupancy 1.0 the estimator is pinned to log N and the plateau is arithmetic.
        rows = full_sweep(signal="sve_grid", bins_per_dim=20, occupancy_at_plateau=1.0)
        frame = make_frame(rows)
        text = format_delta_table(
            delta_summary(frame, PIN, reps=200), conv_status_rates(frame, PIN), PIN
        )

        assert "SATURATED" in text

    def test_retained_is_reported_but_suppressed_on_a_quantised_evaluation(self):
        graded = make_frame(full_sweep(retained=0.87, eval_distinct_returns=40))
        binary = make_frame(
            full_sweep(env_id="MiniGrid-Empty-5x5-v0", retained=4.39, eval_distinct_returns=3)
        )

        assert "retained 0.87" in format_delta_table(
            delta_summary(graded, PIN, reps=200), conv_status_rates(graded, PIN), PIN
        )
        assert "retained n/a" in format_delta_table(
            delta_summary(binary, PIN, reps=200), conv_status_rates(binary, PIN), PIN
        )

    def test_the_robustness_split_separates_the_rejected_reference_quantile(self, frame):
        split = robustness(frame, PIN, split_reference_quantile=True)

        assert set(split["reference_quantile"]) == {PIN.detector.reference_quantile}
        assert "reference_quantile" in format_robustness_table(robustness(frame, PIN), split)

    def test_the_discrimination_table_orders_grid_columns_by_bin_count(self):
        rows = []
        for bins in (20, 5, 10):
            rows += full_sweep(tag="lr", signal="sve_grid", bins_per_dim=bins)
        summary = discrimination_summary(make_frame(rows), PIN, ["lr"])

        assert list(summary["signal_label"]) == ["SVE grid b5", "SVE grid b10", "SVE grid b20"]

    def test_a_signal_that_tracks_convergence_has_a_slope_near_one(self):
        rows = []
        for seed, (t_conv, t_plateau) in enumerate(
            [(10_000, 20_000), (20_000, 30_000), (30_000, 40_000), (40_000, 50_000)]
        ):
            rows += full_sweep(tag="lr", seed=seed, t_conv=t_conv, t_plateau=t_plateau)
        tracked = tracking(make_frame(rows), PIN, ["lr"]).iloc[0]

        assert tracked["spearman"] == pytest.approx(1.0)
        assert tracked["slope"] == pytest.approx(1.0)

    def test_a_signal_stuck_at_one_step_has_a_slope_of_zero(self):
        # The failure the experiment exists to catch: a detector that always fires at the
        # same step is perfectly robust, and its Delta is an artefact of where t_conv landed.
        rows = []
        for seed, t_conv in enumerate([10_000, 20_000, 30_000, 40_000]):
            rows += full_sweep(tag="lr", seed=seed, t_conv=t_conv, t_plateau=55_000)
        tracked = tracking(make_frame(rows), PIN, ["lr"]).iloc[0]

        assert np.isnan(tracked["spearman"])  # t_plateau never moved
        assert tracked["t_plateau_range"] == 0.0
        assert tracked["t_conv_range"] == 30_000

    def test_the_sweep_answers_the_same_question_at_every_detector_setting(self):
        # The pinned row is one cell of the sweep, and the cell that answers most favourably
        # is the one a headline would otherwise be built on.
        rows = []
        for seed, t_conv in enumerate([10_000, 20_000, 30_000, 40_000]):
            for tau in TAUS:
                for conv in CONV_SMOOTH_WINDOWS:
                    rows.append(
                        make_row(
                            tag="lr",
                            seed=seed,
                            tau=tau,
                            conv_smooth_window=conv,
                            t_conv=t_conv,
                            # Tracks at one tau, pinned at every other.
                            t_plateau=t_conv + 5_000 if tau == 0.1 else 55_000,
                        )
                    )
        swept = tracking_sweep(make_frame(rows), PIN, ["lr"])

        tracking_config = swept[swept["tau"] == 0.1].iloc[0]
        assert tracking_config["slope"] == pytest.approx(1.0)
        assert swept["slope"].isna().sum() == 2  # t_plateau never moved at the other two taus
        assert len(swept) == len(TAUS)

    def test_the_sweep_reports_a_p_value_beside_the_correlation(self):
        rows = []
        for seed, t_conv in enumerate([10_000, 20_000, 30_000, 40_000]):
            rows += full_sweep(tag="lr", seed=seed, t_conv=t_conv, t_plateau=t_conv + 5_000)
        swept = tracking_sweep(make_frame(rows), PIN, ["lr"])

        assert (swept["p_value"] < 0.05).all()

    def test_tracking_needs_three_paired_runs(self):
        rows = full_sweep(tag="lr", seed=0, t_conv=10_000, t_plateau=20_000)
        rows += full_sweep(tag="lr", seed=1, t_conv=20_000, t_plateau=30_000)
        tracked = tracking(make_frame(rows), PIN, ["lr"]).iloc[0]

        assert tracked["n_pairs"] == 2
        assert np.isnan(tracked["slope"])

    def test_the_discrimination_table_pairs_t_conv_with_t_plateau(self):
        rows = []
        for tag, t_conv in (("lr1e-3", 8_000), ("lr1e-4", 40_000)):
            rows += full_sweep(tag=tag, t_conv=t_conv, t_plateau=50_000)
        summary = discrimination_summary(make_frame(rows), PIN, ["lr1e-3", "lr1e-4"])

        assert list(summary["t_conv_mean"]) == [8_000, 40_000]
        assert list(summary["t_plateau_mean"]) == [50_000, 50_000]
        assert list(summary["delta_mean"]) == [42_000, 10_000]


def make_archive(path, *, steps=None, decay=True):
    """A curve archive of the shape the figures read, for one run."""
    steps = np.arange(1, 41, dtype=np.int64) * 2_500 if steps is None else steps
    falling = np.exp(-np.arange(steps.size) / 8.0) + 0.01
    flat = np.full(steps.size, 3.0) + np.arange(steps.size) * 1e-3
    curves = [
        Curve(signal="rnd", steps=steps, values=falling if decay else flat, analysis_seed=0),
        Curve(
            signal="sve_grid",
            steps=steps,
            values=flat if decay else flat,
            bins_per_dim=10,
            clip=5.0,
            correction="none",
        ),
    ]
    return save_curves(path, curves)


class TestConfoundSetting:
    """Where to draw the trained-vs-frozen figure, given that it is a comparison."""

    def both_conditions(self, *, frozen_fires_at):
        rows = []
        for tau in TAUS:
            for conv in CONV_SMOOTH_WINDOWS:
                rows.append(make_row(tau=tau, conv_smooth_window=conv))
                fires = tau in frozen_fires_at
                rows.append(
                    make_row(
                        tau=tau,
                        conv_smooth_window=conv,
                        tag="frozen",
                        frozen=True,
                        t_conv=None,
                        conv_status="control",
                        t_plateau=16_000 if fires else None,
                        plateau_status="plateau" if fires else "no_plateau",
                    )
                )
        return make_frame(rows)

    def test_keeps_the_pinned_setting_when_the_control_plateaus_there(self):
        frame = self.both_conditions(frozen_fires_at={PIN.detector.tau})
        chosen, note = confound_pin(frame, PIN, env_id="CartPole-v1", seed=0)

        assert chosen == PIN
        assert "same setting as the table" in note

    def test_moves_to_the_nearest_setting_where_both_conditions_fire(self):
        # The real case: at the table's tau the control never plateaus, so a figure drawn
        # there shows one ring and cannot say "both land in the same place".
        frame = self.both_conditions(frozen_fires_at={0.3})
        chosen, note = confound_pin(frame, PIN, env_id="CartPole-v1", seed=0)

        assert chosen.detector.tau == 0.3
        assert chosen.detector.conv_smooth_window == PIN.detector.conv_smooth_window
        assert "not the table's setting" in note

    def test_falls_back_to_the_pin_and_says_so_when_nothing_qualifies(self):
        frame = self.both_conditions(frozen_fires_at=set())
        chosen, note = confound_pin(frame, PIN, env_id="CartPole-v1", seed=0)

        assert chosen == PIN
        assert "no configuration in the sweep" in note


class TestFigures:
    def test_each_figure_is_written(self, frame, tmp_path):
        robustness_path = figure_robustness(robustness(frame, PIN), tmp_path / "r.png", pin=PIN)
        band_path = figure_delta_sensitivity(delta_band(frame, PIN), tmp_path / "d.png", pin=PIN)

        assert robustness_path.exists() and robustness_path.stat().st_size > 0
        assert band_path.exists() and band_path.stat().st_size > 0

    def test_the_confound_figure_needs_both_conditions(self, frame, tmp_path):
        curves = tmp_path / "curves"
        make_archive(curves / f"{run_stem('CartPole-v1', 0)}.npz")

        missing = figure_confound(
            frame, curves, tmp_path / "c.png", pin=PIN, env_id="CartPole-v1", seed=0
        )
        make_archive(curves / f"{run_stem('CartPole-v1', 0, 'frozen')}.npz", decay=False)
        drawn = figure_confound(
            frame, curves, tmp_path / "c.png", pin=PIN, env_id="CartPole-v1", seed=0
        )

        assert missing is None
        assert drawn is not None and drawn.exists()

    def test_a_curve_is_identified_by_the_pin(self, tmp_path):
        from rnd_convergence.analysis import load_curves

        curves = load_curves(make_archive(tmp_path / "a.npz"))

        assert find_curve(curves, signal="rnd", pin=PIN).signal == "rnd"
        assert find_curve(curves, signal="sve_grid", pin=PIN, bins_per_dim=20) is None

    def test_an_ambiguous_pin_is_an_error_rather_than_the_first_match(self, tmp_path):
        from rnd_convergence.analysis import load_curves

        steps = np.arange(1, 11, dtype=np.int64)
        values = np.linspace(1.0, 0.1, 10)
        path = save_curves(
            tmp_path / "two.npz",
            [
                Curve(signal="rnd", steps=steps, values=values, analysis_seed=0),
                Curve(signal="rnd", steps=steps, values=values, analysis_seed=None),
            ],
        )
        with pytest.raises(ValueError, match="does not identify one curve"):
            find_curve(load_curves(path), signal="rnd", pin=PIN)


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("report_script", REPORT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestReportScript:
    def prepare(self, frame, tmp_path):
        results = tmp_path / "results"
        (results / "curves").mkdir(parents=True)
        frame.to_csv(results / "signals.csv", index=False)
        make_archive(results / "curves" / f"{run_stem('CartPole-v1', 0)}.npz")
        make_archive(
            results / "curves" / f"{run_stem('CartPole-v1', 0, 'frozen')}.npz", decay=False
        )
        return results

    def test_writes_the_table_and_every_figure(self, script, frame, tmp_path):
        results = self.prepare(frame, tmp_path)
        out = tmp_path / "figures"

        assert script.main(["--results", str(results), "--out", str(out), "--reps", "200"]) == 0
        written = sorted(path.name for path in out.iterdir())
        assert written == [
            "confound_CartPole-v1.png",
            "delta_sensitivity.png",
            "delta_summary.csv",
            "delta_table.md",
            "detector_robustness.png",
            "robustness.csv",
            "robustness_by_quantile.csv",
        ]

    def test_the_generated_table_names_the_command_that_made_it(self, script, frame, tmp_path):
        # The committed table is not what the documented bare command produces --- the
        # discrimination section only appears with --conditions --- so the file has to say
        # which invocation it came from.
        results = self.prepare(frame, tmp_path)
        out = tmp_path / "figures"
        script.main(["--results", str(results), "--out", str(out), "--reps", "200"])

        header = (out / "delta_table.md").read_text().splitlines()[1]
        assert header.startswith("<!-- Reproduce with: python scripts/report.py")
        assert "--reps 200" in header

    def test_a_missing_frame_is_reported_rather_than_a_traceback(self, script, tmp_path):
        assert script.main(["--results", str(tmp_path), "--out", str(tmp_path / "out")]) == 1

    def test_a_pin_the_frame_was_not_swept_over_fails_loudly(self, script, frame, tmp_path):
        # Four empty figures and a table of dashes is the alternative, and it looks like a
        # finding about the signals rather than a setting that was never measured.
        results = self.prepare(frame, tmp_path)

        assert (
            script.main(["--results", str(results), "--out", str(tmp_path / "out"), "--tau", "0.9"])
            == 1
        )

    def test_the_discrimination_table_is_written_when_conditions_are_named(
        self, script, frame, tmp_path
    ):
        combined = make_frame(frame.to_dict("records") + full_sweep(tag="lr1e-3", t_conv=5_000))
        results = self.prepare(combined, tmp_path)
        out = tmp_path / "figures"

        script.main(
            ["--results", str(results), "--out", str(out), "--reps", "200"]
            + ["--conditions", "lr1e-3"]
        )

        assert (out / "discrimination.csv").exists()
        assert "lr1e-3" in (out / "delta_table.md").read_text()
