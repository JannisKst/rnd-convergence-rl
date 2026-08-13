"""Tests for rnd_convergence.analysis — the layer that turns runs into table rows.

Two groups carry the weight.

``TestPlateauStatuses`` provokes each of ``plateau_time``'s failure conditions with real
data and asserts the status it lands under. The mapping is by message substring, which only
stays honest if something exercises it: without these tests a reworded message would quietly
turn every affected cell into ``detector_error`` and the sweep would keep running.

``TestFrozenControl`` pins the one thing the pilot showed is easy to get wrong. An untrained
network is still a *consistent* policy, so the frozen control can and does satisfy the
convergence criterion --- the CartPole control reported ``converged`` at 12 288. Deciding
comparability from the convergence status rather than from the ``frozen`` flag would put a
meaningless ``Delta`` into the table on the run whose entire job is to be a control.

``TestAnalyzeScript`` covers the entry point end to end. The library functions below are
tested in isolation; the script is the seam they are wired together across, and a digest
filter that selects nothing or an archive written under the wrong stem raises nothing until
someone runs it.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rnd_convergence
from rnd_convergence.analysis import (
    Curve,
    DetectorSpec,
    analyse_directory,
    analyse_run,
    build_curves,
    detect_plateau,
    detector_grid,
    find_runs,
    iter_status_counts,
    load_curves,
    save_curves,
)
from rnd_convergence.runs import RunMeta, save_run_meta
from rnd_convergence.streams import (
    EvalCurve,
    StateStream,
    run_stem,
    save_eval_curve,
    save_state_stream,
)

SPEC = DetectorSpec()
ANALYZE_SCRIPT = Path(rnd_convergence.__file__).parent.parent / "scripts" / "analyze.py"


def make_curve(values, signal="rnd", **kwargs):
    values = np.asarray(values, dtype=np.float64)
    steps = (np.arange(values.size, dtype=np.int64) + 1) * 100
    return Curve(signal=signal, steps=steps, values=values, **kwargs)


def decaying(n=60):
    """A signal that decays and then flattens --- the shape a plateau is defined on."""
    return np.exp(-np.arange(n) / 6.0) + 0.01


def make_stream(n=4_000, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    return StateStream(
        steps=np.arange(n, dtype=np.int64),
        observations=rng.normal(size=(n, dim)).astype(np.float32),
        episode_ends=np.zeros(n, dtype=bool),
        env_id="CartPole-v1",
        seed=0,
    )


def converging(n=12):
    """A return curve that rises and then holds --- a run with a real ``t_conv``."""
    half = n // 2
    return np.concatenate([np.linspace(0.0, 10.0, n - half), np.full(half, 10.0)])


def make_eval_curve(returns, random_return=0.0, n=12):
    return EvalCurve(
        steps=(np.arange(n, dtype=np.int64) + 1) * 500,
        returns=np.full((n, 3), 0.0, dtype=np.float32)
        + np.asarray(returns, dtype=np.float32)[:, None],
        random_return=random_return,
        env_id="CartPole-v1",
        seed=0,
    )


def make_meta(**overrides):
    fields = {
        "env_id": "CartPole-v1",
        "seed": 0,
        "tag": None,
        "frozen": False,
        "total_steps": 4_000,
        "eval_interval": 500,
        "eval_episodes": 3,
        "rollout_steps": 100,
        "signal_window": 200,
        "signal_stride": 100,
        "n_states": 4_000,
        "wall_clock_seconds": 1.0,
    }
    return RunMeta(**(fields | overrides))


class TestCurve:
    def test_label_is_unique_across_the_built_set(self):
        curves = build_curves(make_stream(), window=400, stride=200, bin_counts=(5, 10))
        labels = [curve.label for curve in curves]
        assert len(labels) == len(set(labels))

    def test_label_records_the_settings_that_distinguish_a_curve(self):
        assert make_curve([1.0], signal="sve_grid", bins_per_dim=10).label == "sve_grid_b10_sliding"
        assert make_curve([1.0], signal="sve_knn", k=4, mode="cumulative").label == (
            "sve_knn_k4_cumulative"
        )
        assert make_curve([1.0]).label == "rnd_sliding"

    def test_label_separates_curves_that_differ_only_in_clip_or_seed(self):
        # Both are swept, so both have to reach the name. A label that dropped either would
        # put two different measurements under one legend entry and one dict key.
        grid = [make_curve([1.0], signal="sve_grid", bins_per_dim=10, clip=c) for c in (3.0, 5.0)]
        assert [curve.label for curve in grid] == [
            "sve_grid_b10_c3_sliding",
            "sve_grid_b10_c5_sliding",
        ]
        rnd = [make_curve([1.0], analysis_seed=s) for s in (0, 1)]
        assert [curve.label for curve in rnd] == ["rnd_s0_sliding", "rnd_s1_sliding"]

    def test_params_round_trip_through_the_constructor(self):
        # save_curves stores exactly this dict as JSON and load_curves splats it back, so
        # every key has to be a constructor argument.
        curve = make_curve([1.0], signal="sve_knn", k=4, max_samples=100, analysis_seed=2)
        assert Curve(steps=curve.steps, values=curve.values, **curve.params).label == curve.label

    def test_occupancy_is_a_diagnostic_and_nothing_else_is(self):
        assert make_curve([1.0], signal="occupancy", bins_per_dim=5).is_diagnostic
        assert not make_curve([1.0], signal="sve_grid", bins_per_dim=5).is_diagnostic
        assert not make_curve([1.0]).is_diagnostic


class TestDetectorSpec:
    @pytest.mark.parametrize(
        "kwargs",
        [{"tau": 0.0}, {"tau": -1.0}, {"reference_quantile": 0.0}, {"smooth_window": 0}],
    )
    def test_rejects_a_malformed_sweep_up_front(self, kwargs):
        # Otherwise every row of the sweep comes back detector_error, which reads like a
        # finding about the data rather than a typo in the sweep definition.
        with pytest.raises(ValueError):
            DetectorSpec(**kwargs)

    def test_grid_is_the_full_cross_product(self):
        grid = detector_grid(taus=(0.1, 0.2), smooth_windows=(1, 5), reference_quantiles=(0.5,))
        assert len(grid) == 2 * 2 * 1 * len(set(spec.conv_smooth_window for spec in grid))
        assert len(set(grid)) == len(grid)


class TestPlateauStatuses:
    def test_a_decaying_signal_plateaus(self):
        t_plateau, status = detect_plateau(make_curve(decaying()), SPEC)
        assert status == "plateau"
        assert t_plateau is not None

    def test_a_steady_ramp_never_plateaus(self):
        # Constant rate of change: the criterion is normalised against exactly that, so it
        # must not fire. This is the false early stop the study exists to measure.
        _, status = detect_plateau(make_curve(np.arange(60, dtype=float)), SPEC)
        assert status == "no_plateau"

    def test_a_curve_shorter_than_the_detector_needs_is_too_short(self):
        _, status = detect_plateau(make_curve(decaying(6)), SPEC)
        assert status == "too_short"

    def test_a_constant_signal_is_reported_as_constant_not_as_a_plateau(self):
        # Calling a signal with no rate of change "plateaued at step 0" would be the
        # study's worst false positive. Note the scope: this catches curves that are
        # constant to within arithmetic, which is *not* the same as a saturated grid --
        # that one wobbles with the per-window sample count and comes back as an ordinary
        # plateau/no_plateau, to be caught by occupancy_max instead.
        _, status = detect_plateau(make_curve(np.ones(60)), SPEC)
        assert status == "constant_signal"

    @pytest.mark.parametrize("smooth_window", [1, 5, 9])
    def test_constant_status_does_not_depend_on_how_the_curve_was_smoothed(self, smooth_window):
        # -60.607334 is the frozen MiniGrid-Empty kNN curve. It is not exactly
        # representable, so under the old exact test it was constant_signal at
        # smooth_window=1 and no_plateau at 5 and 9 -- the same curve filed under two
        # statuses depending on the bit pattern of its value.
        curve = make_curve(np.full(60, -60.607334), signal="sve_knn", k=4)
        _, status = detect_plateau(curve, DetectorSpec(smooth_window=smooth_window))
        assert status == "constant_signal"

    def test_a_saturated_grid_curve_is_not_mistaken_for_a_constant_one(self):
        # Pinned to log N with N wobbling by a few states per window: real movement, far
        # above the arithmetic floor. It has to reach the detector so that occupancy_max
        # is what identifies it, and the status has to stay a statement about the curve.
        rng = np.random.default_rng(0)
        values = np.log(5_000 + rng.integers(-3, 4, 80).astype(np.float64))
        _, status = detect_plateau(make_curve(values, signal="sve_grid", bins_per_dim=20), SPEC)
        assert status in ("plateau", "no_plateau")

    def test_a_curve_with_a_nan_is_reported_as_non_finite(self):
        # knn_entropy returns NaN for windows with <= k samples.
        values = decaying()
        values[10] = np.nan
        _, status = detect_plateau(make_curve(values), SPEC)
        assert status == "non_finite"

    def test_an_empty_curve_is_reported_as_empty(self):
        assert detect_plateau(make_curve([]), SPEC) == (None, "empty")

    def test_no_data_condition_raises(self):
        # The whole point: one unmeasurable cell must not take out the rest of the sweep.
        for values in ([], np.ones(60), decaying(6), np.full(60, np.nan)):
            assert detect_plateau(make_curve(values), SPEC)[1] != "plateau"


class TestBuildCurves:
    def test_every_signal_lands_on_the_same_step_axis(self):
        # The shared grid is what makes t_plateau comparable between RND and SVE; if the
        # curves ever drift onto different axes, part of every Delta becomes an artefact of
        # how each curve was built.
        curves = build_curves(make_stream(), window=400, stride=200, bin_counts=(5, 10))
        axes = {curve.label: curve.steps.tobytes() for curve in curves}
        assert len(set(axes.values())) == 1

    def test_builds_one_grid_and_one_occupancy_curve_per_bin_count(self):
        curves = build_curves(make_stream(), window=400, stride=200, bin_counts=(5, 10, 20))
        signals = pd.Series([curve.signal for curve in curves]).value_counts()
        assert signals["sve_grid"] == 3
        assert signals["occupancy"] == 3
        assert signals["sve_knn"] == 1
        assert signals["rnd"] == 1

    def test_a_second_mode_adds_sve_curves_but_not_a_second_rnd_curve(self):
        # RND is sliding-only by construction; a cumulative variant would be a lagging
        # integral of the question being asked.
        curves = build_curves(
            make_stream(), window=400, stride=200, bin_counts=(5,), modes=("sliding", "cumulative")
        )
        assert sum(curve.signal == "rnd" for curve in curves) == 1
        assert {curve.mode for curve in curves if curve.signal == "sve_grid"} == {
            "sliding",
            "cumulative",
        }

    def test_a_second_clip_adds_grid_curves_but_not_knn_or_rnd_curves(self):
        # clip only means anything to the estimator that lays a bounded grid over the
        # observations; kNN is explicitly not clipped and RND never sees the parameter.
        curves = build_curves(
            make_stream(), window=400, stride=200, bin_counts=(5,), clips=(3.0, 5.0)
        )
        counts = pd.Series([curve.signal for curve in curves]).value_counts()
        assert counts["sve_grid"] == 2
        assert counts["occupancy"] == 2
        assert counts["sve_knn"] == 1
        assert counts["rnd"] == 1
        assert len({curve.label for curve in curves}) == len(curves)

    def test_seeds_sweep_only_the_estimators_that_use_one(self):
        # A grid histogram is deterministic given the stream, so building it once per seed
        # would be the same numbers at several times the cost.
        curves = build_curves(
            make_stream(), window=400, stride=200, bin_counts=(5,), seeds=(0, 1, 2)
        )
        counts = pd.Series([curve.signal for curve in curves]).value_counts()
        assert counts["rnd"] == 3
        assert counts["sve_knn"] == 3
        assert counts["sve_grid"] == 1
        assert {curve.analysis_seed for curve in curves if curve.signal == "rnd"} == {0, 1, 2}
        assert all(curve.analysis_seed is None for curve in curves if curve.signal == "sve_grid")

    def test_the_seed_actually_moves_the_rnd_curve(self):
        # The premise of recording it. If two target draws gave the same curve there would
        # be nothing for the column to explain.
        curves = build_curves(make_stream(), window=400, stride=200, bin_counts=(5,), seeds=(0, 1))
        first, second = (curve for curve in curves if curve.signal == "rnd")
        assert not np.allclose(first.values, second.values)

    def test_rnd_overrides_are_recorded_so_two_frames_cannot_be_confused(self):
        curves = build_curves(
            make_stream(), window=400, stride=200, bin_counts=(5,), rnd_kwargs={"lr": 1e-3}
        )
        rnd = next(curve for curve in curves if curve.signal == "rnd")
        assert rnd.rnd_overrides == '{"lr": 0.001}'
        assert next(c for c in curves if c.signal == "sve_grid").rnd_overrides == ""


class TestAnalyseRun:
    def test_one_row_per_signal_and_detector_setting(self):
        curves = build_curves(make_stream(), window=400, stride=200, bin_counts=(5, 10))
        detectors = detector_grid(taus=(0.1,), smooth_windows=(5,), reference_quantiles=(0.5,))
        rows = analyse_run(
            make_stream(),
            make_eval_curve(np.linspace(0, 10, 12)),
            make_meta(),
            curves=curves,
            detectors=detectors,
        )
        # Four detectable signals: rnd, two grid bin counts, knn. Occupancy is excluded.
        assert len(rows) == 4 * len(detectors)
        assert "occupancy" not in {row["signal"] for row in rows}

    def test_occupancy_is_attached_to_the_grid_rows_it_diagnoses(self):
        rows = analyse_run(
            make_stream(),
            make_eval_curve(np.linspace(0, 10, 12)),
            make_meta(),
            detectors=[SPEC],
            bin_counts=(5,),
        )
        by_signal = {row["signal"]: row for row in rows}
        assert by_signal["sve_grid"]["occupancy_max"] is not None
        # Nothing to diagnose for an estimator that does not discretize.
        assert by_signal["sve_knn"]["occupancy_max"] is None
        assert by_signal["rnd"]["occupancy_max"] is None

    def test_occupancy_is_paired_by_clip_as_well_as_by_bin_count(self):
        # Two clips over the same bin count are two different grids, so they saturate at
        # different rates. Keying the pairing on bins alone would give both grid rows
        # whichever occupancy curve happened to be built last.
        rows = analyse_run(
            make_stream(dim=3),
            make_eval_curve(np.linspace(0, 10, 12)),
            make_meta(),
            detectors=[SPEC],
            bin_counts=(20,),
            clips=(0.5, 5.0),
        )
        grid = {row["clip"]: row for row in rows if row["signal"] == "sve_grid"}
        assert set(grid) == {0.5, 5.0}
        # A tight clip folds everything into fewer cells, so it cannot occupy more of them.
        assert grid[0.5]["occupancy_max"] < grid[5.0]["occupancy_max"]

    def test_passing_curves_and_build_settings_together_is_refused(self):
        # The settings would be silently dropped, leaving a frame whose columns describe a
        # sweep that never ran.
        curves = build_curves(make_stream(), window=400, stride=200, bin_counts=(5,))
        with pytest.raises(TypeError, match="mutually exclusive"):
            analyse_run(
                make_stream(),
                make_eval_curve(np.linspace(0, 10, 12)),
                make_meta(),
                curves=curves,
                detectors=[SPEC],
                bin_counts=(10,),
            )

    def test_measuring_at_another_resolution_is_refused(self):
        with pytest.raises(TypeError, match="RunMeta"):
            analyse_run(
                make_stream(),
                make_eval_curve(np.linspace(0, 10, 12)),
                make_meta(),
                detectors=[SPEC],
                window=400,
                stride=200,
            )

    def test_convergence_is_computed_once_per_smoothing_not_once_per_row(self):
        rows = analyse_run(
            make_stream(),
            # Rises, then holds: a curve that actually converges, so t_conv is a step
            # rather than None and the assertion below has something to compare.
            make_eval_curve(np.concatenate([np.linspace(0, 10, 6), np.full(6, 10.0)])),
            make_meta(),
            detectors=detector_grid(
                taus=(0.1, 0.2), smooth_windows=(5,), reference_quantiles=(0.5,)
            ),
            bin_counts=(5,),
        )
        frame = pd.DataFrame(rows)
        # t_conv depends on conv_smooth_window alone, never on the plateau settings.
        # dropna=False so that "every row is None" counts as one value rather than none.
        assert frame.groupby("conv_smooth_window").t_conv.nunique(dropna=False).max() == 1
        assert frame.t_conv.notna().any()


class TestFrozenControl:
    """The control is excluded from Delta by its flag, never by its convergence status."""

    def evaluate(self, frozen):
        # A flat curve well above the random baseline: exactly the frozen CartPole pilot,
        # which reports `converged` because a constant policy trivially holds above its own
        # threshold for the patience window.
        return analyse_run(
            make_stream(),
            make_eval_curve(np.full(12, 168.0), random_return=20.7),
            make_meta(frozen=frozen, tag="frozen" if frozen else None),
            detectors=[SPEC],
            bin_counts=(5,),
        )

    def test_the_control_would_otherwise_be_reported_as_converged(self):
        # Guards the premise of the test below: if this ever stops holding, the exclusion
        # is no longer load-bearing and this whole class is testing nothing.
        assert all(row["conv_status"] == "converged" for row in self.evaluate(frozen=False))

    def test_a_frozen_run_yields_no_delta_and_no_convergence_claim(self):
        rows = self.evaluate(frozen=True)
        assert rows
        assert all(row["conv_status"] == "control" for row in rows)
        assert all(row["t_conv"] is None for row in rows)
        assert all(row["delta"] is None for row in rows)
        assert all(row["retained"] is None for row in rows)

    def test_a_frozen_run_still_reports_its_plateau(self):
        # The control's whole purpose: does the signal flatten anyway, with no learning to
        # explain it? That question is answered by t_plateau on its own.
        rows = self.evaluate(frozen=True)
        assert all(row["plateau_status"] for row in rows)
        assert all(row["n_points"] > 0 for row in rows)


class TestDiscoveryAndIo:
    def write_run(self, directory, tag=None, complete=True):
        stem = directory / run_stem("CartPole-v1", 0, tag)
        save_run_meta(f"{stem}.run.json", make_meta(tag=tag))
        if complete:
            save_state_stream(f"{stem}.states.npz", make_stream(n=100))
            save_eval_curve(f"{stem}.evals.npz", make_eval_curve(np.linspace(0, 1, 12)))
        return stem

    def test_finds_completed_runs(self, tmp_path):
        self.write_run(tmp_path)
        self.write_run(tmp_path, tag="frozen")
        found = find_runs(tmp_path)
        assert [meta.tag for _, meta in found] == ["frozen", None]

    def test_skips_a_run_that_died_before_writing_its_stream(self, tmp_path):
        # A sidecar without a stream is a job that was killed mid-write. Analysing it as if
        # it were complete would put a partial run into the table.
        self.write_run(tmp_path, complete=False)
        assert find_runs(tmp_path) == []

    def test_curves_survive_a_round_trip_with_their_settings(self, tmp_path):
        curves = build_curves(
            make_stream(),
            window=400,
            stride=200,
            bin_counts=(5, 20),
            clips=(3.0, 5.0),
            seeds=(0, 1),
        )
        loaded = load_curves(save_curves(tmp_path / "run.npz", curves))

        assert [curve.label for curve in loaded] == [curve.label for curve in curves]
        assert [curve.params for curve in loaded] == [curve.params for curve in curves]
        for before, after in zip(curves, loaded):
            assert np.array_equal(before.steps, after.steps)
            assert np.allclose(before.values, after.values)


class TestAnalyseDirectory:
    def write_run(self, directory, seed=0, tag=None):
        stem = directory / run_stem("CartPole-v1", seed, tag)
        save_run_meta(f"{stem}.run.json", make_meta(seed=seed, tag=tag, frozen=tag == "frozen"))
        save_state_stream(f"{stem}.states.npz", make_stream(n=2_000, seed=seed))
        save_eval_curve(f"{stem}.evals.npz", make_eval_curve(converging()))
        return stem

    def analyse(self, tmp_path, **kwargs):
        self.write_run(tmp_path)
        self.write_run(tmp_path, tag="frozen")
        return analyse_directory(tmp_path, detectors=[SPEC], bin_counts=(5,), **kwargs)

    def test_step_counts_stay_integers_despite_the_missing_ones(self, tmp_path):
        # The frozen control has no t_conv, and one None is enough to widen a plain int
        # column to float -- at which point every step count in the report reads "20480.0".
        frame, _ = self.analyse(tmp_path)
        for column in ("t_conv", "t_plateau", "delta"):
            assert frame[column].dtype == "Int64"
        assert frame.t_conv.isna().any()
        assert frame.t_conv.notna().any()

    def test_both_callbacks_fire_once_per_run_in_order(self, tmp_path):
        # on_run_start exists because the wait is inside a run, not between runs: a
        # notification that only arrives on completion arrives once the waiting is over.
        events = []
        frame, curves_by_run = self.analyse(
            tmp_path,
            on_run_start=lambda stem, meta: events.append(("start", stem.name)),
            on_run=lambda stem, meta, rows, curves: events.append(("done", stem.name)),
        )
        assert [phase for phase, _ in events] == ["start", "done", "start", "done"]
        assert set(curves_by_run) == {name for _, name in events}

    def test_the_callback_gets_the_curves_so_they_can_be_written_as_they_land(self, tmp_path):
        written = {}
        self.analyse(
            tmp_path,
            on_run=lambda stem, meta, rows, curves: written.update({stem.name: len(curves)}),
        )
        # rnd + sve_grid + occupancy + sve_knn at one bin count.
        assert set(written.values()) == {4}

    def test_an_empty_directory_gives_an_empty_frame_rather_than_raising(self, tmp_path):
        frame, curves_by_run = analyse_directory(tmp_path, detectors=[SPEC])
        assert frame.empty
        assert curves_by_run == {}


@pytest.fixture(scope="module")
def script():
    """``scripts/analyze.py`` as an importable module, so main() can be called directly."""
    spec = importlib.util.spec_from_file_location("analyze_script", ANALYZE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def results(tmp_path):
    """A results directory holding one trained run and its frozen control."""
    directory = tmp_path / "results"
    directory.mkdir()
    for tag in (None, "frozen"):
        stem = directory / run_stem("CartPole-v1", 0, tag)
        save_run_meta(f"{stem}.run.json", make_meta(tag=tag, frozen=tag == "frozen"))
        save_state_stream(f"{stem}.states.npz", make_stream(n=2_000))
        save_eval_curve(f"{stem}.evals.npz", make_eval_curve(converging()))
    return directory


class TestAnalyzeScript:
    # Two runs, one bin count: rnd + sve_grid + sve_knn each, occupancy excluded.
    ROWS_PER_SETTING = 2 * 3

    def analyse(self, script, results, out, *extra):
        return script.main(["--results", str(results), "--out", str(out), "--bins", "5", *extra])

    def test_writes_the_frame_and_one_archive_per_run(self, script, results, tmp_path):
        out = tmp_path / "out"
        assert self.analyse(script, results, out, "--quick") == 0

        frame = pd.read_csv(out / "signals.csv")
        assert len(frame) == self.ROWS_PER_SETTING
        assert sorted(p.name for p in (out / "curves").glob("*.npz")) == [
            "CartPole-v1__frozen__seed0.npz",
            "CartPole-v1__seed0.npz",
        ]

    def test_the_written_frame_keeps_step_counts_as_integers(self, script, results, tmp_path):
        # The CSV is what the report and the figures read, and one missing t_conv from the
        # frozen control is enough to write every step count in the column as "4000.0".
        out = tmp_path / "out"
        self.analyse(script, results, out, "--quick")
        t_conv = pd.read_csv(out / "signals.csv", dtype={"t_conv": str}).t_conv.dropna()

        assert not t_conv.empty
        assert not t_conv.str.contains(".", regex=False).any()

    def test_reports_a_missing_results_directory_instead_of_crashing(self, script, tmp_path):
        assert script.main(["--results", str(tmp_path / "nothing"), "--out", str(tmp_path)]) == 1

    def test_quick_really_is_a_single_detector_setting(self, script):
        # The help text says so, and a --quick that quietly ran six settings would make the
        # digest a selection rather than the whole frame.
        assert len(script.quick_grid()) == 1

    def test_the_digest_reads_at_a_smoothed_eval_curve(self, script, results, tmp_path):
        # conv_smooth_window=1 is in the sweep but must not be what the console offers as
        # the headline: on an all-or-nothing greedy return it makes `retained` a
        # single-evaluation reading.
        out = tmp_path / "out"
        self.analyse(script, results, out)
        frame = pd.read_csv(out / "signals.csv")

        assert 1 in set(frame.conv_smooth_window)
        assert script.digest_setting(frame)["conv_smooth_window"] > 1

    def test_the_digest_names_its_setting_and_selects_rows(self, script, results, tmp_path):
        out = tmp_path / "out"
        self.analyse(script, results, out)
        digest = script.summarise(pd.read_csv(out / "signals.csv"))

        assert "showing the whole frame" not in digest
        assert "conv_smooth_window=" in digest.splitlines()[0]
        # One line naming the setting, one of column names, then the selected rows.
        assert len(digest.splitlines()) == 2 + self.ROWS_PER_SETTING

    def test_a_sweep_without_the_default_setting_says_so_rather_than_misreporting(self, script):
        # The fallback prints everything. It has to announce that, or the reader takes a
        # 400-row dump for the default-setting digest.
        frame = pd.DataFrame(
            {
                "tau": [0.9],
                "smooth_window": [7],
                "reference_quantile": [0.5],
                "conv_smooth_window": [1],
                "env_id": ["CartPole-v1"],
                "seed": [0],
                "frozen": [False],
                "label": ["rnd_s0_sliding"],
                "n_points": [20],
                "t_conv": [1000],
                "conv_status": ["converged"],
                "t_plateau": [2000],
                "plateau_status": ["plateau"],
                "delta": [1000],
                "retained": [1.0],
                "occupancy_max": [None],
            }
        )
        assert "showing the whole frame" in script.summarise(frame)


def test_status_counts_cover_both_columns():
    frame = pd.DataFrame(
        {
            "conv_status": ["converged", "converged", "not_learned"],
            "plateau_status": ["plateau", "no_plateau", "constant_signal"],
        }
    )
    counts = {(column, status): count for column, status, count in iter_status_counts(frame)}
    assert counts[("conv_status", "converged")] == 2
    assert counts[("plateau_status", "constant_signal")] == 1
