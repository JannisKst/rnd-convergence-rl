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
"""

import numpy as np
import pandas as pd
import pytest

from rnd_convergence.analysis import (
    Curve,
    DetectorSpec,
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
        # The saturated-grid case: grid SVE pinned to log N is arithmetic about sample
        # counts. Calling that "plateaued at step 0" would be the study's worst false
        # positive, and calling it "never plateaued" would hide it entirely.
        _, status = detect_plateau(make_curve(np.ones(60)), SPEC)
        assert status == "constant_signal"

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
        curves = build_curves(make_stream(), window=400, stride=200, bin_counts=(5, 20))
        loaded = load_curves(save_curves(tmp_path / "run.npz", curves))

        assert [curve.label for curve in loaded] == [curve.label for curve in curves]
        assert [curve.params for curve in loaded] == [curve.params for curve in curves]
        for before, after in zip(curves, loaded):
            assert np.array_equal(before.steps, after.steps)
            assert np.allclose(before.values, after.values)


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
