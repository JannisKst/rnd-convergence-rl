"""Tests for rnd_convergence.runs — run metadata and the pre-flight resolution check.

The shipped-config test at the bottom is the one that matters most. Both detectors fail
*quietly* when a run is sized wrong: too few evaluation points and ``t_conv`` is
unresolvable, too few signal points and ``plateau_time`` cannot fire. The result is a
blank cell in the results table, discovered after the cluster time has been spent. That
check moves the discovery to the test suite, and it globs the config directory so a rung
added later cannot skip it.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import rnd_convergence
from rnd_convergence.runs import (
    MIN_EVAL_POINTS,
    MIN_SIGNAL_POINTS,
    RunMeta,
    check_resolution,
    load_run_meta,
    save_run_meta,
)

CONFIG_DIR = Path(rnd_convergence.__file__).parent / "configs"
ENV_CONFIGS = sorted(path.stem for path in (CONFIG_DIR / "env").glob("*.yaml"))

GOOD = {
    "total_steps": 200_000,
    "eval_interval": 2_000,
    "rollout_steps": 2_048,
    "signal_window": 5_000,
    "signal_stride": 2_000,
}


def make_meta(**overrides):
    fields = {
        "env_id": "CartPole-v1",
        "seed": 3,
        "tag": None,
        "frozen": False,
        "total_steps": 200_000,
        "eval_interval": 2_000,
        "eval_episodes": 10,
        "rollout_steps": 2_048,
        "signal_window": 5_000,
        "signal_stride": 2_000,
        "n_states": 200_000,
        "wall_clock_seconds": 123.5,
    }
    return RunMeta(**(fields | overrides))


class TestRunMeta:
    def test_roundtrip_preserves_contents(self, tmp_path):
        meta = make_meta(tag="frozen", frozen=True)
        assert load_run_meta(save_run_meta(tmp_path / "run.run.json", meta)) == meta

    def test_creates_missing_parent_directories(self, tmp_path):
        path = save_run_meta(tmp_path / "nested" / "deeper" / "run.run.json", make_meta())
        assert path.exists()


class TestCheckResolution:
    def test_accepts_a_well_sized_run_and_reports_the_point_counts(self):
        eval_points, signal_points = check_resolution(**GOOD)
        assert eval_points == 200_000 // 2_048
        assert signal_points == 100

    def test_evaluation_spacing_is_bounded_below_by_the_rollout_length(self):
        # Asking for a grid finer than one rollout does not get one: evaluation can only
        # run on a rollout boundary, so the realised spacing is rollout_steps.
        fine, _ = check_resolution(**(GOOD | {"eval_interval": 10}))
        coarse, _ = check_resolution(**(GOOD | {"eval_interval": 2_048}))
        assert fine == coarse == 200_000 // 2_048

    def test_rejects_too_few_evaluation_points(self):
        with pytest.raises(ValueError, match=f"below the {MIN_EVAL_POINTS} needed"):
            check_resolution(**(GOOD | {"total_steps": 20_000}))

    def test_names_rollout_steps_when_that_is_what_limits_the_grid(self):
        # The actionable fix differs: here eval_interval is already fine and lowering it
        # further would change nothing, so the message has to point at rollout_steps.
        with pytest.raises(ValueError, match="lower rollout_steps"):
            check_resolution(
                **(GOOD | {"total_steps": 40_000, "eval_interval": 500, "rollout_steps": 2_048})
            )

    def test_rejects_too_few_signal_points(self):
        with pytest.raises(ValueError, match=f"below the {MIN_SIGNAL_POINTS} needed"):
            check_resolution(**(GOOD | {"signal_stride": 20_000, "signal_window": 20_000}))

    def test_rejects_a_window_narrower_than_the_stride(self):
        # Consecutive windows would then leave gaps that no measurement ever covers.
        with pytest.raises(ValueError, match="narrower than"):
            check_resolution(**(GOOD | {"signal_window": 1_000}))

    def test_rejects_nonsensical_settings(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            check_resolution(**(GOOD | {"signal_stride": 0}))


class TestShippedConfigs:
    """Every rung config in the repo must compose and be sized to be measurable."""

    @pytest.mark.parametrize("name", ENV_CONFIGS)
    def test_env_config_composes_and_resolves(self, name):
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
            cfg = compose(config_name="train", overrides=[f"env={name}"])
        assert cfg.env.id
        assert cfg.frozen is False
        assert cfg.agent.hidden_size > 0

    @pytest.mark.parametrize("name", ENV_CONFIGS)
    def test_env_config_yields_enough_points_to_measure(self, name):
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
            cfg = compose(config_name="train", overrides=[f"env={name}"])
        check_resolution(
            total_steps=cfg.env.total_steps,
            eval_interval=cfg.env.eval_interval,
            rollout_steps=cfg.env.rollout_steps,
            signal_window=cfg.env.signal_window,
            signal_stride=cfg.env.signal_stride,
        )

    def test_the_ladder_is_covered(self):
        # A missing rung is a silently incomplete results table, so the set is asserted
        # rather than merely iterated over.
        assert set(ENV_CONFIGS) == {
            "marsrover",
            "minigrid_empty",
            "minigrid_doorkey5",
            "minigrid_doorkey8",
            "cartpole",
            "lunarlander",
        }
