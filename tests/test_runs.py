"""Tests for rnd_convergence.runs — run metadata and the pre-flight resolution check.

The shipped-config test at the bottom is the one that matters most. Both detectors fail
*quietly* when a run is sized wrong: too few evaluation points and ``t_conv`` is
unresolvable, too few signal points and ``plateau_time`` cannot fire. The result is a
blank cell in the results table, discovered after the cluster time has been spent. That
check moves the discovery to the test suite, and it globs the config directory so a rung
added later cannot skip it.
"""

import inspect
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import rnd_convergence
from rnd_convergence.ppo import PPOAgent
from rnd_convergence.runs import (
    FROZEN_TAG,
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
        "random_episodes": 20,
        "agent": {"lr_actor": 3e-4, "gamma": 0.99, "device": "cpu"},
    }
    return RunMeta(**(fields | overrides))


class TestRunMeta:
    def test_roundtrip_preserves_contents(self, tmp_path):
        meta = make_meta(tag=FROZEN_TAG, frozen=True)
        assert load_run_meta(save_run_meta(tmp_path / "run.run.json", meta)) == meta

    def test_creates_missing_parent_directories(self, tmp_path):
        path = save_run_meta(tmp_path / "nested" / "deeper" / "run.run.json", make_meta())
        assert path.exists()

    def test_records_the_hyperparameters_the_run_was_trained_with(self, tmp_path):
        # The point of the field: Hydra's copy lives in a different tree, keyed to the
        # second, so a job array can put two simultaneous tasks in one directory. A
        # sidecar that cannot answer "at what learning rate?" is not self-contained.
        meta = make_meta(agent={"lr_actor": 1e-3, "clip_eps": 0.2})
        assert load_run_meta(save_run_meta(tmp_path / "run.run.json", meta)).agent == {
            "lr_actor": 1e-3,
            "clip_eps": 0.2,
        }

    def test_sidecars_written_before_the_newer_fields_still_load(self, tmp_path):
        # Pilot artefacts on disk predate random_episodes and agent; reading them must not
        # require regenerating the runs that produced them.
        path = tmp_path / "old.run.json"
        fields = asdict(make_meta())
        del fields["random_episodes"], fields["agent"]
        path.write_text(json.dumps(fields))

        meta = load_run_meta(path)
        assert meta.random_episodes is None
        assert meta.agent == {}


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


def compose_rung(name):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="train", overrides=[f"env={name}"])


class TestShippedConfigs:
    """Every rung config in the repo must compose and be sized to be measurable."""

    @pytest.mark.parametrize("name", ENV_CONFIGS)
    def test_env_config_composes_and_resolves(self, name):
        cfg = compose_rung(name)
        assert cfg.env.id
        assert cfg.frozen is False
        assert cfg.agent.hidden_size > 0

    @pytest.mark.parametrize("name", ENV_CONFIGS)
    def test_train_yaml_keys_the_entry_point_reads_are_all_present(self, name):
        # scripts/train.py is a Hydra entry point, so a key renamed in train.yaml raises
        # nowhere until a run is actually launched -- on the cluster, after the queue wait.
        cfg = compose_rung(name)
        assert cfg.out_dir
        assert cfg.random_episodes > 0
        assert cfg.run_tag is None
        for key in ("total_steps", "eval_interval", "eval_episodes", "rollout_steps"):
            assert cfg.env[key] > 0

    def test_agent_config_matches_the_ppo_agent_signature(self):
        # train.py splats cfg.agent into PPOAgent. An unrecognised key is a TypeError at
        # the first environment step rather than at compose time, so pin the contract here.
        cfg = compose_rung(ENV_CONFIGS[0])
        accepted = set(inspect.signature(PPOAgent.__init__).parameters) - {"self", "env_fn"}
        assert set(cfg.agent) <= accepted, f"unknown agent keys: {set(cfg.agent) - accepted}"

    @pytest.mark.parametrize("name", ENV_CONFIGS)
    def test_env_config_yields_enough_points_to_measure(self, name):
        cfg = compose_rung(name)
        check_resolution(
            total_steps=cfg.env.total_steps,
            eval_interval=cfg.env.eval_interval,
            rollout_steps=cfg.env.rollout_steps,
            signal_window=cfg.env.signal_window,
            signal_stride=cfg.env.signal_stride,
        )

    @pytest.mark.parametrize("name", ENV_CONFIGS)
    def test_window_overlap_is_the_same_on_every_rung(self, name):
        # check_resolution counts curve points, which is not the same as counting
        # independent ones. plateau_time differences consecutive points, so the window
        # overlap sets how much new data separates them -- and Delta is compared *across*
        # rungs, so a ratio that varied by rung would put the detector's effective time
        # scale into the dimensionality axis the table exists to vary.
        cfg = compose_rung(name)
        assert cfg.env.signal_window == 2 * cfg.env.signal_stride

    def test_the_ladder_is_covered(self):
        # A missing rung is a silently incomplete results table, so the committed rungs are
        # asserted rather than merely iterated over. Subset, not equality: minigrid_doorkey5
        # is a reserve variant the pilot is meant to choose between keeping and dropping,
        # and acting on that evidence should not have to edit this test.
        assert {
            "marsrover",
            "minigrid_empty",
            "minigrid_doorkey8",
            "cartpole",
            "lunarlander",
        } <= set(ENV_CONFIGS)
