"""End-to-end tests for scripts/train.py, the entry point that spends the cluster time.

Everything else in the suite tests a library function. This tests the seam those functions
are wired together across, which is where a Hydra entry point fails: a renamed config key,
a kwarg the agent no longer takes or an artefact written under the wrong stem raises
nothing until a run is actually launched, and by then the job has queued, trained and
exited non-zero for a reason no log explains.

The run below is a real one --- MarsRover, sized down to the smallest budget
``check_resolution`` will accept --- because the parts worth testing here are exactly the
ones a mocked agent would stub out.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

import rnd_convergence
from rnd_convergence.runs import RUN_META_SUFFIX, UPDATE_LOG_SUFFIX, load_run_meta
from rnd_convergence.streams import (
    EVAL_CURVE_SUFFIX,
    STATE_STREAM_SUFFIX,
    load_eval_curve,
    load_state_stream,
)

CONFIG_DIR = Path(rnd_convergence.__file__).parent / "configs"
TRAIN_SCRIPT = Path(rnd_convergence.__file__).parent.parent / "scripts" / "train.py"

# The smallest MarsRover run that still satisfies check_resolution: 2000 steps at a
# spacing of 50 is 40 evaluation points and 40 signal points, both exactly at the floor.
SMALL_RUN = [
    "env=marsrover",
    "env.total_steps=2000",
    "env.eval_interval=50",
    "env.eval_episodes=2",
    "env.rollout_steps=50",
    "env.signal_window=100",
    "env.signal_stride=50",
    "random_episodes=2",
]


@pytest.fixture(scope="module")
def train_main():
    """The undecorated ``main``, so it can be called with a composed config."""
    spec = importlib.util.spec_from_file_location("train_script", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main.__wrapped__


def run(train_main, tmp_path, *overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[*SMALL_RUN, f"out_dir={tmp_path}", *overrides],
        )
    return train_main(cfg)


class TestTrainScript:
    def test_writes_all_four_artefacts_and_they_load_back(self, train_main, tmp_path):
        run(train_main, tmp_path)
        stem = "MarsRover__seed0"

        stream = load_state_stream(tmp_path / f"{stem}{STATE_STREAM_SUFFIX}")
        curve = load_eval_curve(tmp_path / f"{stem}{EVAL_CURVE_SUFFIX}")
        meta = load_run_meta(tmp_path / f"{stem}{RUN_META_SUFFIX}")

        assert stream.n_steps == 2000
        assert stream.env_id == "MarsRover" == curve.env_id == meta.env_id
        assert curve.steps.size >= 20
        assert (tmp_path / f"{stem}{UPDATE_LOG_SUFFIX}").exists()

    def test_run_meta_records_the_hyperparameters_and_measurement_settings(
        self, train_main, tmp_path
    ):
        # The sidecar's reason to exist: enough to reconstruct the run without the Hydra
        # log directory, which lives in a different tree and is keyed only to the second.
        run(train_main, tmp_path, "agent.lr_actor=1e-3")
        meta = load_run_meta(tmp_path / f"MarsRover__seed0{RUN_META_SUFFIX}")

        assert meta.agent["lr_actor"] == 1e-3
        assert meta.agent["gamma"] == 0.99
        assert meta.random_episodes == 2
        assert (meta.signal_window, meta.signal_stride) == (100, 50)
        assert meta.frozen is False and meta.tag is None
        assert meta.wall_clock_seconds > 0

    def test_frozen_control_is_tagged_so_it_cannot_overwrite_the_trained_run(
        self, train_main, tmp_path
    ):
        run(train_main, tmp_path)
        run(train_main, tmp_path, "frozen=true")

        trained = load_run_meta(tmp_path / f"MarsRover__seed0{RUN_META_SUFFIX}")
        frozen = load_run_meta(tmp_path / f"MarsRover__frozen__seed0{RUN_META_SUFFIX}")
        assert trained.frozen is False and frozen.frozen is True
        assert frozen.tag == "frozen"

        # The control's update log is the one place its premise is visible: nan losses
        # because no gradient step was taken, and a live entropy column near log 2 showing
        # the unupdated policy really did stay near-uniform over the whole run.
        log = pd.read_csv(tmp_path / f"MarsRover__frozen__seed0{UPDATE_LOG_SUFFIX}")
        assert log["policy_loss"].isna().all()
        assert np.allclose(log["entropy"], np.log(2), rtol=1e-2)

    def test_rejects_a_run_tag_that_would_clobber_the_frozen_control(self, train_main, tmp_path):
        with pytest.raises(ValueError, match="reserved for the frozen-policy control"):
            run(train_main, tmp_path, "run_tag=frozen")

    def test_refuses_to_start_a_run_it_could_not_measure(self, train_main, tmp_path):
        # The check has to fire *before* training, so nothing may be written.
        with pytest.raises(ValueError, match="below the .* needed"):
            run(train_main, tmp_path, "env.signal_stride=500", "env.signal_window=1000")
        assert list(tmp_path.iterdir()) == []

    def test_seed_reaches_both_the_filename_and_the_artefacts(self, train_main, tmp_path):
        run(train_main, tmp_path, "seed=3")
        meta = json.loads((tmp_path / f"MarsRover__seed3{RUN_META_SUFFIX}").read_text())
        assert meta["seed"] == 3
        assert load_state_stream(tmp_path / f"MarsRover__seed3{STATE_STREAM_SUFFIX}").seed == 3
