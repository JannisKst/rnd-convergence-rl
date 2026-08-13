"""Run-level metadata, and the resolution check that guards a run before it is spent.

:mod:`~rnd_convergence.streams` owns the two artefacts a run produces. This module owns
what has to be said *about* a run and does not fit in either of them:

``RunMeta``
    A sidecar written next to the ``.npz`` files. It records the settings the offline
    analysis needs but that the artefacts themselves cannot carry --- the measurement
    resolution the run was sized for, and whether it is the frozen-policy control. Reading
    it beats parsing filenames, and it keeps the ``.npz`` schema frozen.

``check_resolution``
    The training budget and the analysis resolution are jointly constrained, and both
    detectors fail *quietly* when the constraint is missed: too few evaluation points and
    ``t_conv`` is unresolvable, too few signal points and ``plateau_time`` cannot fire.
    Either way the result is a blank cell in the table discovered after the compute has
    been spent. This runs in milliseconds, before the first environment step.

    It guards the window/stride ratio for the same reason. That ratio is a property of the
    study rather than of a run --- it decides the time scale ``plateau_time`` responds to,
    and ``Delta`` is compared across rungs and between signals --- but it reaches a run
    through the config, where a CLI override sails straight past the tests that pin it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RUN_META_SUFFIX = ".run.json"
UPDATE_LOG_SUFFIX = ".updates.csv"

# The run_stem tag the frozen-policy control is written under. Named here rather than
# spelled out at the two call sites that have to agree on it: the control and the trained
# run it is compared against differ only by this string in their filenames.
FROZEN_TAG = "frozen"

# The signal curves want far more points than plateau_time's hard floor of
# min_points + patience. At the defaults that floor is 10 points, at which the detector
# can only ever answer at the very last one; 40 is where a plateau is actually locatable
# rather than merely representable.
MIN_SIGNAL_POINTS = 40

# convergence_time needs `patience` consecutive points above the threshold and reads its
# reference off the final `n_final`. Twenty points is the level below which t_conv is
# quantised too coarsely for Delta to mean anything.
MIN_EVAL_POINTS = 20

# signal_window / signal_stride. plateau_time reads a plateau off the differences between
# consecutive curve points, so the overlap between their windows is what decides how much
# new data separates them --- it sets the time scale the detector responds to. Delta is
# compared both across rungs and between RND and SVE, so the ratio has to be one number for
# the whole study; which number matters far less than that it does not vary. Enforced here
# rather than only in the configs because a CLI override reaches the run and not the tests.
WINDOW_STRIDE_RATIO = 2


@dataclass(frozen=True)
class RunMeta:
    """What the offline analysis needs to know about a completed run.

    Parameters
    ----------
    env_id
        Gymnasium id (or ``MarsRover``) --- always a real environment identifier, so the
        environment can be rebuilt from it without stripping a condition suffix.
    seed, tag
        Together with ``env_id`` these form the file stem; see
        :func:`~rnd_convergence.streams.run_stem`.
    frozen
        True for the frozen-policy control. Such a run has no ``t_conv`` by construction,
        so it is reported as a ``t_plateau`` on its own rather than as a ``Delta``.
    total_steps, eval_interval, eval_episodes, rollout_steps
        The training budget as configured. ``eval_interval`` is the requested grid; the
        realised spacing is ``max(eval_interval, rollout_steps)``.
    signal_window, signal_stride
        The resolution the RND and entropy curves are to be measured at. Carried here
        because it is a property of how the run was *sized*, not a free choice at
        analysis time --- picking a different stride afterwards silently changes every
        ``t_plateau``.
    n_states, wall_clock_seconds
        Recorded for triage and for sizing the full matrix off the pilots.
    random_episodes
        Episodes averaged for the random-policy baseline ``R_0``. Belongs with the other
        measurement settings: ``R_0`` sets the convergence threshold, so a run measured
        against a noisier baseline has a noisier ``t_conv``.
    agent
        The PPO hyperparameters the run was trained with. Hydra already writes these to
        ``.hydra/config.yaml``, but under its own log directory rather than next to the
        artefacts, linked to them only by wall-clock timestamp --- and that directory is
        keyed to the second, so a job array launching tasks simultaneously can land two of
        them in the same one. Copying the values here makes the sidecar self-contained,
        which is the whole reason it exists.

    ``random_episodes`` and ``agent`` carry defaults so that sidecars written before they
    were added still load; an empty ``agent`` means "not recorded", not "no hyperparameters".
    """

    env_id: str
    seed: int
    tag: str | None
    frozen: bool
    total_steps: int
    eval_interval: int
    eval_episodes: int
    rollout_steps: int
    signal_window: int
    signal_stride: int
    n_states: int
    wall_clock_seconds: float
    random_episodes: int | None = None
    agent: dict[str, Any] = field(default_factory=dict)


def save_run_meta(path: str | Path, meta: RunMeta) -> Path:
    """Write a :class:`RunMeta` sidecar as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(meta), indent=2) + "\n")
    return path


def load_run_meta(path: str | Path) -> RunMeta:
    """Read a :class:`RunMeta` written by :func:`save_run_meta`."""
    return RunMeta(**json.loads(Path(path).read_text()))


def check_resolution(
    *,
    total_steps: int,
    eval_interval: int,
    rollout_steps: int,
    signal_window: int,
    signal_stride: int,
    min_eval_points: int = MIN_EVAL_POINTS,
    min_signal_points: int = MIN_SIGNAL_POINTS,
    window_stride_ratio: int | None = WINDOW_STRIDE_RATIO,
) -> tuple[int, int]:
    """Verify a run will yield enough points to measure, and return how many of each.

    Returns ``(eval_points, signal_points)``. Raises :class:`ValueError` naming the setting
    to change when either count is too coarse, or when ``signal_window`` and
    ``signal_stride`` are not in the ratio the whole study is measured at.

    ``signal_points`` covers *both* monitoring signals. It is derived from the stride alone
    because :func:`~rnd_convergence.windows.iter_windows` lays out one grid for the RND
    error curve and the SVE curves alike, and the stride is what sets the number of points
    on it. Pass ``window_stride_ratio=None`` to check a run measured at a deliberately
    different overlap; nothing in the normal path does, since the analysis re-measures the
    logged stream offline at whatever resolution a sensitivity sweep wants and does not
    need the training run to have been sized differently.

    The evaluation spacing is ``max(eval_interval, rollout_steps)`` rather than
    ``eval_interval``: evaluation can only run on a rollout boundary, so asking for a grid
    finer than one rollout silently gets one point per rollout instead.
    """
    if min(total_steps, eval_interval, rollout_steps, signal_window, signal_stride) < 1:
        raise ValueError("all resolution settings must be >= 1")
    if signal_window < signal_stride:
        raise ValueError(
            f"signal_window ({signal_window}) is narrower than signal_stride "
            f"({signal_stride}), so consecutive windows would leave gaps of "
            f"{signal_stride - signal_window} steps that no measurement ever covers"
        )
    if window_stride_ratio is not None and signal_window != window_stride_ratio * signal_stride:
        raise ValueError(
            f"signal_window ({signal_window}) must be {window_stride_ratio}x signal_stride "
            f"({signal_stride}), i.e. {window_stride_ratio * signal_stride}. The overlap "
            "between consecutive windows sets the time scale plateau_time responds to, and "
            "Delta is compared across rungs and between signals, so it has to be the same "
            "everywhere; a run measured at a different ratio is not comparable to the rest "
            "of the table"
        )

    eval_spacing = max(eval_interval, rollout_steps)
    eval_points = total_steps // eval_spacing
    signal_points = total_steps // signal_stride

    if eval_points < min_eval_points:
        detail = (
            f"eval_interval={eval_interval} is finer than rollout_steps={rollout_steps}, "
            "so the realised spacing is one evaluation per rollout; lower rollout_steps"
            if rollout_steps > eval_interval
            else f"lower eval_interval (currently {eval_interval})"
        )
        raise ValueError(
            f"{total_steps} steps at a spacing of {eval_spacing} yields {eval_points} "
            f"evaluation points, below the {min_eval_points} needed to resolve t_conv: "
            f"{detail}, or raise total_steps"
        )
    if signal_points < min_signal_points:
        raise ValueError(
            f"{total_steps} steps at signal_stride={signal_stride} yields {signal_points} "
            f"signal points, below the {min_signal_points} needed to locate a plateau: "
            "lower signal_stride, or raise total_steps"
        )
    return eval_points, signal_points
