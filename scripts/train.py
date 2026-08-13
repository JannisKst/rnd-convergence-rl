"""Training entry point: one PPO run on one rung, written to disk for offline analysis.

Training is the only expensive step in this project. Every signal --- RND prediction
error, every entropy variant, every detector threshold --- is recomputed later from the
artefacts this script writes, so a run is never repeated to try a different analysis.

    python scripts/train.py env=cartpole seed=0
    python scripts/train.py -m env=cartpole seed=0,1,2,3,4      # one run per seed
    python scripts/train.py env=cartpole frozen=true            # frozen-policy control

Writes four files per run to ``out_dir``, keyed by
:func:`~rnd_convergence.streams.run_stem`:

    <stem>.states.npz    the visited-state stream
    <stem>.evals.npz     the evaluation-return curve plus the random-policy baseline
    <stem>.run.json      how the run was configured and how long it took
    <stem>.updates.csv   per-update losses, for triaging a run that went wrong

The last one is diagnostics rather than data: a run in a job array that produces a flat
return curve is otherwise impossible to explain after the fact, and a collapsed policy
entropy and a diverging value loss say very different things about why.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from rnd_convergence.convergence import convergence_report
from rnd_convergence.envs import make_env
from rnd_convergence.ppo import PPOAgent
from rnd_convergence.runs import (
    RUN_META_SUFFIX,
    UPDATE_LOG_SUFFIX,
    RunMeta,
    check_resolution,
    save_run_meta,
)
from rnd_convergence.streams import (
    EVAL_CURVE_SUFFIX,
    STATE_STREAM_SUFFIX,
    run_stem,
    save_eval_curve,
    save_state_stream,
)
from rnd_convergence.utils import set_seed

logger = logging.getLogger(__name__)


@hydra.main(config_path="../rnd_convergence/configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> float:
    """Train one agent and write its artefacts. Returns the final mean evaluation return."""
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Before the first environment step: a run sized so that t_conv or t_plateau cannot be
    # resolved is worse than a run that fails, because it fails silently and only shows up
    # as a blank cell in the results table.
    eval_points, signal_points = check_resolution(
        total_steps=cfg.env.total_steps,
        eval_interval=cfg.env.eval_interval,
        rollout_steps=cfg.env.rollout_steps,
        signal_window=cfg.env.signal_window,
        signal_stride=cfg.env.signal_stride,
    )
    logger.info(
        "Resolution: ~%d evaluation points, ~%d signal points at stride %d",
        eval_points,
        signal_points,
        cfg.env.signal_stride,
    )

    set_seed(cfg.seed)
    env_kwargs = OmegaConf.to_container(cfg.env.kwargs, resolve=True) or {}

    def env_fn():
        return make_env(cfg.env.id, seed=cfg.seed, **env_kwargs)

    agent = PPOAgent(
        env_fn,
        seed=cfg.seed,
        rollout_steps=cfg.env.rollout_steps,
        **OmegaConf.to_container(cfg.agent, resolve=True),
    )

    started = time.perf_counter()
    stream, curve = agent.train(
        total_steps=cfg.env.total_steps,
        eval_interval=cfg.env.eval_interval,
        eval_episodes=cfg.env.eval_episodes,
        random_episodes=cfg.random_episodes,
        env_id=cfg.env.id,
        freeze_policy=cfg.frozen,
    )
    elapsed = time.perf_counter() - started

    tag = cfg.run_tag if cfg.run_tag is not None else ("frozen" if cfg.frozen else None)
    out_dir = Path(to_absolute_path(cfg.out_dir))
    stem = run_stem(cfg.env.id, cfg.seed, tag)

    save_state_stream(out_dir / f"{stem}{STATE_STREAM_SUFFIX}", stream)
    save_eval_curve(out_dir / f"{stem}{EVAL_CURVE_SUFFIX}", curve)
    save_run_meta(
        out_dir / f"{stem}{RUN_META_SUFFIX}",
        RunMeta(
            env_id=cfg.env.id,
            seed=int(cfg.seed),
            tag=tag,
            frozen=bool(cfg.frozen),
            total_steps=int(cfg.env.total_steps),
            eval_interval=int(cfg.env.eval_interval),
            eval_episodes=int(cfg.env.eval_episodes),
            rollout_steps=int(cfg.env.rollout_steps),
            signal_window=int(cfg.env.signal_window),
            signal_stride=int(cfg.env.signal_stride),
            n_states=stream.n_steps,
            wall_clock_seconds=elapsed,
        ),
    )
    pd.DataFrame(agent.update_log).to_csv(out_dir / f"{stem}{UPDATE_LOG_SUFFIX}", index=False)

    # Report t_conv now rather than at analysis time. It costs nothing, and on a pilot it
    # is the answer being waited for; on a matrix run it is what makes a bad seed visible
    # in the job log instead of a fortnight later.
    report = convergence_report(curve)
    final_return = float(curve.mean_returns[-1]) if curve.steps.size else float("nan")
    logger.info(
        "%s in %.1fs | %d states, %d evaluations | random %.2f -> final %.2f | t_conv=%s (%s)",
        stem,
        elapsed,
        stream.n_steps,
        curve.steps.size,
        curve.random_return,
        final_return,
        report.t_conv,
        report.status,
    )
    logger.info("Artefacts written to %s", out_dir)
    return final_return


if __name__ == "__main__":
    main()
