"""Smoke-test entry point: roll out a random policy for a few episodes.

Verifies that the package, Hydra config loading, and Gymnasium environments
are wired up correctly before any actual agent code exists. Will be replaced
by a real training entry point later.

Usage:
    python scripts/random_rollout.py
    python scripts/random_rollout.py env_id=LunarLander-v3 n_episodes=3
"""

from __future__ import annotations

import logging

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from rnd_convergence.utils import make_env, set_seed

logger = logging.getLogger(__name__)


@hydra.main(config_path="../rnd_convergence/configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> float:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    env = make_env(cfg.env_id, seed=cfg.seed)
    returns = []
    for episode in range(cfg.n_episodes):
        _, _ = env.reset(seed=cfg.seed + episode)
        episode_return, terminated, truncated = 0.0, False, False
        while not (terminated or truncated):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
        returns.append(episode_return)
        logger.info("Episode %d: return = %.2f", episode, episode_return)
    env.close()

    mean_return = float(np.mean(returns))
    logger.info("Mean return over %d episodes: %.2f", cfg.n_episodes, mean_return)
    return mean_return


if __name__ == "__main__":
    main()
