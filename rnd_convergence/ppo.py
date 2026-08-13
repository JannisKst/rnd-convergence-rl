"""PPO agent (Schulman et al., 2017) for discrete-action environments.

The agent structure follows the ``week_6/ppo.py`` scaffold of the course exercise repo
(automl-edu/RL-exercises) — GAE advantages, a clipped surrogate objective, and one Adam
optimiser carrying separate actor and critic learning rates. The implementation is our
own; the scaffold defines the interface, not the code.

Training is the *only* expensive step in this project. Everything downstream — the RND
prediction-error curve, every entropy variant, every detector threshold — is recomputed
offline from the artefacts :meth:`PPOAgent.train` writes, so a run never has to be
repeated to try a different analysis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.utils import flatdim, flatten
from torch import nn

from rnd_convergence.networks import Policy, ValueNetwork
from rnd_convergence.streams import EvalCurve, StateStream

StateFn = Callable[[Any], np.ndarray]


@dataclass
class Rollout:
    """One batch of on-policy experience, in visit order."""

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    dones: np.ndarray
    states: np.ndarray
    steps: np.ndarray


class PPOAgent:
    """Proximal Policy Optimization with a Categorical policy.

    Parameters
    ----------
    env_fn
        Zero-argument factory returning a fresh environment. Called twice, so training
        and evaluation never share an environment instance and evaluation cannot disturb
        the training episode in progress.
    state_fn
        Maps a raw observation to the state representation written to the
        :class:`~rnd_convergence.streams.StateStream`. Defaults to the flattened
        observation. MiniGrid passes ``(x, y, dir)`` here instead, which is both the
        exact state for the entropy baseline and far smaller on disk.
    """

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        *,
        state_fn: StateFn | None = None,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        rollout_steps: int = 2_048,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_size: int = 128,
        n_layers: int = 2,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.env = env_fn()
        self.eval_env = env_fn()
        if not isinstance(self.env.action_space, gym.spaces.Discrete):
            raise TypeError(
                f"PPOAgent requires a Discrete action space, got {self.env.action_space}. "
                "Continuous-action environments would need a Gaussian policy head and are "
                "deliberately out of scope for this study."
            )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.device = torch.device(device)

        self.obs_dim = flatdim(self.env.observation_space)
        self.n_actions = int(self.env.action_space.n)
        self.state_fn = state_fn if state_fn is not None else self._flatten

        torch.manual_seed(seed)
        self.policy = Policy(self.obs_dim, self.n_actions, hidden_size, n_layers).to(self.device)
        self.value_net = ValueNetwork(self.obs_dim, hidden_size, n_layers).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": lr_actor},
                {"params": self.value_net.parameters(), "lr": lr_critic},
            ]
        )

        self.rng = np.random.default_rng(seed)
        self.global_step = 0
        self._obs, _ = self.env.reset(seed=seed)
        self.eval_env.reset(seed=seed + 10_000)
        self.state_dim = int(np.asarray(self.state_fn(self._obs)).ravel().size)

    def _flatten(self, obs: Any) -> np.ndarray:
        """Flatten an observation into the vector the networks consume."""
        return np.asarray(flatten(self.env.observation_space, obs), dtype=np.float32)

    def _as_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict(self, obs: Any, greedy: bool = False) -> tuple[int, float, float]:
        """Sample (or take the modal) action for a single observation.

        Returns the action, its log-probability under the current policy, and the
        critic's value estimate.
        """
        flat = self._as_tensor(self._flatten(obs)).unsqueeze(0)
        distribution = self.policy.distribution(flat)
        action = distribution.probs.argmax(dim=-1) if greedy else distribution.sample()  # type: ignore[attr-defined]
        value = self.value_net(flat)
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())

    def collect_rollout(self, n_steps: int | None = None) -> Rollout:
        """Run the current policy for ``n_steps`` environment steps."""
        n_steps = self.rollout_steps if n_steps is None else n_steps
        observations = np.zeros((n_steps, self.obs_dim), dtype=np.float32)
        next_observations = np.zeros((n_steps, self.obs_dim), dtype=np.float32)
        actions = np.zeros(n_steps, dtype=np.int64)
        log_probs = np.zeros(n_steps, dtype=np.float32)
        values = np.zeros(n_steps, dtype=np.float32)
        rewards = np.zeros(n_steps, dtype=np.float32)
        terminated = np.zeros(n_steps, dtype=bool)
        dones = np.zeros(n_steps, dtype=bool)
        states = np.zeros((n_steps, self.state_dim), dtype=np.float32)
        steps = np.zeros(n_steps, dtype=np.int64)

        for t in range(n_steps):
            observations[t] = self._flatten(self._obs)
            states[t] = np.asarray(self.state_fn(self._obs), dtype=np.float32).ravel()
            steps[t] = self.global_step

            action, log_prob, value = self.predict(self._obs)
            next_obs, reward, term, trunc, _ = self.env.step(action)

            actions[t] = action
            log_probs[t] = log_prob
            values[t] = value
            rewards[t] = float(reward)
            terminated[t] = term
            dones[t] = term or trunc
            # The true successor state, captured before any reset, so that a truncated
            # episode bootstraps from where it actually stopped rather than from the
            # start of the next one.
            next_observations[t] = self._flatten(next_obs)

            self.global_step += 1
            self._obs = self.env.reset()[0] if (term or trunc) else next_obs

        return Rollout(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            terminated=terminated,
            dones=dones,
            states=states,
            steps=steps,
        )

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        terminated: np.ndarray,
        dones: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generalised advantage estimates and value targets.

        ``terminated`` and ``dones`` play different roles and conflating them is a
        classic source of silent bias. A *terminated* episode has no future reward, so
        its bootstrap value is zero; a *truncated* one was cut off by a time limit and
        must still bootstrap from the value of the state it stopped in. Both kinds of
        ending break the advantage recursion, which is what ``dones`` marks.
        """
        advantages = np.zeros_like(rewards, dtype=np.float64)
        running = 0.0
        for t in reversed(range(rewards.size)):
            bootstrap = 0.0 if terminated[t] else float(next_values[t])
            delta = rewards[t] + self.gamma * bootstrap - values[t]
            running = delta + self.gamma * self.gae_lambda * (0.0 if dones[t] else running)
            advantages[t] = running
        return advantages, advantages + values

    def update(self, rollout: Rollout) -> dict[str, float]:
        """Run the clipped-surrogate update over one rollout."""
        with torch.no_grad():
            next_values = self.value_net(self._as_tensor(rollout.next_observations)).cpu().numpy()

        advantages, returns = self.compute_gae(
            rollout.rewards, rollout.values, next_values, rollout.terminated, rollout.dones
        )

        obs = self._as_tensor(rollout.observations)
        actions = torch.as_tensor(rollout.actions, dtype=torch.int64, device=self.device)
        old_log_probs = self._as_tensor(rollout.log_probs)
        advantage_t = self._as_tensor(advantages)
        return_t = self._as_tensor(returns)
        advantage_t = (advantage_t - advantage_t.mean()) / (advantage_t.std() + 1e-8)

        n = obs.shape[0]
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_batches = 0

        for _ in range(self.epochs):
            for batch in np.array_split(self.rng.permutation(n), max(n // self.batch_size, 1)):
                index = torch.as_tensor(batch, dtype=torch.int64, device=self.device)
                distribution = self.policy.distribution(obs[index])
                new_log_probs = distribution.log_prob(actions[index])
                entropy = distribution.entropy().mean()

                ratio = (new_log_probs - old_log_probs[index]).exp()
                unclipped = ratio * advantage_t[index]
                clipped = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantage_t[index]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(self.value_net(obs[index]), return_t[index])
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                n_batches += 1

        return {key: value / max(n_batches, 1) for key, value in totals.items()}

    def evaluate(self, n_episodes: int = 10, greedy: bool = True) -> np.ndarray:
        """Return the per-episode returns of the current policy on the evaluation env."""
        returns = np.zeros(n_episodes, dtype=np.float32)
        for episode in range(n_episodes):
            obs, _ = self.eval_env.reset()
            total, done = 0.0, False
            while not done:
                action, _, _ = self.predict(obs, greedy=greedy)
                obs, reward, term, trunc, _ = self.eval_env.step(action)
                total += float(reward)
                done = term or trunc
            returns[episode] = total
        return returns

    def random_policy_return(self, n_episodes: int = 20) -> float:
        """Mean return of a uniformly random policy — the anchor for ``t_conv``.

        Both the reset states and the sampled actions are seeded, so the baseline is a
        deterministic function of the agent's seed. It has to be: ``R_0`` sets the
        convergence threshold for the whole run, and a baseline that drifted between
        calls would move ``t_conv`` without anything about the run having changed.
        """
        returns = np.zeros(n_episodes, dtype=np.float64)
        self.eval_env.action_space.seed(self.seed + 30_000)
        for episode in range(n_episodes):
            self.eval_env.reset(seed=self.seed + 20_000 + episode)
            total, done = 0.0, False
            while not done:
                _, reward, term, trunc, _ = self.eval_env.step(self.eval_env.action_space.sample())
                total += float(reward)
                done = term or trunc
            returns[episode] = total
        return float(returns.mean())

    def train(
        self,
        total_steps: int,
        *,
        eval_interval: int = 5_000,
        eval_episodes: int = 10,
        random_episodes: int = 20,
        env_id: str | None = None,
    ) -> tuple[StateStream, EvalCurve]:
        """Train for ``total_steps`` environment steps and return the logged artefacts.

        The random-policy baseline is measured *before* training, since ``t_conv`` is
        undefined without it.
        """
        env_id = env_id or getattr(self.env.spec, "id", type(self.env).__name__)
        random_return = self.random_policy_return(random_episodes)

        state_chunks: list[np.ndarray] = []
        step_chunks: list[np.ndarray] = []
        end_chunks: list[np.ndarray] = []
        eval_steps: list[int] = []
        eval_returns: list[np.ndarray] = []
        next_eval = eval_interval

        while self.global_step < total_steps:
            rollout = self.collect_rollout(min(self.rollout_steps, total_steps - self.global_step))
            self.update(rollout)

            state_chunks.append(rollout.states)
            step_chunks.append(rollout.steps)
            end_chunks.append(rollout.dones)

            while self.global_step >= next_eval and next_eval <= total_steps:
                eval_steps.append(next_eval)
                eval_returns.append(self.evaluate(eval_episodes))
                next_eval += eval_interval

        stream = StateStream(
            steps=np.concatenate(step_chunks),
            observations=np.concatenate(state_chunks),
            episode_ends=np.concatenate(end_chunks),
            env_id=env_id,
            seed=self.seed,
        )
        curve = EvalCurve(
            steps=np.asarray(eval_steps, dtype=np.int64),
            returns=np.asarray(eval_returns, dtype=np.float32).reshape(len(eval_steps), -1),
            random_return=random_return,
            env_id=env_id,
            seed=self.seed,
        )
        return stream, curve
