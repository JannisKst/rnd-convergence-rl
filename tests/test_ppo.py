"""Tests for rnd_convergence.ppo.

The GAE tests carry most of the weight. A sign error or a mishandled episode boundary
there yields an agent that *almost* learns, which is expensive to diagnose from training
curves and would quietly corrupt every downstream measurement, so the advantages are
checked against values computed by hand.
"""

import gymnasium as gym
import numpy as np
import pytest

from rnd_convergence.convergence import convergence_report
from rnd_convergence.ppo import PPOAgent
from rnd_convergence.streams import EvalCurve, StateStream


@pytest.fixture
def agent():
    return PPOAgent(
        lambda: gym.make("CartPole-v1"),
        rollout_steps=64,
        batch_size=32,
        epochs=2,
        seed=0,
    )


class TestConstruction:
    def test_rejects_continuous_action_spaces(self):
        with pytest.raises(TypeError, match="Discrete action space"):
            PPOAgent(lambda: gym.make("Pendulum-v1"))

    def test_infers_dimensions_from_the_environment(self, agent):
        assert agent.obs_dim == 4
        assert agent.n_actions == 2

    def test_training_and_evaluation_envs_are_separate(self, agent):
        assert agent.env is not agent.eval_env


class TestComputeGae:
    gamma = 0.9
    gae_lambda = 0.8

    def build(self, **kwargs):
        return PPOAgent(
            lambda: gym.make("CartPole-v1"),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            **kwargs,
        )

    def test_matches_hand_computed_advantages(self):
        # delta_2 = 1 + 0.9*0    - 0.5 = 0.5     (terminated, so no bootstrap)
        # delta_1 = 1 + 0.9*0.5  - 0.5 = 0.95 -> A_1 = 0.95 + 0.72*0.5  = 1.31
        # delta_0 = 0.95              -> A_0 = 0.95 + 0.72*1.31 = 1.8932
        rewards = np.array([1.0, 1.0, 1.0])
        values = np.array([0.5, 0.5, 0.5])
        next_values = np.array([0.5, 0.5, 0.5])
        terminated = np.array([False, False, True])
        dones = np.array([False, False, True])

        advantages, returns = self.build().compute_gae(
            rewards, values, next_values, terminated, dones
        )

        assert advantages == pytest.approx([1.8932, 1.31, 0.5])
        assert returns == pytest.approx([2.3932, 1.81, 1.0])

    def test_truncation_bootstraps_but_termination_does_not(self):
        rewards = np.array([1.0])
        values = np.array([0.5])
        next_values = np.array([0.5])

        terminated_adv, _ = self.build().compute_gae(
            rewards, values, next_values, np.array([True]), np.array([True])
        )
        truncated_adv, _ = self.build().compute_gae(
            rewards, values, next_values, np.array([False]), np.array([True])
        )

        assert terminated_adv[0] == pytest.approx(0.5)
        assert truncated_adv[0] == pytest.approx(0.95)
        assert truncated_adv[0] > terminated_adv[0]

    def test_episode_boundary_breaks_the_recursion(self):
        # Two episodes in one rollout; rewards after the boundary must not leak backwards.
        rewards = np.array([1.0, 1.0, 5.0, 5.0])
        values = np.zeros(4)
        next_values = np.zeros(4)
        terminated = np.array([False, True, False, True])
        dones = np.array([False, True, False, True])

        advantages, _ = self.build().compute_gae(rewards, values, next_values, terminated, dones)
        # A_1 closes its episode, so it equals its own delta and ignores the 5.0s.
        assert advantages[1] == pytest.approx(1.0)
        assert advantages[3] == pytest.approx(5.0)

    def test_zero_lambda_reduces_to_one_step_td_error(self):
        rewards = np.array([1.0, 1.0, 1.0])
        values = np.array([0.2, 0.4, 0.6])
        next_values = np.array([0.4, 0.6, 0.8])
        flags = np.zeros(3, dtype=bool)

        agent = PPOAgent(lambda: gym.make("CartPole-v1"), gamma=0.9, gae_lambda=0.0)
        advantages, _ = agent.compute_gae(rewards, values, next_values, flags, flags)

        assert advantages == pytest.approx(rewards + 0.9 * next_values - values)


class TestRollout:
    def test_shapes_and_step_indices(self, agent):
        rollout = agent.collect_rollout(50)

        assert rollout.observations.shape == (50, 4)
        assert rollout.next_observations.shape == (50, 4)
        assert rollout.actions.shape == rollout.rewards.shape == (50,)
        assert np.array_equal(rollout.steps, np.arange(50))
        assert agent.global_step == 50

    def test_step_counter_continues_across_rollouts(self, agent):
        agent.collect_rollout(20)
        second = agent.collect_rollout(20)
        assert np.array_equal(second.steps, np.arange(20, 40))

    def test_actions_are_within_the_action_space(self, agent):
        rollout = agent.collect_rollout(80)
        assert rollout.actions.min() >= 0
        assert rollout.actions.max() < agent.n_actions

    def test_episodes_end_within_a_long_rollout(self, agent):
        # An untrained CartPole policy fails quickly, so boundaries must appear.
        assert agent.collect_rollout(300).dones.sum() > 0


class TestUpdate:
    def test_reports_finite_losses(self, agent):
        losses = agent.update(agent.collect_rollout(64))
        assert set(losses) == {"policy_loss", "value_loss", "entropy"}
        assert all(np.isfinite(value) for value in losses.values())

    def test_changes_the_policy_parameters(self, agent):
        before = [p.detach().clone() for p in agent.policy.parameters()]
        agent.update(agent.collect_rollout(64))
        after = list(agent.policy.parameters())
        assert any(not np.allclose(a.numpy(), b.detach().numpy()) for a, b in zip(before, after))

    def test_entropy_is_positive_for_a_fresh_policy(self, agent):
        # Two actions, near-uniform at initialisation, so entropy starts close to log 2.
        losses = agent.update(agent.collect_rollout(64))
        assert 0.0 < losses["entropy"] <= np.log(2) + 1e-6


class TestPredictAndEvaluate:
    def test_greedy_prediction_is_deterministic(self, agent):
        obs, _ = agent.env.reset(seed=3)
        first = agent.predict(obs, greedy=True)[0]
        assert all(agent.predict(obs, greedy=True)[0] == first for _ in range(5))

    def test_evaluate_returns_one_value_per_episode(self, agent):
        returns = agent.evaluate(n_episodes=3)
        assert returns.shape == (3,)
        assert np.all(returns > 0)

    def test_random_policy_return_is_reproducible(self, agent):
        assert agent.random_policy_return(3) == agent.random_policy_return(3)


class TestTrain:
    def test_emits_well_formed_artefacts(self, agent):
        stream, curve = agent.train(
            600, eval_interval=200, eval_episodes=2, random_episodes=2, env_id="CartPole-v1"
        )

        assert isinstance(stream, StateStream)
        assert isinstance(curve, EvalCurve)
        assert stream.n_steps == 600
        assert np.array_equal(stream.steps, np.arange(600))
        assert stream.obs_dim == 4
        assert stream.env_id == curve.env_id == "CartPole-v1"
        assert stream.seed == curve.seed == 0

    def test_evaluations_are_stamped_with_the_step_they_ran_at(self, agent):
        # Evaluation can only happen on a rollout boundary, so the recorded step is the
        # first boundary at or after each grid point -- never the grid point itself,
        # which the policy was not at. A mislabel here is a systematic, one-directional
        # offset in t_conv and therefore in every Delta the project reports.
        _, curve = agent.train(600, eval_interval=200, eval_episodes=2, random_episodes=2)

        assert curve.returns.shape == (3, 2)
        assert np.array_equal(curve.steps, [256, 448, 600])  # rollout_steps=64
        for step, grid_point in zip(curve.steps, [200, 400, 600]):
            assert grid_point <= step < grid_point + agent.rollout_steps
            assert step % agent.rollout_steps == 0 or step == 600

    def test_one_evaluation_per_grid_point_even_when_rollouts_overshoot(self):
        # rollout_steps > eval_interval: a single rollout jumps several grid points. The
        # policy is unchanged across them, so re-evaluating once per skipped point would
        # hand convergence_time's "5 consecutive evaluations" patience free duplicates.
        agent = PPOAgent(lambda: gym.make("CartPole-v1"), rollout_steps=256, seed=0)
        _, curve = agent.train(512, eval_interval=50, eval_episodes=1, random_episodes=1)

        assert np.array_equal(curve.steps, [256, 512])
        assert np.all(np.diff(curve.steps) > 0)

    def test_records_the_random_policy_baseline(self, agent):
        _, curve = agent.train(400, eval_interval=200, eval_episodes=2, random_episodes=3)
        assert curve.random_return > 0  # CartPole pays +1 per surviving step.

    def test_custom_state_fn_controls_what_is_logged(self):
        # MiniGrid will use this to log (x, y, dir) rather than the raw observation.
        agent = PPOAgent(
            lambda: gym.make("CartPole-v1"),
            state_fn=lambda obs: np.asarray(obs[:2], dtype=np.float32),
            rollout_steps=64,
            seed=0,
        )
        stream, _ = agent.train(128, eval_interval=128, eval_episodes=1, random_episodes=1)
        assert stream.obs_dim == 2


class TestDegenerateRolloutGuard:
    def test_single_step_rollout_does_not_poison_the_policy(self):
        # torch.std of one element is NaN, so normalising advantages over a 1-step
        # rollout used to NaN out every parameter. It happens on the final partial
        # rollout when total_steps is not a multiple of rollout_steps -- i.e. after all
        # the compute has been spent.
        agent = PPOAgent(lambda: gym.make("CartPole-v1"), rollout_steps=32, seed=0)
        losses = agent.update(agent.collect_rollout(1))

        assert all(np.isfinite(v) for v in losses.values())
        assert all(np.isfinite(p.detach().numpy()).all() for p in agent.policy.parameters())
        assert all(np.isfinite(p.detach().numpy()).all() for p in agent.value_net.parameters())

    def test_training_survives_a_ragged_step_budget(self):
        agent = PPOAgent(lambda: gym.make("CartPole-v1"), rollout_steps=64, seed=0)
        stream, _ = agent.train(129, eval_interval=64, eval_episodes=1, random_episodes=1)

        assert stream.n_steps == 129
        assert all(np.isfinite(p.detach().numpy()).all() for p in agent.policy.parameters())


class TestLearning:
    @pytest.mark.slow
    def test_cartpole_return_improves(self):
        # The suite can verify every component in isolation and still not tell us the
        # agent learns; this is the only test that does. If it regresses, every Delta in
        # the results table is measured against a policy that never converged.
        agent = PPOAgent(lambda: gym.make("CartPole-v1"), seed=0)
        baseline = agent.evaluate(n_episodes=5).mean()
        _, curve = agent.train(30_000, eval_interval=5_000, eval_episodes=5, random_episodes=5)

        assert curve.mean_returns[-1] > baseline + 50
        assert curve.mean_returns[-1] > 150

    def test_run_shorter_than_one_eval_interval_produces_an_empty_curve(self):
        # No evaluation fires, so eval_returns is empty and reshape cannot infer the
        # episode count. Crashing here would discard a completed training run.
        agent = PPOAgent(lambda: gym.make("CartPole-v1"), rollout_steps=64, seed=0)
        stream, curve = agent.train(128, eval_interval=5_000, eval_episodes=3, random_episodes=1)

        assert stream.n_steps == 128
        assert curve.steps.size == 0
        assert curve.returns.shape == (0, 3)
        assert convergence_report(curve).status == "not_learned"

    def test_update_losses_are_retained_for_debugging(self):
        agent = PPOAgent(lambda: gym.make("CartPole-v1"), rollout_steps=64, seed=0)
        agent.train(256, eval_interval=128, eval_episodes=1, random_episodes=1)

        assert len(agent.update_log) == 4
        assert set(agent.update_log[0]) == {
            "step",
            "mean_reward",
            "policy_loss",
            "value_loss",
            "entropy",
        }
        assert all(np.isfinite(list(entry.values())).all() for entry in agent.update_log)
