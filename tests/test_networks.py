"""Tests for rnd_convergence.networks."""

import numpy as np
import pytest
import torch

from rnd_convergence.networks import Policy, ValueNetwork


class TestPolicy:
    def test_logits_have_one_entry_per_action(self):
        assert Policy(obs_dim=4, n_actions=3)(torch.randn(8, 4)).shape == (8, 3)

    def test_distribution_is_normalised(self):
        distribution = Policy(obs_dim=4, n_actions=3).distribution(torch.randn(5, 4))
        assert torch.allclose(distribution.probs.sum(dim=-1), torch.ones(5))

    def test_starts_close_to_uniform(self):
        # The 0.01-gain head keeps the initial policy near-uniform, so early exploration
        # is not biased by initialisation.
        distribution = Policy(obs_dim=6, n_actions=4).distribution(torch.randn(64, 6))
        assert distribution.probs.std().item() < 0.02

    def test_sampled_actions_are_valid_indices(self):
        actions = Policy(obs_dim=4, n_actions=3).distribution(torch.randn(32, 4)).sample()
        assert actions.min().item() >= 0
        assert actions.max().item() < 3

    def test_is_deterministic_given_a_seed(self):
        batch = torch.randn(4, 5)
        torch.manual_seed(7)
        first = Policy(obs_dim=5, n_actions=2)(batch)
        torch.manual_seed(7)
        assert torch.allclose(first, Policy(obs_dim=5, n_actions=2)(batch))

    def test_depth_is_configurable(self):
        deep = Policy(obs_dim=4, n_actions=2, n_layers=3)
        assert sum(1 for module in deep.trunk if isinstance(module, torch.nn.Linear)) == 3

    def test_rejects_zero_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            Policy(obs_dim=4, n_actions=2, n_layers=0)


class TestValueNetwork:
    def test_returns_one_scalar_per_state(self):
        assert ValueNetwork(obs_dim=8)(torch.randn(16, 8)).shape == (16,)

    def test_gradients_reach_every_parameter(self):
        network = ValueNetwork(obs_dim=4)
        network(torch.randn(8, 4)).mean().backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters()
        )

    def test_weights_are_orthogonally_initialised(self):
        network = ValueNetwork(obs_dim=6, hidden_size=6, n_layers=1)
        weight = network.trunk[0].weight.detach()
        product = (weight @ weight.T) / 2.0  # gain sqrt(2) scales the product by 2
        assert np.allclose(product.numpy(), np.eye(6), atol=1e-5)

    def test_biases_start_at_zero(self):
        network = ValueNetwork(obs_dim=4)
        assert all(
            torch.all(m.bias == 0) for m in network.modules() if isinstance(m, torch.nn.Linear)
        )
