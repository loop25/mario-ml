"""Tests for the Rainbow DQN network architecture.

Validates forward pass shapes, feature toggle combinations,
noise reset behavior, and Q-value computation.
"""
import pytest
import torch

from src.algorithms.rainbow.rainbow_network import RainbowNetwork
from src.algorithms.rainbow.noisy_linear import NoisyLinear


# ─── Helper ──────────────────────────────────────────────────────


def _make_net(**kwargs):
    """Create a RainbowNetwork with small defaults for testing."""
    defaults = dict(
        input_channels=4,
        num_actions=7,
        input_height=84,
        input_width=84,
        num_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        use_dueling=True,
        use_noisy=True,
        use_distributional=True,
        sigma_init=0.5,
    )
    defaults.update(kwargs)
    return RainbowNetwork(**defaults)


def _dummy_input(batch_size=4, channels=4, h=84, w=84):
    return torch.rand(batch_size, channels, h, w)


# ─── Forward Pass Shape Tests ────────────────────────────────────


class TestRainbowNetworkShapes:
    def test_full_rainbow_output_shape(self):
        """Full Rainbow: (batch, actions, atoms) log-probs."""
        net = _make_net()
        x = _dummy_input()
        out = net(x)
        assert out.shape == (4, 7, 51)

    def test_distributional_output_sums_to_one(self):
        """Log-probs should exponentiate to valid distributions."""
        net = _make_net()
        x = _dummy_input()
        log_probs = net(x)
        probs = log_probs.exp()
        # Sum over atoms for each action should be ~1.0
        sums = probs.sum(dim=2)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_q_values_shape(self):
        """Q-values for action selection: (batch, actions)."""
        net = _make_net()
        x = _dummy_input()
        q = net.q_values(x)
        assert q.shape == (4, 7)

    def test_single_sample_forward(self):
        net = _make_net()
        x = _dummy_input(batch_size=1)
        out = net(x)
        assert out.shape == (1, 7, 51)


# ─── Feature Toggle Tests ───────────────────────────────────────


class TestRainbowFeatureToggles:
    def test_dueling_only(self):
        """Dueling without distributional: (batch, actions) Q-values."""
        net = _make_net(use_distributional=False, use_noisy=False)
        x = _dummy_input()
        out = net(x)
        assert out.shape == (4, 7)

    def test_distributional_only(self):
        """C51 without dueling: (batch, actions, atoms) log-probs."""
        net = _make_net(use_dueling=False, use_noisy=False)
        x = _dummy_input()
        out = net(x)
        assert out.shape == (4, 7, 51)

    def test_noisy_only(self):
        """Noisy nets without dueling or distributional."""
        net = _make_net(use_dueling=False, use_distributional=False)
        x = _dummy_input()
        out = net(x)
        assert out.shape == (4, 7)

    def test_vanilla_dqn_mode(self):
        """All toggles off = vanilla DQN: (batch, actions) Q-values."""
        net = _make_net(
            use_dueling=False, use_noisy=False, use_distributional=False,
        )
        x = _dummy_input()
        out = net(x)
        assert out.shape == (4, 7)

    def test_dueling_distributional_no_noisy(self):
        """Dueling + C51 without noisy."""
        net = _make_net(use_noisy=False)
        x = _dummy_input()
        out = net(x)
        assert out.shape == (4, 7, 51)

    def test_all_toggles_on(self):
        """All features enabled (default)."""
        net = _make_net()
        x = _dummy_input()
        out = net(x)
        assert out.shape == (4, 7, 51)
        q = net.q_values(x)
        assert q.shape == (4, 7)


# ─── Noise Behavior Tests ───────────────────────────────────────


class TestRainbowNoise:
    def test_reset_noise_changes_output(self):
        """After resetting noise, outputs should change."""
        net = _make_net(use_noisy=True)
        net.train()
        x = _dummy_input()

        out1 = net(x).detach()
        net.reset_noise()
        out2 = net(x).detach()

        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_reset_noise_noop_when_disabled(self):
        """reset_noise() should be safe to call without noisy nets."""
        net = _make_net(use_noisy=False)
        # Should not raise
        net.reset_noise()

    def test_eval_mode_is_deterministic_with_noisy(self):
        """In eval mode, noisy nets should not add noise."""
        net = _make_net(use_noisy=True)
        net.eval()
        x = _dummy_input()

        out1 = net(x).detach()
        out2 = net(x).detach()
        assert torch.allclose(out1, out2)

    def test_contains_noisy_layers(self):
        """When use_noisy=True, model should contain NoisyLinear layers."""
        net = _make_net(use_noisy=True)
        noisy_layers = [m for m in net.modules() if isinstance(m, NoisyLinear)]
        assert len(noisy_layers) >= 2  # At least hidden + output per stream

    def test_no_noisy_layers_when_disabled(self):
        """When use_noisy=False, model should use regular Linear layers."""
        net = _make_net(use_noisy=False)
        noisy_layers = [m for m in net.modules() if isinstance(m, NoisyLinear)]
        assert len(noisy_layers) == 0


# ─── Q-Value Tests ───────────────────────────────────────────────


class TestRainbowQValues:
    def test_q_values_distributional(self):
        """Q-values should be expected value of distribution."""
        net = _make_net(use_distributional=True)
        x = _dummy_input()
        q = net.q_values(x)
        assert q.shape == (4, 7)
        # Q-values should be within [v_min, v_max] (expected value)
        assert q.min() >= -10.0 - 1e-3
        assert q.max() <= 10.0 + 1e-3

    def test_q_values_non_distributional(self):
        """Without distributional, q_values() returns forward() directly."""
        net = _make_net(use_distributional=False)
        x = _dummy_input()
        q = net.q_values(x)
        fwd = net(x)
        assert torch.allclose(q, fwd)


# ─── Support Buffer Tests ───────────────────────────────────────


class TestRainbowSupport:
    def test_support_shape(self):
        net = _make_net(num_atoms=51)
        assert net.support.shape == (51,)

    def test_support_range(self):
        net = _make_net(v_min=-10.0, v_max=10.0, num_atoms=51)
        assert abs(net.support[0].item() - (-10.0)) < 1e-5
        assert abs(net.support[-1].item() - 10.0) < 1e-5

    def test_delta_z(self):
        net = _make_net(v_min=-10.0, v_max=10.0, num_atoms=51)
        expected = 20.0 / 50
        assert abs(net.delta_z - expected) < 1e-5

    def test_non_distributional_single_atom(self):
        """Without distributional, num_atoms should be 1."""
        net = _make_net(use_distributional=False)
        assert net.num_atoms == 1


# ─── Misc Tests ──────────────────────────────────────────────────


class TestRainbowNetworkMisc:
    def test_different_action_counts(self):
        for n_actions in [2, 4, 7, 12]:
            net = _make_net(num_actions=n_actions)
            x = _dummy_input()
            q = net.q_values(x)
            assert q.shape == (4, n_actions)

    def test_different_atom_counts(self):
        for n_atoms in [11, 21, 51]:
            net = _make_net(num_atoms=n_atoms)
            x = _dummy_input()
            out = net(x)
            assert out.shape == (4, 7, n_atoms)

    def test_gradient_flows(self):
        """Verify gradients propagate through the full network."""
        net = _make_net()
        x = _dummy_input()
        out = net(x)
        loss = out.sum()
        loss.backward()
        # Check at least one conv layer has gradients
        assert net.conv[0].weight.grad is not None
