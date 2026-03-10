"""Tests for the NoisyLinear layer used in Rainbow DQN.

Validates factorized Gaussian noise, parameter initialization,
noise reset behavior, and train/eval mode differences.
"""
import pytest
import torch

from src.algorithms.rainbow.noisy_linear import NoisyLinear


class TestNoisyLinearConstruction:
    def test_creates_correct_parameter_shapes(self):
        layer = NoisyLinear(64, 32)
        assert layer.weight_mu.shape == (32, 64)
        assert layer.weight_sigma.shape == (32, 64)
        assert layer.bias_mu.shape == (32,)
        assert layer.bias_sigma.shape == (32,)

    def test_creates_noise_buffers(self):
        layer = NoisyLinear(64, 32)
        assert layer.weight_epsilon.shape == (32, 64)
        assert layer.bias_epsilon.shape == (32,)

    def test_sigma_init_stored(self):
        layer = NoisyLinear(64, 32, sigma_init=0.3)
        assert layer.sigma_init == 0.3

    def test_default_sigma_init(self):
        layer = NoisyLinear(64, 32)
        assert layer.sigma_init == 0.5


class TestNoisyLinearForward:
    def test_output_shape(self):
        layer = NoisyLinear(64, 32)
        x = torch.randn(8, 64)
        out = layer(x)
        assert out.shape == (8, 32)

    def test_single_sample_output(self):
        layer = NoisyLinear(64, 32)
        x = torch.randn(1, 64)
        out = layer(x)
        assert out.shape == (1, 32)

    def test_output_differs_from_mu_only_in_train_mode(self):
        """In training mode, noise should cause different output than mu-only."""
        layer = NoisyLinear(64, 32)
        layer.train()
        x = torch.randn(4, 64)

        # Get output with noise
        out_noisy = layer(x)

        # Get output without noise (eval mode uses mu only)
        layer.eval()
        out_clean = layer(x)

        # They should (almost certainly) differ due to noise
        assert not torch.allclose(out_noisy, out_clean, atol=1e-6)

    def test_eval_mode_is_deterministic(self):
        """In eval mode, output should be deterministic (no noise)."""
        layer = NoisyLinear(64, 32)
        layer.eval()
        x = torch.randn(4, 64)

        out1 = layer(x)
        out2 = layer(x)
        assert torch.allclose(out1, out2)


class TestNoisyLinearNoise:
    def test_reset_noise_changes_epsilon(self):
        layer = NoisyLinear(64, 32)
        old_weight_eps = layer.weight_epsilon.clone()
        old_bias_eps = layer.bias_epsilon.clone()

        layer.reset_noise()

        # Noise should change (extremely unlikely to be identical)
        assert not torch.allclose(old_weight_eps, layer.weight_epsilon)
        assert not torch.allclose(old_bias_eps, layer.bias_epsilon)

    def test_reset_noise_changes_output(self):
        """Different noise should produce different outputs."""
        layer = NoisyLinear(64, 32)
        layer.train()
        x = torch.randn(4, 64)

        out1 = layer(x)
        layer.reset_noise()
        out2 = layer(x)

        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_factorized_noise_has_correct_shape(self):
        noise = NoisyLinear._factorized_noise(100)
        assert noise.shape == (100,)

    def test_factorized_noise_preserves_sign(self):
        """f(x) = sign(x) * sqrt(|x|) should preserve sign."""
        torch.manual_seed(42)
        noise = NoisyLinear._factorized_noise(1000)
        # Should have both positive and negative values
        assert (noise > 0).any()
        assert (noise < 0).any()


class TestNoisyLinearInitialization:
    def test_mu_initialized_within_bounds(self):
        layer = NoisyLinear(100, 50)
        bound = 1.0 / (100 ** 0.5)
        assert layer.weight_mu.data.max() <= bound + 1e-6
        assert layer.weight_mu.data.min() >= -bound - 1e-6

    def test_sigma_initialized_correctly(self):
        layer = NoisyLinear(100, 50, sigma_init=0.5)
        expected = 0.5 / (100 ** 0.5)
        assert torch.allclose(
            layer.weight_sigma.data,
            torch.full_like(layer.weight_sigma.data, expected),
        )

    def test_reset_parameters_restores_init(self):
        layer = NoisyLinear(64, 32)
        # Corrupt parameters
        layer.weight_mu.data.fill_(999)
        layer.reset_parameters()
        # Should be back in range
        bound = 1.0 / (64 ** 0.5)
        assert layer.weight_mu.data.max() <= bound + 1e-6


class TestNoisyLinearRepr:
    def test_extra_repr(self):
        layer = NoisyLinear(64, 32, sigma_init=0.3)
        r = layer.extra_repr()
        assert '64' in r
        assert '32' in r
        assert '0.3' in r
