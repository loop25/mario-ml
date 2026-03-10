"""
Noisy Linear Layer for Rainbow DQN.

Implements factorized Gaussian noise as described in:
    Fortunato et al., 2018 - "Noisy Networks for Exploration"

Instead of epsilon-greedy exploration (which explores randomly),
NoisyLinear adds learned noise to the network weights. The network
learns WHEN and HOW MUCH to explore by adjusting the noise magnitude
(sigma parameters) through gradient descent.

Factorized noise is used for efficiency:
    - Independent noise: requires p * q random numbers per layer
    - Factorized noise: requires only p + q random numbers
    - Uses f(x) = sign(x) * sqrt(|x|) to generate the noise matrix

The key insight is that exploration becomes a function of the state,
not random chance. The agent learns to explore more in unfamiliar
states and exploit in well-understood ones.

Usage:
    layer = NoisyLinear(512, 7, sigma_init=0.5)
    output = layer(features)     # Forward with noise
    layer.reset_noise()          # New noise for next episode
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Linear layer with factorized Gaussian noise on weights and biases.

    Parameters are split into deterministic (mu) and noisy (sigma) parts:
        weight = weight_mu + weight_sigma * epsilon_weight
        bias   = bias_mu   + bias_sigma   * epsilon_bias

    where epsilon values are factorized noise vectors resampled each episode.

    Args:
        in_features: Size of input.
        out_features: Size of output.
        sigma_init: Initial value for sigma parameters.
            Controls starting noise magnitude. Default 0.5 from the paper.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters: mu (mean) and sigma (noise scale)
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features),
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features),
        )
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not learnable, regenerated each episode)
        self.register_buffer(
            'weight_epsilon', torch.empty(out_features, in_features),
        )
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize mu and sigma parameters.

        mu is initialized uniformly in [-1/sqrt(fan_in), 1/sqrt(fan_in)],
        matching PyTorch's default Linear initialization.
        sigma is initialized to sigma_init / sqrt(fan_in), ensuring
        noise starts proportional to the signal magnitude.
        """
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

        sigma_value = self.sigma_init / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma_value)
        self.bias_sigma.data.fill_(sigma_value)

    @staticmethod
    def _factorized_noise(size: int) -> torch.Tensor:
        """Generate factorized noise vector.

        f(x) = sign(x) * sqrt(|x|)

        This transformation preserves the sign (positive/negative
        exploration directions) while compressing the magnitude through
        the square root, keeping noise bounded.

        Args:
            size: Number of noise values to generate.

        Returns:
            Noise tensor of shape (size,).
        """
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Regenerate noise vectors.

        Call once per episode (not per step) so the agent maintains
        consistent exploration behavior within an episode.

        Uses factorized noise: generates p and q noise vectors,
        then computes outer product for the weight noise matrix.
        This requires only p + q random numbers instead of p * q.
        """
        epsilon_in = self._factorized_noise(self.in_features)
        epsilon_out = self._factorized_noise(self.out_features)

        # Outer product: (out, 1) * (1, in) → (out, in)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights.

        weight = mu + sigma * epsilon
        bias   = mu + sigma * epsilon

        During training, noise provides exploration.
        For evaluation, use the mu parameters directly (no noise).

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, out_features).
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # During evaluation, use only the learned means (no noise)
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'sigma_init={self.sigma_init}'
        )
