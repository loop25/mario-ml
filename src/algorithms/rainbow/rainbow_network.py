"""
Rainbow DQN Network Architecture.

Combines three network-level improvements over vanilla DQN:
    1. Dueling Architecture (Wang et al., 2016):
       Separates state-value V(s) and advantage A(s,a) estimation.
       Q(s,a) = V(s) + A(s,a) - mean(A). This helps the network learn
       which states are valuable independent of actions.

    2. Distributional RL / C51 (Bellemare et al., 2017):
       Instead of predicting scalar Q-values, predicts the full
       distribution of returns using 51 atoms over a fixed support
       [Vmin, Vmax]. Richer signal leads to better learning.

    3. Noisy Networks (Fortunato et al., 2018):
       Replaces epsilon-greedy with learned noise on weights.
       Exploration becomes state-dependent rather than random.

The convolutional backbone is identical to the vanilla DQN
(Mnih et al., 2015): Conv(32,8,4) → Conv(64,4,2) → Conv(64,3,1).
Only the fully connected head is replaced.

Feature toggles allow enabling/disabling each improvement
independently for ablation studies.

Usage:
    net = RainbowNetwork(input_channels=4, num_actions=7)
    log_probs = net(state)        # (batch, actions, atoms)
    q_values = net.q_values(state) # (batch, actions) - for action selection
    net.reset_noise()              # New noise each episode
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.algorithms.rainbow.noisy_linear import NoisyLinear


class RainbowNetwork(nn.Module):
    """Rainbow DQN network with dueling, distributional, and noisy heads.

    Architecture:
        Conv backbone (shared with vanilla DQN):
            Conv2d(in, 32, 8, 4) → ReLU
            Conv2d(32, 64, 4, 2) → ReLU
            Conv2d(64, 64, 3, 1) → ReLU
            Flatten → 3136 features (for 84x84 input)

        FC head (configurable):
            Dueling + C51 + Noisy: Two-stream distributional head
            Dueling only: Two-stream scalar head
            C51 only: Single-stream distributional head
            Vanilla: Standard 2-layer FC head

    Args:
        input_channels: Number of stacked frames (typically 4).
        num_actions: Number of possible actions.
        input_height: Height of input frames. Default 84.
        input_width: Width of input frames. Default 84.
        num_atoms: Number of atoms for distributional RL. Default 51.
        v_min: Minimum return value for C51 support. Default -10.
        v_max: Maximum return value for C51 support. Default 10.
        use_dueling: Enable dueling architecture. Default True.
        use_noisy: Use NoisyLinear instead of nn.Linear. Default True.
        use_distributional: Enable C51 distributional output. Default True.
        sigma_init: Initial noise magnitude for NoisyLinear. Default 0.5.
    """

    def __init__(
        self,
        input_channels: int = 4,
        num_actions: int = 7,
        input_height: int = 84,
        input_width: int = 84,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        use_dueling: bool = True,
        use_noisy: bool = True,
        use_distributional: bool = True,
        sigma_init: float = 0.5,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms if use_distributional else 1
        self.v_min = v_min
        self.v_max = v_max
        self.use_dueling = use_dueling
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional

        # Register the atom support as a buffer (moved to device with model)
        if use_distributional:
            support = torch.linspace(v_min, v_max, num_atoms)
            self.register_buffer('support', support)
            self.delta_z = (v_max - v_min) / (num_atoms - 1)
        else:
            self.register_buffer('support', torch.ones(1))
            self.delta_z = 0.0

        # ================================================================
        # Convolutional Feature Extractor (same as vanilla DQN)
        # ================================================================
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Compute conv output size via dummy pass
        conv_out_size = self._get_conv_output_size(
            input_channels, input_height, input_width,
        )

        # ================================================================
        # FC Head — configured by feature toggles
        # ================================================================
        # Choose linear layer type
        LinearLayer = NoisyLinear if use_noisy else nn.Linear
        linear_kwargs = {'sigma_init': sigma_init} if use_noisy else {}

        if use_dueling:
            # Dueling: separate V(s) and A(s,a) streams
            self.value_hidden = LinearLayer(
                conv_out_size, 512, **linear_kwargs,
            )
            self.value_output = LinearLayer(
                512, self.num_atoms, **linear_kwargs,
            )

            self.advantage_hidden = LinearLayer(
                conv_out_size, 512, **linear_kwargs,
            )
            self.advantage_output = LinearLayer(
                512, num_actions * self.num_atoms, **linear_kwargs,
            )
        else:
            # Single-stream head
            self.fc_hidden = LinearLayer(
                conv_out_size, 512, **linear_kwargs,
            )
            self.fc_output = LinearLayer(
                512, num_actions * self.num_atoms, **linear_kwargs,
            )

    def _get_conv_output_size(
        self, channels: int, height: int, width: int,
    ) -> int:
        """Calculate flattened conv output size via dummy tensor."""
        dummy = torch.zeros(1, channels, height, width)
        output = self.conv(dummy)
        return int(np.prod(output.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning distributional output.

        Args:
            x: Input frames, shape (batch, channels, height, width).
               Pixel values in [0, 1].

        Returns:
            If distributional: log-probabilities over atoms for each action,
                shape (batch, num_actions, num_atoms).
            If not distributional: Q-values for each action,
                shape (batch, num_actions).
        """
        batch_size = x.size(0)
        features = self.conv(x)
        features = features.reshape(batch_size, -1)

        if self.use_dueling:
            # Value stream: V(s) → (batch, num_atoms)
            value = F.relu(self.value_hidden(features))
            value = self.value_output(value)
            value = value.view(batch_size, 1, self.num_atoms)

            # Advantage stream: A(s,a) → (batch, num_actions, num_atoms)
            advantage = F.relu(self.advantage_hidden(features))
            advantage = self.advantage_output(advantage)
            advantage = advantage.view(
                batch_size, self.num_actions, self.num_atoms,
            )

            # Combine: Q = V + A - mean(A)
            # Per-atom combination for distributional
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            hidden = F.relu(self.fc_hidden(features))
            q_atoms = self.fc_output(hidden)
            q_atoms = q_atoms.view(
                batch_size, self.num_actions, self.num_atoms,
            )

        if self.use_distributional:
            # Softmax over atoms → probability distribution per action
            log_probs = F.log_softmax(q_atoms, dim=2)
            return log_probs
        else:
            # Squeeze single atom → scalar Q-values
            return q_atoms.squeeze(2)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Compute scalar Q-values for action selection.

        For distributional: Q(s,a) = sum(p_i * z_i) — expected value
        of the return distribution.
        For non-distributional: returns Q-values directly.

        Args:
            x: Input frames, shape (batch, channels, height, width).

        Returns:
            Q-values, shape (batch, num_actions).
        """
        if self.use_distributional:
            log_probs = self.forward(x)
            probs = log_probs.exp()
            # Expected value: sum of probs * support atoms
            return (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        else:
            return self.forward(x)

    def reset_noise(self) -> None:
        """Reset noise in all NoisyLinear layers.

        Call once per episode to regenerate exploration noise.
        No-op if use_noisy=False (regular nn.Linear layers).
        """
        if not self.use_noisy:
            return

        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
