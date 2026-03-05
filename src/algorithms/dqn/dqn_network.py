"""
DQN Convolutional Neural Network.

Implements the Q-network architecture from the original DQN paper
(Mnih et al., 2015 - "Human-level control through deep RL").

Architecture:
    Input: (batch, 4, 84, 84) — 4 stacked grayscale frames
    Conv1: 32 filters, 8x8 kernel, stride 4 → (batch, 32, 20, 20)
    Conv2: 64 filters, 4x4 kernel, stride 2 → (batch, 64, 9, 9)
    Conv3: 64 filters, 3x3 kernel, stride 1 → (batch, 64, 7, 7)
    Flatten → (batch, 3136)
    FC1: 512 neurons with ReLU
    FC2: num_actions outputs (Q-value per action)

The network maps game frames to Q-values. The action with the
highest Q-value is selected during exploitation (non-random play).

Usage:
    net = DQNNetwork(input_channels=4, num_actions=7)
    q_values = net(state_tensor)  # shape: (batch, 7)
    best_action = q_values.argmax(dim=1)
"""

import torch
import torch.nn as nn
import numpy as np


class DQNNetwork(nn.Module):
    """
    Deep Q-Network with convolutional feature extraction.

    Processes stacked game frames through 3 convolutional layers
    to extract spatial features (platforms, enemies, gaps), then
    maps to Q-values through fully connected layers.

    Args:
        input_channels: Number of stacked frames (typically 4).
        num_actions: Number of possible actions (7 for Mario).
        input_height: Height of input frames. Default 84.
        input_width: Width of input frames. Default 84.

    Example:
        >>> net = DQNNetwork(input_channels=4, num_actions=7)
        >>> state = torch.randn(1, 4, 84, 84)  # batch of 1
        >>> q_values = net(state)
        >>> print(q_values.shape)  # torch.Size([1, 7])
        >>> action = q_values.argmax(dim=1).item()  # best action
    """

    def __init__(
        self,
        input_channels: int = 4,
        num_actions: int = 7,
        input_height: int = 84,
        input_width: int = 84,
    ):
        super().__init__()

        # ================================================================
        # Convolutional Feature Extractor
        # ================================================================
        # These layers learn to recognize visual patterns in the game:
        # Conv1: Large 8x8 filters detect gross features (ground, sky)
        # Conv2: Medium 4x4 filters detect structures (pipes, blocks)
        # Conv3: Small 3x3 filters detect fine details (enemies, coins)
        self.conv = nn.Sequential(
            # Conv1: (batch, 4, 84, 84) → (batch, 32, 20, 20)
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Conv2: (batch, 32, 20, 20) → (batch, 64, 9, 9)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Conv3: (batch, 64, 9, 9) → (batch, 64, 7, 7)
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate the flattened size after convolutions
        # We pass a dummy tensor through the conv layers to determine this
        conv_output_size = self._get_conv_output_size(
            input_channels, input_height, input_width,
        )

        # ================================================================
        # Fully Connected Q-Value Head
        # ================================================================
        # Maps extracted features to Q-values for each action
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def _get_conv_output_size(
        self, channels: int, height: int, width: int,
    ) -> int:
        """
        Calculate the flattened output size of the convolutional layers.

        Passes a dummy tensor through the conv layers to automatically
        compute the output dimensions, rather than calculating manually.

        Args:
            channels: Number of input channels.
            height: Input height.
            width: Input width.

        Returns:
            int: Total number of features after flattening.
        """
        dummy = torch.zeros(1, channels, height, width)
        output = self.conv(dummy)
        return int(np.prod(output.shape[1:]))  # Exclude batch dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: frames → Q-values.

        Args:
            x: Input tensor of shape (batch, channels, height, width).
               Pixel values should be normalized to [0, 1].

        Returns:
            torch.Tensor: Q-values for each action, shape (batch, num_actions).
        """
        # Extract features with convolutions
        features = self.conv(x)
        # Flatten spatial dimensions: (batch, 64, 7, 7) → (batch, 3136)
        features = features.reshape(features.size(0), -1)
        # Map to Q-values
        q_values = self.fc(features)
        return q_values
