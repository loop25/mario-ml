"""
Experience Replay Buffer for DQN.

Stores past experiences (state, action, reward, next_state, done) and
provides random sampling for training. This breaks the temporal correlation
between consecutive experiences, which is critical for stable DQN training.

Without replay: The network trains on correlated consecutive frames,
leading to unstable learning and catastrophic forgetting.

With replay: Random sampling from a large buffer provides diverse,
uncorrelated training batches, leading to stable convergence.

Implementation uses numpy arrays for memory-efficient storage with
a circular buffer pattern (oldest experiences are overwritten when full).

Memory usage estimate (84x84x4 uint8 observations):
    - 100K experiences ≈ 2.8 GB (state + next_state)
    - 50K experiences ≈ 1.4 GB
    - Adjust buffer_size based on available RAM

Usage:
    buffer = ReplayBuffer(capacity=100000)
    buffer.push(state, action, reward, next_state, done)
    batch = buffer.sample(batch_size=32)
"""

import numpy as np
from typing import Tuple, NamedTuple
import torch


class Experience(NamedTuple):
    """
    A single experience tuple from the environment.

    Attributes:
        states: Batch of states, shape (batch, C, H, W).
        actions: Batch of actions taken, shape (batch,).
        rewards: Batch of rewards received, shape (batch,).
        next_states: Batch of next states, shape (batch, C, H, W).
        dones: Batch of done flags, shape (batch,).
    """
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """
    Fixed-size circular experience replay buffer.

    Stores experiences as numpy arrays for memory efficiency,
    converts to PyTorch tensors on sampling.

    Args:
        capacity: Maximum number of experiences to store.
                  Oldest experiences are overwritten when full.
        observation_shape: Shape of a single observation (C, H, W).
                          Default (4, 84, 84) for stacked frames.

    Attributes:
        capacity: Buffer size.
        size: Current number of stored experiences.
        position: Next write position in the circular buffer.

    Example:
        >>> buffer = ReplayBuffer(capacity=100000)
        >>> buffer.push(state, action=2, reward=1.0, next_state, done=False)
        >>> print(len(buffer))  # 1
        >>> batch = buffer.sample(32)  # Random batch of 32
    """

    def __init__(
        self,
        capacity: int = 100000,
        observation_shape: Tuple[int, ...] = (4, 84, 84),
    ):
        self.capacity = capacity
        self.position = 0
        self.size = 0

        # Pre-allocate numpy arrays for all data
        # Using uint8 for observations saves 4x memory vs float32
        self.states = np.zeros(
            (capacity, *observation_shape), dtype=np.uint8,
        )
        self.next_states = np.zeros(
            (capacity, *observation_shape), dtype=np.uint8,
        )
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add one experience to the buffer.

        If the buffer is full, the oldest experience is overwritten.

        Args:
            state: Current observation, shape (C, H, W).
            action: Action taken (integer index).
            reward: Reward received after taking the action.
            next_state: Observation after taking the action, shape (C, H, W).
            done: Whether the episode ended.
        """
        # Convert float observations to uint8 for storage efficiency
        if state.dtype == np.float32 or state.dtype == np.float64:
            state = (state * 255).astype(np.uint8)
        if next_state.dtype == np.float32 or next_state.dtype == np.float64:
            next_state = (next_state * 255).astype(np.uint8)

        # Store at current position (circular overwrite)
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        # Advance position with wraparound
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: str = 'cpu',
    ) -> Experience:
        """
        Sample a random batch of experiences.

        Converts numpy arrays to PyTorch tensors and normalizes
        pixel values from [0, 255] uint8 to [0, 1] float32.

        Args:
            batch_size: Number of experiences to sample.
            device: PyTorch device ('cpu' or 'cuda').

        Returns:
            Experience: Named tuple with batched tensors.

        Raises:
            ValueError: If batch_size > number of stored experiences.
        """
        if batch_size > self.size:
            raise ValueError(
                f'Cannot sample {batch_size} from buffer with {self.size} experiences'
            )

        # Random indices without replacement
        indices = np.random.choice(self.size, batch_size, replace=False)

        # Convert to tensors and normalize observations to [0, 1]
        states = torch.FloatTensor(
            self.states[indices].astype(np.float32) / 255.0
        ).to(device)
        next_states = torch.FloatTensor(
            self.next_states[indices].astype(np.float32) / 255.0
        ).to(device)
        actions = torch.LongTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        dones = torch.BoolTensor(self.dones[indices]).to(device)

        return Experience(states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """Return the number of stored experiences."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if the buffer has enough experiences for sampling.

        Args:
            batch_size: Required batch size.

        Returns:
            bool: True if enough experiences are stored.
        """
        return self.size >= batch_size
