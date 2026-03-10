"""
Prioritized Experience Replay Buffer for Rainbow DQN.

Implements priority-based sampling using a SumTree data structure,
as described in:
    Schaul et al., 2016 - "Prioritized Experience Replay"

Key idea: Instead of sampling uniformly, sample experiences proportional
to their TD-error (how "surprising" they are). The agent learns faster
by focusing on transitions it currently understands least.

Components:
    SumTree: Binary tree for O(log N) weighted sampling.
    PrioritizedReplayBuffer: Drop-in replacement for ReplayBuffer with
        priority-based sampling and importance sampling weight correction.

The buffer also stores pre-computed n-step returns for multi-step
learning, a key Rainbow DQN improvement.

Memory optimization: States stored as uint8 (0-255), converted to
float32 (0.0-1.0) only when sampled, saving 4x memory.

Usage:
    buffer = PrioritizedReplayBuffer(capacity=100000)
    buffer.push(state, action, reward, next_state, done,
                n_step_return, n_step_next_state, n_step_done)
    batch, weights, indices = buffer.sample(32, beta=0.4)
    buffer.update_priorities(indices, td_errors)
"""

import numpy as np
import torch
from typing import Tuple, NamedTuple, Optional


class PrioritizedExperience(NamedTuple):
    """Experience tuple with n-step return data.

    Attributes:
        states: Batch of current states, shape (batch, C, H, W).
        actions: Batch of actions, shape (batch,).
        rewards: Batch of immediate rewards, shape (batch,).
        next_states: Batch of next states, shape (batch, C, H, W).
        dones: Batch of done flags, shape (batch,).
        n_step_returns: Batch of n-step discounted returns, shape (batch,).
        n_step_next_states: Batch of n-step lookahead states, shape (batch, C, H, W).
        n_step_dones: Batch of n-step done flags, shape (batch,).
    """
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    n_step_returns: torch.Tensor
    n_step_next_states: torch.Tensor
    n_step_dones: torch.Tensor


class SumTree:
    """Binary tree where each leaf holds a priority value and each
    internal node holds the sum of its children.

    Enables O(log N) operations:
        - add/update a priority
        - sample an index proportional to priorities

    The tree is stored as a flat array of size (2 * capacity - 1):
        - Internal nodes: indices 0 to capacity - 2
        - Leaf nodes: indices capacity - 1 to 2 * capacity - 2

    Args:
        capacity: Maximum number of leaf entries (experiences).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write_position = 0
        self.size = 0

    @property
    def total(self) -> float:
        """Sum of all priorities (root node value)."""
        return float(self.tree[0])

    def _propagate(self, tree_index: int, change: float) -> None:
        """Update parent nodes after a leaf value changes."""
        parent = (tree_index - 1) // 2
        self.tree[parent] += change
        if parent > 0:
            self._propagate(parent, change)

    def add(self, priority: float) -> int:
        """Add a new entry with the given priority.

        Returns the tree index of the added leaf.
        If the tree is full, overwrites the oldest entry.
        """
        tree_index = self.write_position + self.capacity - 1
        self.update(tree_index, priority)
        self.write_position = (self.write_position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return tree_index

    def update(self, tree_index: int, priority: float) -> None:
        """Update the priority of an existing leaf.

        Args:
            tree_index: Index in the tree array (leaf index).
            priority: New priority value.
        """
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    def get(self, cumulative_sum: float) -> Tuple[int, float, int]:
        """Find the leaf corresponding to a cumulative sum value.

        Walk down the tree: go left if cumsum < left child,
        otherwise subtract left child and go right.

        Args:
            cumulative_sum: Random value in [0, total_priority).

        Returns:
            (tree_index, priority, data_index) tuple.
        """
        node = 0  # Start at root
        while True:
            left = 2 * node + 1
            right = left + 1

            # Reached a leaf node
            if left >= len(self.tree):
                break

            if cumulative_sum <= self.tree[left]:
                node = left
            else:
                cumulative_sum -= self.tree[left]
                node = right

        data_index = node - (self.capacity - 1)
        return node, self.tree[node], data_index

    @property
    def min_priority(self) -> float:
        """Minimum priority among stored entries."""
        leaf_start = self.capacity - 1
        leaves = self.tree[leaf_start:leaf_start + self.size]
        nonzero = leaves[leaves > 0]
        if len(nonzero) == 0:
            return 1.0
        return float(nonzero.min())


class PrioritizedReplayBuffer:
    """Replay buffer with priority-based sampling via SumTree.

    Matches the interface pattern of ReplayBuffer (push/sample/len)
    but adds:
        - Priority-weighted sampling instead of uniform
        - Importance sampling weights to correct for bias
        - N-step return storage for multi-step learning

    Args:
        capacity: Maximum stored experiences.
        observation_shape: Shape of a single observation (C, H, W).
        alpha: Priority exponent. 0 = uniform, 1 = full prioritization.
        epsilon: Small constant added to TD errors for numerical stability.
    """

    def __init__(
        self,
        capacity: int = 100000,
        observation_shape: Tuple[int, ...] = (4, 84, 84),
        alpha: float = 0.6,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.tree = SumTree(capacity)

        # Pre-allocate numpy arrays (uint8 for states = 4x memory savings)
        self.states = np.zeros(
            (capacity, *observation_shape), dtype=np.uint8,
        )
        self.next_states = np.zeros(
            (capacity, *observation_shape), dtype=np.uint8,
        )
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        # N-step return data
        self.n_step_returns = np.zeros(capacity, dtype=np.float32)
        self.n_step_next_states = np.zeros(
            (capacity, *observation_shape), dtype=np.uint8,
        )
        self.n_step_dones = np.zeros(capacity, dtype=np.bool_)

        self._max_priority = 1.0
        self.position = 0
        self.size = 0

    def _to_uint8(self, arr: np.ndarray) -> np.ndarray:
        """Convert float observations to uint8 for storage."""
        if arr.dtype in (np.float32, np.float64):
            return (arr * 255).astype(np.uint8)
        return arr

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        n_step_return: float = 0.0,
        n_step_next_state: Optional[np.ndarray] = None,
        n_step_done: bool = False,
    ) -> None:
        """Add an experience with maximum priority.

        New experiences get max priority so they're sampled at least
        once before their priority is updated with the actual TD error.

        Args:
            state: Current observation (C, H, W).
            action: Action taken.
            reward: Immediate reward.
            next_state: Next observation (C, H, W).
            done: Episode ended flag.
            n_step_return: Pre-computed n-step discounted return.
            n_step_next_state: State n steps ahead.
            n_step_done: Whether episode ended within n steps.
        """
        idx = self.position

        self.states[idx] = self._to_uint8(state)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = self._to_uint8(next_state)
        self.dones[idx] = done

        # N-step data (defaults to 1-step if not provided)
        self.n_step_returns[idx] = n_step_return
        if n_step_next_state is not None:
            self.n_step_next_states[idx] = self._to_uint8(n_step_next_state)
        else:
            self.n_step_next_states[idx] = self._to_uint8(next_state)
        self.n_step_dones[idx] = n_step_done

        # New experiences get max priority
        priority = self._max_priority ** self.alpha
        self.tree.add(priority)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
        device: str = 'cpu',
    ) -> Tuple[PrioritizedExperience, torch.Tensor, np.ndarray]:
        """Sample a batch weighted by priority.

        Divides the total priority range into batch_size equal segments
        and samples one transition from each segment (stratified sampling).

        Args:
            batch_size: Number of experiences to sample.
            beta: IS weight exponent. 0 = no correction, 1 = full.
                Annealed from ~0.4 to 1.0 during training.
            device: PyTorch device.

        Returns:
            (experience, is_weights, tree_indices):
                experience: PrioritizedExperience named tuple.
                is_weights: Importance sampling weights, shape (batch,).
                tree_indices: SumTree indices for priority updates.
        """
        if batch_size > self.size:
            raise ValueError(
                f'Cannot sample {batch_size} from buffer with {self.size} experiences'
            )

        tree_indices = np.zeros(batch_size, dtype=np.int64)
        data_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        # Stratified sampling: divide total priority into equal segments
        segment_size = self.tree.total / batch_size

        for i in range(batch_size):
            low = segment_size * i
            high = segment_size * (i + 1)
            cumsum = np.random.uniform(low, high)
            tree_idx, priority, data_idx = self.tree.get(cumsum)
            tree_indices[i] = tree_idx
            data_indices[i] = data_idx
            priorities[i] = priority

        # Importance sampling weights: w_i = (N * P(i))^(-beta) / max(w)
        # P(i) = priority_i / sum(priorities)
        sampling_probs = priorities / self.tree.total
        # Clamp to avoid division by zero
        sampling_probs = np.clip(sampling_probs, 1e-10, None)
        is_weights = (self.size * sampling_probs) ** (-beta)
        # Normalize by max weight for stability
        is_weights = is_weights / is_weights.max()
        is_weights = torch.FloatTensor(is_weights).to(device)

        # Convert stored data to tensors
        states = torch.FloatTensor(
            self.states[data_indices].astype(np.float32) / 255.0
        ).to(device)
        next_states = torch.FloatTensor(
            self.next_states[data_indices].astype(np.float32) / 255.0
        ).to(device)
        actions = torch.LongTensor(self.actions[data_indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[data_indices]).to(device)
        dones = torch.BoolTensor(self.dones[data_indices]).to(device)

        n_step_returns = torch.FloatTensor(
            self.n_step_returns[data_indices],
        ).to(device)
        n_step_next_states = torch.FloatTensor(
            self.n_step_next_states[data_indices].astype(np.float32) / 255.0
        ).to(device)
        n_step_dones = torch.BoolTensor(
            self.n_step_dones[data_indices],
        ).to(device)

        experience = PrioritizedExperience(
            states, actions, rewards, next_states, dones,
            n_step_returns, n_step_next_states, n_step_dones,
        )

        return experience, is_weights, tree_indices

    def update_priorities(
        self,
        tree_indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        """Update priorities based on new TD errors.

        priority = (|td_error| + epsilon) ^ alpha

        Args:
            tree_indices: SumTree indices from the last sample() call.
            td_errors: New TD errors for the sampled batch.
        """
        td_errors = np.abs(td_errors) + self.epsilon
        for idx, td_error in zip(tree_indices, td_errors):
            priority = td_error ** self.alpha
            self.tree.update(idx, priority)
            self._max_priority = max(self._max_priority, td_error)

    def __len__(self) -> int:
        """Return the number of stored experiences."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return self.size >= batch_size
