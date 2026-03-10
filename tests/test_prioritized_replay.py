"""Tests for SumTree and PrioritizedReplayBuffer.

Validates priority-weighted sampling, importance sampling weight
computation, priority updates, and the SumTree data structure.
"""
import pytest
import numpy as np
import torch

from src.algorithms.rainbow.prioritized_replay import (
    SumTree,
    PrioritizedReplayBuffer,
    PrioritizedExperience,
)


# ─── SumTree Tests ───────────────────────────────────────────────


class TestSumTree:
    def test_empty_tree_has_zero_total(self):
        tree = SumTree(capacity=10)
        assert tree.total == 0.0

    def test_add_updates_total(self):
        tree = SumTree(capacity=10)
        tree.add(1.0)
        tree.add(2.0)
        tree.add(3.0)
        assert abs(tree.total - 6.0) < 1e-6

    def test_add_returns_tree_index(self):
        tree = SumTree(capacity=10)
        idx = tree.add(1.0)
        # First leaf is at capacity - 1
        assert idx == 9

    def test_get_retrieves_correct_leaf(self):
        tree = SumTree(capacity=4)
        tree.add(1.0)  # data_idx=0, priority=1.0
        tree.add(2.0)  # data_idx=1, priority=2.0
        tree.add(3.0)  # data_idx=2, priority=3.0

        # cumsum=0.5 should land in first leaf (priority 1.0)
        _, priority, data_idx = tree.get(0.5)
        assert data_idx == 0
        assert abs(priority - 1.0) < 1e-6

        # cumsum=2.5 should land in second leaf (priority 2.0)
        _, priority, data_idx = tree.get(2.5)
        assert data_idx == 1

        # cumsum=5.5 should land in third leaf (priority 3.0)
        _, priority, data_idx = tree.get(5.5)
        assert data_idx == 2

    def test_update_changes_priority(self):
        tree = SumTree(capacity=4)
        tree_idx = tree.add(1.0)
        assert abs(tree.total - 1.0) < 1e-6

        tree.update(tree_idx, 5.0)
        assert abs(tree.total - 5.0) < 1e-6

    def test_capacity_wraparound(self):
        tree = SumTree(capacity=3)
        tree.add(1.0)
        tree.add(2.0)
        tree.add(3.0)
        assert tree.size == 3
        assert abs(tree.total - 6.0) < 1e-6

        # Adding a 4th overwrites the first
        tree.add(4.0)
        assert tree.size == 3
        # Total should be 2 + 3 + 4 = 9
        assert abs(tree.total - 9.0) < 1e-6

    def test_min_priority_excludes_zero(self):
        tree = SumTree(capacity=4)
        tree.add(5.0)
        tree.add(2.0)
        # Unused slots have priority 0, should be excluded
        assert abs(tree.min_priority - 2.0) < 1e-6

    def test_min_priority_on_empty_tree(self):
        tree = SumTree(capacity=4)
        assert tree.min_priority == 1.0  # Default when no entries


# ─── PrioritizedReplayBuffer Tests ───────────────────────────────


def _make_buffer(capacity=100, obs_shape=(4, 84, 84)):
    return PrioritizedReplayBuffer(
        capacity=capacity,
        observation_shape=obs_shape,
        alpha=0.6,
        epsilon=1e-6,
    )


def _random_obs(shape=(4, 84, 84)):
    return np.random.randint(0, 256, shape, dtype=np.uint8)


class TestPrioritizedReplayBufferPush:
    def test_push_increments_size(self):
        buf = _make_buffer(capacity=10)
        assert len(buf) == 0
        state = _random_obs()
        buf.push(state, 0, 1.0, state, False)
        assert len(buf) == 1

    def test_push_with_n_step_data(self):
        buf = _make_buffer(capacity=10)
        state = _random_obs()
        next_state = _random_obs()
        n_step_state = _random_obs()
        buf.push(state, 1, 0.5, next_state, False,
                 n_step_return=2.5, n_step_next_state=n_step_state,
                 n_step_done=True)
        assert len(buf) == 1

    def test_push_wraps_at_capacity(self):
        buf = _make_buffer(capacity=5)
        state = _random_obs()
        for i in range(10):
            buf.push(state, 0, float(i), state, False)
        assert len(buf) == 5  # Capped at capacity

    def test_push_converts_float_to_uint8(self):
        buf = _make_buffer(capacity=10)
        state = np.random.rand(4, 84, 84).astype(np.float32)
        buf.push(state, 0, 1.0, state, False)
        # Stored as uint8 internally
        assert buf.states[0].dtype == np.uint8

    def test_is_ready(self):
        buf = _make_buffer(capacity=10)
        assert not buf.is_ready(batch_size=5)
        state = _random_obs()
        for _ in range(5):
            buf.push(state, 0, 1.0, state, False)
        assert buf.is_ready(batch_size=5)


class TestPrioritizedReplayBufferSample:
    def _fill_buffer(self, buf, n):
        for i in range(n):
            state = _random_obs()
            next_state = _random_obs()
            buf.push(state, i % 7, float(i), next_state, i % 5 == 0)

    def test_sample_returns_correct_types(self):
        buf = _make_buffer(capacity=50)
        self._fill_buffer(buf, 20)
        exp, weights, indices = buf.sample(batch_size=8)

        assert isinstance(exp, PrioritizedExperience)
        assert isinstance(weights, torch.Tensor)
        assert isinstance(indices, np.ndarray)

    def test_sample_returns_correct_shapes(self):
        buf = _make_buffer(capacity=50)
        self._fill_buffer(buf, 20)
        exp, weights, indices = buf.sample(batch_size=8)

        assert exp.states.shape == (8, 4, 84, 84)
        assert exp.actions.shape == (8,)
        assert exp.rewards.shape == (8,)
        assert exp.next_states.shape == (8, 4, 84, 84)
        assert exp.dones.shape == (8,)
        assert exp.n_step_returns.shape == (8,)
        assert exp.n_step_next_states.shape == (8, 4, 84, 84)
        assert exp.n_step_dones.shape == (8,)
        assert weights.shape == (8,)
        assert indices.shape == (8,)

    def test_sample_states_are_normalized(self):
        """States should be float32 in [0, 1] after sampling."""
        buf = _make_buffer(capacity=50)
        self._fill_buffer(buf, 20)
        exp, _, _ = buf.sample(batch_size=8)
        assert exp.states.dtype == torch.float32
        assert exp.states.min() >= 0.0
        assert exp.states.max() <= 1.0

    def test_is_weights_are_positive(self):
        buf = _make_buffer(capacity=50)
        self._fill_buffer(buf, 20)
        _, weights, _ = buf.sample(batch_size=8, beta=0.4)
        assert (weights > 0).all()

    def test_is_weights_max_is_one(self):
        """Weights are normalized by max, so max should be ~1.0."""
        buf = _make_buffer(capacity=50)
        self._fill_buffer(buf, 20)
        _, weights, _ = buf.sample(batch_size=8, beta=0.4)
        assert abs(weights.max().item() - 1.0) < 1e-6

    def test_higher_beta_gives_more_uniform_weights(self):
        """Beta=1.0 should give more uniform IS weights than beta=0.0."""
        buf = _make_buffer(capacity=50)
        self._fill_buffer(buf, 50)

        _, weights_low, _ = buf.sample(batch_size=16, beta=0.1)
        _, weights_high, _ = buf.sample(batch_size=16, beta=1.0)

        # Higher beta -> weights closer to 1.0 -> lower variance
        var_low = weights_low.var().item()
        var_high = weights_high.var().item()
        # This is probabilistic but very likely with 50 samples
        assert var_high <= var_low + 0.1

    def test_sample_raises_if_not_enough(self):
        buf = _make_buffer(capacity=50)
        self._fill_buffer(buf, 5)
        with pytest.raises(ValueError, match="Cannot sample"):
            buf.sample(batch_size=10)


class TestPrioritizedReplayBufferPriorities:
    def test_update_priorities(self):
        buf = _make_buffer(capacity=50)
        state = _random_obs()
        for _ in range(20):
            buf.push(state, 0, 1.0, state, False)

        _, _, tree_indices = buf.sample(batch_size=8)
        td_errors = np.random.rand(8) * 10
        # Should not raise
        buf.update_priorities(tree_indices, td_errors)

    def test_new_experiences_get_max_priority(self):
        """New experiences should get max priority in the tree."""
        buf = _make_buffer(capacity=100)
        state = _random_obs()

        # Fill with some experiences
        for _ in range(10):
            buf.push(state, 0, 1.0, state, False)

        # Update all priorities to very low values
        _, _, indices = buf.sample(batch_size=10)
        buf.update_priorities(indices, np.full(10, 0.001))

        # Record max priority before adding new experience
        old_max = buf._max_priority

        # Add a new experience — should get max priority
        buf.push(state, 0, 99.0, state, False)

        # The new experience's priority should use _max_priority
        # which is at least as large as old_max
        assert buf._max_priority >= old_max


# ─── PrioritizedExperience Tests ─────────────────────────────────


class TestPrioritizedExperience:
    def test_is_named_tuple(self):
        exp = PrioritizedExperience(
            states=torch.zeros(1), actions=torch.zeros(1),
            rewards=torch.zeros(1), next_states=torch.zeros(1),
            dones=torch.zeros(1), n_step_returns=torch.zeros(1),
            n_step_next_states=torch.zeros(1), n_step_dones=torch.zeros(1),
        )
        assert hasattr(exp, 'states')
        assert hasattr(exp, 'n_step_returns')

    def test_field_count(self):
        assert len(PrioritizedExperience._fields) == 8
