"""Tests for the Shared Experience Store (src/experience/experience_store.py).

Validates:
    - Adding and loading trajectories
    - Disk persistence (index.json + .npz files)
    - Capacity eviction (lowest-return removed)
    - Batch sampling (correct shapes, padding, masking)
    - Game filtering and return filtering
    - Statistics and query API
    - Clear operation
"""

import os
import json
import numpy as np
import pytest

from games.reward_config import TokenConfig
from src.experience.tokenizer import Tokenizer, TokenizedTrajectory
from src.experience.experience_store import ExperienceStore, TrajectoryMeta


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def store_dir(tmp_path):
    """Temp directory for experience store."""
    return str(tmp_path / 'test_store')


@pytest.fixture
def store(store_dir):
    return ExperienceStore(store_dir=store_dir, max_trajectories=100)


def make_trajectory(game_token_id=42, length=10, total_return=50.0):
    """Create a minimal TokenizedTrajectory for testing."""
    return TokenizedTrajectory(
        game_token_id=game_token_id,
        observations=np.random.randint(0, 255, (length, 1, 84, 84), dtype=np.uint8),
        actions=np.random.randint(0, 7, length, dtype=np.int32),
        rewards=np.ones(length, dtype=np.float32) * (total_return / length),
        returns_to_go=np.linspace(total_return, total_return / length, length).astype(np.float32),
        timesteps=np.arange(length, dtype=np.int32),
        dones=np.zeros(length, dtype=bool),
        total_return=total_return,
        length=length,
    )


# ── Add & Load ────────────────────────────────────────────────────────


class TestAddAndLoad:

    def test_add_returns_id(self, store):
        traj = make_trajectory()
        traj_id = store.add_trajectory(traj)
        assert traj_id == 0

    def test_sequential_ids(self, store):
        id0 = store.add_trajectory(make_trajectory())
        id1 = store.add_trajectory(make_trajectory())
        assert id0 == 0
        assert id1 == 1

    def test_len_after_add(self, store):
        assert len(store) == 0
        store.add_trajectory(make_trajectory())
        assert len(store) == 1
        store.add_trajectory(make_trajectory())
        assert len(store) == 2

    def test_load_trajectory_data(self, store):
        original = make_trajectory(game_token_id=7, length=15, total_return=75.0)
        store.add_trajectory(original)

        loaded = store.load_trajectory(store._index[0])
        assert loaded.game_token_id == 7
        assert loaded.length == 15
        assert loaded.total_return == 75.0
        np.testing.assert_array_equal(loaded.actions, original.actions)
        np.testing.assert_array_equal(loaded.observations, original.observations)

    def test_npz_file_created(self, store, store_dir):
        store.add_trajectory(make_trajectory())
        assert os.path.exists(os.path.join(store_dir, 'traj_000000.npz'))


# ── Disk Persistence ─────────────────────────────────────────────────


class TestPersistence:

    def test_index_json_created(self, store, store_dir):
        store.add_trajectory(make_trajectory())
        index_path = os.path.join(store_dir, 'index.json')
        assert os.path.exists(index_path)

        with open(index_path, 'r') as f:
            data = json.load(f)
        assert len(data['trajectories']) == 1
        assert data['next_id'] == 1

    def test_reload_from_disk(self, store_dir):
        """Create store, add data, create new store from same dir — data persists."""
        store1 = ExperienceStore(store_dir=store_dir, max_trajectories=100)
        store1.add_trajectory(make_trajectory(total_return=99.0))
        store1.add_trajectory(make_trajectory(total_return=88.0))

        store2 = ExperienceStore(store_dir=store_dir, max_trajectories=100)
        assert len(store2) == 2
        assert store2._next_id == 2

    def test_reload_preserves_data(self, store_dir):
        store1 = ExperienceStore(store_dir=store_dir, max_trajectories=100)
        traj = make_trajectory(game_token_id=99, total_return=42.0)
        store1.add_trajectory(traj)

        store2 = ExperienceStore(store_dir=store_dir, max_trajectories=100)
        loaded = store2.load_trajectory(store2._index[0])
        assert loaded.game_token_id == 99
        assert loaded.total_return == 42.0


# ── Capacity & Eviction ──────────────────────────────────────────────


class TestEviction:

    def test_eviction_removes_lowest_return(self):
        """When over capacity, lowest-return trajectories are removed."""
        store = ExperienceStore(
            store_dir='test_evict_tmp', max_trajectories=3,
        )
        try:
            store.add_trajectory(make_trajectory(total_return=10.0))
            store.add_trajectory(make_trajectory(total_return=50.0))
            store.add_trajectory(make_trajectory(total_return=30.0))
            # At capacity (3/3)
            assert len(store) == 3

            # Adding one more should evict the lowest (return=10.0)
            store.add_trajectory(make_trajectory(total_return=40.0))
            assert len(store) == 3

            returns = [m.total_return for m in store._index]
            assert 10.0 not in returns
            assert 50.0 in returns
            assert 40.0 in returns
        finally:
            store.clear()
            import shutil
            if os.path.exists('test_evict_tmp'):
                shutil.rmtree('test_evict_tmp')

    def test_eviction_removes_npz_file(self, store_dir):
        store = ExperienceStore(store_dir=store_dir, max_trajectories=2)
        store.add_trajectory(make_trajectory(total_return=10.0))
        store.add_trajectory(make_trajectory(total_return=50.0))

        # This should evict the first (return=10)
        store.add_trajectory(make_trajectory(total_return=30.0))

        # The evicted file should be deleted
        assert not os.path.exists(os.path.join(store_dir, 'traj_000000.npz'))


# ── Batch Sampling ────────────────────────────────────────────────────


class TestSampling:

    def test_sample_batch_shapes(self, store):
        for _ in range(5):
            store.add_trajectory(make_trajectory(length=30))

        batch = store.sample_batch(batch_size=4, context_length=10)
        assert batch['observations'].shape == (4, 10, 1, 84, 84)
        assert batch['actions'].shape == (4, 10)
        assert batch['returns_to_go'].shape == (4, 10)
        assert batch['timesteps'].shape == (4, 10)
        assert batch['game_tokens'].shape == (4,)
        assert batch['mask'].shape == (4, 10)

    def test_sample_batch_dtypes(self, store):
        for _ in range(3):
            store.add_trajectory(make_trajectory())
        batch = store.sample_batch(batch_size=2, context_length=5)
        assert batch['observations'].dtype == np.float32
        assert batch['actions'].dtype == np.int64
        assert batch['returns_to_go'].dtype == np.float32
        assert batch['mask'].dtype == bool

    def test_mask_for_short_trajectory(self, store):
        """Short trajectories should be padded with mask=False."""
        store.add_trajectory(make_trajectory(length=3))
        batch = store.sample_batch(batch_size=1, context_length=10)
        mask = batch['mask'][0]
        assert mask[:3].all()   # First 3 valid
        assert not mask[3:].any()  # Rest padded

    def test_game_filter(self, store):
        store.add_trajectory(make_trajectory(game_token_id=1))
        store.add_trajectory(make_trajectory(game_token_id=2))
        store.add_trajectory(make_trajectory(game_token_id=1))

        batch = store.sample_batch(batch_size=5, context_length=5, game_filter=1)
        # All sampled should be game 1
        assert (batch['game_tokens'] == 1).all()

    def test_return_filter(self, store):
        store.add_trajectory(make_trajectory(total_return=10.0))
        store.add_trajectory(make_trajectory(total_return=100.0))
        store.add_trajectory(make_trajectory(total_return=50.0))

        batch = store.sample_batch(batch_size=3, context_length=5, min_return=50.0)
        # Should only sample from trajectories with return >= 50
        # (game_tokens won't tell us which, but the call should not crash)
        assert batch['observations'].shape[0] == 3

    def test_empty_store_raises(self, store):
        with pytest.raises(ValueError, match="No eligible trajectories"):
            store.sample_batch(batch_size=1, context_length=5)

    def test_obs_normalized(self, store):
        """Observations in sampled batch should be in [0, 1] range."""
        store.add_trajectory(make_trajectory())
        batch = store.sample_batch(batch_size=1, context_length=5)
        assert batch['observations'].max() <= 1.0
        assert batch['observations'].min() >= 0.0


# ── Stats & Query ─────────────────────────────────────────────────────


class TestStatsAndQuery:

    def test_empty_stats(self, store):
        stats = store.get_stats()
        assert stats['total_trajectories'] == 0

    def test_stats_after_add(self, store):
        store.add_trajectory(make_trajectory(game_token_id=1, length=10, total_return=50.0))
        store.add_trajectory(make_trajectory(game_token_id=1, length=20, total_return=80.0))
        store.add_trajectory(make_trajectory(game_token_id=2, length=15, total_return=60.0))

        stats = store.get_stats()
        assert stats['total_trajectories'] == 3
        assert stats['total_timesteps'] == 45
        assert 1 in stats['games']
        assert 2 in stats['games']
        assert stats['games'][1]['count'] == 2
        assert stats['games'][2]['count'] == 1

    def test_get_game_ids(self, store):
        store.add_trajectory(make_trajectory(game_token_id=10))
        store.add_trajectory(make_trajectory(game_token_id=20))
        store.add_trajectory(make_trajectory(game_token_id=10))
        ids = store.get_game_ids()
        assert set(ids) == {10, 20}

    def test_clear(self, store, store_dir):
        store.add_trajectory(make_trajectory())
        store.add_trajectory(make_trajectory())
        assert len(store) == 2

        store.clear()
        assert len(store) == 0
        # npz files should be deleted
        npz_files = [f for f in os.listdir(store_dir) if f.endswith('.npz')]
        assert len(npz_files) == 0
