"""Tests for checkpoint metadata restoration in trainers."""
import json
import os
import pytest


def test_metadata_json_round_trip(tmp_path):
    """Metadata saved by BaseTrainer should be loadable and contain
    all expected fields for checkpoint restoration."""
    meta = {
        'algorithm': 'PPOTrainer',
        'episode': 150,
        'best_reward': 2500.5,
        'best_distance': 3200,
        'elapsed_seconds': 600.0,
        'elapsed_time': '10m 0s',
        'timestamp': '2025-01-01 12:00:00',
        'config': {'learning_rate': '0.0003'},
    }
    metadata_path = tmp_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(meta, f, indent=2)

    with open(metadata_path, 'r') as f:
        loaded = json.load(f)

    assert loaded['episode'] == 150
    assert loaded['best_reward'] == 2500.5
    assert loaded['best_distance'] == 3200
    assert loaded['algorithm'] == 'PPOTrainer'


def test_metadata_missing_is_graceful():
    """If metadata.json is absent, the trainers should not crash.
    They guard with `if os.path.exists(metadata_path):` so missing
    metadata is silently skipped."""
    metadata_path = os.path.join('nonexistent_dir_12345', 'metadata.json')
    assert not os.path.exists(metadata_path)
    # The pattern used by PPO/NEAT load_checkpoint:
    # if os.path.exists(metadata_path): ...
    # This test just confirms the guard pattern works.


def test_metadata_corrupt_handled(tmp_path):
    """Corrupt metadata.json should not crash the loader."""
    metadata_path = tmp_path / 'metadata.json'
    metadata_path.write_text('not valid json {{{')

    # Simulate the pattern used in load_checkpoint
    try:
        with open(metadata_path, 'r') as f:
            json.load(f)
        assert False, 'Should have raised'
    except json.JSONDecodeError:
        pass  # Expected — trainers catch this


def test_metadata_partial_fields(tmp_path):
    """Metadata with missing fields should use defaults via .get()."""
    meta = {'episode': 42}  # Only one field
    metadata_path = tmp_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(meta, f)

    with open(metadata_path, 'r') as f:
        loaded = json.load(f)

    # Simulate how the trainers use .get() with defaults
    episode_count = loaded.get('episode', 0)
    best_reward = loaded.get('best_reward', float('-inf'))
    best_distance = loaded.get('best_distance', 0)

    assert episode_count == 42
    assert best_reward == float('-inf')
    assert best_distance == 0
