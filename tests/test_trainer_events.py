"""Tests for BaseTrainer event bus integration."""

import pytest

from src.algorithms.base_trainer import BaseTrainer
from src.achievements.event_bus import EventBus


# ── Minimal stubs ────────────────────────────────────────────────

class FakeEnv:
    action_space = type('AS', (), {'n': 4})()
    observation_space = type('OS', (), {'shape': (84, 84, 1)})()


class DummyTrainer(BaseTrainer):
    """Concrete trainer for testing (implements all abstract methods)."""

    def train(self, num_episodes=1):
        pass

    def evaluate(self, num_episodes=1):
        pass

    def save_checkpoint(self, path):
        pass

    def load_checkpoint(self, path):
        pass


def _make_trainer(event_bus=None):
    """Create a DummyTrainer with optional event bus."""
    return DummyTrainer(
        env=FakeEnv(),
        config={'save_freq': 50},
        event_bus=event_bus,
    )


# ── Tests ────────────────────────────────────────────────────────

class TestTrainerEvents:

    def test_no_event_bus(self):
        """publish_event should be a no-op when no event bus is set."""
        trainer = _make_trainer()
        # Should not raise
        trainer.publish_event({'type': 'test'})
        trainer.publish_episode_complete(10.0)
        trainer.publish_new_best(20.0)
        trainer.publish_training_start()
        trainer.publish_training_end(100)

    def test_publish_episode_complete(self):
        bus = EventBus()
        received = []
        bus.subscribe('episode_complete', received.append)

        trainer = _make_trainer(event_bus=bus)
        trainer.publish_episode_complete(42.0)

        assert len(received) == 1
        event = received[0]
        assert event['type'] == 'episode_complete'
        assert event['reward'] == 42.0

    def test_publish_new_best(self):
        bus = EventBus()
        received = []
        bus.subscribe('new_best_reward', received.append)

        trainer = _make_trainer(event_bus=bus)
        trainer.publish_new_best(99.5)

        assert len(received) == 1
        event = received[0]
        assert event['type'] == 'new_best_reward'
        assert event['reward'] == 99.5

    def test_publish_training_start(self):
        bus = EventBus()
        received = []
        bus.subscribe('training_start', received.append)

        trainer = _make_trainer(event_bus=bus)
        trainer.publish_training_start()

        assert len(received) == 1
        event = received[0]
        assert event['type'] == 'training_start'
        assert event['algorithm'] == 'DummyTrainer'

    def test_publish_training_end(self):
        bus = EventBus()
        received = []
        bus.subscribe('training_end', received.append)

        trainer = _make_trainer(event_bus=bus)
        trainer.best_reward = 150.0
        trainer.publish_training_end(total_episodes=200)

        assert len(received) == 1
        event = received[0]
        assert event['type'] == 'training_end'
        assert event['total_episodes'] == 200
        assert event['best_reward'] == 150.0
        assert 'elapsed' in event

    def test_auto_injects_game_id(self):
        bus = EventBus()
        received = []
        bus.subscribe('test', received.append)

        trainer = _make_trainer(event_bus=bus)
        trainer.game_id = 'snake'
        trainer.publish_event({'type': 'test'})

        assert len(received) == 1
        assert received[0]['game_id'] == 'snake'

    def test_auto_injects_episode(self):
        bus = EventBus()
        received = []
        bus.subscribe('test', received.append)

        trainer = _make_trainer(event_bus=bus)
        trainer.episode_count = 50
        trainer.publish_event({'type': 'test'})

        assert len(received) == 1
        assert received[0]['episode'] == 50
