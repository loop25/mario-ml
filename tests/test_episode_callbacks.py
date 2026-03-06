"""Tests for the episode callback mechanism in BaseTrainer."""
import threading
import pytest


class FakeTrainerWithCallbacks:
    """Minimal trainer stub with the callback mechanism."""

    def __init__(self):
        self._episode_callbacks = []

    def add_episode_callback(self, callback):
        self._episode_callbacks.append(callback)

    def _fire_episode_complete(self, reward, distance=0, completed=False):
        for cb in self._episode_callbacks:
            try:
                cb(reward=reward, distance=distance, completed=completed)
            except Exception:
                pass  # Swallow errors (mirrors BaseTrainer behavior)


def test_callback_receives_reward():
    """Callback should receive the reward from the completed episode."""
    received = []
    t = FakeTrainerWithCallbacks()
    t.add_episode_callback(lambda reward, **kw: received.append(reward))
    t._fire_episode_complete(reward=123.5, distance=400, completed=False)
    assert received == [123.5]


def test_callback_receives_all_kwargs():
    """Callback should receive reward, distance, and completed."""
    received = {}
    def capture(**kwargs):
        received.update(kwargs)
    t = FakeTrainerWithCallbacks()
    t.add_episode_callback(capture)
    t._fire_episode_complete(reward=500.0, distance=1200, completed=True)
    assert received == {'reward': 500.0, 'distance': 1200, 'completed': True}


def test_multiple_callbacks_all_called():
    """All registered callbacks should fire on each episode."""
    counts = [0, 0]
    t = FakeTrainerWithCallbacks()
    t.add_episode_callback(lambda **kw: counts.__setitem__(0, counts[0] + 1))
    t.add_episode_callback(lambda **kw: counts.__setitem__(1, counts[1] + 1))
    for _ in range(5):
        t._fire_episode_complete(reward=100.0)
    assert counts == [5, 5]


def test_callback_error_does_not_stop_others():
    """If one callback raises, remaining callbacks should still fire."""
    called = []
    def bad_callback(**kw):
        raise ValueError("boom")
    def good_callback(**kw):
        called.append(True)

    t = FakeTrainerWithCallbacks()
    t.add_episode_callback(bad_callback)
    t.add_episode_callback(good_callback)
    t._fire_episode_complete(reward=42.0)
    assert called == [True]


def test_no_callbacks_does_not_error():
    """Firing with no registered callbacks should be a no-op."""
    t = FakeTrainerWithCallbacks()
    t._fire_episode_complete(reward=0.0)  # Should not raise
