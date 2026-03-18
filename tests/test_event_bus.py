"""Tests for the EventBus publish/subscribe system."""
import pytest
from src.achievements.event_bus import EventBus


class TestEventBus:
    """EventBus test suite."""

    def test_subscribe_and_publish(self):
        """Basic pub/sub works: subscriber receives the published event."""
        bus = EventBus()
        received = []
        bus.subscribe('episode_done', lambda e: received.append(e))
        event = {'type': 'episode_done', 'reward': 42}
        bus.publish(event)
        assert received == [event]

    def test_multiple_subscribers(self):
        """Multiple handlers for the same event type all get called."""
        bus = EventBus()
        results_a, results_b = [], []
        bus.subscribe('tick', lambda e: results_a.append(e))
        bus.subscribe('tick', lambda e: results_b.append(e))
        event = {'type': 'tick', 'step': 1}
        bus.publish(event)
        assert results_a == [event]
        assert results_b == [event]

    def test_no_crosstalk(self):
        """Subscribing to 'alpha' does not receive 'beta' events."""
        bus = EventBus()
        received = []
        bus.subscribe('alpha', lambda e: received.append(e))
        bus.publish({'type': 'beta', 'val': 1})
        assert received == []

    def test_wildcard_subscriber(self):
        """A '*' subscriber receives all events regardless of type."""
        bus = EventBus()
        received = []
        bus.subscribe('*', lambda e: received.append(e))
        bus.publish({'type': 'foo'})
        bus.publish({'type': 'bar'})
        assert len(received) == 2
        assert received[0]['type'] == 'foo'
        assert received[1]['type'] == 'bar'

    def test_unsubscribe(self):
        """An unsubscribed callback is no longer called."""
        bus = EventBus()
        received = []
        cb = lambda e: received.append(e)
        bus.subscribe('x', cb)
        bus.unsubscribe('x', cb)
        bus.publish({'type': 'x'})
        assert received == []

    def test_publish_without_subscribers(self):
        """Publishing with no subscribers does not raise."""
        bus = EventBus()
        bus.publish({'type': 'lonely_event'})  # should not raise

    def test_subscriber_exception_does_not_crash(self):
        """A failing subscriber does not prevent other subscribers from running."""
        bus = EventBus()
        received = []

        def bad_handler(e):
            raise RuntimeError('boom')

        def good_handler(e):
            received.append(e)

        bus.subscribe('test', bad_handler)
        bus.subscribe('test', good_handler)
        event = {'type': 'test', 'data': 123}
        bus.publish(event)  # should not raise
        assert received == [event]
