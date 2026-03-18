"""
Lightweight publish/subscribe event bus for training events.

Events are dicts with a 'type' key and arbitrary data. Subscribers
register for specific event types or '*' for all events. Subscriber
exceptions are caught and logged to prevent one bad handler from
breaking the pipeline.
"""
from collections import defaultdict
from typing import Callable, Dict, List


class EventBus:
    """Thread-safe publish/subscribe event hub."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Register a callback for an event type. Use '*' for all events."""
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Remove a callback for an event type."""
        subs = self._subscribers.get(event_type, [])
        if callback in subs:
            subs.remove(callback)

    def publish(self, event: dict) -> None:
        """Publish an event to all matching subscribers."""
        event_type = event.get('type', '')
        for cb in self._subscribers.get(event_type, []):
            try:
                cb(event)
            except Exception as e:
                print(f'  [EventBus] Subscriber error on {event_type}: {e}')
        if event_type != '*':
            for cb in self._subscribers.get('*', []):
                try:
                    cb(event)
                except Exception as e:
                    print(f'  [EventBus] Wildcard subscriber error: {e}')
