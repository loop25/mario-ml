"""Tests for the scheduler engine."""

import tempfile
import threading
import time
from datetime import datetime, timedelta

from src.achievements.event_bus import EventBus
from src.scheduler.calendar_store import CalendarStore
from src.scheduler.scheduler_engine import SchedulerEngine
from src.scheduler.session import create_session


def _make_store(tmp_path: str) -> CalendarStore:
    """Create a CalendarStore backed by a temp file."""
    return CalendarStore(path=f"{tmp_path}/schedule.json")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_initial_state():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        engine = SchedulerEngine(store)
        assert engine.is_running() is False
        assert engine.get_current_session() is None


def test_start_stop():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        engine = SchedulerEngine(store)
        engine.start()
        assert engine.is_running() is True
        engine.stop()
        assert engine.is_running() is False


def test_run_next_executes_callback():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        session = create_session("snake", "ppo", episodes=10)
        store.add_session(session)

        engine = SchedulerEngine(store)
        engine.set_execute_callback(lambda s: {"reward": 42})

        result = engine.run_next()

        assert result is not None
        assert result.status == "completed"
        assert result.result_summary == {"reward": 42}
        assert result.actual_start is not None
        assert result.actual_end is not None

        # Verify the store was updated too
        stored = store.get_session(session.session_id)
        assert stored.status == "completed"


def test_run_next_no_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        engine = SchedulerEngine(store)
        assert engine.run_next() is None


def test_failed_callback():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        session = create_session("tetris", "dqn")
        store.add_session(session)

        def _boom(s):
            raise RuntimeError("GPU on fire")

        engine = SchedulerEngine(store)
        engine.set_execute_callback(_boom)

        result = engine.run_next()

        assert result is not None
        assert result.status == "failed"
        assert "error" in result.result_summary
        assert "GPU on fire" in result.result_summary["error"]


def test_cancel_current():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        session = create_session("mario", "ppo")
        store.add_session(session)

        cancel_happened = threading.Event()

        def _slow(s):
            # Spin until cancelled or 5 seconds
            for _ in range(50):
                if s.status == "cancelled":
                    cancel_happened.set()
                    return {"cancelled": True}
                time.sleep(0.1)
            return {}

        engine = SchedulerEngine(store)
        engine.set_execute_callback(_slow)

        # Run in background thread so we can cancel from main thread
        t = threading.Thread(target=engine.run_next)
        t.start()

        # Wait a bit for the session to start running
        time.sleep(0.3)
        assert engine.get_current_session() is not None
        assert engine.cancel_current() is True

        t.join(timeout=5)
        cancel_happened.wait(timeout=5)

        assert session.status == "cancelled"


def test_session_events_published():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        bus = EventBus()
        events = []
        bus.subscribe("session_start", lambda e: events.append(e))
        bus.subscribe("session_complete", lambda e: events.append(e))

        session = create_session("chess", "ppo")
        store.add_session(session)

        engine = SchedulerEngine(store, event_bus=bus)
        engine.set_execute_callback(lambda s: {"elo": 1200})
        engine.run_next()

        assert len(events) == 2
        assert events[0]["type"] == "session_start"
        assert events[0]["session_id"] == session.session_id
        assert events[0]["game_id"] == "chess"
        assert events[1]["type"] == "session_complete"
        assert events[1]["status"] == "completed"


def test_should_start_future_session():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        engine = SchedulerEngine(store)

        session = create_session(
            "snake", "ppo",
            scheduled_start=datetime.now() + timedelta(hours=1),
        )
        assert engine._should_start(session) is False


def test_should_start_past_session():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        engine = SchedulerEngine(store)

        session = create_session(
            "snake", "ppo",
            scheduled_start=datetime.now() - timedelta(hours=1),
        )
        assert engine._should_start(session) is True


def test_should_start_no_time():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        engine = SchedulerEngine(store)

        session = create_session("snake", "ppo")
        assert session.scheduled_start is None
        assert engine._should_start(session) is False
