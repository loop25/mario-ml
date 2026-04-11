"""Tests for CalendarStore — JSON-backed training schedule persistence."""

import os
import tempfile

import pytest

from datetime import datetime, timedelta

from src.scheduler.calendar_store import CalendarStore
from src.scheduler.session import TrainingSession, create_session


@pytest.fixture()
def store_path():
    """Return a temporary file path for a CalendarStore."""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "schedule.json")
    yield path


# ------------------------------------------------------------------
# Basic CRUD
# ------------------------------------------------------------------

def test_empty_store(store_path):
    store = CalendarStore(path=store_path)
    assert store.get_all_sessions() == []


def test_add_and_get(store_path):
    store = CalendarStore(path=store_path)
    session = create_session("mario", "ppo", episodes=2000)
    store.add_session(session)

    retrieved = store.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.game_id == "mario"
    assert retrieved.algorithm == "ppo"
    assert retrieved.episodes == 2000


def test_remove_session(store_path):
    store = CalendarStore(path=store_path)
    session = create_session("snake", "dqn")
    store.add_session(session)

    assert store.remove_session(session.session_id) is True
    assert store.get_session(session.session_id) is None


def test_remove_nonexistent(store_path):
    store = CalendarStore(path=store_path)
    assert store.remove_session("does_not_exist") is False


def test_update_session(store_path):
    store = CalendarStore(path=store_path)
    session = create_session("tetris", "a2c", episodes=1000)
    store.add_session(session)

    assert store.update_session(session.session_id, episodes=5000) is True
    updated = store.get_session(session.session_id)
    assert updated is not None
    assert updated.episodes == 5000


# ------------------------------------------------------------------
# Query helpers
# ------------------------------------------------------------------

def test_get_pending(store_path):
    store = CalendarStore(path=store_path)
    s1 = create_session("mario", "ppo", status="pending")
    s2 = create_session("snake", "dqn", status="pending")
    s3 = create_session("tetris", "a2c", status="completed")
    store.add_session(s1)
    store.add_session(s2)
    store.add_session(s3)

    pending = store.get_pending_sessions()
    assert len(pending) == 2
    ids = {s.session_id for s in pending}
    assert s1.session_id in ids
    assert s2.session_id in ids


def test_get_next(store_path):
    store = CalendarStore(path=store_path)
    now = datetime.now()
    s1 = create_session("mario", "ppo", scheduled_start=now + timedelta(hours=2))
    s2 = create_session("snake", "dqn", scheduled_start=now + timedelta(hours=1))
    s3 = create_session("tetris", "a2c", scheduled_start=now + timedelta(hours=3))
    store.add_session(s1)
    store.add_session(s2)
    store.add_session(s3)

    nxt = store.get_next_session()
    assert nxt is not None
    assert nxt.session_id == s2.session_id


def test_get_sessions_for_date(store_path):
    store = CalendarStore(path=store_path)
    today = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)

    s1 = create_session("mario", "ppo", scheduled_start=today)
    s2 = create_session("snake", "dqn", scheduled_start=tomorrow)
    s3 = create_session("tetris", "a2c", scheduled_start=today.replace(hour=14))
    store.add_session(s1)
    store.add_session(s2)
    store.add_session(s3)

    today_sessions = store.get_sessions_for_date(today.date())
    assert len(today_sessions) == 2
    ids = {s.session_id for s in today_sessions}
    assert s1.session_id in ids
    assert s3.session_id in ids


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def test_persistence(store_path):
    store = CalendarStore(path=store_path)
    session = create_session("chess", "ppo", episodes=3000)
    store.add_session(session)

    # Create a brand-new store pointing to the same file
    store2 = CalendarStore(path=store_path)
    retrieved = store2.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.game_id == "chess"
    assert retrieved.episodes == 3000


def test_corrupt_file(store_path):
    # Write garbage to the file
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, "w") as f:
        f.write("{{{ not valid json !!!")

    store = CalendarStore(path=store_path)
    assert store.get_all_sessions() == []


# ------------------------------------------------------------------
# Bulk operations
# ------------------------------------------------------------------

def test_clear_completed(store_path):
    store = CalendarStore(path=store_path)
    s1 = create_session("mario", "ppo", status="completed")
    s2 = create_session("snake", "dqn", status="pending")
    s3 = create_session("tetris", "a2c", status="failed")
    store.add_session(s1)
    store.add_session(s2)
    store.add_session(s3)

    removed = store.clear_completed()
    assert removed == 2
    assert store.get_session(s1.session_id) is None
    assert store.get_session(s3.session_id) is None
    assert store.get_session(s2.session_id) is not None


def test_add_preset(store_path):
    store = CalendarStore(path=store_path)
    sessions = [
        create_session("mario", "ppo"),
        create_session("snake", "dqn"),
        create_session("tetris", "a2c"),
    ]
    store.add_preset(sessions)

    assert len(store.get_all_sessions()) == 3
    for s in sessions:
        assert store.get_session(s.session_id) is not None
