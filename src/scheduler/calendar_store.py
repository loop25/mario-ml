"""Persistent store for training schedule sessions (JSON-backed)."""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from typing import Dict, List, Optional

from .session import TrainingSession, session_from_dict, session_to_dict

logger = logging.getLogger(__name__)


class CalendarStore:
    """Persistent store for training schedule sessions."""

    def __init__(self, path: str = "config/training_schedule.json") -> None:
        self._path = path
        self._sessions: Dict[str, TrainingSession] = {}
        self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_session(self, session: TrainingSession) -> None:
        """Add a session to the schedule. Saves immediately."""
        self._sessions[session.session_id] = session
        self.save()

    def remove_session(self, session_id: str) -> bool:
        """Remove a session by ID. Returns True if found and removed."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self.save()
            return True
        return False

    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update fields on a session. Returns True if found."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)
        self.save()
        return True

    def get_session(self, session_id: str) -> Optional[TrainingSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_all_sessions(self) -> List[TrainingSession]:
        """Get all sessions, ordered by scheduled_start (None last)."""
        return self._sorted(list(self._sessions.values()))

    def get_pending_sessions(self) -> List[TrainingSession]:
        """Get sessions with status='pending', ordered by scheduled_start."""
        return self._sorted(
            [s for s in self._sessions.values() if s.status == "pending"]
        )

    def get_next_session(self) -> Optional[TrainingSession]:
        """Get the next pending session (earliest scheduled_start, or first if no times set)."""
        pending = self.get_pending_sessions()
        return pending[0] if pending else None

    def get_sessions_for_date(self, target_date: date) -> List[TrainingSession]:
        """Get all sessions scheduled for a specific date."""
        results: List[TrainingSession] = []
        for session in self._sessions.values():
            if (
                session.scheduled_start is not None
                and session.scheduled_start.date() == target_date
            ):
                results.append(session)
        return self._sorted(results)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def clear_completed(self) -> int:
        """Remove all completed/failed/cancelled sessions. Returns count removed."""
        remove_statuses = {"completed", "failed", "cancelled"}
        to_remove = [
            sid
            for sid, s in self._sessions.items()
            if s.status in remove_statuses
        ]
        for sid in to_remove:
            del self._sessions[sid]
        if to_remove:
            self.save()
        return len(to_remove)

    def add_preset(self, sessions: List[TrainingSession]) -> None:
        """Add multiple sessions from a preset."""
        for session in sessions:
            self._sessions[session.session_id] = session
        self.save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist to disk."""
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        data = [session_to_dict(s) for s in self._sessions.values()]
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load from disk. Handles missing/corrupt file gracefully."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                session = session_from_dict(item)
                self._sessions[session.session_id] = session
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning(
                "Corrupt schedule file %s — starting empty: %s", self._path, exc
            )
            self._sessions = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sorted(sessions: List[TrainingSession]) -> List[TrainingSession]:
        """Sort sessions by scheduled_start; None-start sessions go to end."""

        def _sort_key(s: TrainingSession):
            if s.scheduled_start is None:
                return (1, datetime.min)
            return (0, s.scheduled_start)

        return sorted(sessions, key=_sort_key)
