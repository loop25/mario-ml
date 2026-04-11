"""Scheduler engine — daemon that manages training session lifecycle."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Callable, Dict, Optional

from .calendar_store import CalendarStore
from .session import TrainingSession

logger = logging.getLogger(__name__)


class SchedulerEngine:
    """Daemon that manages training session lifecycle.

    The engine checks the calendar store for pending sessions, starts them
    when their scheduled time arrives, and updates statuses throughout.
    Actual training is delegated to an external callback set via
    :meth:`set_execute_callback`.
    """

    def __init__(
        self,
        calendar_store: CalendarStore,
        event_bus=None,
    ) -> None:
        self._store = calendar_store
        self._bus = event_bus
        self._running = False
        self._current_session: Optional[TrainingSession] = None
        self._thread: Optional[threading.Thread] = None
        self._execute_callback: Optional[Callable[[TrainingSession], Dict]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_execute_callback(self, callback: Callable[[TrainingSession], Dict]) -> None:
        """Set the function that actually runs a training session.

        *callback* receives a :class:`TrainingSession` and should return a
        result dict (or ``None``).
        """
        self._execute_callback = callback

    def start(self) -> None:
        """Start the scheduler daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the scheduler daemon.  Waits for current session to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def is_running(self) -> bool:
        return self._running

    def get_current_session(self) -> Optional[TrainingSession]:
        return self._current_session

    def run_next(self) -> Optional[TrainingSession]:
        """Manually trigger the next pending session (for "Run Now" button)."""
        next_session = self._store.get_next_session()
        if next_session is None:
            return None
        self._execute_session(next_session)
        return next_session

    def cancel_current(self) -> bool:
        """Cancel the currently running session."""
        session = self._current_session
        if session is None:
            return False
        session.status = "cancelled"
        self._store.update_session(session.session_id, status="cancelled")
        return True

    # ------------------------------------------------------------------
    # Daemon loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main daemon loop: check schedule, execute sessions."""
        while self._running:
            next_session = self._store.get_next_session()
            if next_session and self._should_start(next_session):
                self._execute_session(next_session)
            time.sleep(10)

    def _should_start(self, session: TrainingSession) -> bool:
        """Check if a session should start now."""
        if session.scheduled_start is None:
            return False  # Manual-only sessions don't auto-start
        return datetime.now() >= session.scheduled_start

    # ------------------------------------------------------------------
    # Session execution
    # ------------------------------------------------------------------

    def _execute_session(self, session: TrainingSession) -> None:
        """Run a single training session through the callback."""
        session.status = "running"
        session.actual_start = datetime.now()
        self._current_session = session
        self._store.update_session(
            session.session_id,
            status="running",
            actual_start=session.actual_start,
        )

        if self._bus:
            self._bus.publish(
                {
                    "type": "session_start",
                    "session_id": session.session_id,
                    "game_id": session.game_id,
                    "algorithm": session.algorithm,
                }
            )

        try:
            result: Dict = {}
            if self._execute_callback:
                result = self._execute_callback(session) or {}

            # If the session was cancelled while the callback was running,
            # don't overwrite the cancelled status.
            if session.status == "cancelled":
                return

            session.status = "completed"
            session.actual_end = datetime.now()
            session.result_summary = result
            self._store.update_session(
                session.session_id,
                status="completed",
                actual_end=session.actual_end,
                result_summary=result,
            )
        except Exception as exc:
            session.status = "failed"
            session.actual_end = datetime.now()
            session.result_summary = {"error": str(exc)}
            self._store.update_session(
                session.session_id,
                status="failed",
                actual_end=session.actual_end,
                result_summary={"error": str(exc)},
            )
        finally:
            self._current_session = None
            if self._bus:
                self._bus.publish(
                    {
                        "type": "session_complete",
                        "session_id": session.session_id,
                        "status": session.status,
                        "result": session.result_summary,
                    }
                )
