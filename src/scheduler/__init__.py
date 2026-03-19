"""Training scheduler — session dataclasses and schedule presets."""

from .session import (
    TrainingSession,
    SchedulePreset,
    create_session,
    overnight_all_games,
    dt_generalist_run,
    session_to_dict,
    session_from_dict,
    BUILT_IN_GAMES,
    BUILT_IN_ALGORITHMS,
)

__all__ = [
    "TrainingSession",
    "SchedulePreset",
    "create_session",
    "overnight_all_games",
    "dt_generalist_run",
    "session_to_dict",
    "session_from_dict",
    "BUILT_IN_GAMES",
    "BUILT_IN_ALGORITHMS",
]
