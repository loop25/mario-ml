"""Training session and schedule preset dataclasses."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


BUILT_IN_GAMES: List[str] = [
    "mario",
    "snake",
    "tetris",
    "chess",
    "checkers",
    "connect4",
    "tictactoe",
]

BUILT_IN_ALGORITHMS: List[str] = [
    "ppo",
    "dqn",
    "a2c",
    "rainbow",
    "neat",
    "dt",
]


@dataclass
class TrainingSession:
    """A single scheduled training session."""

    session_id: str = ""
    game_id: str = ""
    algorithm: str = ""
    episodes: int = 1000
    duration_minutes: int = 0
    device: str = "auto"
    stream: bool = False
    opponent_type: str = "random"
    opponent_config: Dict = field(default_factory=dict)
    notes: str = ""

    # Scheduling fields
    scheduled_start: Optional[datetime] = None
    recurring: Optional[str] = None

    # Runtime fields
    status: str = "pending"
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    result_summary: Dict = field(default_factory=dict)


@dataclass
class SchedulePreset:
    """Pre-built schedule templates for common workflows."""

    name: str = ""
    description: str = ""
    sessions: List[TrainingSession] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_session(game_id: str, algorithm: str, **kwargs) -> TrainingSession:
    """Create a *TrainingSession* with an auto-generated *session_id*."""
    return TrainingSession(
        session_id=uuid.uuid4().hex[:12],
        game_id=game_id,
        algorithm=algorithm,
        **kwargs,
    )


def overnight_all_games(episodes_per_game: int = 500) -> List[TrainingSession]:
    """Return one PPO session per built-in game."""
    return [
        create_session(game_id=g, algorithm="ppo", episodes=episodes_per_game)
        for g in BUILT_IN_GAMES
    ]


def dt_generalist_run(episodes_per_game: int = 100) -> List[TrainingSession]:
    """Collection phase for every game (PPO) followed by a single DT training session."""
    sessions: List[TrainingSession] = [
        create_session(game_id=g, algorithm="ppo", episodes=episodes_per_game)
        for g in BUILT_IN_GAMES
    ]
    sessions.append(
        create_session(
            game_id="all",
            algorithm="dt",
            episodes=episodes_per_game,
            notes="Decision-transformer generalist training on collected trajectories",
        )
    )
    return sessions


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

_DATETIME_FIELDS = ("scheduled_start", "actual_start", "actual_end")


def session_to_dict(session: TrainingSession) -> dict:
    """Serialize a *TrainingSession* to a JSON-friendly dict."""
    d: dict = {}
    for k, v in session.__dict__.items():
        if k in _DATETIME_FIELDS and isinstance(v, datetime):
            d[k] = v.isoformat()
        else:
            d[k] = v
    return d


def session_from_dict(d: dict) -> TrainingSession:
    """Deserialize a dict (e.g. from JSON) back into a *TrainingSession*."""
    kwargs = dict(d)
    for k in _DATETIME_FIELDS:
        val = kwargs.get(k)
        if isinstance(val, str):
            kwargs[k] = datetime.fromisoformat(val)
    return TrainingSession(**kwargs)
