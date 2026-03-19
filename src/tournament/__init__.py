"""
Tournament engine for pitting multiple agents against each other.

Supports round-robin and bracket formats across any board game
(Chess, Checkers, Connect4, TicTacToe) with pluggable opponents.
"""

from .tournament_engine import (
    MatchResult,
    TournamentEngine,
    TournamentParticipant,
    TournamentResult,
)

__all__ = [
    "MatchResult",
    "TournamentEngine",
    "TournamentParticipant",
    "TournamentResult",
]
