"""
Opponents
=========
Pluggable opponent system for board games (Chess, Checkers, Connect4, TicTacToe).

Provides a base interface and concrete implementations so that any game adapter
can be paired with random, minimax, trained-model, or human opponents.
"""

from .base_opponent import BaseOpponent
from .human_opponent import HumanOpponent
from .minimax_opponent import MinimaxOpponent
from .model_opponent import ModelOpponent
from .random_opponent import RandomOpponent

__all__ = [
    "BaseOpponent",
    "HumanOpponent",
    "MinimaxOpponent",
    "ModelOpponent",
    "RandomOpponent",
]
