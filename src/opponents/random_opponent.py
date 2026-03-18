"""Random opponent — picks uniformly from valid actions."""

import random

from .base_opponent import BaseOpponent


class RandomOpponent(BaseOpponent):
    """Baseline opponent that selects a random legal move."""

    @property
    def name(self) -> str:
        return "Random"

    @property
    def difficulty_tier(self) -> int:
        return 0

    def pick_action(self, board_state: dict) -> int:
        return random.choice(board_state["valid_actions"])
