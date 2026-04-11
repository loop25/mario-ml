"""Tests for the pluggable opponent system."""

import numpy as np
import pytest

from src.opponents import RandomOpponent


def _make_board_state(valid_actions=None):
    """Helper to build a minimal board_state dict."""
    if valid_actions is None:
        valid_actions = [0, 1, 2, 3]
    return {
        "board": np.zeros((3, 3), dtype=np.int8),
        "valid_actions": valid_actions,
        "game_id": "tictactoe",
        "turn": 1,
    }


class TestRandomOpponent:
    def test_picks_from_valid_actions(self):
        opp = RandomOpponent()
        state = _make_board_state([2, 5, 7])
        for _ in range(50):
            action = opp.pick_action(state)
            assert action in state["valid_actions"]

    def test_single_valid_action(self):
        opp = RandomOpponent()
        state = _make_board_state([4])
        assert opp.pick_action(state) == 4

    def test_name(self):
        assert RandomOpponent().name == "Random"

    def test_difficulty_tier(self):
        assert RandomOpponent().difficulty_tier == 0

    def test_reset_does_not_crash(self):
        opp = RandomOpponent()
        opp.reset()  # should not raise
