"""Tests for HumanOpponent with callback-based input."""

import random

import pytest

from src.opponents.human_opponent import HumanOpponent


# ------------------------------------------------------------------
# Property / metadata tests
# ------------------------------------------------------------------

class TestHumanProperties:
    def test_name(self):
        opp = HumanOpponent(input_callback=lambda s: 0)
        assert opp.name == "Human"

    def test_difficulty_tier(self):
        opp = HumanOpponent(input_callback=lambda s: 0)
        assert opp.difficulty_tier == -1


# ------------------------------------------------------------------
# Callback action tests
# ------------------------------------------------------------------

class TestHumanPickAction:
    def test_valid_callback_action(self):
        """Callback returns a valid action — pick_action returns it directly."""
        callback = lambda state: 3
        opp = HumanOpponent(input_callback=callback)
        state = {"valid_actions": [1, 2, 3, 4], "game_id": "tictactoe", "turn": 1}
        assert opp.pick_action(state) == 3

    def test_invalid_callback_retries(self):
        """Callback returns invalid action first, then valid on second call."""
        call_count = 0

        def callback(state):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 99  # invalid
            return 2       # valid

        opp = HumanOpponent(input_callback=callback)
        state = {"valid_actions": [1, 2, 3], "game_id": "tictactoe", "turn": 1}
        assert opp.pick_action(state) == 2
        assert call_count == 2

    def test_callback_exception_falls_back(self):
        """Callback raises an exception — falls back to random valid action."""
        def bad_callback(state):
            raise ValueError("oops")

        random.seed(42)
        opp = HumanOpponent(input_callback=bad_callback)
        state = {"valid_actions": [5, 6, 7], "game_id": "tictactoe", "turn": 1}
        action = opp.pick_action(state)
        assert action in [5, 6, 7]

    def test_no_valid_actions_returns_zero(self):
        """Empty valid_actions list — pick_action returns 0."""
        callback = lambda state: 0
        opp = HumanOpponent(input_callback=callback)
        state = {"valid_actions": [], "game_id": "tictactoe", "turn": 1}
        assert opp.pick_action(state) == 0


# ------------------------------------------------------------------
# set_input_callback
# ------------------------------------------------------------------

class TestSetInputCallback:
    def test_set_input_callback(self):
        """Changing the callback after construction uses the new one."""
        opp = HumanOpponent(input_callback=lambda s: 1)
        state = {"valid_actions": [1, 2, 3], "game_id": "tictactoe", "turn": 1}
        assert opp.pick_action(state) == 1

        opp.set_input_callback(lambda s: 3)
        assert opp.pick_action(state) == 3


# ------------------------------------------------------------------
# Console fallback (default callback)
# ------------------------------------------------------------------

class TestConsoleFallback:
    def test_default_uses_console_input(self, monkeypatch):
        """When no callback is given, _console_input is used (mock input())."""
        monkeypatch.setattr("builtins.input", lambda prompt="": "2")
        opp = HumanOpponent()  # no callback
        state = {"valid_actions": [1, 2, 3], "game_id": "tictactoe", "turn": 1}
        assert opp.pick_action(state) == 2
