"""Tests for MinimaxOpponent with alpha-beta pruning."""

import numpy as np
import pytest

from src.opponents.minimax_opponent import MinimaxOpponent


# ------------------------------------------------------------------
# Property / metadata tests
# ------------------------------------------------------------------

class TestMinimaxProperties:
    def test_name_includes_depth(self):
        opp = MinimaxOpponent(depth=4, game_id="tictactoe")
        assert opp.name == "Minimax (depth=4)"

    def test_difficulty_tier_depth_1(self):
        assert MinimaxOpponent(depth=1).difficulty_tier == 1

    def test_difficulty_tier_depth_3(self):
        assert MinimaxOpponent(depth=3).difficulty_tier == 2

    def test_difficulty_tier_depth_5(self):
        assert MinimaxOpponent(depth=5).difficulty_tier == 3

    def test_difficulty_tier_scales_with_depth(self):
        assert MinimaxOpponent(depth=1).difficulty_tier == 1
        assert MinimaxOpponent(depth=2).difficulty_tier == 2
        assert MinimaxOpponent(depth=3).difficulty_tier == 2
        assert MinimaxOpponent(depth=4).difficulty_tier == 3
        assert MinimaxOpponent(depth=5).difficulty_tier == 3


# ------------------------------------------------------------------
# TicTacToe tests
# ------------------------------------------------------------------

class TestMinimaxTicTacToe:
    def test_blocks_winning_move(self):
        """Player 1 has [1,1,0] in top row — opponent must pick action 2."""
        board = np.array([
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 0],
        ])
        state = {
            "board": board,
            "valid_actions": [2, 3, 5, 6, 7, 8],
            "game_id": "tictactoe",
            "turn": 2,
        }
        opp = MinimaxOpponent(depth=5, game_id="tictactoe")
        action = opp.pick_action(state)
        assert action == 2, f"Expected block at 2, got {action}"

    def test_takes_winning_move(self):
        """Player 2 has two in a column and can win by completing it."""
        board = np.array([
            [0, 2, 1],
            [0, 2, 1],
            [0, 0, 0],
        ])
        state = {
            "board": board,
            "valid_actions": [0, 3, 6, 7, 8],
            "game_id": "tictactoe",
            "turn": 2,
        }
        opp = MinimaxOpponent(depth=5, game_id="tictactoe")
        action = opp.pick_action(state)
        # Winning move is action 7 (row 2, col 1) to complete column 1.
        assert action == 7, f"Expected winning move 7, got {action}"

    def test_returns_valid_action(self):
        board = np.array([
            [1, 2, 1],
            [2, 1, 2],
            [0, 0, 0],
        ])
        state = {
            "board": board,
            "valid_actions": [6, 7, 8],
            "game_id": "tictactoe",
            "turn": 2,
        }
        opp = MinimaxOpponent(depth=3, game_id="tictactoe")
        action = opp.pick_action(state)
        assert action in [6, 7, 8]

    def test_empty_valid_actions(self):
        """Edge case: no valid actions should return 0."""
        board = np.array([
            [1, 2, 1],
            [2, 1, 2],
            [2, 1, 2],
        ])
        state = {
            "board": board,
            "valid_actions": [],
            "game_id": "tictactoe",
            "turn": 2,
        }
        opp = MinimaxOpponent(depth=3, game_id="tictactoe")
        assert opp.pick_action(state) == 0


# ------------------------------------------------------------------
# Connect4 tests
# ------------------------------------------------------------------

class TestMinimaxConnect4:
    def test_blocks_vertical_three(self):
        """Player 1 has 3 in column 3 — opponent must block on top."""
        board = np.zeros((6, 7), dtype=int)
        board[5, 3] = 1
        board[4, 3] = 1
        board[3, 3] = 1
        # Add some opponent pieces elsewhere so it's realistic
        board[5, 0] = 2
        board[5, 1] = 2
        state = {
            "board": board,
            "valid_actions": [0, 1, 2, 3, 4, 5, 6],
            "game_id": "connect4",
            "turn": 2,
        }
        opp = MinimaxOpponent(depth=3, game_id="connect4")
        action = opp.pick_action(state)
        assert action == 3, f"Expected block at column 3, got {action}"

    def test_takes_winning_column(self):
        """Player 2 has 3 in column 2 — should complete it."""
        board = np.zeros((6, 7), dtype=int)
        board[5, 2] = 2
        board[4, 2] = 2
        board[3, 2] = 2
        # Some player 1 pieces elsewhere
        board[5, 0] = 1
        board[5, 1] = 1
        board[5, 3] = 1
        state = {
            "board": board,
            "valid_actions": [0, 1, 2, 3, 4, 5, 6],
            "game_id": "connect4",
            "turn": 2,
        }
        opp = MinimaxOpponent(depth=3, game_id="connect4")
        action = opp.pick_action(state)
        assert action == 2, f"Expected winning move at column 2, got {action}"

    def test_returns_valid_column(self):
        board = np.zeros((6, 7), dtype=int)
        state = {
            "board": board,
            "valid_actions": [0, 1, 2, 3, 4, 5, 6],
            "game_id": "connect4",
            "turn": 2,
        }
        opp = MinimaxOpponent(depth=2, game_id="connect4")
        action = opp.pick_action(state)
        assert 0 <= action <= 6


# ------------------------------------------------------------------
# Checkers tests (basic)
# ------------------------------------------------------------------

class TestMinimaxCheckers:
    def test_returns_valid_action(self):
        """Smoke test: opponent returns a legal move on a simple board."""
        board = np.zeros((8, 8), dtype=int)
        board[0, 1] = 2  # player 2 piece
        board[7, 0] = 1  # player 1 piece
        opp = MinimaxOpponent(depth=2, game_id="checkers")
        # Player 2 piece at (0,1) can move to (1,0) or (1,2) potentially
        # but only on dark squares — our simple implementation doesn't enforce
        # dark-square-only, so both should be returned as valid.
        valid_moves = opp._get_valid_moves(board, "checkers", player=2)
        if valid_moves:
            state = {
                "board": board,
                "valid_actions": valid_moves,
                "game_id": "checkers",
                "turn": 2,
            }
            action = opp.pick_action(state)
            assert action in valid_moves


# ------------------------------------------------------------------
# General integration
# ------------------------------------------------------------------

class TestMinimaxGeneral:
    def test_game_id_override_from_board_state(self):
        """game_id in board_state overrides constructor game_id."""
        opp = MinimaxOpponent(depth=3, game_id="connect4")
        board = np.array([
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 0],
        ])
        state = {
            "board": board,
            "valid_actions": [2, 3, 5, 6, 7, 8],
            "game_id": "tictactoe",
            "turn": 2,
        }
        action = opp.pick_action(state)
        assert action == 2  # should still block using ttt logic
