"""
Checkers (Draughts) game engine as a Gym environment.

Simplified American Checkers on an 8x8 board:
- Agent plays dark pieces (player 1), random opponent plays light (player 2).
- Regular pieces move diagonally forward one square.
- Kings (promoted on back row) move diagonally in any direction.
- Captures are mandatory when available (jump over opponent piece).
- Multi-jump sequences are allowed (one action = one jump).
- Win by capturing all opponent pieces or leaving them with no moves.

Actions are encoded as: from_pos * 32 + to_pos, where positions 0-31
map to the 32 dark squares of the 8x8 board (row-major, only playable
squares). The action space is Discrete(1024) — most actions are invalid
at any given time.

Observations: 84x84 grayscale.
"""
import random

import cv2
import numpy as np
try:
    import gymnasium as gym
except ImportError:
    import gym  # Legacy fallback

from gymnasium.spaces import Box, Discrete


# Board constants
BOARD_SIZE = 8
NUM_SQUARES = 32  # Only dark squares are playable

# Piece values
EMPTY = 0
P1_MAN = 1
P1_KING = 2
P2_MAN = 3
P2_KING = 4


def _pos_to_rc(pos: int):
    """Convert position (0-31) to row, col on 8x8 board."""
    row = pos // 4
    col = (pos % 4) * 2 + (1 - row % 2)
    return row, col


def _rc_to_pos(row: int, col: int) -> int:
    """Convert row, col to position (0-31). Returns -1 if not a dark square."""
    if (row + col) % 2 == 0:
        return -1  # Light square
    return row * 4 + (col // 2)


class CheckersEnv(gym.Env):
    """Checkers as a Gym environment with a random opponent."""

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, render_size: int = 84):
        super().__init__()
        self.render_size = render_size
        self.action_space = Discrete(NUM_SQUARES * NUM_SQUARES)
        self.observation_space = Box(
            low=0, high=255,
            shape=(render_size, render_size, 1),
            dtype=np.uint8,
        )
        self.board = np.zeros(NUM_SQUARES, dtype=np.int8)
        self._winner = 0
        self._moves_played = 0
        self._max_moves = 200

    def reset(self):
        self.board = np.zeros(NUM_SQUARES, dtype=np.int8)
        # Player 2 (opponent): rows 0-2 (positions 0-11)
        for i in range(12):
            self.board[i] = P2_MAN
        # Player 1 (agent): rows 5-7 (positions 20-31)
        for i in range(20, 32):
            self.board[i] = P1_MAN
        self._winner = 0
        self._moves_played = 0
        return self._render_obs()

    def step(self, action):
        from_pos = action // NUM_SQUARES
        to_pos = action % NUM_SQUARES

        # Validate move
        valid_moves = self._get_valid_moves(player=1)
        if (from_pos, to_pos) not in valid_moves:
            # Invalid move — forfeit
            self._winner = 2
            return self._render_obs(), -1.0, True, self._info()

        # Execute agent move
        self._execute_move(from_pos, to_pos)
        self._moves_played += 1

        # Check promotion
        self._check_promotion()

        # Check if opponent has no pieces or no moves
        if self._player_lost(player=2):
            self._winner = 1
            return self._render_obs(), 1.0, True, self._info()

        # Max move limit
        if self._moves_played >= self._max_moves:
            return self._render_obs(), 0.0, True, self._info()

        # Opponent turn (random valid move)
        opp_moves = self._get_valid_moves(player=2)
        if not opp_moves:
            self._winner = 1
            return self._render_obs(), 1.0, True, self._info()

        opp_from, opp_to = random.choice(opp_moves)
        self._execute_move(opp_from, opp_to)
        self._moves_played += 1
        self._check_promotion()

        # Check if agent has no pieces or no moves
        if self._player_lost(player=1):
            self._winner = 2
            return self._render_obs(), -1.0, True, self._info()

        if self._moves_played >= self._max_moves:
            return self._render_obs(), 0.0, True, self._info()

        return self._render_obs(), 0.0, False, self._info()

    def _get_valid_moves(self, player: int):
        """Get all valid moves for a player. Captures are mandatory."""
        pieces = (P1_MAN, P1_KING) if player == 1 else (P2_MAN, P2_KING)
        captures = []
        regular = []

        for pos in range(NUM_SQUARES):
            if self.board[pos] not in pieces:
                continue
            piece = self.board[pos]
            r, c = _pos_to_rc(pos)
            directions = self._get_directions(piece)

            for dr, dc in directions:
                # Check capture (jump)
                mid_r, mid_c = r + dr, c + dc
                land_r, land_c = r + 2 * dr, c + 2 * dc
                if 0 <= land_r < 8 and 0 <= land_c < 8:
                    mid_pos = _rc_to_pos(mid_r, mid_c)
                    land_pos = _rc_to_pos(land_r, land_c)
                    if mid_pos >= 0 and land_pos >= 0:
                        if self._is_opponent(self.board[mid_pos], player):
                            if self.board[land_pos] == EMPTY:
                                captures.append((pos, land_pos))

                # Check regular move
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < 8 and 0 <= new_c < 8:
                    new_pos = _rc_to_pos(new_r, new_c)
                    if new_pos >= 0 and self.board[new_pos] == EMPTY:
                        regular.append((pos, new_pos))

        # Mandatory captures
        return captures if captures else regular

    def _get_directions(self, piece: int):
        """Movement directions for a piece."""
        if piece == P1_MAN:
            return [(-1, -1), (-1, 1)]  # Forward (toward row 0)
        elif piece == P2_MAN:
            return [(1, -1), (1, 1)]    # Forward (toward row 7)
        else:
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # King

    def _is_opponent(self, piece: int, player: int) -> bool:
        if player == 1:
            return piece in (P2_MAN, P2_KING)
        return piece in (P1_MAN, P1_KING)

    def _execute_move(self, from_pos: int, to_pos: int):
        from_r, from_c = _pos_to_rc(from_pos)
        to_r, to_c = _pos_to_rc(to_pos)

        self.board[to_pos] = self.board[from_pos]
        self.board[from_pos] = EMPTY

        # If jump (distance > 1), remove captured piece
        if abs(to_r - from_r) == 2:
            mid_r = (from_r + to_r) // 2
            mid_c = (from_c + to_c) // 2
            mid_pos = _rc_to_pos(mid_r, mid_c)
            if mid_pos >= 0:
                self.board[mid_pos] = EMPTY

    def _check_promotion(self):
        """Promote men to kings on their respective back rows."""
        for pos in range(4):  # Row 0 (positions 0-3)
            if self.board[pos] == P1_MAN:
                self.board[pos] = P1_KING
        for pos in range(28, 32):  # Row 7 (positions 28-31)
            if self.board[pos] == P2_MAN:
                self.board[pos] = P2_KING

    def _player_lost(self, player: int) -> bool:
        pieces = (P1_MAN, P1_KING) if player == 1 else (P2_MAN, P2_KING)
        if not any(self.board[i] in pieces for i in range(NUM_SQUARES)):
            return True  # No pieces left
        if not self._get_valid_moves(player):
            return True  # No valid moves
        return False

    def _render_obs(self) -> np.ndarray:
        """Render board as 84x84 grayscale for ML input."""
        grid = np.zeros((8, 8), dtype=np.uint8)
        for pos in range(NUM_SQUARES):
            r, c = _pos_to_rc(pos)
            if self.board[pos] == P1_MAN:
                grid[r, c] = 200
            elif self.board[pos] == P1_KING:
                grid[r, c] = 255
            elif self.board[pos] == P2_MAN:
                grid[r, c] = 100
            elif self.board[pos] == P2_KING:
                grid[r, c] = 128
        obs = cv2.resize(grid, (self.render_size, self.render_size),
                         interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(obs, axis=-1)

    def _render_rgb(self) -> np.ndarray:
        """Render a colorful checkers board for dashboard display."""
        size = self.render_size
        cell = size // 8
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Draw board squares
        for r in range(8):
            for c in range(8):
                x1, y1 = c * cell, r * cell
                x2, y2 = x1 + cell, y1 + cell
                if (r + c) % 2 == 0:
                    img[y1:y2, x1:x2] = (200, 190, 160)  # Light tan
                else:
                    img[y1:y2, x1:x2] = (60, 100, 60)    # Dark green

        # Draw pieces
        radius = max(2, cell // 2 - 2)
        for pos in range(NUM_SQUARES):
            r, c = _pos_to_rc(pos)
            cx = c * cell + cell // 2
            cy = r * cell + cell // 2
            piece = self.board[pos]

            if piece == P1_MAN:
                cv2.circle(img, (cx, cy), radius, (40, 40, 40), -1)
                cv2.circle(img, (cx, cy), radius, (80, 80, 80), 1)
            elif piece == P1_KING:
                cv2.circle(img, (cx, cy), radius, (40, 40, 40), -1)
                cv2.circle(img, (cx, cy), radius, (200, 200, 60), 2)
            elif piece == P2_MAN:
                cv2.circle(img, (cx, cy), radius, (180, 50, 50), -1)
                cv2.circle(img, (cx, cy), radius, (220, 100, 100), 1)
            elif piece == P2_KING:
                cv2.circle(img, (cx, cy), radius, (180, 50, 50), -1)
                cv2.circle(img, (cx, cy), radius, (200, 200, 60), 2)

        return img

    def render(self, mode='rgb_array'):
        return self._render_rgb()

    def _info(self) -> dict:
        p1_count = sum(1 for p in self.board if p in (P1_MAN, P1_KING))
        p2_count = sum(1 for p in self.board if p in (P2_MAN, P2_KING))
        return {
            'winner': self._winner,
            'moves_played': self._moves_played,
            'p1_pieces': p1_count,
            'p2_pieces': p2_count,
        }

    def close(self):
        pass
