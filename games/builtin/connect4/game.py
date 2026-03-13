"""
Connect Four game engine as a Gym environment.

Two-player game on a 6x7 grid. Players alternate dropping pieces
into columns. First to get four in a row (horizontal, vertical,
or diagonal) wins.

The agent plays as Player 1. A simple random opponent plays as
Player 2 (opponent auto-plays after each agent action).

Observations are rendered as 84x84 grayscale images:
- Player 1 pieces: white (255)
- Player 2 pieces: gray (128)
- Empty: black (0)
"""
import random

import cv2
import numpy as np
try:
    import gymnasium as gym
except ImportError:
    import gym  # Legacy fallback

from gymnasium.spaces import Box, Discrete


ROWS = 6
COLS = 7


class ConnectFourEnv(gym.Env):
    """Connect Four as a Gym environment with a random opponent."""

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, render_size: int = 84):
        super().__init__()
        self.render_size = render_size
        self.action_space = Discrete(COLS)
        self.observation_space = Box(
            low=0, high=255,
            shape=(render_size, render_size, 1),
            dtype=np.uint8,
        )
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self._pieces_played = 0
        self._winner = 0

    def reset(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self._pieces_played = 0
        self._winner = 0
        return self._render_obs()

    def step(self, action):
        # Player 1 move
        if not self._is_valid_column(action):
            # Illegal move — game over with penalty
            return self._render_obs(), -1.0, True, self._info()

        self._drop_piece(action, player=1)
        self._pieces_played += 1

        # Check if player 1 wins
        if self._check_winner(1):
            self._winner = 1
            return self._render_obs(), 1.0, True, self._info()

        # Check draw
        if self._pieces_played >= ROWS * COLS:
            return self._render_obs(), 0.0, True, self._info()

        # Opponent move (random valid column)
        valid_cols = [c for c in range(COLS) if self._is_valid_column(c)]
        if not valid_cols:
            return self._render_obs(), 0.0, True, self._info()

        opp_col = random.choice(valid_cols)
        self._drop_piece(opp_col, player=2)
        self._pieces_played += 1

        # Check if opponent wins
        if self._check_winner(2):
            self._winner = 2
            return self._render_obs(), -1.0, True, self._info()

        # Check draw after opponent move
        if self._pieces_played >= ROWS * COLS:
            return self._render_obs(), 0.0, True, self._info()

        return self._render_obs(), 0.0, False, self._info()

    def _is_valid_column(self, col: int) -> bool:
        return 0 <= col < COLS and self.board[0, col] == 0

    def _drop_piece(self, col: int, player: int):
        for row in range(ROWS - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = player
                return

    def _check_winner(self, player: int) -> bool:
        """Check all four-in-a-row possibilities for player."""
        b = self.board
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if all(b[r, c + i] == player for i in range(4)):
                    return True
        # Vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                if all(b[r + i, c] == player for i in range(4)):
                    return True
        # Diagonal (down-right)
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(b[r + i, c + i] == player for i in range(4)):
                    return True
        # Diagonal (down-left)
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                if all(b[r + i, c - i] == player for i in range(4)):
                    return True
        return False

    def _get_winning_cells(self):
        """Return list of (row, col) for the winning four-in-a-row, or None."""
        for player in (1, 2):
            b = self.board
            for r in range(ROWS):
                for c in range(COLS - 3):
                    if all(b[r, c + i] == player for i in range(4)):
                        return [(r, c + i) for i in range(4)]
            for r in range(ROWS - 3):
                for c in range(COLS):
                    if all(b[r + i, c] == player for i in range(4)):
                        return [(r + i, c) for i in range(4)]
            for r in range(ROWS - 3):
                for c in range(COLS - 3):
                    if all(b[r + i, c + i] == player for i in range(4)):
                        return [(r + i, c + i) for i in range(4)]
            for r in range(ROWS - 3):
                for c in range(3, COLS):
                    if all(b[r + i, c - i] == player for i in range(4)):
                        return [(r + i, c - i) for i in range(4)]
        return None

    def _render_obs(self) -> np.ndarray:
        """Render board as 84x84 grayscale."""
        grid = np.zeros((ROWS, COLS), dtype=np.uint8)
        grid[self.board == 1] = 255
        grid[self.board == 2] = 128
        obs = cv2.resize(grid, (self.render_size, self.render_size),
                         interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(obs, axis=-1)

    # High-res rendering for the dashboard
    DISPLAY_SIZE = 480

    def _render_rgb(self) -> np.ndarray:
        """Render a polished Connect Four board for dashboard/stream.

        Renders at 480px. Rich blue board with rounded slot holes,
        red/yellow pieces with 3D highlight and shadow effects.
        """
        size = self.DISPLAY_SIZE
        cell = size // max(ROWS, COLS)  # ~68px per cell
        board_w = cell * COLS
        board_h = cell * ROWS
        # Center the board in a square canvas
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = (15, 15, 32)  # Dark background behind board

        # Offset to center the board
        ox = (size - board_w) // 2
        oy = (size - board_h) // 2

        # Board background with rounded rect effect
        cv2.rectangle(img, (ox - 4, oy - 4),
                      (ox + board_w + 4, oy + board_h + 4),
                      (25, 50, 160), -1)  # Outer blue
        cv2.rectangle(img, (ox, oy),
                      (ox + board_w, oy + board_h),
                      (35, 65, 190), -1)  # Inner blue

        radius = max(4, cell // 2 - 5)

        for r in range(ROWS):
            for c in range(COLS):
                cx = ox + c * cell + cell // 2
                cy = oy + r * cell + cell // 2

                if self.board[r, c] == 0:
                    # Empty slot: dark recessed hole
                    cv2.circle(img, (cx, cy), radius + 2, (20, 35, 120),
                               -1, cv2.LINE_AA)
                    cv2.circle(img, (cx, cy), radius, (12, 12, 28),
                               -1, cv2.LINE_AA)
                elif self.board[r, c] == 1:
                    # Player 1: red piece with 3D effect
                    cv2.circle(img, (cx + 1, cy + 1), radius, (120, 20, 20),
                               -1, cv2.LINE_AA)  # Shadow
                    cv2.circle(img, (cx, cy), radius, (220, 45, 45),
                               -1, cv2.LINE_AA)  # Main
                    cv2.circle(img, (cx, cy), radius, (240, 70, 70),
                               2, cv2.LINE_AA)   # Edge
                    # Highlight for 3D
                    hl_r = max(3, radius // 3)
                    cv2.circle(img, (cx - radius // 4, cy - radius // 4),
                               hl_r, (255, 140, 140), -1, cv2.LINE_AA)
                elif self.board[r, c] == 2:
                    # Player 2: yellow piece with 3D effect
                    cv2.circle(img, (cx + 1, cy + 1), radius, (120, 105, 15),
                               -1, cv2.LINE_AA)  # Shadow
                    cv2.circle(img, (cx, cy), radius, (240, 210, 35),
                               -1, cv2.LINE_AA)  # Main
                    cv2.circle(img, (cx, cy), radius, (250, 230, 80),
                               2, cv2.LINE_AA)   # Edge
                    hl_r = max(3, radius // 3)
                    cv2.circle(img, (cx - radius // 4, cy - radius // 4),
                               hl_r, (255, 245, 150), -1, cv2.LINE_AA)

        # Winning cells: bright white ring around the four connected pieces
        winning = self._get_winning_cells()
        if winning:
            for wr, wc in winning:
                wx = ox + wc * cell + cell // 2
                wy = oy + wr * cell + cell // 2
                cv2.circle(img, (wx, wy), radius + 3, (255, 255, 255),
                           3, cv2.LINE_AA)

        return img

    def render(self, mode='rgb_array'):
        return self._render_rgb()

    def _info(self) -> dict:
        return {
            'pieces_played': self._pieces_played,
            'winner': self._winner,
        }

    def close(self):
        pass
