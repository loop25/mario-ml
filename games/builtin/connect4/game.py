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
import gym
from gym.spaces import Box, Discrete


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

    def _render_obs(self) -> np.ndarray:
        """Render board as 84x84 grayscale."""
        grid = np.zeros((ROWS, COLS), dtype=np.uint8)
        grid[self.board == 1] = 255
        grid[self.board == 2] = 128
        obs = cv2.resize(grid, (self.render_size, self.render_size),
                         interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(obs, axis=-1)

    def render(self, mode='rgb_array'):
        gray = self._render_obs()[:, :, 0]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def _info(self) -> dict:
        return {
            'pieces_played': self._pieces_played,
            'winner': self._winner,
        }

    def close(self):
        pass
