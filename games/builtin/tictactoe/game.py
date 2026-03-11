"""
Tic-Tac-Toe game engine as a Gym environment.

Classic 3x3 grid game. Agent plays as X (player 1), random opponent
plays as O (player 2). Opponent auto-plays after each agent move.

Observations are rendered as 84x84 grayscale images:
- X pieces: white (255)
- O pieces: gray (128)
- Empty: black (0)

The game is intentionally simple — a perfect agent should achieve
near-100% win rate against a random opponent.
"""
import random

import cv2
import numpy as np
try:
    import gymnasium as gym
except ImportError:
    import gym  # Legacy fallback

from gymnasium.spaces import Box, Discrete


class TicTacToeEnv(gym.Env):
    """Tic-Tac-Toe as a Gym environment with a random opponent."""

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, render_size: int = 84):
        super().__init__()
        self.render_size = render_size
        self.action_space = Discrete(9)  # 3x3 positions
        self.observation_space = Box(
            low=0, high=255,
            shape=(render_size, render_size, 1),
            dtype=np.uint8,
        )
        self.board = np.zeros((3, 3), dtype=np.int8)
        self._moves_played = 0
        self._winner = 0

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.int8)
        self._moves_played = 0
        self._winner = 0
        return self._render_obs()

    def step(self, action):
        row, col = divmod(action, 3)

        # Invalid move (already occupied or out of range)
        if not (0 <= action < 9) or self.board[row, col] != 0:
            return self._render_obs(), -1.0, True, self._info()

        # Player 1 (agent) move
        self.board[row, col] = 1
        self._moves_played += 1

        # Check if player 1 wins
        if self._check_winner(1):
            self._winner = 1
            return self._render_obs(), 1.0, True, self._info()

        # Check draw
        if self._moves_played >= 9:
            return self._render_obs(), 0.0, True, self._info()

        # Opponent move (random valid cell)
        empty = list(zip(*np.where(self.board == 0)))
        if not empty:
            return self._render_obs(), 0.0, True, self._info()

        opp_r, opp_c = random.choice(empty)
        self.board[opp_r, opp_c] = 2
        self._moves_played += 1

        # Check if opponent wins
        if self._check_winner(2):
            self._winner = 2
            return self._render_obs(), -1.0, True, self._info()

        # Check draw after opponent move
        if self._moves_played >= 9:
            return self._render_obs(), 0.0, True, self._info()

        return self._render_obs(), 0.0, False, self._info()

    def _check_winner(self, player: int) -> bool:
        b = self.board
        # Rows
        for r in range(3):
            if all(b[r, c] == player for c in range(3)):
                return True
        # Columns
        for c in range(3):
            if all(b[r, c] == player for r in range(3)):
                return True
        # Diagonals
        if all(b[i, i] == player for i in range(3)):
            return True
        if all(b[i, 2 - i] == player for i in range(3)):
            return True
        return False

    def _render_obs(self) -> np.ndarray:
        """Render board as 84x84 grayscale for ML input."""
        grid = np.zeros((3, 3), dtype=np.uint8)
        grid[self.board == 1] = 255
        grid[self.board == 2] = 128
        obs = cv2.resize(grid, (self.render_size, self.render_size),
                         interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(obs, axis=-1)

    def _render_rgb(self) -> np.ndarray:
        """Render a colorful Tic-Tac-Toe board for dashboard display.

        Dark background with a grid, blue X marks and red O marks.
        """
        size = self.render_size
        cell = size // 3
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = (25, 25, 40)  # Dark background

        # Draw grid lines
        line_color = (80, 80, 120)
        for i in range(1, 3):
            cv2.line(img, (i * cell, 0), (i * cell, size), line_color, 2)
            cv2.line(img, (0, i * cell), (size, i * cell), line_color, 2)

        pad = max(4, cell // 6)

        for r in range(3):
            for c in range(3):
                x1 = c * cell + pad
                y1 = r * cell + pad
                x2 = (c + 1) * cell - pad
                y2 = (r + 1) * cell - pad
                cx = c * cell + cell // 2
                cy = r * cell + cell // 2
                radius = cell // 2 - pad

                if self.board[r, c] == 1:
                    # X — two diagonal lines (blue)
                    cv2.line(img, (x1, y1), (x2, y2), (100, 160, 255), 2)
                    cv2.line(img, (x2, y1), (x1, y2), (100, 160, 255), 2)
                elif self.board[r, c] == 2:
                    # O — circle (red)
                    cv2.circle(img, (cx, cy), radius, (255, 80, 80), 2)

        return img

    def render(self, mode='rgb_array'):
        return self._render_rgb()

    def _info(self) -> dict:
        return {
            'moves_played': self._moves_played,
            'winner': self._winner,
        }

    def close(self):
        pass
