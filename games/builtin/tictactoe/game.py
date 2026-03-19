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
    """Tic-Tac-Toe as a Gym environment with a pluggable opponent."""

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, render_size: int = 84, opponent=None):
        super().__init__()
        if opponent is None:
            from src.opponents import RandomOpponent
            opponent = RandomOpponent()
        self.opponent = opponent
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
        self._last_action = None

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.int8)
        self._moves_played = 0
        self._winner = 0
        self._last_action = None
        return self._render_obs()

    def step(self, action):
        self._last_action = action
        row, col = divmod(action, 3)

        # Invalid move (already occupied or out of range)
        if not (0 <= action < 9) or self.board[row, col] != 0:
            self._winner = 2  # Opponent wins by forfeit
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

        # Opponent move
        empty = list(zip(*np.where(self.board == 0)))
        if not empty:
            valid_actions = [r * 3 + c for r, c in empty]
            board_state = {
                'board': self.board,
                'valid_actions': valid_actions,
                'game_id': 'tictactoe',
                'turn': 2,
            }
            opp_action = self.opponent.pick_action(board_state)
            opp_r, opp_c = divmod(opp_action, 3)
        else:
            return self._render_obs(), 0.0, True, self._info()

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

    # High-res rendering for the dashboard
    DISPLAY_SIZE = 480

    def _render_rgb(self) -> np.ndarray:
        """Render a visually rich chalkboard-style Tic-Tac-Toe board.

        Renders at 480px with:
        - Dark textured chalkboard background with grain noise
        - Hand-drawn chalk grid lines (multi-stroke imperfections)
        - Chalk-blue X marks with glow and slight hand-drawn offsets
        - Warm coral O marks with glow and thickness variation
        - Golden animated-feel winning line
        - Score/status overlay bar at the bottom
        - Chalk dust particle specks
        - Last-played cell highlight
        """
        size = self.DISPLAY_SIZE
        cell = size // 3  # 160px per cell

        # --- Seeded RNG for deterministic "hand-drawn" jitter per board state ---
        board_seed = int(self.board.tobytes().hex(), 16) % (2**31)
        rng = np.random.RandomState(board_seed)

        # --- 1. Chalkboard background with noise/grain ---
        img = np.zeros((size, size, 3), dtype=np.uint8)
        # Base dark slate-green chalkboard color
        img[:, :, 0] = 38   # B
        img[:, :, 1] = 42   # G
        img[:, :, 2] = 35   # R
        # Add subtle grain noise
        noise = rng.randint(-12, 13, (size, size), dtype=np.int16)
        for ch in range(3):
            plane = img[:, :, ch].astype(np.int16) + noise
            img[:, :, ch] = np.clip(plane, 0, 255).astype(np.uint8)

        # --- 2. Last-played cell highlight ---
        if self._last_action is not None and 0 <= self._last_action < 9:
            lr, lc = divmod(self._last_action, 3)
            overlay = img.copy()
            x1h = lc * cell
            y1h = lr * cell
            x2h = x1h + cell
            y2h = y1h + cell
            cv2.rectangle(overlay, (x1h, y1h), (x2h, y2h),
                          (70, 85, 60), -1)  # subtle greenish highlight
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

        # --- 3. Hand-drawn chalk grid lines ---
        chalk_white = (190, 190, 180)  # slightly warm white
        chalk_dim = (130, 130, 120)
        margin = 12
        for i in range(1, 3):
            pos = i * cell
            # Draw multiple offset strokes to simulate hand-drawn look
            for offset in [-1, 0, 1]:
                # Vertical lines
                jx1 = rng.randint(-2, 3)
                jx2 = rng.randint(-2, 3)
                cv2.line(img,
                         (pos + offset + jx1, margin),
                         (pos + offset + jx2, size - margin),
                         chalk_dim, 2, cv2.LINE_AA)
                # Horizontal lines
                jy1 = rng.randint(-2, 3)
                jy2 = rng.randint(-2, 3)
                cv2.line(img,
                         (margin, pos + offset + jy1),
                         (size - margin, pos + offset + jy2),
                         chalk_dim, 2, cv2.LINE_AA)
            # Core line on top (thicker, brighter)
            cv2.line(img, (pos, margin), (pos, size - margin),
                     chalk_white, 3, cv2.LINE_AA)
            cv2.line(img, (margin, pos), (size - margin, pos),
                     chalk_white, 3, cv2.LINE_AA)

        # --- 4. Draw X and O marks ---
        pad = cell // 5
        stroke = max(5, cell // 16)

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
                    # --- X mark: chalk-blue with glow, hand-drawn jitter ---
                    jitter = lambda: int(rng.randint(-3, 4))
                    p1a = (x1 + jitter(), y1 + jitter())
                    p1b = (x2 + jitter(), y2 + jitter())
                    p2a = (x2 + jitter(), y1 + jitter())
                    p2b = (x1 + jitter(), y2 + jitter())

                    # Outer glow (diffuse blue)
                    cv2.line(img, p1a, p1b, (80, 50, 30),
                             stroke + 8, cv2.LINE_AA)
                    cv2.line(img, p2a, p2b, (80, 50, 30),
                             stroke + 8, cv2.LINE_AA)
                    # Mid glow
                    cv2.line(img, p1a, p1b, (170, 130, 60),
                             stroke + 3, cv2.LINE_AA)
                    cv2.line(img, p2a, p2b, (170, 130, 60),
                             stroke + 3, cv2.LINE_AA)
                    # Core bright chalk-blue stroke
                    cv2.line(img, p1a, p1b, (230, 200, 120),
                             stroke, cv2.LINE_AA)
                    cv2.line(img, p2a, p2b, (230, 200, 120),
                             stroke, cv2.LINE_AA)

                elif self.board[r, c] == 2:
                    # --- O mark: warm coral/red chalk with glow ---
                    jitter_r = int(rng.randint(-2, 3))
                    ocx = cx + jitter_r
                    ocy = cy + int(rng.randint(-2, 3))
                    orad = radius + int(rng.randint(-2, 3))

                    # Outer glow
                    cv2.circle(img, (ocx, ocy), orad, (60, 45, 120),
                               stroke + 8, cv2.LINE_AA)
                    # Mid glow
                    cv2.circle(img, (ocx, ocy), orad, (90, 80, 200),
                               stroke + 3, cv2.LINE_AA)
                    # Core bright coral stroke
                    cv2.circle(img, (ocx, ocy), orad, (120, 120, 245),
                               stroke, cv2.LINE_AA)

        # --- 5. Winning line: dramatic golden glow ---
        winning_line = self._get_winning_line()
        if winning_line:
            (r1, c1), (r2, c2) = winning_line
            p1 = (c1 * cell + cell // 2, r1 * cell + cell // 2)
            p2 = (c2 * cell + cell // 2, r2 * cell + cell // 2)
            # Wide golden glow (outermost)
            cv2.line(img, p1, p2, (30, 170, 220), stroke + 16, cv2.LINE_AA)
            # Mid glow
            cv2.line(img, p1, p2, (55, 210, 250), stroke + 8, cv2.LINE_AA)
            # Bright core
            cv2.line(img, p1, p2, (100, 240, 255), stroke + 2, cv2.LINE_AA)
            # Hot white center
            cv2.line(img, p1, p2, (180, 255, 255), stroke - 2, cv2.LINE_AA)

        # --- 6. Chalk dust particles ---
        for _ in range(25):
            dx = rng.randint(0, size)
            dy = rng.randint(0, size)
            brightness = int(rng.randint(55, 100))
            radius_d = int(rng.randint(1, 3))
            cv2.circle(img, (int(dx), int(dy)), radius_d,
                       (brightness, brightness, brightness - 10), -1,
                       cv2.LINE_AA)

        # --- 7. Score/status overlay bar at bottom ---
        bar_h = 38
        bar_y = size - bar_h
        overlay = img.copy()
        cv2.rectangle(overlay, (0, bar_y), (size, size), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        # Count pieces
        x_count = int(np.sum(self.board == 1))
        o_count = int(np.sum(self.board == 2))

        # Status text
        if self._winner == 1:
            status = "X WINS!"
        elif self._winner == 2:
            status = "O WINS!"
        elif self._moves_played >= 9:
            status = "DRAW"
        else:
            status = "X's turn" if self._moves_played % 2 == 0 else "O's turn"

        # "X vs O" with scores on left side
        score_text = f"X:{x_count}  vs  O:{o_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness_t = 1

        # Score on the left
        cv2.putText(img, score_text, (10, bar_y + 25),
                    font, font_scale, (230, 200, 120), thickness_t,
                    cv2.LINE_AA)

        # Status on the right
        (tw, _), _ = cv2.getTextSize(status, font, font_scale, thickness_t)
        cv2.putText(img, status, (size - tw - 10, bar_y + 25),
                    font, font_scale, (120, 220, 250), thickness_t,
                    cv2.LINE_AA)

        # Move counter in center
        move_text = f"Move {self._moves_played}/9"
        (mw, _), _ = cv2.getTextSize(move_text, font, 0.40, 1)
        cv2.putText(img, move_text, ((size - mw) // 2, bar_y + 25),
                    font, 0.40, (160, 160, 150), 1, cv2.LINE_AA)

        return img

    def _get_winning_line(self):
        """Return ((r1,c1),(r2,c2)) of the winning line, or None."""
        for player in (1, 2):
            b = self.board
            # Rows
            for r in range(3):
                if all(b[r, c] == player for c in range(3)):
                    return ((r, 0), (r, 2))
            # Columns
            for c in range(3):
                if all(b[r, c] == player for r in range(3)):
                    return ((0, c), (2, c))
            # Diagonals
            if all(b[i, i] == player for i in range(3)):
                return ((0, 0), (2, 2))
            if all(b[i, 2 - i] == player for i in range(3)):
                return ((0, 2), (2, 0))
        return None

    def render(self, mode='rgb_array'):
        return self._render_rgb()

    def _info(self) -> dict:
        return {
            'moves_played': self._moves_played,
            'winner': self._winner,
        }

    def close(self):
        pass
