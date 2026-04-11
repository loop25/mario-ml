"""
Tetris game engine as a Gym environment.

A self-contained Tetris implementation following the Gym interface.
Standard 10-wide, 20-tall board with 7 tetrominoes (I, O, T, S, Z, J, L).

Actions: 0=left, 1=right, 2=rotate_cw, 3=rotate_ccw, 4=drop (hard drop)

Observations: 84x84 grayscale images (board rendering).
"""
import random

import cv2
import numpy as np
try:
    import gymnasium as gym
except ImportError:
    import gym  # Legacy fallback

from gymnasium.spaces import Box, Discrete


# Board dimensions
BOARD_W = 10
BOARD_H = 20

# Tetromino definitions: each piece has rotations as lists of (row, col) offsets
# relative to the piece's origin.
TETROMINOES = {
    'I': [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ],
    'O': [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    'T': [
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (1, 1)],
        [(1, 0), (1, 1), (1, 2), (0, 1)],
        [(0, 0), (1, 0), (2, 0), (1, -1)],
    ],
    'S': [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    'Z': [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    'J': [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, -1)],
    ],
    'L': [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}

PIECE_NAMES = list(TETROMINOES.keys())

# Colors for each piece type (used in _render_rgb)
PIECE_COLORS = {
    'I': (0, 240, 240),    # Cyan
    'O': (240, 240, 0),    # Yellow
    'T': (160, 0, 240),    # Purple
    'S': (0, 240, 0),      # Green
    'Z': (240, 0, 0),      # Red
    'J': (0, 0, 240),      # Blue
    'L': (240, 160, 0),    # Orange
}


class TetrisEnv(gym.Env):
    """Tetris as a Gym environment.

    Args:
        render_size: Pixel size of ML observation. Default 84.
        max_steps: Maximum steps per episode. Default 10000.
    """

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, render_size: int = 84, max_steps: int = 10000):
        super().__init__()
        self.render_size = render_size
        self.max_steps = max_steps

        # 5 actions: left, right, rotate_cw, rotate_ccw, hard_drop
        self.action_space = Discrete(5)
        self.observation_space = Box(
            low=0, high=255,
            shape=(render_size, render_size, 1),
            dtype=np.uint8,
        )

        self.board = np.zeros((BOARD_H, BOARD_W), dtype=np.int8)
        self._piece = None
        self._piece_type = None
        self._piece_row = 0
        self._piece_col = 0
        self._rotation = 0
        self._next_piece_type = None
        self._score = 0
        self._lines_cleared = 0
        self._total_steps = 0
        self._level = 1
        self._combo = 0
        self._last_clear_count = 0
        self._gravity_counter = 0  # Steps since last gravity tick

    def reset(self):
        self.board = np.zeros((BOARD_H, BOARD_W), dtype=np.int8)
        self._score = 0
        self._lines_cleared = 0
        self._total_steps = 0
        self._level = 1
        self._combo = 0
        self._last_clear_count = 0
        self._gravity_counter = 0
        self._next_piece_type = random.choice(PIECE_NAMES)
        self._spawn_piece()
        return self._render_obs()

    def step(self, action):
        self._total_steps += 1
        self._last_clear_count = 0  # Reset flash from previous frame
        reward = 0.0

        # Apply action to current piece
        hard_dropped = False
        if action == 0:    # Left
            self._try_move(0, -1)
        elif action == 1:  # Right
            self._try_move(0, 1)
        elif action == 2:  # Rotate CW
            self._try_rotate(1)
        elif action == 3:  # Rotate CCW
            self._try_rotate(-1)
        elif action == 4:  # Hard drop — instant landing, bypasses gravity
            drop_rows = 0
            while self._try_move(1, 0):
                drop_rows += 1
            reward += drop_rows * 0.02  # Small reward for dropping
            hard_dropped = True

        # Gravity: piece falls based on level speed.
        # Higher levels = faster drops. Level 1: every 20 steps, Level 10+: every 2.
        # Hard drops skip gravity and lock immediately.
        if not hard_dropped:
            gravity_interval = max(2, 22 - self._level * 2)
            self._gravity_counter += 1
            if self._gravity_counter < gravity_interval:
                # No gravity this step — piece stays in place
                if self._total_steps >= self.max_steps:
                    return self._render_obs(), 0.0, True, self._info()
                return self._render_obs(), reward, False, self._info()
            self._gravity_counter = 0

        # Gravity tick (or hard drop): piece falls one row
        if not self._try_move(1, 0):
            # Piece landed — lock it
            self._lock_piece()
            cleared = self._clear_lines()
            self._last_clear_count = cleared

            # Scoring: line clears are the ONLY significant reward.
            # Survival reward removed — it caused agents to learn "stack
            # randomly for guaranteed +50" instead of "clear lines for +10-80".
            if cleared > 0:
                self._combo += 1
                line_rewards = {1: 10.0, 2: 30.0, 3: 50.0, 4: 80.0}
                reward += line_rewards.get(cleared, cleared * 20.0)
                reward += self._combo * 2.0  # Combo bonus
            else:
                self._combo = 0

            # Height penalty: discourage stacking too high.
            # Count occupied cells in the top 4 rows — penalize tall stacks.
            top_cells = int(np.sum(self.board[:4] != 0))
            if top_cells > 0:
                reward -= top_cells * 0.05

            # Level up every 10 lines
            self._level = 1 + self._lines_cleared // 10

            # Spawn next piece
            if not self._spawn_piece():
                # Game over — can't place new piece
                return self._render_obs(), -2.0, True, self._info()

        # Max steps limit
        if self._total_steps >= self.max_steps:
            return self._render_obs(), 0.0, True, self._info()

        return self._render_obs(), reward, False, self._info()

    def _spawn_piece(self) -> bool:
        """Spawn a new piece at the top. Returns False if blocked (game over)."""
        self._piece_type = self._next_piece_type
        self._next_piece_type = random.choice(PIECE_NAMES)
        self._rotation = 0
        self._piece = TETROMINOES[self._piece_type][0]
        self._piece_row = 0
        self._piece_col = BOARD_W // 2 - 1

        # Check if spawn position is blocked
        if not self._is_valid_position(self._piece_row, self._piece_col,
                                       self._rotation):
            return False
        return True

    def _try_move(self, dr: int, dc: int) -> bool:
        """Try to move the piece. Returns True if successful."""
        new_r = self._piece_row + dr
        new_c = self._piece_col + dc
        if self._is_valid_position(new_r, new_c, self._rotation):
            self._piece_row = new_r
            self._piece_col = new_c
            return True
        return False

    def _try_rotate(self, direction: int) -> bool:
        """Try to rotate the piece. direction: 1=CW, -1=CCW."""
        new_rot = (self._rotation + direction) % 4
        if self._is_valid_position(self._piece_row, self._piece_col, new_rot):
            self._rotation = new_rot
            self._piece = TETROMINOES[self._piece_type][new_rot]
            return True
        # Wall kick: try shifting left/right
        for offset in [1, -1, 2, -2]:
            if self._is_valid_position(self._piece_row,
                                       self._piece_col + offset, new_rot):
                self._piece_col += offset
                self._rotation = new_rot
                self._piece = TETROMINOES[self._piece_type][new_rot]
                return True
        return False

    def _is_valid_position(self, row: int, col: int, rotation: int) -> bool:
        """Check if piece at given position/rotation is valid."""
        blocks = TETROMINOES[self._piece_type][rotation]
        for dr, dc in blocks:
            r, c = row + dr, col + dc
            if r < 0 or r >= BOARD_H or c < 0 or c >= BOARD_W:
                return False
            if self.board[r, c] != 0:
                return False
        return True

    def _lock_piece(self):
        """Lock the current piece into the board."""
        piece_id = PIECE_NAMES.index(self._piece_type) + 1
        blocks = TETROMINOES[self._piece_type][self._rotation]
        for dr, dc in blocks:
            r, c = self._piece_row + dr, self._piece_col + dc
            if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
                self.board[r, c] = piece_id

    def _clear_lines(self) -> int:
        """Clear completed lines. Returns number of lines cleared."""
        cleared = 0
        new_board = []
        for row in range(BOARD_H):
            if np.all(self.board[row] != 0):
                cleared += 1
            else:
                new_board.append(self.board[row].copy())

        if cleared > 0:
            self._lines_cleared += cleared
            self._score += cleared * 100 * self._level
            # Rebuild board with empty rows on top
            empty_rows = [np.zeros(BOARD_W, dtype=np.int8)
                          for _ in range(cleared)]
            self.board = np.array(empty_rows + new_board, dtype=np.int8)

        return cleared

    def _info(self) -> dict:
        return {
            'score': self._score,
            'lines_cleared': self._lines_cleared,
            'level': self._level,
            'total_steps': self._total_steps,
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_obs(self) -> np.ndarray:
        """Render board as an 84x84 grayscale image for ML."""
        grid = np.zeros((BOARD_H, BOARD_W), dtype=np.uint8)

        # Draw locked pieces
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                if self.board[r, c] != 0:
                    grid[r, c] = 200

        # Draw current piece
        if self._piece_type is not None:
            blocks = TETROMINOES[self._piece_type][self._rotation]
            for dr, dc in blocks:
                r, c = self._piece_row + dr, self._piece_col + dc
                if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
                    grid[r, c] = 255

        obs = cv2.resize(grid, (self.render_size, self.render_size),
                         interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(obs, axis=-1)

    def render(self, mode='rgb_array'):
        """Render as RGB for dashboard display."""
        return self._render_rgb()

    # Display constants
    DISPLAY_SIZE = 480
    _CELL = 20           # Pixels per cell on the board
    _BOARD_X = 40        # Left margin for the board
    _BOARD_Y = 30        # Top margin for the board

    def _render_rgb(self) -> np.ndarray:
        """Render a polished Tetris board for dashboard/stream display.

        Features: dark background with subtle grid, colorful glossy pieces,
        ghost piece preview, next-piece panel, score/lines/level HUD,
        line-clear flash effect, and gradient frame.
        """
        size = self.DISPLAY_SIZE
        cell = self._CELL
        bx, by = self._BOARD_X, self._BOARD_Y
        board_w = BOARD_W * cell
        board_h = BOARD_H * cell

        # --- Background: dark gradient ---
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for y in range(size):
            t = y / size
            img[y, :] = (int(10 + 8 * t), int(10 + 12 * t), int(25 + 15 * t))

        # --- Board outline + playfield background ---
        # Outer frame (subtle glow)
        cv2.rectangle(img, (bx - 3, by - 3),
                      (bx + board_w + 2, by + board_h + 2),
                      (60, 80, 120), 2)
        # Playfield fill (very dark blue)
        cv2.rectangle(img, (bx, by),
                      (bx + board_w - 1, by + board_h - 1),
                      (12, 14, 22), -1)

        # --- Grid lines (subtle) ---
        for c in range(1, BOARD_W):
            x = bx + c * cell
            cv2.line(img, (x, by), (x, by + board_h - 1), (25, 28, 38), 1)
        for r in range(1, BOARD_H):
            y = by + r * cell
            cv2.line(img, (bx, y), (bx + board_w - 1, y), (25, 28, 38), 1)

        # --- Draw locked pieces ---
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                piece_id = self.board[r, c]
                if piece_id != 0:
                    piece_name = PIECE_NAMES[piece_id - 1]
                    color = PIECE_COLORS[piece_name]
                    self._draw_block(img, bx + c * cell, by + r * cell,
                                     cell, color)

        # --- Draw ghost piece (drop preview) ---
        if self._piece_type is not None:
            ghost_row = self._piece_row
            while self._is_valid_position(ghost_row + 1, self._piece_col,
                                          self._rotation):
                ghost_row += 1
            if ghost_row != self._piece_row:
                blocks = TETROMINOES[self._piece_type][self._rotation]
                color = PIECE_COLORS[self._piece_type]
                ghost_color = (color[0] // 4, color[1] // 4, color[2] // 4)
                for dr, dc in blocks:
                    r, c = ghost_row + dr, self._piece_col + dc
                    if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
                        x0, y0 = bx + c * cell, by + r * cell
                        cv2.rectangle(img, (x0 + 1, y0 + 1),
                                      (x0 + cell - 2, y0 + cell - 2),
                                      ghost_color, 1)

            # --- Draw current piece ---
            blocks = TETROMINOES[self._piece_type][self._rotation]
            color = PIECE_COLORS[self._piece_type]
            for dr, dc in blocks:
                r, c = self._piece_row + dr, self._piece_col + dc
                if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
                    self._draw_block(img, bx + c * cell, by + r * cell,
                                     cell, color)

        # --- Line clear flash effect ---
        if self._last_clear_count > 0:
            flash_alpha = 0.4
            flash = np.full_like(img, 255)
            # Flash the cleared area
            y_start = by + (BOARD_H - self._last_clear_count) * cell
            y_end = by + BOARD_H * cell
            img[y_start:y_end, bx:bx + board_w] = cv2.addWeighted(
                img[y_start:y_end, bx:bx + board_w], 1.0 - flash_alpha,
                flash[y_start:y_end, bx:bx + board_w], flash_alpha, 0,
            )

        # --- Right panel: Next piece ---
        panel_x = bx + board_w + 20
        panel_y = by + 10

        # "NEXT" label
        cv2.putText(img, 'NEXT', (panel_x, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 160, 180), 1,
                    cv2.LINE_AA)

        # Next piece preview box
        preview_y = panel_y + 10
        preview_size = cell * 4
        cv2.rectangle(img, (panel_x, preview_y),
                      (panel_x + preview_size, preview_y + preview_size),
                      (30, 35, 50), -1)
        cv2.rectangle(img, (panel_x, preview_y),
                      (panel_x + preview_size, preview_y + preview_size),
                      (50, 60, 80), 1)

        # Draw next piece centered in preview
        if self._next_piece_type is not None:
            next_blocks = TETROMINOES[self._next_piece_type][0]
            next_color = PIECE_COLORS[self._next_piece_type]
            # Center the piece in the preview box
            min_r = min(dr for dr, dc in next_blocks)
            max_r = max(dr for dr, dc in next_blocks)
            min_c = min(dc for dr, dc in next_blocks)
            max_c = max(dc for dr, dc in next_blocks)
            piece_h = max_r - min_r + 1
            piece_w = max_c - min_c + 1
            offset_r = (4 - piece_h) // 2 - min_r
            offset_c = (4 - piece_w) // 2 - min_c
            for dr, dc in next_blocks:
                px = panel_x + (dc + offset_c) * cell
                py = preview_y + (dr + offset_r) * cell
                self._draw_block(img, px, py, cell, next_color)

        # --- Stats panel ---
        stats_y = preview_y + preview_size + 30
        stats = [
            ('SCORE', f'{self._score:,}'),
            ('LINES', f'{self._lines_cleared}'),
            ('LEVEL', f'{self._level}'),
        ]
        for i, (label, value) in enumerate(stats):
            y_off = stats_y + i * 50
            cv2.putText(img, label, (panel_x, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 110, 130), 1,
                        cv2.LINE_AA)
            cv2.putText(img, value, (panel_x, y_off + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 230, 240), 1,
                        cv2.LINE_AA)

        # --- Combo indicator ---
        if self._combo > 1:
            combo_y = stats_y + len(stats) * 50 + 10
            combo_text = f'COMBO x{self._combo}'
            # Pulsing color based on combo count
            pulse = min(self._combo * 30, 255)
            cv2.putText(img, combo_text, (panel_x, combo_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (pulse, 255 - pulse // 2, 50), 1, cv2.LINE_AA)

        # --- Bottom status bar ---
        bar_y = by + board_h + 12
        status = f'Tetris  |  Score: {self._score:,}  |  Lines: {self._lines_cleared}  |  Level: {self._level}'
        cv2.putText(img, status, (bx, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 130, 150), 1,
                    cv2.LINE_AA)

        # Resize to standard display size
        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

        return img

    @staticmethod
    def _draw_block(img, x, y, cell, color):
        """Draw a single Tetris block with 3D-ish bevel effect."""
        r, g, b = color
        # Fill
        cv2.rectangle(img, (x + 1, y + 1), (x + cell - 2, y + cell - 2),
                      (r, g, b), -1)
        # Highlight (top-left edges) — brighter
        hi = (min(r + 60, 255), min(g + 60, 255), min(b + 60, 255))
        cv2.line(img, (x + 1, y + 1), (x + cell - 2, y + 1), hi, 1)
        cv2.line(img, (x + 1, y + 1), (x + 1, y + cell - 2), hi, 1)
        # Shadow (bottom-right edges) — darker
        lo = (max(r - 60, 0), max(g - 60, 0), max(b - 60, 0))
        cv2.line(img, (x + cell - 2, y + 1), (x + cell - 2, y + cell - 2),
                 lo, 1)
        cv2.line(img, (x + 1, y + cell - 2), (x + cell - 2, y + cell - 2),
                 lo, 1)
        # Center shine spot
        cx, cy = x + cell // 2, y + cell // 2
        shine = (min(r + 90, 255), min(g + 90, 255), min(b + 90, 255))
        cv2.circle(img, (cx - 1, cy - 1), max(cell // 6, 1), shine, -1)
