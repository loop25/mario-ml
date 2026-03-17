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
        self._last_col = -1

    def reset(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self._pieces_played = 0
        self._winner = 0
        self._last_col = -1
        return self._render_obs()

    def step(self, action):
        self._last_col = action
        # Player 1 move
        if not self._is_valid_column(action):
            # Illegal move — opponent wins by forfeit
            self._winner = 2
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
        """Render a premium Connect Four board for dashboard/stream.

        Renders at 480px with physical board aesthetics: gradient blue
        board, recessed slot holes, glossy 3D pieces, drop indicator,
        winning highlight with connecting line, and score overlay.
        """
        S = self.DISPLAY_SIZE  # 480

        # ── Layout constants ──────────────────────────────────────
        frame_t = 8            # outer frame thickness
        top_bar = 40           # space above board for col numbers + arrow
        bot_bar = 36           # bottom score/turn overlay
        cell = (S - top_bar - bot_bar - frame_t * 2) // ROWS  # ~56
        board_w = cell * COLS
        board_h = cell * ROWS
        ox = (S - board_w) // 2
        oy = top_bar + frame_t
        radius = max(8, cell // 2 - 6)

        # ── Canvas ────────────────────────────────────────────────
        img = np.zeros((S, S, 3), dtype=np.uint8)
        # Dark navy background
        img[:] = (20, 16, 12)

        # ── Outer frame (dark blue stand) ─────────────────────────
        fx1, fy1 = ox - frame_t, oy - frame_t
        fx2, fy2 = ox + board_w + frame_t, oy + board_h + frame_t
        cv2.rectangle(img, (fx1, fy1), (fx2, fy2),
                      (100, 50, 10), -1, cv2.LINE_AA)  # dark blue
        # Inner bevel highlight on frame
        cv2.rectangle(img, (fx1 + 2, fy1 + 2), (fx2 - 2, fy2 - 2),
                      (120, 60, 15), 2, cv2.LINE_AA)

        # ── Board gradient (royal blue, darker at bottom) ─────────
        for y in range(board_h):
            t = y / max(1, board_h - 1)  # 0 top → 1 bottom
            b = int(200 - 55 * t)
            g = int(75 - 25 * t)
            r = int(42 - 15 * t)
            cv2.line(img, (ox, oy + y), (ox + board_w - 1, oy + y),
                     (r, g, b), 1)

        # ── Column numbers above board ────────────────────────────
        font = cv2.FONT_HERSHEY_SIMPLEX
        for c in range(COLS):
            cx = ox + c * cell + cell // 2
            label = str(c + 1)
            ts = cv2.getTextSize(label, font, 0.4, 1)[0]
            tx = cx - ts[0] // 2
            ty = oy - frame_t - 6
            cv2.putText(img, label, (tx, ty), font, 0.4,
                        (160, 170, 190), 1, cv2.LINE_AA)

        # ── Drop column indicator (triangle) ──────────────────────
        if 0 <= self._last_col < COLS:
            ax = ox + self._last_col * cell + cell // 2
            ay = oy - frame_t - 14
            tri = np.array([[ax, ay + 8], [ax - 6, ay], [ax + 6, ay]],
                           dtype=np.int32)
            cv2.fillPoly(img, [tri], (200, 210, 230), cv2.LINE_AA)

        # ── Slot rendering ────────────────────────────────────────
        for r in range(ROWS):
            for c in range(COLS):
                cx = ox + c * cell + cell // 2
                cy = oy + r * cell + cell // 2
                val = self.board[r, c]

                if val == 0:
                    # Recessed empty hole with bevel ring
                    cv2.circle(img, (cx, cy), radius + 2,
                               (15, 30, 90), -1, cv2.LINE_AA)   # outer ring
                    cv2.circle(img, (cx, cy), radius,
                               (8, 8, 22), -1, cv2.LINE_AA)     # dark center
                    # Top-left light bevel arc
                    cv2.ellipse(img, (cx - 1, cy - 1), (radius, radius),
                                0, 200, 340, (30, 55, 130), 1, cv2.LINE_AA)
                    # Bottom-right shadow arc
                    cv2.ellipse(img, (cx + 1, cy + 1), (radius, radius),
                                0, 20, 160, (5, 5, 15), 1, cv2.LINE_AA)

                elif val == 1:
                    # Red piece (BGR: high B=0, G=40, R=210)
                    self._draw_glossy_piece(img, cx, cy, radius,
                                            base=(50, 40, 210),
                                            light=(80, 80, 255),
                                            highlight=(140, 150, 255),
                                            shadow=(15, 15, 100),
                                            rim=(60, 60, 240))

                elif val == 2:
                    # Yellow piece (BGR: high B=0, G=210, R=230)
                    self._draw_glossy_piece(img, cx, cy, radius,
                                            base=(30, 210, 230),
                                            light=(50, 235, 255),
                                            highlight=(120, 250, 255),
                                            shadow=(10, 110, 120),
                                            rim=(40, 225, 245))

        # ── Winning highlight ─────────────────────────────────────
        winning = self._get_winning_cells()
        if winning:
            pulse = int(128 + 127 * np.sin(self._pieces_played * 1.2))
            glow = (pulse, pulse, 255)
            # White ring around each winning piece
            for wr, wc in winning:
                wx = ox + wc * cell + cell // 2
                wy = oy + wr * cell + cell // 2
                cv2.circle(img, (wx, wy), radius + 4, glow,
                           3, cv2.LINE_AA)
            # Connecting line through winning cells
            if len(winning) >= 2:
                p1 = (ox + winning[0][1] * cell + cell // 2,
                       oy + winning[0][0] * cell + cell // 2)
                p2 = (ox + winning[-1][1] * cell + cell // 2,
                       oy + winning[-1][0] * cell + cell // 2)
                cv2.line(img, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)

        # ── Bottom score / turn overlay ───────────────────────────
        bar_y = oy + board_h + frame_t + 4
        red_count = int(np.sum(self.board == 1))
        yel_count = int(np.sum(self.board == 2))

        # Red indicator dot + count
        cv2.circle(img, (ox + 10, bar_y + 12), 7,
                   (40, 40, 210), -1, cv2.LINE_AA)
        cv2.putText(img, f"Red {red_count}",
                    (ox + 22, bar_y + 17), font, 0.42,
                    (130, 140, 220), 1, cv2.LINE_AA)

        # "vs" in center
        vs_ts = cv2.getTextSize("vs", font, 0.4, 1)[0]
        vs_x = ox + board_w // 2 - vs_ts[0] // 2
        cv2.putText(img, "vs", (vs_x, bar_y + 17), font, 0.4,
                    (140, 140, 150), 1, cv2.LINE_AA)

        # Yellow indicator dot + count
        cv2.circle(img, (ox + board_w - 70, bar_y + 12), 7,
                   (30, 200, 230), -1, cv2.LINE_AA)
        cv2.putText(img, f"Yel {yel_count}",
                    (ox + board_w - 58, bar_y + 17), font, 0.42,
                    (80, 210, 230), 1, cv2.LINE_AA)

        # Turn indicator (only if game not over)
        if self._winner == 0 and self._pieces_played < ROWS * COLS:
            whose = "Red" if self._pieces_played % 2 == 0 else "Yellow"
            turn_txt = f"{whose}'s turn"
            tt_s = cv2.getTextSize(turn_txt, font, 0.35, 1)[0]
            tx = ox + board_w // 2 - tt_s[0] // 2
            cv2.putText(img, turn_txt, (tx, bar_y + 30), font, 0.35,
                        (170, 170, 180), 1, cv2.LINE_AA)
        elif self._winner:
            w_name = "Red" if self._winner == 1 else "Yellow"
            win_txt = f"{w_name} wins!"
            wt_s = cv2.getTextSize(win_txt, font, 0.45, 1)[0]
            tx = ox + board_w // 2 - wt_s[0] // 2
            cv2.putText(img, win_txt, (tx, bar_y + 30), font, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)

        return img

    @staticmethod
    def _draw_glossy_piece(img, cx, cy, radius, *, base, light,
                           highlight, shadow, rim):
        """Draw a single 3D glossy game piece with full shading.

        Args:
            img: Canvas (BGR).
            cx, cy: Center.
            radius: Piece radius.
            base: Main fill colour (BGR).
            light: Lighter shade for upper gradient band.
            highlight: Specular highlight colour (BGR).
            shadow: Dark underside colour (BGR).
            rim: Subtle reflected-light colour at bottom edge.
        """
        # Drop shadow offset behind piece
        cv2.circle(img, (cx + 2, cy + 3), radius, shadow,
                   -1, cv2.LINE_AA)

        # Main body
        cv2.circle(img, (cx, cy), radius, base, -1, cv2.LINE_AA)

        # Upper lighter hemisphere (simulate gradient)
        cv2.ellipse(img, (cx, cy - radius // 6),
                    (radius - 2, radius // 2), 0, 180, 360,
                    light, -1, cv2.LINE_AA)

        # Specular highlight — top-left
        hl_r = max(3, radius // 3)
        hl_x = cx - radius // 3
        hl_y = cy - radius // 3
        cv2.circle(img, (hl_x, hl_y), hl_r,
                   highlight, -1, cv2.LINE_AA)
        # Softer secondary spec
        cv2.circle(img, (hl_x + 1, hl_y + 1), max(2, hl_r // 2),
                   (min(255, highlight[0] + 40),
                    min(255, highlight[1] + 40),
                    min(255, highlight[2] + 40)),
                   -1, cv2.LINE_AA)

        # Reflected light at bottom edge
        cv2.ellipse(img, (cx, cy + radius // 2),
                    (radius // 2, radius // 5), 0, 0, 180,
                    rim, -1, cv2.LINE_AA)

        # Crisp edge ring
        cv2.circle(img, (cx, cy), radius, light, 1, cv2.LINE_AA)

    def render(self, mode='rgb_array'):
        return self._render_rgb()

    def _info(self) -> dict:
        return {
            'pieces_played': self._pieces_played,
            'winner': self._winner,
        }

    def close(self):
        pass
