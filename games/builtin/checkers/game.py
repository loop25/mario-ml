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

    # High-res rendering for the dashboard
    DISPLAY_SIZE = 480

    def _render_rgb(self) -> np.ndarray:
        """Render a polished checkers board for dashboard/stream display.

        Classic dark-green and cream board with a dark wooden frame,
        3D-shaded red and black pieces with gradient highlights, and
        gold crown symbols for kings. Score overlay shows piece counts.
        """
        import math

        size = self.DISPLAY_SIZE
        border = 10  # Dark wooden frame thickness
        inner = size - 2 * border
        cell = inner // 8
        inner = cell * 8  # Snap to exact multiple
        total = inner + 2 * border
        img = np.zeros((total, total, 3), dtype=np.uint8)
        img[:] = (20, 15, 10)  # Dark background

        # Board colors
        CREAM = (220, 210, 185)
        GREEN = (30, 80, 30)

        # Draw board squares
        for r in range(8):
            for c in range(8):
                x1 = border + c * cell
                y1 = border + r * cell
                x2, y2 = x1 + cell, y1 + cell
                if (r + c) % 2 == 0:
                    img[y1:y2, x1:x2] = CREAM
                else:
                    img[y1:y2, x1:x2] = GREEN

        # Dark wooden frame border
        cv2.rectangle(img, (0, 0), (total - 1, total - 1),
                      (60, 40, 20), border)
        # Inner highlight line
        cv2.rectangle(img, (border - 1, border - 1),
                      (border + inner, border + inner),
                      (90, 65, 35), 1)

        # Draw pieces with 3D gradient shading
        radius = max(6, cell // 2 - 6)

        for pos in range(NUM_SQUARES):
            r, c = _pos_to_rc(pos)
            cx = border + c * cell + cell // 2
            cy = border + r * cell + cell // 2
            piece = self.board[pos]

            if piece == EMPTY:
                continue

            if piece in (P2_MAN, P2_KING):
                # Red piece
                base_color = (200, 50, 50)
                shadow_color = (100, 20, 20)
                highlight_color = (240, 130, 120)
                edge_color = (230, 80, 70)
                dark_edge = (140, 30, 25)
            else:
                # Black piece
                base_color = (50, 50, 50)
                shadow_color = (20, 20, 20)
                highlight_color = (100, 100, 100)
                edge_color = (80, 80, 80)
                dark_edge = (30, 30, 30)

            # Drop shadow (offset down-right)
            cv2.circle(img, (cx + 2, cy + 3), radius, shadow_color,
                       -1, cv2.LINE_AA)

            # Base circle
            cv2.circle(img, (cx, cy), radius, base_color,
                       -1, cv2.LINE_AA)

            # 3D gradient effect: darker bottom-right arc
            cv2.ellipse(img, (cx + 1, cy + 1), (radius - 1, radius - 1),
                        0, 30, 210, dark_edge, 2, cv2.LINE_AA)

            # Lighter top-left highlight arc
            cv2.ellipse(img, (cx - 1, cy - 1), (radius - 2, radius - 2),
                        0, 200, 350, highlight_color, 2, cv2.LINE_AA)

            # Specular highlight (top-left blob)
            hl_r = max(3, radius // 3)
            cv2.circle(img, (cx - radius // 4, cy - radius // 4),
                       hl_r, highlight_color, -1, cv2.LINE_AA)

            # Edge ring
            cv2.circle(img, (cx, cy), radius, edge_color,
                       1, cv2.LINE_AA)

            # King: gold crown/star symbol
            if piece in (P1_KING, P2_KING):
                gold = (220, 180, 50)
                gold_bright = (255, 220, 60)
                crown_r = max(4, radius * 2 // 3)
                # Crown ring
                cv2.circle(img, (cx, cy), crown_r, gold,
                           2, cv2.LINE_AA)
                # Star points (5-pointed)
                for i in range(5):
                    angle = math.radians(i * 72 - 90)
                    px = int(cx + crown_r * 0.55 * math.cos(angle))
                    py = int(cy + crown_r * 0.55 * math.sin(angle))
                    cv2.circle(img, (px, py), max(1, radius // 8),
                               gold_bright, -1, cv2.LINE_AA)
                # Center dot
                cv2.circle(img, (cx, cy), max(1, radius // 6),
                           gold_bright, -1, cv2.LINE_AA)

        # Score overlay — piece counts
        p1_count = sum(1 for p in self.board if p in (P1_MAN, P1_KING))
        p2_count = sum(1 for p in self.board if p in (P2_MAN, P2_KING))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Dark pieces count (top-left)
        dark_text = f"Dark: {p1_count}"
        cv2.rectangle(img, (border, border),
                      (border + 88, border + 22), (20, 15, 10), -1)
        cv2.putText(img, dark_text, (border + 4, border + 16),
                    font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Red pieces count (top-right)
        red_text = f"Red: {p2_count}"
        tw = cv2.getTextSize(red_text, font, 0.45, 1)[0][0]
        rx = border + inner - tw - 8
        cv2.rectangle(img, (rx - 4, border),
                      (border + inner, border + 22), (20, 15, 10), -1)
        cv2.putText(img, red_text, (rx, border + 16),
                    font, 0.45, (220, 100, 90), 1, cv2.LINE_AA)

        # Move counter (bottom-center)
        move_text = f"Move {self._moves_played}"
        mtw = cv2.getTextSize(move_text, font, 0.40, 1)[0][0]
        mx = total // 2 - mtw // 2
        my = border + inner - 4
        cv2.rectangle(img, (mx - 4, my - 14), (mx + mtw + 4, my + 4),
                      (20, 15, 10), -1)
        cv2.putText(img, move_text, (mx, my),
                    font, 0.40, (160, 150, 130), 1, cv2.LINE_AA)

        # Resize to exact DISPLAY_SIZE if frame differs
        if total != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

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
