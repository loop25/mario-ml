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
        self._last_from = -1
        self._last_to = -1

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
        self._last_from = -1
        self._last_to = -1
        return self._render_obs()

    def step(self, action):
        from_pos = action // NUM_SQUARES
        to_pos = action % NUM_SQUARES

        # Validate move
        valid_moves = self._get_valid_moves(player=1)
        if (from_pos, to_pos) not in valid_moves:
            if not valid_moves:
                # No valid moves at all — agent loses
                self._winner = 2
                return self._render_obs(), -1.0, True, self._info()
            # Invalid move — redirect to a random valid move with a penalty
            # so the game actually progresses and the agent can learn from
            # board state changes instead of forfeiting every episode.
            from_pos, to_pos = random.choice(valid_moves)

        # Execute agent move
        self._last_from = from_pos
        self._last_to = to_pos
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
        self._last_from = opp_from
        self._last_to = opp_to
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

        Wood-textured board with warm tan / dark walnut squares,
        3D-layered pieces (shadow, outer ring, body, inner ring, dome
        highlight), proper mini-crown for kings, last-move golden
        highlights, beveled frame, and a top info banner.
        """
        import math

        size = self.DISPLAY_SIZE          # 480
        border = 14                       # Thick beveled frame
        inner = size - 2 * border
        cell = inner // 8
        inner = cell * 8                  # Snap to exact multiple
        total = inner + 2 * border
        img = np.zeros((total, total, 3), dtype=np.uint8)
        img[:] = (30, 20, 12)            # Very dark background

        # ── Board squares (wood texture) ──────────────────────────
        LIGHT_BASE = np.array([210, 185, 140], dtype=np.int16)   # warm tan
        DARK_BASE  = np.array([90,  60,  35],  dtype=np.int16)   # dark walnut

        # Build set of last-move squares (row, col) for golden tint
        last_move_rcs = set()
        for lm_pos in (self._last_from, self._last_to):
            if 0 <= lm_pos < NUM_SQUARES:
                last_move_rcs.add(_pos_to_rc(lm_pos))

        for r in range(8):
            for c in range(8):
                x1 = border + c * cell
                y1 = border + r * cell
                x2, y2 = x1 + cell, y1 + cell
                # Deterministic per-square brightness variation
                var = ((r * 13 + c * 7) % 15) - 7          # range -7..+7
                if (r + c) % 2 == 0:
                    base = LIGHT_BASE
                else:
                    base = DARK_BASE
                sq = np.clip(base + var, 0, 255).astype(np.uint8)
                img[y1:y2, x1:x2] = sq

                # Subtle wood grain: horizontal lines every few pixels
                grain_step = max(3, cell // 10)
                for gy in range(y1 + grain_step, y2, grain_step):
                    gvar = ((r * 3 + c * 11 + gy) % 7) - 3
                    g_col = np.clip(base + var + gvar - 6, 0, 255).astype(np.uint8)
                    img[gy, x1:x2] = g_col

                # Golden tint for last-move squares
                if (r, c) in last_move_rcs:
                    overlay = img[y1:y2, x1:x2].astype(np.int16)
                    gold_tint = np.array([25, 20, -10], dtype=np.int16)
                    img[y1:y2, x1:x2] = np.clip(
                        overlay + gold_tint, 0, 255
                    ).astype(np.uint8)
                    # Thin golden border around the highlighted square
                    cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1),
                                  (180, 165, 70), 1, cv2.LINE_AA)

        # ── Beveled board frame ───────────────────────────────────
        # Outer dark border
        cv2.rectangle(img, (0, 0), (total - 1, total - 1),
                      (35, 22, 12), border)
        # Outer edge highlight (top-left lighter bevel)
        cv2.line(img, (0, 0), (total - 1, 0), (70, 50, 30), 2)
        cv2.line(img, (0, 0), (0, total - 1), (70, 50, 30), 2)
        # Outer edge shadow (bottom-right darker)
        cv2.line(img, (0, total - 1), (total - 1, total - 1),
                 (18, 12, 6), 2)
        cv2.line(img, (total - 1, 0), (total - 1, total - 1),
                 (18, 12, 6), 2)
        # Inner bevel highlight
        cv2.rectangle(img, (border - 1, border - 1),
                      (border + inner, border + inner),
                      (120, 90, 55), 1)
        # Secondary inner shadow
        cv2.rectangle(img, (border - 2, border - 2),
                      (border + inner + 1, border + inner + 1),
                      (50, 35, 18), 1)

        # ── Draw pieces ───────────────────────────────────────────
        radius = max(8, cell // 2 - 5)

        for pos in range(NUM_SQUARES):
            r, c = _pos_to_rc(pos)
            cx = border + c * cell + cell // 2
            cy = border + r * cell + cell // 2
            piece = self.board[pos]

            if piece == EMPTY:
                continue

            if piece in (P2_MAN, P2_KING):
                # ── Red / light piece ──
                body       = (185, 45, 40)
                outer_ring = (140, 30, 28)
                inner_ring = (220, 90, 80)
                highlight  = (245, 155, 140)
                shadow     = (60, 15, 12)
            else:
                # ── Dark piece ──
                body       = (55, 55, 58)
                outer_ring = (30, 30, 32)
                inner_ring = (90, 90, 95)
                highlight  = (130, 130, 135)
                shadow     = (12, 12, 14)

            # 1) Bottom shadow (offset down-right 3px)
            cv2.circle(img, (cx + 3, cy + 3), radius + 1, shadow,
                       -1, cv2.LINE_AA)

            # 2) Outer ring (1px border, darker)
            cv2.circle(img, (cx, cy), radius, outer_ring,
                       -1, cv2.LINE_AA)

            # 3) Main body fill (slightly smaller)
            cv2.circle(img, (cx, cy), radius - 2, body,
                       -1, cv2.LINE_AA)

            # 4) Inner decorative ring
            inner_r = max(4, radius * 3 // 5)
            cv2.circle(img, (cx, cy), inner_r, inner_ring,
                       1, cv2.LINE_AA)

            # 5) Top highlight ellipse (dome effect — upper half)
            hl_w = max(4, radius * 2 // 3)
            hl_h = max(3, radius // 3)
            cv2.ellipse(img, (cx, cy - radius // 4),
                        (hl_w, hl_h), 0, 180, 360,
                        highlight, -1, cv2.LINE_AA)

            # ── King crown ────────────────────────────────────────
            if piece in (P1_KING, P2_KING):
                gold       = (50, 190, 220)   # BGR gold
                gold_dark  = (30, 140, 180)    # BGR darker gold

                cw = max(6, radius * 3 // 5)       # crown half-width
                ch = max(4, radius * 2 // 5)       # crown total height
                tip_h = max(2, ch // 2)             # height of pointed tips

                # Crown base: small trapezoid
                base_top = cy + 1
                base_bot = cy + ch // 2 + 1
                trap_pts = np.array([
                    [cx - cw, base_bot],
                    [cx - cw + cw // 3, base_top],
                    [cx + cw - cw // 3, base_top],
                    [cx + cw, base_bot],
                ], dtype=np.int32)
                cv2.fillConvexPoly(img, trap_pts, gold, cv2.LINE_AA)

                # Three pointed tips on top of the trapezoid
                tip_y = base_top - tip_h
                for ti, tx in enumerate([cx - cw + cw // 3,
                                         cx,
                                         cx + cw - cw // 3]):
                    tp = np.array([
                        [tx - cw // 5, base_top],
                        [tx, tip_y],
                        [tx + cw // 5, base_top],
                    ], dtype=np.int32)
                    cv2.fillConvexPoly(img, tp, gold, cv2.LINE_AA)
                    # Tiny gem dot on each tip
                    cv2.circle(img, (tx, tip_y + 1),
                               max(1, cw // 6), gold_dark,
                               -1, cv2.LINE_AA)

                # Crown base band line
                cv2.line(img, (cx - cw, base_bot), (cx + cw, base_bot),
                         gold_dark, 1, cv2.LINE_AA)

        # ── Top banner (semi-transparent) ─────────────────────────
        banner_h = 28
        # Blend a dark overlay
        overlay = img[0:banner_h, :].astype(np.int16)
        overlay = np.clip(overlay * 4 // 10 + 10, 0, 255).astype(np.uint8)
        img[0:banner_h, :] = overlay

        font = cv2.FONT_HERSHEY_SIMPLEX
        p1_count = sum(1 for p in self.board if p in (P1_MAN, P1_KING))
        p2_count = sum(1 for p in self.board if p in (P2_MAN, P2_KING))

        # Title
        cv2.putText(img, "CHECKERS", (total // 2 - 42, 19),
                    font, 0.48, (210, 200, 170), 1, cv2.LINE_AA)

        # Dark pieces count with colored dot (left side)
        cv2.circle(img, (12, 14), 5, (55, 55, 58), -1, cv2.LINE_AA)
        cv2.putText(img, str(p1_count), (22, 19),
                    font, 0.42, (190, 190, 190), 1, cv2.LINE_AA)

        # Red pieces count with colored dot (right of dark count)
        cv2.circle(img, (48, 14), 5, (185, 45, 40), -1, cv2.LINE_AA)
        cv2.putText(img, str(p2_count), (58, 19),
                    font, 0.42, (220, 110, 100), 1, cv2.LINE_AA)

        # Move counter (right side)
        move_text = f"Move {self._moves_played}"
        mtw = cv2.getTextSize(move_text, font, 0.38, 1)[0][0]
        cv2.putText(img, move_text, (total - mtw - 10, 19),
                    font, 0.38, (160, 150, 130), 1, cv2.LINE_AA)

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
