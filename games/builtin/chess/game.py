"""
Chess game engine as a Gym environment.

Wraps the `python-chess` library for correct rule enforcement including
castling, en passant, promotion, check, checkmate, stalemate, threefold
repetition, and the fifty-move rule.

Agent plays as White (player 1). A random opponent plays Black (player 2).

Actions are encoded as: from_square * 64 + to_square, where squares use
python-chess ordering (a1=0, b1=1, ..., h8=63). The action space is
Discrete(4096) — most actions are invalid at any given time. Invalid
moves result in a forfeit (reward -1).

Promotions try queen first, then knight, rook, bishop (for underpromotion).

Observations: 84x84 grayscale image.
"""
import random

import cv2
import numpy as np
try:
    import gymnasium as gym
except ImportError:
    import gym  # Legacy fallback

from gymnasium.spaces import Box, Discrete

try:
    import chess
    HAS_CHESS = True
except ImportError:
    HAS_CHESS = False

NUM_SQUARES = 64
ACTION_SPACE_SIZE = NUM_SQUARES * NUM_SQUARES  # 4096


class ChessEnv(gym.Env):
    """Chess as a Gym environment with a pluggable opponent."""

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, render_size: int = 84, max_moves: int = 200,
                 opponent=None):
        super().__init__()
        if not HAS_CHESS:
            raise ImportError(
                "python-chess is required for the Chess game. "
                "Install it with: pip install python-chess"
            )
        if opponent is None:
            from src.opponents import RandomOpponent
            opponent = RandomOpponent()
        self.opponent = opponent
        self.render_size = render_size
        self.max_moves = max_moves
        self.action_space = Discrete(ACTION_SPACE_SIZE)
        self.observation_space = Box(
            low=0, high=255,
            shape=(render_size, render_size, 1),
            dtype=np.uint8,
        )
        self.board = chess.Board()
        self._winner = 0
        self._moves_played = 0
        self._last_move = None  # Track last move for highlight
        self._captured_white = []  # White pieces captured by black
        self._captured_black = []  # Black pieces captured by white

    def reset(self):
        self.board = chess.Board()
        self._winner = 0
        self._moves_played = 0
        self._last_move = None
        self._captured_white = []
        self._captured_black = []
        return self._render_obs()

    def step(self, action):
        from_sq = action // NUM_SQUARES
        to_sq = action % NUM_SQUARES

        # Build the move — try promotion if pawn reaching last rank
        move = chess.Move(from_sq, to_sq)
        piece = self.board.piece_at(from_sq)
        if (piece is not None and piece.piece_type == chess.PAWN
                and chess.square_rank(to_sq) in (0, 7)):
            # Try queen first (most common), then knight (useful for forks),
            # then rook, then bishop
            for promo in [chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP]:
                candidate = chess.Move(from_sq, to_sq, promotion=promo)
                if candidate in self.board.legal_moves:
                    move = candidate
                    break

        # Validate move
        if move not in self.board.legal_moves:
            self._winner = 2
            return self._render_obs(), -1.0, True, self._info()

        # Piece value table for intermediate capture rewards.
        # IMPORTANT: These must be very small relative to the terminal
        # win/loss (±1.0). If too large, the agent learns "maximize
        # captures" instead of "win the game." Total possible captures
        # over a game should sum to << 1.0.
        _PIECE_VALUES = {
            chess.PAWN: 0.001, chess.KNIGHT: 0.003, chess.BISHOP: 0.003,
            chess.ROOK: 0.005, chess.QUEEN: 0.009,
        }

        # Track captured piece and compute capture reward
        step_reward = 0.0
        captured = self.board.piece_at(move.to_square)
        if captured is not None:
            if captured.color == chess.WHITE:
                self._captured_white.append(captured.piece_type)
            else:
                self._captured_black.append(captured.piece_type)
                step_reward += _PIECE_VALUES.get(captured.piece_type, 0.0)

        # Tiny reward for giving check (encourages aggression without
        # dominating the terminal win signal)
        self.board.push(move)
        if self.board.is_check():
            step_reward += 0.002
        self.board.pop()

        # Execute agent move (White)
        self._last_move = move
        self.board.push(move)
        self._moves_played += 1

        # Check terminal after agent move
        done, reward = self._check_terminal()
        if done:
            return self._render_obs(), reward, True, self._info()

        # Max move limit
        if self._moves_played >= self.max_moves:
            return self._render_obs(), 0.0, True, self._info()

        # Opponent move (Black)
        opp_moves = list(self.board.legal_moves)
        if not opp_moves:
            self._winner = 1
            return self._render_obs(), 1.0, True, self._info()

        board_state = {
            'board': self.board,
            'valid_actions': list(range(len(opp_moves))),
            'game_id': 'chess',
            'turn': 2,
        }
        opp_idx = self.opponent.pick_action(board_state)
        opp_move = opp_moves[opp_idx]

        # Track opponent capture — penalize agent for losing pieces
        opp_captured = self.board.piece_at(opp_move.to_square)
        if opp_captured is not None:
            if opp_captured.color == chess.WHITE:
                self._captured_white.append(opp_captured.piece_type)
                step_reward -= _PIECE_VALUES.get(opp_captured.piece_type, 0.0)
            else:
                self._captured_black.append(opp_captured.piece_type)

        self._last_move = opp_move
        self.board.push(opp_move)
        self._moves_played += 1

        # Check terminal after opponent move
        done, reward = self._check_terminal()
        if done:
            return self._render_obs(), reward + step_reward, True, self._info()

        if self._moves_played >= self.max_moves:
            return self._render_obs(), step_reward, True, self._info()

        return self._render_obs(), step_reward, False, self._info()

    def _check_terminal(self):
        """Check if the game is over. Returns (done, reward_for_agent)."""
        if self.board.is_checkmate():
            # The side to move is in checkmate — they lost
            if self.board.turn == chess.BLACK:
                # Black to move and checkmated => White (agent) wins
                self._winner = 1
                return True, 1.0
            else:
                # White to move and checkmated => Black (opponent) wins
                self._winner = 2
                return True, -1.0

        if self.board.is_stalemate():
            return True, 0.0

        if self.board.is_insufficient_material():
            return True, 0.0

        if self.board.is_fivefold_repetition():
            return True, 0.0

        if self.board.is_seventyfive_moves():
            return True, 0.0

        return False, 0.0

    def _render_obs(self) -> np.ndarray:
        """Render board as 84x84 grayscale for ML input.

        Encoding:
        - White pieces: 200 (pawns/bishops/knights/rooks), 255 (queen/king)
        - Black pieces: 80 (pawns/bishops/knights/rooks), 128 (queen/king)
        - Empty: 0
        """
        grid = np.zeros((8, 8), dtype=np.uint8)
        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece is None:
                continue
            # python-chess: rank 0 = row 0 in grid, file 0 = col 0
            row = 7 - chess.square_rank(sq)  # Flip: rank 7 at top
            col = chess.square_file(sq)
            if piece.color == chess.WHITE:
                if piece.piece_type in (chess.QUEEN, chess.KING):
                    grid[row, col] = 255
                else:
                    grid[row, col] = 200
            else:
                if piece.piece_type in (chess.QUEEN, chess.KING):
                    grid[row, col] = 128
                else:
                    grid[row, col] = 80

        obs = cv2.resize(grid, (self.render_size, self.render_size),
                         interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(obs, axis=-1)

    # High-res rendering for the dashboard (independent of ML obs size)
    DISPLAY_SIZE = 480

    # Board frame and margin sizes
    _FRAME = 16   # Thick wood frame around the board
    _MARGIN = 24  # Coordinate label area (inside the frame)

    # Piece-type symbols for the status bar (ASCII fallback labels)
    # Uses integer constants (PAWN=1..KING=6) so the class can be defined
    # even when python-chess is not installed.
    _PIECE_LABELS = {1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K'}
    # Piece ordering for material display (most valuable first)
    _PIECE_ORDER = [5, 4, 3, 2, 1]  # QUEEN, ROOK, BISHOP, KNIGHT, PAWN

    def _render_rgb(self) -> np.ndarray:
        """Render a polished chess board for dashboard/stream display.

        Features: dark wood frame, textured warm board, coordinate labels,
        last-move highlights, check indicator, geometric pieces with drop
        shadows, captured-pieces tray, and a full-width status banner.
        """
        size = self.DISPLAY_SIZE
        frame = self._FRAME
        margin = self._MARGIN
        # Board area sits inside frame + label margin
        inner = size - 2 * frame  # Area inside the outer frame
        board_px = inner - margin  # Board squares area (labels on left/bottom)
        cell = board_px // 8
        board_px = cell * 8  # Snap to exact multiple

        # Canvas is the full DISPLAY_SIZE
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = (30, 25, 20)

        # --- Dark wood frame ---
        # Outer frame fill: polished dark wood gradient look
        frame_color_outer = (45, 32, 22)
        frame_color_inner = (65, 48, 35)
        frame_highlight = (85, 65, 48)

        # Fill outer frame region
        img[0:size, 0:size] = frame_color_outer
        # Inner bevel (lighter strip)
        cv2.rectangle(img, (frame - 3, frame - 3),
                      (size - frame + 2, size - frame + 2),
                      frame_highlight, 2, cv2.LINE_AA)
        cv2.rectangle(img, (frame - 1, frame - 1),
                      (size - frame, size - frame),
                      frame_color_inner, 1, cv2.LINE_AA)
        # Outer edge highlight (top-left light, bottom-right dark)
        cv2.line(img, (0, 0), (size - 1, 0), (75, 58, 42), 1)
        cv2.line(img, (0, 0), (0, size - 1), (75, 58, 42), 1)
        cv2.line(img, (size - 1, 0), (size - 1, size - 1), (25, 18, 12), 1)
        cv2.line(img, (0, size - 1), (size - 1, size - 1), (25, 18, 12), 1)

        # Clear the inner area (labels + board) to dark bg
        img[frame:size - frame, frame:size - frame] = (30, 25, 20)

        # Board origin: labels on left and bottom inside the inner area
        ox = frame + margin  # Board squares start after frame + label margin
        oy = frame           # Board squares start at top of inner area

        # --- Warm wooden square colors ---
        LIGHT_SQ = np.array([222, 196, 160], dtype=np.int16)
        DARK_SQ = np.array([160, 120, 80], dtype=np.int16)
        HIGHLIGHT_LIGHT = np.array([240, 220, 130], dtype=np.int16)
        HIGHLIGHT_DARK = np.array([200, 180, 80], dtype=np.int16)
        CHECK_TINT = (180, 80, 80)

        # Determine last-move squares and check square
        last_from = last_to = -1
        if self._last_move is not None:
            last_from = self._last_move.from_square
            last_to = self._last_move.to_square

        check_sq = -1
        if self.board.is_check():
            king_sq = self.board.king(self.board.turn)
            if king_sq is not None:
                check_sq = king_sq

        # --- Draw board squares with per-square texture variation ---
        for row in range(8):
            for col in range(8):
                sq = chess.square(col, 7 - row)
                x1 = ox + col * cell
                y1 = oy + row * cell
                x2, y2 = x1 + cell, y1 + cell

                is_light = (row + col) % 2 == 0

                # Choose base color
                if sq == check_sq:
                    color = CHECK_TINT
                    img[y1:y2, x1:x2] = color
                    continue
                elif sq == last_from or sq == last_to:
                    base = HIGHLIGHT_LIGHT if is_light else HIGHLIGHT_DARK
                else:
                    base = LIGHT_SQ if is_light else DARK_SQ

                # Deterministic per-square texture variation
                variation = ((row * 7 + col * 13) % 8 - 4)
                color = tuple(int(max(0, min(255, int(c) + variation)))
                              for c in base)
                img[y1:y2, x1:x2] = color

        # --- Draw pieces with drop shadows ---
        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece is None:
                continue
            row = 7 - chess.square_rank(sq)
            col = chess.square_file(sq)
            cx = ox + col * cell + cell // 2
            cy = oy + row * cell + cell // 2
            self._draw_piece(img, piece, cx, cy, cell)

        # --- Coordinate labels ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_scale = 0.38
        label_color = (180, 165, 140)
        label_thick = 1
        banner_h = 22  # Status banner height (used for label positioning)
        files = 'abcdefgh'
        label_row_y = oy + 8 * cell + banner_h + 4  # Below the status banner
        for col in range(8):
            lx = ox + col * cell + cell // 2 - 4
            cv2.putText(img, files[col], (lx, label_row_y + 8), font,
                        label_scale, label_color, label_thick, cv2.LINE_AA)
        for row in range(8):
            rank = str(8 - row)
            lx = frame + 5
            ly = oy + row * cell + cell // 2 + 4
            cv2.putText(img, rank, (lx, ly), font, label_scale,
                        label_color, label_thick, cv2.LINE_AA)

        # --- Captured pieces tray (in the frame edges) ---
        cap_font = cv2.FONT_HERSHEY_SIMPLEX
        cap_scale = 0.30

        # Captured white pieces shown in top frame strip
        cap_x = ox
        cap_y_top = 11  # Vertically centered in top frame
        for pt in sorted(self._captured_white,
                         key=lambda t: self._PIECE_ORDER.index(t)
                         if t in self._PIECE_ORDER else 99):
            label = self._PIECE_LABELS.get(pt, '?')
            cv2.putText(img, label, (cap_x, cap_y_top), cap_font,
                        cap_scale, (200, 190, 170), 1, cv2.LINE_AA)
            cap_x += 12

        # Captured black pieces shown in right frame strip (vertical)
        cap_ry = oy + 4
        cap_rx = size - frame + 3  # In the right frame strip
        for pt in sorted(self._captured_black,
                         key=lambda t: self._PIECE_ORDER.index(t)
                         if t in self._PIECE_ORDER else 99):
            label = self._PIECE_LABELS.get(pt, '?')
            cv2.putText(img, label, (cap_rx, cap_ry + 8), cap_font,
                        cap_scale, (120, 110, 100), 1, cv2.LINE_AA)
            cap_ry += 12

        # --- Full-width status banner at the bottom of the board area ---
        banner_y = oy + 8 * cell  # Just below the board squares
        banner_x1 = ox
        banner_x2 = ox + board_px

        # Semi-transparent dark banner (blend with background)
        overlay = img[banner_y:banner_y + banner_h,
                      banner_x1:banner_x2].astype(np.int16)
        overlay = np.clip(overlay * 4 // 10, 0, 255).astype(np.uint8)
        img[banner_y:banner_y + banner_h, banner_x1:banner_x2] = overlay

        # Side-to-move indicator dot
        dot_cx = banner_x1 + 8
        dot_cy = banner_y + banner_h // 2
        dot_color = (230, 225, 210) if self.board.turn == chess.WHITE \
            else (50, 45, 40)
        cv2.circle(img, (dot_cx, dot_cy), 5, dot_color, -1, cv2.LINE_AA)
        cv2.circle(img, (dot_cx, dot_cy), 5, (140, 130, 110), 1, cv2.LINE_AA)

        # Move number
        move_num = self._moves_played // 2 + 1
        status_parts = [f"#{move_num}"]

        # Check / mate / draw status
        if self.board.is_checkmate():
            status_parts.append("CHECKMATE")
        elif self.board.is_stalemate():
            status_parts.append("STALEMATE")
        elif self.board.is_check():
            status_parts.append("CHECK")

        status_text = " ".join(status_parts)
        cv2.putText(img, status_text, (dot_cx + 10, banner_y + 15),
                    font, 0.38, (220, 200, 160), 1, cv2.LINE_AA)

        # Material balance on the right side of the banner
        mat_parts = []
        for pt in self._PIECE_ORDER:
            w = sum(1 for s in range(64)
                    if self.board.piece_at(s) is not None
                    and self.board.piece_at(s).color == chess.WHITE
                    and self.board.piece_at(s).piece_type == pt)
            b = sum(1 for s in range(64)
                    if self.board.piece_at(s) is not None
                    and self.board.piece_at(s).color == chess.BLACK
                    and self.board.piece_at(s).piece_type == pt)
            label = self._PIECE_LABELS[pt]
            if w > 0 or b > 0:
                mat_parts.append(f"{label}{w}/{b}")
        mat_text = " ".join(mat_parts)
        tw = cv2.getTextSize(mat_text, font, 0.32, 1)[0][0]
        cv2.putText(img, mat_text, (banner_x2 - tw - 4, banner_y + 15),
                    font, 0.32, (190, 175, 150), 1, cv2.LINE_AA)

        # Resize to exact DISPLAY_SIZE if needed (should already be 480)
        h, w = img.shape[:2]
        if h != size or w != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

        return img

    def _draw_piece(self, img, piece, cx, cy, cell):
        """Draw a single chess piece with a drop shadow underneath."""
        is_white = piece.color == chess.WHITE
        if is_white:
            fill = (240, 235, 220)
            dark = (180, 170, 150)
            outline = (60, 50, 40)
            highlight = (255, 252, 245)
        else:
            fill = (50, 45, 40)
            dark = (30, 28, 26)
            outline = (100, 90, 80)
            highlight = (80, 75, 70)

        r = cell // 2 - 4
        pt = piece.piece_type

        # Shadow color (very dark, semi-visible)
        shadow = (20, 18, 15)
        shadow_dx, shadow_dy = 2, 2

        # Draw shadow first (same shape offset by 2px)
        if pt == chess.PAWN:
            self._draw_pawn(img, cx + shadow_dx, cy + shadow_dy, r,
                            shadow, shadow, shadow, shadow)
            self._draw_pawn(img, cx, cy, r, fill, dark, outline, highlight)
        elif pt == chess.ROOK:
            self._draw_rook(img, cx + shadow_dx, cy + shadow_dy, r,
                            shadow, shadow, shadow, shadow)
            self._draw_rook(img, cx, cy, r, fill, dark, outline, highlight)
        elif pt == chess.KNIGHT:
            self._draw_knight(img, cx + shadow_dx, cy + shadow_dy, r,
                              shadow, shadow, shadow, shadow)
            self._draw_knight(img, cx, cy, r, fill, dark, outline, highlight)
        elif pt == chess.BISHOP:
            self._draw_bishop(img, cx + shadow_dx, cy + shadow_dy, r,
                              shadow, shadow, shadow, shadow)
            self._draw_bishop(img, cx, cy, r, fill, dark, outline, highlight)
        elif pt == chess.QUEEN:
            self._draw_queen(img, cx + shadow_dx, cy + shadow_dy, r,
                             shadow, shadow, shadow, shadow)
            self._draw_queen(img, cx, cy, r, fill, dark, outline, highlight)
        elif pt == chess.KING:
            self._draw_king(img, cx + shadow_dx, cy + shadow_dy, r,
                            shadow, shadow, shadow, shadow)
            self._draw_king(img, cx, cy, r, fill, dark, outline, highlight)

    def _draw_base(self, img, cx, cy, r, fill, dark, outline):
        """Draw the common base/pedestal of a chess piece."""
        cv2.ellipse(img, (cx + 1, cy + r - 1), (r - 2, r // 4),
                    0, 0, 360, dark, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy + r - 2), (r - 2, r // 4),
                    0, 0, 360, fill, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy + r - 2), (r - 2, r // 4),
                    0, 0, 360, outline, 1, cv2.LINE_AA)

    def _draw_pawn(self, img, cx, cy, r, fill, dark, outline, highlight):
        """Pawn: small circle on a tapered stem with base."""
        self._draw_base(img, cx, cy, r, fill, dark, outline)
        hw = r // 3
        pts = np.array([
            [cx - hw, cy + r // 3],
            [cx + hw, cy + r // 3],
            [cx + hw - 2, cy - r // 4],
            [cx - hw + 2, cy - r // 4],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, fill, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, outline, 1, cv2.LINE_AA)
        head_r = r // 3 + 1
        cv2.circle(img, (cx, cy - r // 3), head_r, fill, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy - r // 3), head_r, outline, 1, cv2.LINE_AA)
        cv2.circle(img, (cx - 1, cy - r // 3 - 2), max(2, head_r // 2),
                   highlight, -1, cv2.LINE_AA)

    def _draw_rook(self, img, cx, cy, r, fill, dark, outline, highlight):
        """Rook: rectangular tower with battlements."""
        self._draw_base(img, cx, cy, r, fill, dark, outline)
        hw = r * 2 // 5
        top = cy - r // 2
        bot = cy + r // 3
        cv2.rectangle(img, (cx - hw, top), (cx + hw, bot), fill, -1)
        cv2.rectangle(img, (cx - hw, top), (cx + hw, bot), outline, 1)
        bt = top - r // 4
        notch = hw // 3
        for offset in [-hw, -notch, notch]:
            cv2.rectangle(img, (cx + offset, bt),
                          (cx + offset + notch, top), fill, -1)
            cv2.rectangle(img, (cx + offset, bt),
                          (cx + offset + notch, top), outline, 1)
        cv2.line(img, (cx - hw + 2, top + 2), (cx - hw + 2, bot - 2),
                 highlight, 1, cv2.LINE_AA)

    def _draw_knight(self, img, cx, cy, r, fill, dark, outline, highlight):
        """Knight: horse head shape (stylized L-profile)."""
        self._draw_base(img, cx, cy, r, fill, dark, outline)
        pts = np.array([
            [cx - r // 3, cy + r // 3],
            [cx - r // 3, cy - r // 4],
            [cx - r // 5, cy - r // 2],
            [cx + r // 8, cy - r * 2 // 3],
            [cx + r // 3, cy - r // 3],
            [cx + r // 3, cy - r // 6],
            [cx + r // 5, cy],
            [cx + r // 4, cy + r // 3],
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], fill, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, outline, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy - r // 4), max(1, r // 8),
                   outline, -1, cv2.LINE_AA)
        cv2.line(img, (cx - r // 4, cy - r // 5),
                 (cx - r // 6, cy - r // 2 + 2), highlight, 1, cv2.LINE_AA)

    def _draw_bishop(self, img, cx, cy, r, fill, dark, outline, highlight):
        """Bishop: tall pointed hat on a stem."""
        self._draw_base(img, cx, cy, r, fill, dark, outline)
        hw = r // 3
        pts = np.array([
            [cx - hw, cy + r // 4],
            [cx + hw, cy + r // 4],
            [cx + r // 6, cy - r // 3],
            [cx, cy - r * 2 // 3],
            [cx - r // 6, cy - r // 3],
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], fill, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, outline, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy - r * 2 // 3 - 2), max(2, r // 6),
                   fill, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy - r * 2 // 3 - 2), max(2, r // 6),
                   outline, 1, cv2.LINE_AA)
        cv2.line(img, (cx - r // 5, cy - r // 6),
                 (cx + r // 5, cy - r // 3), outline, 1, cv2.LINE_AA)
        cv2.line(img, (cx - r // 5, cy + r // 6),
                 (cx - r // 6, cy - r // 3), highlight, 1, cv2.LINE_AA)

    def _draw_queen(self, img, cx, cy, r, fill, dark, outline, highlight):
        """Queen: crown with five points atop a body."""
        self._draw_base(img, cx, cy, r, fill, dark, outline)
        hw = r * 2 // 5
        pts = np.array([
            [cx - hw, cy + r // 4],
            [cx + hw, cy + r // 4],
            [cx + r // 5, cy - r // 6],
            [cx - r // 5, cy - r // 6],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, fill, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, outline, 1, cv2.LINE_AA)
        crown_base = cy - r // 6
        crown_top = cy - r * 3 // 4
        mid_y = (crown_base + crown_top) // 2
        points = []
        for i in range(5):
            px = cx + int((i - 2) * r * 2 / 5 / 2)
            points.append([px, crown_top])
            if i < 4:
                mx = (px + cx + int((i - 1) * r * 2 / 5 / 2)) // 2
                points.append([mx, mid_y])
        points.append([cx + int(2 * r * 2 / 5 / 2), crown_top])
        points.append([cx + r // 5, crown_base])
        points.append([cx - r // 5, crown_base])
        pts_crown = np.array(points, dtype=np.int32)
        cv2.fillPoly(img, [pts_crown], fill, cv2.LINE_AA)
        cv2.polylines(img, [pts_crown], True, outline, 1, cv2.LINE_AA)
        for i in range(5):
            px = cx + int((i - 2) * r * 2 / 5 / 2)
            cv2.circle(img, (px, crown_top + 1), max(1, r // 8),
                       outline, -1, cv2.LINE_AA)
        cv2.line(img, (cx - hw + 2, cy + r // 6),
                 (cx - r // 5, cy - r // 6), highlight, 1, cv2.LINE_AA)

    def _draw_king(self, img, cx, cy, r, fill, dark, outline, highlight):
        """King: body with a cross on top."""
        self._draw_base(img, cx, cy, r, fill, dark, outline)
        hw = r * 2 // 5
        pts = np.array([
            [cx - hw, cy + r // 4],
            [cx + hw, cy + r // 4],
            [cx + r // 5, cy - r // 4],
            [cx - r // 5, cy - r // 4],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, fill, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, outline, 1, cv2.LINE_AA)
        cv2.rectangle(img, (cx - hw, cy - r // 8),
                      (cx + hw, cy + r // 8), fill, -1)
        cv2.rectangle(img, (cx - hw, cy - r // 8),
                      (cx + hw, cy + r // 8), outline, 1)
        cross_top = cy - r * 3 // 4
        cross_base = cy - r // 4
        cross_hw = r // 6
        cv2.rectangle(img, (cx - cross_hw // 2, cross_top),
                      (cx + cross_hw // 2, cross_base), fill, -1)
        cv2.rectangle(img, (cx - cross_hw // 2, cross_top),
                      (cx + cross_hw // 2, cross_base), outline, 1)
        cross_mid = cross_top + (cross_base - cross_top) // 3
        cv2.rectangle(img, (cx - cross_hw, cross_mid - cross_hw // 2),
                      (cx + cross_hw, cross_mid + cross_hw // 2), fill, -1)
        cv2.rectangle(img, (cx - cross_hw, cross_mid - cross_hw // 2),
                      (cx + cross_hw, cross_mid + cross_hw // 2), outline, 1)
        cv2.line(img, (cx - hw + 2, cy + r // 6),
                 (cx - r // 5, cy - r // 4), highlight, 1, cv2.LINE_AA)

    def render(self, mode='rgb_array'):
        return self._render_rgb()

    def get_legal_action_mask(self) -> np.ndarray:
        """Return a binary mask over the 4096 action space.

        1 = legal move, 0 = illegal. Useful for action masking in policies.
        """
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for move in self.board.legal_moves:
            action = move.from_square * NUM_SQUARES + move.to_square
            mask[action] = 1.0
        return mask

    def _info(self) -> dict:
        white_pieces = sum(
            1 for sq in range(64)
            if self.board.piece_at(sq) is not None
            and self.board.piece_at(sq).color == chess.WHITE
        )
        black_pieces = sum(
            1 for sq in range(64)
            if self.board.piece_at(sq) is not None
            and self.board.piece_at(sq).color == chess.BLACK
        )
        return {
            'winner': self._winner,
            'moves_played': self._moves_played,
            'p1_pieces': white_pieces,
            'p2_pieces': black_pieces,
            'in_check': self.board.is_check(),
            'action_mask': self.get_legal_action_mask(),
        }

    def close(self):
        pass
