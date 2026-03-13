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

Promotions default to queen. The agent cannot choose underpromotion.

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
    """Chess as a Gym environment with a random opponent."""

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, render_size: int = 84, max_moves: int = 200):
        super().__init__()
        if not HAS_CHESS:
            raise ImportError(
                "python-chess is required for the Chess game. "
                "Install it with: pip install python-chess"
            )
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

    def reset(self):
        self.board = chess.Board()
        self._winner = 0
        self._moves_played = 0
        return self._render_obs()

    def step(self, action):
        from_sq = action // NUM_SQUARES
        to_sq = action % NUM_SQUARES

        # Build the move — try promotion to queen if it's a pawn reaching
        # the last rank
        move = chess.Move(from_sq, to_sq)
        piece = self.board.piece_at(from_sq)
        if (piece is not None and piece.piece_type == chess.PAWN
                and chess.square_rank(to_sq) in (0, 7)):
            move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

        # Validate move
        if move not in self.board.legal_moves:
            self._winner = 2
            return self._render_obs(), -1.0, True, self._info()

        # Execute agent move (White)
        self.board.push(move)
        self._moves_played += 1

        # Check terminal after agent move
        done, reward = self._check_terminal()
        if done:
            return self._render_obs(), reward, True, self._info()

        # Max move limit
        if self._moves_played >= self.max_moves:
            return self._render_obs(), 0.0, True, self._info()

        # Opponent move (Black) — random legal move
        opp_moves = list(self.board.legal_moves)
        if not opp_moves:
            # Shouldn't happen (terminal check above), but safety
            self._winner = 1
            return self._render_obs(), 1.0, True, self._info()

        opp_move = random.choice(opp_moves)
        self.board.push(opp_move)
        self._moves_played += 1

        # Check terminal after opponent move
        done, reward = self._check_terminal()
        if done:
            return self._render_obs(), reward, True, self._info()

        if self._moves_played >= self.max_moves:
            return self._render_obs(), 0.0, True, self._info()

        return self._render_obs(), 0.0, False, self._info()

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

        if self.board.can_claim_threefold_repetition():
            return True, 0.0

        if self.board.can_claim_fifty_moves():
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

    def _render_rgb(self) -> np.ndarray:
        """Render a polished chess board for dashboard display.

        Renders at 480px. Rich wood-toned board with geometric piece
        shapes drawn using cv2 primitives — no font/text rendering.
        White pieces are cream with dark outlines, black pieces are
        dark charcoal with light outlines, both with 3D highlight effects.
        """
        size = self.DISPLAY_SIZE
        cell = size // 8  # 60px per cell
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Light/dark square colors (rich wood tones)
        LIGHT_SQ = (230, 210, 175)  # Warm cream
        DARK_SQ = (160, 110, 65)    # Deep wood brown

        # Draw board squares
        for row in range(8):
            for col in range(8):
                x1, y1 = col * cell, row * cell
                x2, y2 = x1 + cell, y1 + cell
                if (row + col) % 2 == 0:
                    img[y1:y2, x1:x2] = LIGHT_SQ
                else:
                    img[y1:y2, x1:x2] = DARK_SQ

        # Draw each piece as a geometric shape
        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece is None:
                continue
            row = 7 - chess.square_rank(sq)
            col = chess.square_file(sq)
            cx = col * cell + cell // 2
            cy = row * cell + cell // 2
            self._draw_piece(img, piece, cx, cy, cell)

        return img

    def _draw_piece(self, img, piece, cx, cy, cell):
        """Draw a single chess piece using geometric shapes."""
        is_white = piece.color == chess.WHITE
        # Color palette
        if is_white:
            fill = (235, 230, 215)   # Cream white
            dark = (180, 170, 150)   # Shadow
            outline = (80, 70, 55)   # Dark outline
            highlight = (255, 252, 245)
        else:
            fill = (55, 50, 48)      # Dark charcoal
            dark = (30, 28, 26)      # Shadow
            outline = (160, 155, 145)  # Light outline
            highlight = (90, 85, 80)

        r = cell // 2 - 4  # Piece radius fits within cell
        base_y = cy + r - 2  # Base of the piece
        top_offset = r  # How far up the piece extends

        pt = piece.piece_type

        if pt == chess.PAWN:
            self._draw_pawn(img, cx, cy, r, fill, dark, outline, highlight)
        elif pt == chess.ROOK:
            self._draw_rook(img, cx, cy, r, fill, dark, outline, highlight)
        elif pt == chess.KNIGHT:
            self._draw_knight(img, cx, cy, r, fill, dark, outline, highlight)
        elif pt == chess.BISHOP:
            self._draw_bishop(img, cx, cy, r, fill, dark, outline, highlight)
        elif pt == chess.QUEEN:
            self._draw_queen(img, cx, cy, r, fill, dark, outline, highlight)
        elif pt == chess.KING:
            self._draw_king(img, cx, cy, r, fill, dark, outline, highlight)

    def _draw_base(self, img, cx, cy, r, fill, dark, outline):
        """Draw the common base/pedestal of a chess piece."""
        # Shadow ellipse
        cv2.ellipse(img, (cx + 1, cy + r - 1), (r - 2, r // 4),
                    0, 0, 360, dark, -1, cv2.LINE_AA)
        # Base ellipse
        cv2.ellipse(img, (cx, cy + r - 2), (r - 2, r // 4),
                    0, 0, 360, fill, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy + r - 2), (r - 2, r // 4),
                    0, 0, 360, outline, 1, cv2.LINE_AA)

    def _draw_pawn(self, img, cx, cy, r, fill, dark, outline, highlight):
        """Pawn: small circle on a tapered stem with base."""
        self._draw_base(img, cx, cy, r, fill, dark, outline)
        # Stem
        hw = r // 3
        pts = np.array([
            [cx - hw, cy + r // 3],
            [cx + hw, cy + r // 3],
            [cx + hw - 2, cy - r // 4],
            [cx - hw + 2, cy - r // 4],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, fill, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, outline, 1, cv2.LINE_AA)
        # Head circle
        head_r = r // 3 + 1
        cv2.circle(img, (cx, cy - r // 3), head_r, fill, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy - r // 3), head_r, outline, 1, cv2.LINE_AA)
        # Highlight
        cv2.circle(img, (cx - 1, cy - r // 3 - 2), max(2, head_r // 2),
                   highlight, -1, cv2.LINE_AA)

    def _draw_rook(self, img, cx, cy, r, fill, dark, outline, highlight):
        """Rook: rectangular tower with battlements."""
        self._draw_base(img, cx, cy, r, fill, dark, outline)
        # Tower body
        hw = r * 2 // 5
        top = cy - r // 2
        bot = cy + r // 3
        cv2.rectangle(img, (cx - hw, top), (cx + hw, bot), fill, -1)
        cv2.rectangle(img, (cx - hw, top), (cx + hw, bot), outline, 1)
        # Battlements (three notches)
        bw = hw * 2 // 3
        bt = top - r // 4
        notch = hw // 3
        for offset in [-hw, -notch, notch]:
            cv2.rectangle(img, (cx + offset, bt),
                          (cx + offset + notch, top), fill, -1)
            cv2.rectangle(img, (cx + offset, bt),
                          (cx + offset + notch, top), outline, 1)
        # Highlight
        cv2.line(img, (cx - hw + 2, top + 2), (cx - hw + 2, bot - 2),
                 highlight, 1, cv2.LINE_AA)

    def _draw_knight(self, img, cx, cy, r, fill, dark, outline, highlight):
        """Knight: horse head shape (stylized L-profile)."""
        self._draw_base(img, cx, cy, r, fill, dark, outline)
        # Horse head profile (simplified)
        pts = np.array([
            [cx - r // 3, cy + r // 3],       # Bottom left
            [cx - r // 3, cy - r // 4],        # Mid left
            [cx - r // 5, cy - r // 2],        # Ear left
            [cx + r // 8, cy - r * 2 // 3],    # Top (mane)
            [cx + r // 3, cy - r // 3],        # Nose tip
            [cx + r // 3, cy - r // 6],        # Jaw
            [cx + r // 5, cy],                 # Chin
            [cx + r // 4, cy + r // 3],        # Bottom right
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], fill, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, outline, 1, cv2.LINE_AA)
        # Eye
        cv2.circle(img, (cx, cy - r // 4), max(1, r // 8),
                   outline, -1, cv2.LINE_AA)
        # Highlight
        cv2.line(img, (cx - r // 4, cy - r // 5),
                 (cx - r // 6, cy - r // 2 + 2), highlight, 1, cv2.LINE_AA)

    def _draw_bishop(self, img, cx, cy, r, fill, dark, outline, highlight):
        """Bishop: tall pointed hat on a stem."""
        self._draw_base(img, cx, cy, r, fill, dark, outline)
        # Body (tapered)
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
        # Top ball
        cv2.circle(img, (cx, cy - r * 2 // 3 - 2), max(2, r // 6),
                   fill, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy - r * 2 // 3 - 2), max(2, r // 6),
                   outline, 1, cv2.LINE_AA)
        # Diagonal slash (bishop's notch)
        cv2.line(img, (cx - r // 5, cy - r // 6),
                 (cx + r // 5, cy - r // 3), outline, 1, cv2.LINE_AA)
        # Highlight
        cv2.line(img, (cx - r // 5, cy + r // 6),
                 (cx - r // 6, cy - r // 3), highlight, 1, cv2.LINE_AA)

    def _draw_queen(self, img, cx, cy, r, fill, dark, outline, highlight):
        """Queen: crown with five points atop a body."""
        self._draw_base(img, cx, cy, r, fill, dark, outline)
        # Body
        hw = r * 2 // 5
        pts = np.array([
            [cx - hw, cy + r // 4],
            [cx + hw, cy + r // 4],
            [cx + r // 5, cy - r // 6],
            [cx - r // 5, cy - r // 6],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, fill, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, outline, 1, cv2.LINE_AA)
        # Crown points (5 points)
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
        # Close crown
        points.append([cx + int(2 * r * 2 / 5 / 2), crown_top])
        points.append([cx + r // 5, crown_base])
        points.append([cx - r // 5, crown_base])
        pts_crown = np.array(points, dtype=np.int32)
        cv2.fillPoly(img, [pts_crown], fill, cv2.LINE_AA)
        cv2.polylines(img, [pts_crown], True, outline, 1, cv2.LINE_AA)
        # Crown jewel dots
        for i in range(5):
            px = cx + int((i - 2) * r * 2 / 5 / 2)
            cv2.circle(img, (px, crown_top + 1), max(1, r // 8),
                       outline, -1, cv2.LINE_AA)
        # Highlight
        cv2.line(img, (cx - hw + 2, cy + r // 6),
                 (cx - r // 5, cy - r // 6), highlight, 1, cv2.LINE_AA)

    def _draw_king(self, img, cx, cy, r, fill, dark, outline, highlight):
        """King: body with a cross on top."""
        self._draw_base(img, cx, cy, r, fill, dark, outline)
        # Body
        hw = r * 2 // 5
        pts = np.array([
            [cx - hw, cy + r // 4],
            [cx + hw, cy + r // 4],
            [cx + r // 5, cy - r // 4],
            [cx - r // 5, cy - r // 4],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, fill, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, outline, 1, cv2.LINE_AA)
        # Horizontal band
        cv2.rectangle(img, (cx - hw, cy - r // 8),
                      (cx + hw, cy + r // 8), fill, -1)
        cv2.rectangle(img, (cx - hw, cy - r // 8),
                      (cx + hw, cy + r // 8), outline, 1)
        # Cross on top
        cross_top = cy - r * 3 // 4
        cross_base = cy - r // 4
        cross_hw = r // 6
        # Vertical bar
        cv2.rectangle(img, (cx - cross_hw // 2, cross_top),
                      (cx + cross_hw // 2, cross_base), fill, -1)
        cv2.rectangle(img, (cx - cross_hw // 2, cross_top),
                      (cx + cross_hw // 2, cross_base), outline, 1)
        # Horizontal bar
        cross_mid = cross_top + (cross_base - cross_top) // 3
        cv2.rectangle(img, (cx - cross_hw, cross_mid - cross_hw // 2),
                      (cx + cross_hw, cross_mid + cross_hw // 2), fill, -1)
        cv2.rectangle(img, (cx - cross_hw, cross_mid - cross_hw // 2),
                      (cx + cross_hw, cross_mid + cross_hw // 2), outline, 1)
        # Highlight on body
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
