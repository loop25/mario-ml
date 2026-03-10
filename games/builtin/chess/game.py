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
import gym
from gym.spaces import Box, Discrete

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

    def _render_rgb(self) -> np.ndarray:
        """Render a colorful chess board for dashboard display."""
        size = self.render_size
        cell = size // 8
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Piece symbols (unicode)
        PIECE_CHARS = {
            (chess.PAWN, chess.WHITE): 'P',
            (chess.KNIGHT, chess.WHITE): 'N',
            (chess.BISHOP, chess.WHITE): 'B',
            (chess.ROOK, chess.WHITE): 'R',
            (chess.QUEEN, chess.WHITE): 'Q',
            (chess.KING, chess.WHITE): 'K',
            (chess.PAWN, chess.BLACK): 'p',
            (chess.KNIGHT, chess.BLACK): 'n',
            (chess.BISHOP, chess.BLACK): 'b',
            (chess.ROOK, chess.BLACK): 'r',
            (chess.QUEEN, chess.BLACK): 'q',
            (chess.KING, chess.BLACK): 'k',
        }

        PIECE_COLORS = {
            chess.WHITE: (240, 240, 240),  # White pieces
            chess.BLACK: (30, 30, 30),      # Black pieces
        }

        # Light/dark square colors
        LIGHT_SQ = (235, 210, 170)  # Warm cream
        DARK_SQ = (170, 120, 70)    # Wood brown

        # Draw board squares
        for row in range(8):
            for col in range(8):
                x1, y1 = col * cell, row * cell
                x2, y2 = x1 + cell, y1 + cell
                if (row + col) % 2 == 0:
                    img[y1:y2, x1:x2] = LIGHT_SQ
                else:
                    img[y1:y2, x1:x2] = DARK_SQ

        # Draw pieces
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.25, cell / 28.0)
        thickness = max(1, cell // 14)

        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece is None:
                continue
            row = 7 - chess.square_rank(sq)
            col = chess.square_file(sq)

            char = PIECE_CHARS.get((piece.piece_type, piece.color), '?')
            color = PIECE_COLORS[piece.color]

            # Center the character in the cell
            text_size = cv2.getTextSize(char, font, font_scale, thickness)[0]
            tx = col * cell + (cell - text_size[0]) // 2
            ty = row * cell + (cell + text_size[1]) // 2

            # Draw outline for contrast
            outline_color = (0, 0, 0) if piece.color == chess.WHITE else (200, 200, 200)
            cv2.putText(img, char, (tx, ty), font, font_scale,
                        outline_color, thickness + 1, cv2.LINE_AA)
            cv2.putText(img, char, (tx, ty), font, font_scale,
                        color, thickness, cv2.LINE_AA)

        return img

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
