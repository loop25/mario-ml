"""Minimax opponent with alpha-beta pruning for board games."""

import copy

import numpy as np

from .base_opponent import BaseOpponent

# Depth caps per game to keep search tractable.
_DEPTH_CAPS = {
    "checkers": 3,
    "chess": 3,
}

# Piece material values for chess evaluation.
_CHESS_PIECE_VALUES = {
    1: 1,   # pawn
    2: 3,   # knight
    3: 3,   # bishop
    4: 5,   # rook
    5: 9,   # queen
    6: 0,   # king (not counted as material)
}


class MinimaxOpponent(BaseOpponent):
    """Opponent that picks moves using minimax with alpha-beta pruning.

    Parameters
    ----------
    depth : int
        Search depth (may be capped per-game for performance).
    game_id : str
        One of ``'tictactoe'``, ``'connect4'``, ``'checkers'``, ``'chess'``.
    """

    def __init__(self, depth: int = 3, game_id: str = "tictactoe"):
        self.depth = depth
        self.game_id = game_id

    # ------------------------------------------------------------------
    # BaseOpponent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"Minimax (depth={self.depth})"

    @property
    def difficulty_tier(self) -> int:
        if self.depth <= 1:
            return 1
        if self.depth <= 3:
            return 2
        return 3

    def pick_action(self, board_state: dict) -> int:
        valid = board_state["valid_actions"]
        if not valid:
            return 0

        board = board_state["board"]
        game_id = board_state.get("game_id", self.game_id)
        effective_depth = min(self.depth, _DEPTH_CAPS.get(game_id, self.depth))

        best_action = valid[0]
        best_score = float("-inf")

        for action in valid:
            new_board = self._simulate(board, action, player=2, game_id=game_id)
            score = self._minimax(
                new_board, effective_depth - 1, False,
                float("-inf"), float("inf"), game_id,
            )
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # ------------------------------------------------------------------
    # Core minimax with alpha-beta pruning
    # ------------------------------------------------------------------

    def _minimax(self, board, depth, maximizing, alpha, beta, game_id):
        terminal, winner = self._check_terminal(board, game_id)
        if terminal:
            return self._terminal_score(winner)
        if depth == 0:
            return self._evaluate(board, game_id)

        moves = self._get_valid_moves(board, game_id,
                                      player=2 if maximizing else 1)
        if not moves:
            return self._evaluate(board, game_id)

        if maximizing:
            value = float("-inf")
            for move in moves:
                new_board = self._simulate(board, move, player=2,
                                           game_id=game_id)
                value = max(value, self._minimax(
                    new_board, depth - 1, False, alpha, beta, game_id))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float("inf")
            for move in moves:
                new_board = self._simulate(board, move, player=1,
                                           game_id=game_id)
                value = min(value, self._minimax(
                    new_board, depth - 1, True, alpha, beta, game_id))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    @staticmethod
    def _terminal_score(winner):
        """Return score from the perspective of player 2 (the opponent)."""
        if winner == 2:
            return 100
        if winner == 1:
            return -100
        return 0  # draw

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def _simulate(self, board, action, player, game_id):
        dispatch = {
            "tictactoe": self._simulate_ttt,
            "connect4": self._simulate_c4,
            "checkers": self._simulate_checkers,
            "chess": self._simulate_chess,
        }
        return dispatch[game_id](board, action, player)

    def _get_valid_moves(self, board, game_id, player):
        dispatch = {
            "tictactoe": self._valid_ttt,
            "connect4": self._valid_c4,
            "checkers": self._valid_checkers,
            "chess": self._valid_chess,
        }
        return dispatch[game_id](board, player)

    def _check_terminal(self, board, game_id):
        dispatch = {
            "tictactoe": self._terminal_ttt,
            "connect4": self._terminal_c4,
            "checkers": self._terminal_checkers,
            "chess": self._terminal_chess,
        }
        return dispatch[game_id](board)

    def _evaluate(self, board, game_id):
        dispatch = {
            "tictactoe": self._eval_ttt,
            "connect4": self._eval_c4,
            "checkers": self._eval_checkers,
            "chess": self._eval_chess,
        }
        return dispatch[game_id](board)

    # ==================================================================
    # TicTacToe  (3x3, values 0/1/2)
    # ==================================================================

    @staticmethod
    def _simulate_ttt(board, action, player):
        b = board.copy()
        r, c = divmod(action, 3)
        b[r, c] = player
        return b

    @staticmethod
    def _valid_ttt(board, _player):
        return [r * 3 + c
                for r in range(3) for c in range(3) if board[r, c] == 0]

    @staticmethod
    def _check_winner_ttt(board):
        """Return winner (1 or 2) or 0 if none."""
        for p in (1, 2):
            # rows and columns
            for i in range(3):
                if np.all(board[i, :] == p) or np.all(board[:, i] == p):
                    return p
            # diagonals
            if board[0, 0] == board[1, 1] == board[2, 2] == p:
                return p
            if board[0, 2] == board[1, 1] == board[2, 0] == p:
                return p
        return 0

    def _terminal_ttt(self, board):
        w = self._check_winner_ttt(board)
        if w:
            return True, w
        if not np.any(board == 0):
            return True, 0  # draw
        return False, 0

    def _eval_ttt(self, board):
        # Simple: no heuristic beyond terminal — TicTacToe is small enough.
        w = self._check_winner_ttt(board)
        if w == 2:
            return 100
        if w == 1:
            return -100
        return 0

    # ==================================================================
    # Connect4  (6x7, values 0/1/2)
    # ==================================================================

    @staticmethod
    def _simulate_c4(board, action, player):
        b = board.copy()
        col = action
        for r in range(5, -1, -1):
            if b[r, col] == 0:
                b[r, col] = player
                return b
        return b  # column full — shouldn't happen if valid_actions correct

    @staticmethod
    def _valid_c4(board, _player):
        return [c for c in range(7) if board[0, c] == 0]

    @staticmethod
    def _check_winner_c4(board):
        """Return winner (1 or 2) or 0."""
        rows, cols = board.shape
        # horizontal
        for r in range(rows):
            for c in range(cols - 3):
                window = board[r, c:c + 4]
                for p in (1, 2):
                    if np.all(window == p):
                        return p
        # vertical
        for r in range(rows - 3):
            for c in range(cols):
                window = board[r:r + 4, c]
                for p in (1, 2):
                    if np.all(window == p):
                        return p
        # diagonal ↘
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = np.array([board[r + i, c + i] for i in range(4)])
                for p in (1, 2):
                    if np.all(window == p):
                        return p
        # diagonal ↙
        for r in range(3, rows):
            for c in range(cols - 3):
                window = np.array([board[r - i, c + i] for i in range(4)])
                for p in (1, 2):
                    if np.all(window == p):
                        return p
        return 0

    def _terminal_c4(self, board):
        w = self._check_winner_c4(board)
        if w:
            return True, w
        if not np.any(board == 0):
            return True, 0
        return False, 0

    def _eval_c4(self, board):
        w = self._check_winner_c4(board)
        if w == 2:
            return 100
        if w == 1:
            return -100

        score = 0
        rows, cols = board.shape

        # Center column bonus
        center = board[:, cols // 2]
        score += np.sum(center == 2) * 3
        score -= np.sum(center == 1) * 3

        # Count windows of 4
        def _score_window(window):
            s = 0
            p2 = np.sum(window == 2)
            p1 = np.sum(window == 1)
            empty = np.sum(window == 0)
            if p2 == 3 and empty == 1:
                s += 5
            elif p2 == 2 and empty == 2:
                s += 2
            if p1 == 3 and empty == 1:
                s -= 5
            elif p1 == 2 and empty == 2:
                s -= 2
            return s

        for r in range(rows):
            for c in range(cols - 3):
                score += _score_window(board[r, c:c + 4])
        for r in range(rows - 3):
            for c in range(cols):
                score += _score_window(board[r:r + 4, c])
        for r in range(rows - 3):
            for c in range(cols - 3):
                score += _score_window(
                    np.array([board[r + i, c + i] for i in range(4)]))
        for r in range(3, rows):
            for c in range(cols - 3):
                score += _score_window(
                    np.array([board[r - i, c + i] for i in range(4)]))
        return score

    # ==================================================================
    # Checkers  (simple material evaluation)
    # ==================================================================

    @staticmethod
    def _simulate_checkers(board, action, player):
        b = board.copy()
        from_pos = action // 64
        to_pos = action % 64
        fr, fc = divmod(from_pos, 8)
        tr, tc = divmod(to_pos, 8)
        b[tr, tc] = b[fr, fc]
        b[fr, fc] = 0
        # Handle captures (jumped square)
        if abs(fr - tr) == 2:
            mr, mc = (fr + tr) // 2, (fc + tc) // 2
            b[mr, mc] = 0
        return b

    @staticmethod
    def _valid_checkers(board, player):
        """Simplified: find all pieces for player, generate diagonal moves."""
        moves = []
        rows, cols = board.shape
        # Player 1 pieces are positive odd, player 2 positive even (simplified).
        # Use convention: player 1 = values 1,3  player 2 = values 2,4
        for r in range(rows):
            for c in range(cols):
                val = board[r, c]
                if val == 0:
                    continue
                owner = 1 if val in (1, 3) else (2 if val in (2, 4) else 0)
                if owner != player:
                    continue
                # Simple moves
                directions = []
                if val in (1, 3, 4):  # player 1 moves down, kings both
                    directions.append((1, -1))
                    directions.append((1, 1))
                if val in (2, 3, 4):  # player 2 moves up, kings both
                    directions.append((-1, -1))
                    directions.append((-1, 1))
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and board[nr, nc] == 0:
                        moves.append((r * 8 + c) * 64 + (nr * 8 + nc))
                    # Jump
                    jr, jc = r + 2 * dr, c + 2 * dc
                    if (0 <= jr < rows and 0 <= jc < cols
                            and board[nr, nc] != 0
                            and board[jr, jc] == 0):
                        mid_owner = (1 if board[nr, nc] in (1, 3)
                                     else 2 if board[nr, nc] in (2, 4) else 0)
                        if mid_owner != 0 and mid_owner != player:
                            moves.append((r * 8 + c) * 64 + (jr * 8 + jc))
        return moves

    def _terminal_checkers(self, board):
        p1 = np.sum(np.isin(board, [1, 3]))
        p2 = np.sum(np.isin(board, [2, 4]))
        if p1 == 0:
            return True, 2
        if p2 == 0:
            return True, 1
        return False, 0

    @staticmethod
    def _eval_checkers(board):
        p1 = int(np.sum(board == 1)) + 2 * int(np.sum(board == 3))
        p2 = int(np.sum(board == 2)) + 2 * int(np.sum(board == 4))
        return (p2 - p1) * 10

    # ==================================================================
    # Chess  (python-chess Board)
    # ==================================================================

    @staticmethod
    def _simulate_chess(board, action, _player):
        import chess
        b = board.copy()
        moves = list(b.legal_moves)
        if action < len(moves):
            b.push(moves[action])
        return b

    @staticmethod
    def _valid_chess(board, _player):
        return list(range(len(list(board.legal_moves))))

    @staticmethod
    def _terminal_chess(board):
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return True, 1
            if result == "0-1":
                return True, 2
            return True, 0
        return False, 0

    @staticmethod
    def _eval_chess(board):
        import chess
        score = 0
        for piece_type in range(1, 7):
            val = _CHESS_PIECE_VALUES.get(piece_type, 0)
            score -= len(board.pieces(piece_type, chess.WHITE)) * val
            score += len(board.pieces(piece_type, chess.BLACK)) * val
        # Mobility bonus
        mob = board.legal_moves.count()
        if board.turn == chess.BLACK:
            score += mob * 0.1
        else:
            score -= mob * 0.1
        return score
