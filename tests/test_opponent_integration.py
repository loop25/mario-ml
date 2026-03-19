"""Integration tests: pluggable opponent interface in all board games.

Verifies that each game works both with an explicit RandomOpponent and
with the default (opponent=None) backward-compatible path.
"""
import pytest

from src.opponents import RandomOpponent


# ── Tic-Tac-Toe ────────────────────────────────────────────────────

class TestTicTacToeOpponentIntegration:
    """Opponent integration for TicTacToeEnv."""

    def test_ttt_with_random_opponent(self):
        from games.builtin.tictactoe.game import TicTacToeEnv
        env = TicTacToeEnv(opponent=RandomOpponent())
        obs = env.reset()
        assert obs is not None
        done = False
        for _ in range(20):
            obs, reward, done, info = env.step(env.action_space.sample())
            if done:
                break
        env.close()

    def test_ttt_default_opponent(self):
        from games.builtin.tictactoe.game import TicTacToeEnv
        env = TicTacToeEnv()
        obs = env.reset()
        assert obs is not None
        done = False
        for _ in range(20):
            obs, reward, done, info = env.step(env.action_space.sample())
            if done:
                break
        env.close()

    def test_ttt_opponent_stored(self):
        from games.builtin.tictactoe.game import TicTacToeEnv
        opp = RandomOpponent()
        env = TicTacToeEnv(opponent=opp)
        assert env.opponent is opp


# ── Connect Four ────────────────────────────────────────────────────

class TestConnectFourOpponentIntegration:
    """Opponent integration for ConnectFourEnv."""

    def test_c4_with_random_opponent(self):
        from games.builtin.connect4.game import ConnectFourEnv
        env = ConnectFourEnv(opponent=RandomOpponent())
        obs = env.reset()
        assert obs is not None
        done = False
        for _ in range(50):
            obs, reward, done, info = env.step(env.action_space.sample())
            if done:
                break
        env.close()

    def test_c4_default_opponent(self):
        from games.builtin.connect4.game import ConnectFourEnv
        env = ConnectFourEnv()
        obs = env.reset()
        assert obs is not None
        done = False
        for _ in range(50):
            obs, reward, done, info = env.step(env.action_space.sample())
            if done:
                break
        env.close()


# ── Checkers ────────────────────────────────────────────────────────

class TestCheckersOpponentIntegration:
    """Opponent integration for CheckersEnv."""

    def test_checkers_with_random_opponent(self):
        from games.builtin.checkers.game import CheckersEnv
        env = CheckersEnv(opponent=RandomOpponent())
        obs = env.reset()
        assert obs is not None
        done = False
        for _ in range(300):
            obs, reward, done, info = env.step(env.action_space.sample())
            if done:
                break
        env.close()

    def test_checkers_default_opponent(self):
        from games.builtin.checkers.game import CheckersEnv
        env = CheckersEnv()
        obs = env.reset()
        assert obs is not None
        done = False
        for _ in range(300):
            obs, reward, done, info = env.step(env.action_space.sample())
            if done:
                break
        env.close()


# ── Chess ───────────────────────────────────────────────────────────

class TestChessOpponentIntegration:
    """Opponent integration for ChessEnv."""

    def test_chess_with_random_opponent(self):
        from games.builtin.chess.game import ChessEnv
        env = ChessEnv(opponent=RandomOpponent())
        obs = env.reset()
        assert obs is not None
        done = False
        for _ in range(300):
            obs, reward, done, info = env.step(env.action_space.sample())
            if done:
                break
        env.close()

    def test_chess_default_opponent(self):
        from games.builtin.chess.game import ChessEnv
        env = ChessEnv()
        obs = env.reset()
        assert obs is not None
        done = False
        for _ in range(300):
            obs, reward, done, info = env.step(env.action_space.sample())
            if done:
                break
        env.close()


# ── Adapter forwarding ──────────────────────────────────────────────

class TestAdapterOpponentForwarding:
    """Verify adapters forward the opponent kwarg correctly."""

    def test_ttt_adapter_forwards_opponent(self):
        from games.builtin.tictactoe.adapter import TicTacToeAdapter
        opp = RandomOpponent()
        env = TicTacToeAdapter().create_env(opponent=opp)
        assert env.opponent is opp
        env.close()

    def test_c4_adapter_forwards_opponent(self):
        from games.builtin.connect4.adapter import ConnectFourAdapter
        opp = RandomOpponent()
        env = ConnectFourAdapter().create_env(opponent=opp)
        assert env.opponent is opp
        env.close()

    def test_checkers_adapter_forwards_opponent(self):
        from games.builtin.checkers.adapter import CheckersAdapter
        opp = RandomOpponent()
        env = CheckersAdapter().create_env(opponent=opp)
        assert env.opponent is opp
        env.close()

    def test_chess_adapter_forwards_opponent(self):
        from games.builtin.chess.adapter import ChessAdapter
        opp = RandomOpponent()
        env = ChessAdapter().create_env(opponent=opp)
        assert env.opponent is opp
        env.close()
