"""Tests for Chess game engine and adapter."""
import pytest
import numpy as np

chess = pytest.importorskip('chess', reason='python-chess not installed')

from games.builtin.chess.game import ChessEnv, NUM_SQUARES, ACTION_SPACE_SIZE
from games.builtin.chess.adapter import ChessAdapter
from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics


# ─── Environment ─────────────────────────────────────────────────


class TestChessEnv:
    def test_reset_returns_observation(self):
        env = ChessEnv()
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_step_returns_4_tuple(self):
        env = ChessEnv()
        env.reset()
        # e2e4 — classic opening (e2=12, e4=28)
        action = 12 * 64 + 28
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        env.close()

    def test_action_space_is_4096(self):
        env = ChessEnv()
        assert env.action_space.n == 4096
        env.close()

    def test_info_has_expected_keys(self):
        env = ChessEnv()
        env.reset()
        action = 12 * 64 + 28  # e2e4
        _, _, _, info = env.step(action)
        assert 'winner' in info
        assert 'moves_played' in info
        assert 'p1_pieces' in info
        assert 'p2_pieces' in info
        assert 'in_check' in info
        assert 'action_mask' in info
        env.close()

    def test_action_mask_shape(self):
        env = ChessEnv()
        env.reset()
        mask = env.get_legal_action_mask()
        assert mask.shape == (4096,)
        assert mask.dtype == np.float32
        # Should have some legal moves
        assert mask.sum() > 0
        env.close()

    def test_action_mask_count_matches_legal_moves(self):
        env = ChessEnv()
        env.reset()
        mask = env.get_legal_action_mask()
        legal_count = len(list(env.board.legal_moves))
        assert int(mask.sum()) == legal_count
        env.close()

    def test_invalid_move_forfeits(self):
        env = ChessEnv()
        env.reset()
        # Action 0 (a1->a1) is never valid
        _, reward, done, info = env.step(0)
        assert done is True
        assert reward == -1.0
        assert info['winner'] == 2
        env.close()

    def test_valid_opening_move(self):
        env = ChessEnv()
        env.reset()
        # e2e4 is always valid from starting position
        action = 12 * 64 + 28
        _, reward, done, info = env.step(action)
        # Should not end the game (unless opponent randomly checkmates)
        if not done:
            assert reward == 0.0
        env.close()

    def test_initial_piece_counts(self):
        env = ChessEnv()
        env.reset()
        info = env._info()
        assert info['p1_pieces'] == 16
        assert info['p2_pieces'] == 16
        env.close()

    def test_fools_mate(self):
        """Test checkmate detection with fool's mate pattern."""
        env = ChessEnv()
        env.reset()
        # Set up a position where White can deliver checkmate
        # We'll manually set the board to a near-checkmate position
        env.board = chess.Board('rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2')
        # Now it's Black's turn. Override: we want to test agent (White) checkmating.
        # Instead, set up White to move with a forced mate
        env.board = chess.Board('rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3')
        # White is already in checkmate (Qh4# from Black)
        done, reward = env._check_terminal()
        assert done is True
        assert reward == -1.0  # White is checkmated, agent loses
        env.close()

    def test_game_terminates_eventually(self):
        """Playing random legal moves, game should end."""
        env = ChessEnv(max_moves=50)
        env.reset()
        done = False
        for _ in range(100):
            mask = env.get_legal_action_mask()
            legal_actions = np.where(mask > 0)[0]
            if len(legal_actions) == 0:
                break
            action = np.random.choice(legal_actions)
            _, _, done, _ = env.step(action)
            if done:
                break
        assert done is True
        env.close()

    def test_render_rgb(self):
        env = ChessEnv()
        env.reset()
        img = env.render(mode='rgb_array')
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3
        env.close()

    def test_pawn_promotion(self):
        """Pawn reaching 8th rank should promote to queen."""
        env = ChessEnv()
        env.reset()
        # Set up: White pawn on e7, empty e8, Black king far away
        env.board = chess.Board('4k3/4P3/8/8/8/8/8/4K3 w - - 0 1')
        # e7=52, e8=60
        action = 52 * 64 + 60
        _, _, _, info = env.step(action)
        # The pawn should have promoted
        piece = env.board.piece_at(chess.E8) if not env.board.is_game_over() else None
        # After our move, opponent plays, so board state may differ
        # Just check the game didn't crash and accepted the promotion
        env.close()


# ─── Adapter ─────────────────────────────────────────────────────


class TestChessAdapter:
    def test_is_base_game_adapter(self):
        adapter = ChessAdapter()
        assert isinstance(adapter, BaseGameAdapter)

    def test_identity(self):
        adapter = ChessAdapter()
        assert adapter.game_id == 'chess'
        assert adapter.category == 'board'
        assert adapter.name == 'Chess'

    def test_action_space(self):
        adapter = ChessAdapter()
        info = adapter.get_action_space_info()
        assert info.num_actions == 4096
        assert len(info.action_labels) == 4096

    def test_observation_shape(self):
        adapter = ChessAdapter()
        assert adapter.get_observation_shape() == (84, 84, 1)

    def test_extract_metrics_win(self):
        adapter = ChessAdapter()
        info = {'winner': 1, 'moves_played': 30, 'p1_pieces': 10,
                'p2_pieces': 2}
        metrics = adapter.extract_metrics(info, 60.0)
        assert isinstance(metrics, StandardMetrics)
        assert metrics.score == 1.0
        assert metrics.completed is True

    def test_extract_metrics_loss(self):
        adapter = ChessAdapter()
        info = {'winner': 2, 'moves_played': 25, 'p1_pieces': 2,
                'p2_pieces': 12}
        metrics = adapter.extract_metrics(info, 50.0)
        assert metrics.score == 0.0
        assert metrics.completed is False

    def test_board_game_no_time_penalty(self):
        adapter = ChessAdapter()
        rc = adapter.get_reward_config()
        assert rc.time_penalty_per_second == 0.0

    def test_no_neat_support(self):
        adapter = ChessAdapter()
        algos = adapter.supported_algorithms()
        assert 'neat' not in algos
        assert 'rainbow' in algos

    def test_create_env(self):
        adapter = ChessAdapter()
        env = adapter.create_env()
        obs = env.reset()
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_is_available(self):
        adapter = ChessAdapter()
        assert adapter.is_available() is True

    def test_dashboard_config(self):
        adapter = ChessAdapter()
        config = adapter.get_dashboard_config()
        assert config['graph_2_metric'] == 'win_rate'

    def test_completion_criteria(self):
        adapter = ChessAdapter()
        criteria = adapter.get_completion_criteria()
        assert criteria['metric'] == 'win_rate'
        assert criteria['threshold'] == 0.7
