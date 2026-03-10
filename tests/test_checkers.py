"""Tests for Checkers game engine and adapter."""
import pytest
import numpy as np

from games.builtin.checkers.game import (
    CheckersEnv, NUM_SQUARES, BOARD_SIZE,
    P1_MAN, P1_KING, P2_MAN, P2_KING, EMPTY,
    _pos_to_rc, _rc_to_pos,
)
from games.builtin.checkers.adapter import CheckersAdapter
from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics


# ─── Helper ──────────────────────────────────────────────────────


class TestPositionConversion:
    def test_pos_to_rc_first_square(self):
        """Position 0 should be row 0."""
        r, c = _pos_to_rc(0)
        assert r == 0

    def test_pos_to_rc_last_square(self):
        """Position 31 should be row 7."""
        r, c = _pos_to_rc(31)
        assert r == 7

    def test_roundtrip_conversion(self):
        """Converting pos -> rc -> pos should be identity for dark squares."""
        for pos in range(NUM_SQUARES):
            r, c = _pos_to_rc(pos)
            assert _rc_to_pos(r, c) == pos

    def test_light_square_returns_negative(self):
        """Light squares are not playable."""
        # (0, 0) is a light square on standard checkerboard
        assert _rc_to_pos(0, 0) == -1


# ─── Environment ─────────────────────────────────────────────────


class TestCheckersEnv:
    def test_reset_returns_observation(self):
        env = CheckersEnv()
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_initial_piece_counts(self):
        env = CheckersEnv()
        env.reset()
        p1 = sum(1 for p in env.board if p in (P1_MAN, P1_KING))
        p2 = sum(1 for p in env.board if p in (P2_MAN, P2_KING))
        assert p1 == 12
        assert p2 == 12
        env.close()

    def test_step_returns_4_tuple(self):
        env = CheckersEnv()
        env.reset()
        # Get a valid move
        valid = env._get_valid_moves(player=1)
        assert len(valid) > 0
        from_pos, to_pos = valid[0]
        action = from_pos * NUM_SQUARES + to_pos
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        env.close()

    def test_action_space_size(self):
        env = CheckersEnv()
        assert env.action_space.n == NUM_SQUARES * NUM_SQUARES
        env.close()

    def test_info_has_expected_keys(self):
        env = CheckersEnv()
        env.reset()
        valid = env._get_valid_moves(player=1)
        from_pos, to_pos = valid[0]
        action = from_pos * NUM_SQUARES + to_pos
        _, _, _, info = env.step(action)
        assert 'winner' in info
        assert 'moves_played' in info
        assert 'p1_pieces' in info
        assert 'p2_pieces' in info
        env.close()

    def test_invalid_move_forfeits(self):
        """Invalid action should end the game with -1 reward."""
        env = CheckersEnv()
        env.reset()
        # Action 0 (pos 0 -> pos 0) is never a valid move
        _, reward, done, info = env.step(0)
        assert done is True
        assert reward == -1.0
        assert info['winner'] == 2
        env.close()

    def test_mandatory_captures(self):
        """If captures are available, only captures are valid."""
        env = CheckersEnv()
        env.reset()
        # Clear board and set up a forced capture
        env.board[:] = EMPTY
        # P1 man at position 20 (row 5, col 1)
        env.board[20] = P1_MAN
        # P2 man at position 16 (row 4, col 1) - diagonally ahead
        env.board[16] = P2_MAN
        # Landing at position 12 (row 3) should be empty
        env.board[12] = EMPTY

        valid = env._get_valid_moves(player=1)
        # All valid moves should be captures (distance = 2 rows)
        for from_pos, to_pos in valid:
            fr, _ = _pos_to_rc(from_pos)
            tr, _ = _pos_to_rc(to_pos)
            assert abs(fr - tr) == 2
        env.close()

    def test_king_promotion(self):
        """A man reaching the back row becomes a king."""
        env = CheckersEnv()
        env.reset()
        # Place P1 man on position 4 (row 1) — one step from promotion
        env.board[:] = EMPTY
        env.board[4] = P1_MAN
        # Position 0 (row 0) is the promotion row for P1
        # Manually move piece to back row
        env.board[0] = P1_MAN
        env._check_promotion()
        assert env.board[0] == P1_KING
        env.close()

    def test_game_terminates_eventually(self):
        """Playing valid moves, the game should end within max_moves."""
        env = CheckersEnv()
        env.reset()
        done = False
        for _ in range(300):
            valid = env._get_valid_moves(player=1)
            if not valid:
                break
            from_pos, to_pos = valid[0]
            action = from_pos * NUM_SQUARES + to_pos
            _, _, done, _ = env.step(action)
            if done:
                break
        assert done is True
        env.close()

    def test_render_rgb(self):
        env = CheckersEnv()
        env.reset()
        img = env.render(mode='rgb_array')
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3
        env.close()


# ─── Adapter ─────────────────────────────────────────────────────


class TestCheckersAdapter:
    def test_is_base_game_adapter(self):
        adapter = CheckersAdapter()
        assert isinstance(adapter, BaseGameAdapter)

    def test_identity(self):
        adapter = CheckersAdapter()
        assert adapter.game_id == 'checkers'
        assert adapter.category == 'board'
        assert adapter.name == 'Checkers'

    def test_action_space(self):
        adapter = CheckersAdapter()
        info = adapter.get_action_space_info()
        assert info.num_actions == NUM_SQUARES * NUM_SQUARES

    def test_observation_shape(self):
        adapter = CheckersAdapter()
        assert adapter.get_observation_shape() == (84, 84, 1)

    def test_extract_metrics_win(self):
        adapter = CheckersAdapter()
        info = {'winner': 1, 'p1_pieces': 8, 'p2_pieces': 0,
                'moves_played': 50}
        metrics = adapter.extract_metrics(info, 30.0)
        assert isinstance(metrics, StandardMetrics)
        assert metrics.score == 1.0
        assert metrics.completed is True

    def test_extract_metrics_loss(self):
        adapter = CheckersAdapter()
        info = {'winner': 2, 'p1_pieces': 0, 'p2_pieces': 5,
                'moves_played': 40}
        metrics = adapter.extract_metrics(info, 25.0)
        assert metrics.score == 0.0
        assert metrics.completed is False

    def test_board_game_no_time_penalty(self):
        adapter = CheckersAdapter()
        rc = adapter.get_reward_config()
        assert rc.time_penalty_per_second == 0.0

    def test_no_neat_support(self):
        """Checkers obs is 84x84, too large for NEAT."""
        adapter = CheckersAdapter()
        algos = adapter.supported_algorithms()
        assert 'neat' not in algos
        assert 'rainbow' in algos

    def test_create_env(self):
        adapter = CheckersAdapter()
        env = adapter.create_env()
        obs = env.reset()
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_dashboard_config_has_win_rate(self):
        adapter = CheckersAdapter()
        config = adapter.get_dashboard_config()
        assert config['graph_2_metric'] == 'win_rate'

    def test_completion_criteria(self):
        adapter = CheckersAdapter()
        criteria = adapter.get_completion_criteria()
        assert criteria['metric'] == 'win_rate'
        assert criteria['threshold'] == 0.8
