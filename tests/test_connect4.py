"""Tests for Connect Four game engine and adapter."""
import pytest
import numpy as np

from games.builtin.connect4.game import ConnectFourEnv
from games.builtin.connect4.adapter import ConnectFourAdapter
from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics


class TestConnectFourEnv:
    def test_reset_returns_observation(self):
        env = ConnectFourEnv()
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_step_returns_4_tuple(self):
        env = ConnectFourEnv()
        env.reset()
        obs, reward, done, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(done, bool)
        env.close()

    def test_action_space_is_7(self):
        env = ConnectFourEnv()
        assert env.action_space.n == 7
        env.close()

    def test_info_has_pieces_played(self):
        env = ConnectFourEnv()
        env.reset()
        _, _, _, info = env.step(0)
        assert 'pieces_played' in info
        env.close()

    def test_full_column_is_illegal(self):
        """Dropping in a full column returns done=True (illegal move)."""
        env = ConnectFourEnv()
        env.reset()
        # Fill column 0 (6 rows) — each step plays P1 + P2
        # but P2 plays randomly, so we just keep dropping in col 0
        for _ in range(6):
            env.step(0)
        # 7th drop in column 0 should end game (illegal)
        _, _, done, info = env.step(0)
        assert done is True
        env.close()

    def test_horizontal_win(self):
        """Four in a row horizontally wins."""
        env = ConnectFourEnv()
        env.reset()
        # Manually set up board: P1 has 3 in a row on bottom
        env.board[5, 0] = 1
        env.board[5, 1] = 1
        env.board[5, 2] = 1
        # P2 pieces elsewhere so opponent can't block col 3
        env.board[5, 5] = 2
        env.board[5, 6] = 2
        env.board[4, 6] = 2
        env._pieces_played = 6
        # P1 drops in col 3 -> completes horizontal 4 in a row
        _, reward, done, info = env.step(3)
        assert done is True
        assert info.get('winner') == 1
        env.close()


class TestConnectFourAdapter:
    def test_is_base_game_adapter(self):
        adapter = ConnectFourAdapter()
        assert isinstance(adapter, BaseGameAdapter)

    def test_identity(self):
        adapter = ConnectFourAdapter()
        assert adapter.game_id == 'connect4'
        assert adapter.category == 'board'

    def test_action_space(self):
        adapter = ConnectFourAdapter()
        info = adapter.get_action_space_info()
        assert info.num_actions == 7

    def test_observation_shape(self):
        adapter = ConnectFourAdapter()
        assert adapter.get_observation_shape() == (84, 84, 1)

    def test_extract_metrics(self):
        adapter = ConnectFourAdapter()
        info = {'pieces_played': 10, 'winner': 0}
        metrics = adapter.extract_metrics(info, 5.0)
        assert isinstance(metrics, StandardMetrics)
        assert metrics.progress == pytest.approx(10 / 42)

    def test_board_game_no_time_penalty(self):
        adapter = ConnectFourAdapter()
        rc = adapter.get_reward_config()
        assert rc.time_penalty_per_second == 0.0
