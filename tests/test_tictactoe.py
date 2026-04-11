"""Tests for Tic-Tac-Toe game engine and adapter."""
import pytest
import numpy as np

from games.builtin.tictactoe.game import TicTacToeEnv
from games.builtin.tictactoe.adapter import TicTacToeAdapter
from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics


class TestTicTacToeEnv:
    def test_reset_returns_observation(self):
        env = TicTacToeEnv()
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_step_returns_4_tuple(self):
        env = TicTacToeEnv()
        env.reset()
        obs, reward, done, info = env.step(4)  # Center
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        env.close()

    def test_action_space_is_9(self):
        env = TicTacToeEnv()
        assert env.action_space.n == 9
        env.close()

    def test_info_has_winner(self):
        env = TicTacToeEnv()
        env.reset()
        _, _, _, info = env.step(0)
        assert 'winner' in info
        assert 'moves_played' in info
        env.close()

    def test_invalid_move_is_forfeit(self):
        """Playing on an occupied square ends the game."""
        env = TicTacToeEnv()
        env.reset()
        # Place agent on cell 4
        env.board[1, 1] = 1
        # Try to play on cell 4 again
        _, reward, done, info = env.step(4)
        assert done is True
        assert reward == -1.0
        env.close()

    def test_agent_win_horizontal(self):
        """Agent wins with three in a row."""
        env = TicTacToeEnv()
        env.reset()
        # Set up: agent has cells 0, 1. Opponent has cells 3, 4.
        env.board[0, 0] = 1  # cell 0
        env.board[0, 1] = 1  # cell 1
        env.board[1, 0] = 2  # cell 3
        env.board[1, 1] = 2  # cell 4
        env._moves_played = 4
        # Agent plays cell 2 -> wins
        _, reward, done, info = env.step(2)
        assert done is True
        assert reward == 1.0
        assert info['winner'] == 1
        env.close()

    def test_game_terminates_eventually(self):
        """Playing randomly, the game should end within 9 moves."""
        env = TicTacToeEnv()
        env.reset()
        done = False
        for _ in range(20):
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            if done:
                break
        assert done is True
        env.close()

    def test_render_rgb(self):
        env = TicTacToeEnv()
        env.reset()
        img = env.render(mode='rgb_array')
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3  # RGB
        env.close()

    def test_board_starts_empty(self):
        env = TicTacToeEnv()
        env.reset()
        assert np.all(env.board == 0)
        env.close()


class TestTicTacToeAdapter:
    def test_is_base_game_adapter(self):
        adapter = TicTacToeAdapter()
        assert isinstance(adapter, BaseGameAdapter)

    def test_identity(self):
        adapter = TicTacToeAdapter()
        assert adapter.game_id == 'tictactoe'
        assert adapter.category == 'board'
        assert adapter.name == 'Tic-Tac-Toe'

    def test_action_space(self):
        adapter = TicTacToeAdapter()
        info = adapter.get_action_space_info()
        assert info.num_actions == 9
        assert len(info.action_labels) == 9

    def test_observation_shape(self):
        adapter = TicTacToeAdapter()
        assert adapter.get_observation_shape() == (84, 84, 1)

    def test_extract_metrics(self):
        adapter = TicTacToeAdapter()
        info = {'winner': 1, 'moves_played': 5}
        metrics = adapter.extract_metrics(info, 2.0)
        assert isinstance(metrics, StandardMetrics)
        assert metrics.score == 1.0
        assert metrics.completed is True

    def test_extract_metrics_loss(self):
        adapter = TicTacToeAdapter()
        info = {'winner': 2, 'moves_played': 6}
        metrics = adapter.extract_metrics(info, 3.0)
        assert metrics.score == 0.0
        assert metrics.completed is False

    def test_board_game_no_time_penalty(self):
        adapter = TicTacToeAdapter()
        rc = adapter.get_reward_config()
        assert rc.time_penalty_per_second == 0.0

    def test_create_env(self):
        adapter = TicTacToeAdapter()
        env = adapter.create_env()
        obs = env.reset()
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_dashboard_config_has_win_rate(self):
        adapter = TicTacToeAdapter()
        config = adapter.get_dashboard_config()
        assert config['graph_2_metric'] == 'win_rate'

    def test_completion_criteria(self):
        adapter = TicTacToeAdapter()
        criteria = adapter.get_completion_criteria()
        assert criteria['metric'] == 'win_rate'
        assert criteria['threshold'] == 0.95
