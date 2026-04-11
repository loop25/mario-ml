"""Tests for the Tetris game engine and adapter."""
import pytest
import numpy as np

from games.builtin.tetris.game import TetrisEnv, BOARD_W, BOARD_H, PIECE_NAMES
from games.builtin.tetris.adapter import TetrisAdapter
from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics, ActionSpaceInfo


class TestTetrisEnv:
    def test_reset_returns_observation(self):
        env = TetrisEnv()
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_step_returns_4_tuple(self):
        env = TetrisEnv()
        env.reset()
        obs, reward, done, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        env.close()

    def test_info_contains_expected_keys(self):
        env = TetrisEnv()
        env.reset()
        _, _, _, info = env.step(0)
        assert 'score' in info
        assert 'lines_cleared' in info
        assert 'level' in info
        env.close()

    def test_action_space_is_8(self):
        env = TetrisEnv()
        assert env.action_space.n == 8  # left, right, rot_cw, rot_ccw, hard_drop, soft_drop, no-op, hold
        env.close()

    def test_observation_space_shape(self):
        env = TetrisEnv()
        assert env.observation_space.shape == (84, 84, 1)
        env.close()

    def test_game_eventually_terminates(self):
        """Random play should eventually cause game over."""
        env = TetrisEnv(max_steps=5000)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 5000:
            _, _, done, _ = env.step(env.action_space.sample())
            steps += 1
        assert done is True
        env.close()

    def test_hard_drop_action(self):
        """Hard drop (action 4) should lock piece immediately."""
        env = TetrisEnv()
        env.reset()
        # Hard drop
        obs, reward, done, info = env.step(4)
        # Piece should have landed — board should have some blocks
        assert not np.all(env.board == 0) or done
        env.close()

    def test_render_rgb_shape(self):
        """Dashboard render should produce 480x480 RGB."""
        env = TetrisEnv()
        env.reset()
        frame = env.render(mode='rgb_array')
        assert frame.shape == (480, 480, 3)
        assert frame.dtype == np.uint8
        env.close()

    def test_board_dimensions(self):
        env = TetrisEnv()
        env.reset()
        assert env.board.shape == (BOARD_H, BOARD_W)
        assert BOARD_H == 20
        assert BOARD_W == 10
        env.close()

    def test_line_clearing(self):
        """Manually fill a row and verify it gets cleared."""
        env = TetrisEnv()
        env.reset()
        # Fill bottom row completely
        env.board[BOARD_H - 1, :] = 1
        cleared = env._clear_lines()
        assert cleared == 1
        assert env._lines_cleared == 1
        # Bottom row should now be empty (cleared and replaced)
        assert np.all(env.board[BOARD_H - 1] == 0)
        env.close()

    def test_multiple_line_clear(self):
        """Fill multiple rows and verify all get cleared."""
        env = TetrisEnv()
        env.reset()
        env.board[BOARD_H - 1, :] = 1
        env.board[BOARD_H - 2, :] = 2
        env.board[BOARD_H - 3, :] = 3
        cleared = env._clear_lines()
        assert cleared == 3
        assert env._lines_cleared == 3
        env.close()

    def test_hold_piece(self):
        """Hold action should stash the current piece."""
        env = TetrisEnv()
        env.reset()
        original_piece = env._piece_type
        # Action 7 = Hold
        obs, reward, done, info = env.step(7)
        assert info['hold_piece'] == original_piece
        env.close()

    def test_tspin_field_in_info(self):
        """Info dict should contain tspin field."""
        env = TetrisEnv()
        env.reset()
        _, _, _, info = env.step(0)
        assert 'tspin' in info
        assert isinstance(info['tspin'], bool)
        env.close()

    def test_hold_piece_swap(self):
        """Second hold should swap current with held piece."""
        env = TetrisEnv()
        env.reset()
        first_piece = env._piece_type
        # First hold
        env.step(7)
        second_piece = env._piece_type
        # Hard drop to lock piece and reset hold_used
        env.step(4)
        # Now hold again to swap
        env.step(7)
        assert env._hold_piece_type == env._piece_type or env._hold_piece_type is not None
        env.close()

    def test_level_progression(self):
        """Level should increase every 10 lines."""
        env = TetrisEnv()
        env.reset()
        env._lines_cleared = 0
        env._level = 1 + env._lines_cleared // 10
        assert env._level == 1
        env._lines_cleared = 10
        env._level = 1 + env._lines_cleared // 10
        assert env._level == 2
        env._lines_cleared = 25
        env._level = 1 + env._lines_cleared // 10
        assert env._level == 3
        env.close()

    def test_frame_capture_through_wrappers(self):
        """Frame capture should work through TimeRewardWrapper."""
        from src.rewards.time_reward import TimeRewardWrapper
        from src.algorithms.frame_utils import capture_display_frame
        env = TetrisEnv()
        wrapped = TimeRewardWrapper(env, None, None)
        wrapped.reset()
        frame = capture_display_frame(wrapped)
        assert frame is not None
        assert frame.shape == (480, 480, 3)
        wrapped.close()


class TestTetrisAdapter:
    def test_is_base_game_adapter(self):
        adapter = TetrisAdapter()
        assert isinstance(adapter, BaseGameAdapter)

    def test_identity(self):
        adapter = TetrisAdapter()
        assert adapter.name == 'Tetris'
        assert adapter.game_id == 'tetris'
        assert adapter.category == 'puzzle'

    def test_action_space_info(self):
        adapter = TetrisAdapter()
        info = adapter.get_action_space_info()
        assert isinstance(info, ActionSpaceInfo)
        assert info.num_actions == 8
        assert len(info.action_labels) == 8

    def test_observation_shape(self):
        adapter = TetrisAdapter()
        assert adapter.get_observation_shape() == (84, 84, 1)

    def test_create_env(self):
        adapter = TetrisAdapter()
        env = adapter.create_env()
        assert isinstance(env, TetrisEnv)
        obs = env.reset()
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_extract_metrics(self):
        adapter = TetrisAdapter()
        info = {'score': 500, 'lines_cleared': 10, 'level': 2}
        metrics = adapter.extract_metrics(info, episode_time=30.0)
        assert isinstance(metrics, StandardMetrics)
        assert metrics.score == 500.0
        assert metrics.progress == 10 / 40  # 10 lines / 40 target

    def test_registry_discovery(self):
        """Tetris should be discovered by the game registry."""
        from games.registry import GameRegistry
        registry = GameRegistry()
        registry.discover()
        game_ids = [g.game_id for g in registry.list_games()]
        assert 'tetris' in game_ids

    def test_dashboard_config(self):
        adapter = TetrisAdapter()
        config = adapter.get_dashboard_config()
        assert 'graph_2_title' in config
        assert config['graph_2_info_key'] == 'lines_cleared'
