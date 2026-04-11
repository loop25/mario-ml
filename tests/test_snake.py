"""Tests for the Snake game engine and adapter."""
import pytest
import numpy as np

from games.builtin.snake.game import SnakeEnv
from games.builtin.snake.adapter import SnakeAdapter
from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics, ActionSpaceInfo


class TestSnakeEnv:
    def test_reset_returns_observation(self):
        env = SnakeEnv(grid_size=8)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (84, 84, 1)  # Grayscale rendered
        env.close()

    def test_step_returns_4_tuple(self):
        env = SnakeEnv(grid_size=8)
        env.reset()
        obs, reward, done, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        env.close()

    def test_info_contains_score(self):
        env = SnakeEnv(grid_size=8)
        env.reset()
        _, _, _, info = env.step(0)
        assert 'score' in info
        assert 'snake_length' in info
        env.close()

    def test_action_space_is_4(self):
        env = SnakeEnv(grid_size=8)
        assert env.action_space.n == 4
        env.close()

    def test_game_terminates_on_wall_collision(self):
        env = SnakeEnv(grid_size=4)
        env.reset()
        # Move in one direction until wall hit
        done = False
        for _ in range(20):
            _, _, done, _ = env.step(1)  # Right
            if done:
                break
        assert done is True
        env.close()


class TestSnakeAdapter:
    def test_is_base_game_adapter(self):
        adapter = SnakeAdapter()
        assert isinstance(adapter, BaseGameAdapter)

    def test_identity(self):
        adapter = SnakeAdapter()
        assert adapter.game_id == 'snake'
        assert adapter.category == 'arcade'

    def test_action_space_info(self):
        adapter = SnakeAdapter()
        info = adapter.get_action_space_info()
        assert info.num_actions == 4

    def test_observation_shape(self):
        adapter = SnakeAdapter()
        assert adapter.get_observation_shape() == (84, 84, 1)

    def test_extract_metrics(self):
        adapter = SnakeAdapter()
        info = {'score': 5, 'snake_length': 6, 'max_length': 64}
        metrics = adapter.extract_metrics(info, 10.0)
        assert isinstance(metrics, StandardMetrics)
        assert metrics.score == 5
        assert 0 <= metrics.progress <= 1.0

    def test_create_env(self):
        adapter = SnakeAdapter()
        env = adapter.create_env(grid_size=8)
        obs = env.reset()
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_does_not_need_sb3_compat(self):
        adapter = SnakeAdapter()
        assert adapter.needs_sb3_compat() is False
