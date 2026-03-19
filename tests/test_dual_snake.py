"""Tests for the Dual Snake cooperative game engine and adapter."""
import pytest
import numpy as np

from games.builtin.dual_snake.game import DualSnakeEnv
from games.builtin.dual_snake.adapter import DualSnakeAdapter
from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics, ActionSpaceInfo


class TestDualSnakeEnv:
    def test_reset_returns_obs(self):
        env = DualSnakeEnv(grid_size=8)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (84, 84, 3)
        env.close()

    def test_action_space(self):
        env = DualSnakeEnv(grid_size=8)
        assert env.action_space.n == 16
        env.close()

    def test_both_snakes_start_alive(self):
        env = DualSnakeEnv(grid_size=8)
        env.reset()
        # Take one step to get info
        _, _, _, info = env.step(0)
        assert info['snake1_alive'] is True
        assert info['snake2_alive'] is True
        env.close()

    def test_food_eaten_increases_score(self):
        """Place food directly in front of snake1 and verify score."""
        env = DualSnakeEnv(grid_size=16)
        env.reset()
        # Snake1 starts at (4,4) moving right; place food at (4,5)
        env.food = (env.grid_size // 4, env.grid_size // 4 + 1)
        # Action: snake1=right(1), snake2=left(3) -> 1*4+3=7
        _, reward, _, info = env.step(7)
        assert info['score'] >= 1
        assert reward >= 1.0
        env.close()

    def test_wall_collision_kills_snake(self):
        """Move snake1 into wall repeatedly; it should die."""
        env = DualSnakeEnv(grid_size=8)
        env.reset()
        # Snake1 starts at (2,2) moving right on 8x8 grid.
        # Move right until wall collision. Action: s1=right(1), s2=left(3) -> 7
        dead = False
        for _ in range(20):
            _, _, done, info = env.step(7)
            if not info['snake1_alive']:
                dead = True
                break
        assert dead is True
        env.close()

    def test_snake_collision(self):
        """Force snake1 to collide with snake2."""
        env = DualSnakeEnv(grid_size=16)
        env.reset()
        g = env.grid_size
        # Place snake1 right next to snake2 body, facing into it
        env.snake1 = [(3 * g // 4, 3 * g // 4 - 1)]
        env.dir1 = 1  # Moving right, toward snake2 head at (3*g//4, 3*g//4)
        env.snake2 = [(3 * g // 4, 3 * g // 4),
                      (3 * g // 4, 3 * g // 4 + 1),
                      (3 * g // 4, 3 * g // 4 + 2)]
        env._dead1 = False
        env._dead2 = False
        # Action: s1=right(1), s2=left(3) -> 7
        _, _, _, info = env.step(7)
        assert info['snake1_alive'] is False
        env.close()

    def test_both_dead_ends_episode(self):
        """When both snakes are dead, done should be True."""
        env = DualSnakeEnv(grid_size=8)
        env.reset()
        # Manually kill both snakes
        env._dead1 = True
        env._dead2 = True
        _, _, done, _ = env.step(0)
        assert done is True
        env.close()

    def test_game_continues_one_alive(self):
        """One dead but other alive: done should be False."""
        env = DualSnakeEnv(grid_size=16)
        env.reset()
        env._dead1 = True
        env._dead2 = False
        _, _, done, info = env.step(0)
        # Should not be done (snake2 still alive, no timeout yet)
        assert done is False
        assert info['snake1_alive'] is False
        assert info['snake2_alive'] is True
        env.close()

    def test_render_rgb(self):
        """Observation should be 3-channel RGB."""
        env = DualSnakeEnv(grid_size=8)
        obs = env.reset()
        assert obs.shape[2] == 3
        assert obs.dtype == np.uint8
        env.close()


class TestDualSnakeAdapter:
    def test_adapter_properties(self):
        adapter = DualSnakeAdapter()
        assert isinstance(adapter, BaseGameAdapter)
        assert adapter.game_id == 'dual_snake'
        assert adapter.name == 'Dual Snake'
        assert adapter.category == 'cooperative'

    def test_action_space_info(self):
        adapter = DualSnakeAdapter()
        info = adapter.get_action_space_info()
        assert info.num_actions == 16
        assert len(info.action_labels) == 16

    def test_observation_shape(self):
        adapter = DualSnakeAdapter()
        assert adapter.get_observation_shape() == (84, 84, 3)

    def test_extract_metrics(self):
        adapter = DualSnakeAdapter()
        info = {'score': 5, 'snake1_length': 6, 'snake2_length': 4,
                'max_length': 256}
        metrics = adapter.extract_metrics(info, 10.0)
        assert isinstance(metrics, StandardMetrics)
        assert metrics.score == 5.0

    def test_create_env(self):
        adapter = DualSnakeAdapter()
        env = adapter.create_env(grid_size=8)
        obs = env.reset()
        assert obs.shape == (84, 84, 3)
        env.close()

    def test_does_not_need_sb3_compat(self):
        adapter = DualSnakeAdapter()
        assert adapter.needs_sb3_compat() is False
