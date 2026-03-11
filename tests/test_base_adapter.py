"""Tests for BaseGameAdapter abstract interface."""
import pytest
import numpy as np
try:
    import gymnasium as gym
    from gymnasium.spaces import Box, Discrete
except ImportError:
    import gym
    from gym.spaces import Box, Discrete

from games.base_adapter import BaseGameAdapter
from games.reward_config import (
    StandardMetrics, RewardConfig, ActionSpaceInfo, TokenConfig,
)


class FakeEnv(gym.Env):
    """Minimal gym env for testing."""
    def __init__(self):
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = Discrete(4)

    def reset(self):
        return np.zeros((84, 84, 1), dtype=np.uint8)

    def step(self, action):
        obs = np.zeros((84, 84, 1), dtype=np.uint8)
        return obs, 1.0, False, {'score': 10}

    def render(self, mode='rgb_array'):
        return np.zeros((84, 84, 3), dtype=np.uint8)


class StubAdapter(BaseGameAdapter):
    """Concrete adapter for testing the abstract interface."""

    @property
    def name(self):
        return 'Stub Game'

    @property
    def game_id(self):
        return 'stub'

    @property
    def category(self):
        return 'arcade'

    @property
    def description(self):
        return 'A test stub game.'

    def create_env(self, **kwargs):
        return FakeEnv()

    def get_action_space_info(self):
        return ActionSpaceInfo(num_actions=4, action_labels=['up', 'down', 'left', 'right'])

    def get_observation_shape(self):
        return (84, 84, 1)

    def extract_metrics(self, info, episode_time):
        return StandardMetrics(
            progress=0.5,
            score=info.get('score', 0),
            completed=False,
            time_elapsed=episode_time,
        )


class TestBaseGameAdapter:
    def test_identity_properties(self):
        adapter = StubAdapter()
        assert adapter.name == 'Stub Game'
        assert adapter.game_id == 'stub'
        assert adapter.category == 'arcade'
        assert adapter.description == 'A test stub game.'

    def test_create_env_returns_gym_env(self):
        adapter = StubAdapter()
        env = adapter.create_env()
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        env.close()

    def test_get_action_space_info(self):
        adapter = StubAdapter()
        info = adapter.get_action_space_info()
        assert info.num_actions == 4
        assert info.action_labels[0] == 'up'

    def test_get_observation_shape(self):
        adapter = StubAdapter()
        shape = adapter.get_observation_shape()
        assert shape == (84, 84, 1)

    def test_extract_metrics(self):
        adapter = StubAdapter()
        metrics = adapter.extract_metrics({'score': 42}, episode_time=5.0)
        assert isinstance(metrics, StandardMetrics)
        assert metrics.score == 42
        assert metrics.time_elapsed == 5.0

    def test_default_reward_config(self):
        adapter = StubAdapter()
        rc = adapter.get_reward_config()
        assert isinstance(rc, RewardConfig)
        assert rc.time_penalty_per_second == 0.01

    def test_default_curriculum_is_none(self):
        adapter = StubAdapter()
        assert adapter.get_curriculum() is None

    def test_default_game_specific_options_empty(self):
        adapter = StubAdapter()
        assert adapter.get_game_specific_options() == {}

    def test_default_token_config(self):
        adapter = StubAdapter()
        tc = adapter.get_token_config()
        assert isinstance(tc, TokenConfig)
        assert tc.obs_resolution == (84, 84)
        assert tc.obs_channels == 1
        assert tc.action_vocab_size == 4

    def test_default_needs_sb3_compat_false(self):
        adapter = StubAdapter()
        assert adapter.needs_sb3_compat() is False

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseGameAdapter()
