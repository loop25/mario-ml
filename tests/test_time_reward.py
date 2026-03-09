"""Tests for TimeRewardWrapper."""
import time
import pytest
import numpy as np
import gym
from gym.spaces import Box, Discrete
from unittest.mock import MagicMock

from src.rewards.time_reward import TimeRewardWrapper
from games.reward_config import RewardConfig, StandardMetrics


class FakeGameEnv(gym.Env):
    """Minimal env that returns controllable info dicts."""

    def __init__(self):
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = Discrete(4)
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        return np.zeros((84, 84, 1), dtype=np.uint8)

    def step(self, action):
        self._step_count += 1
        obs = np.zeros((84, 84, 1), dtype=np.uint8)
        reward = 1.0
        done = self._step_count >= 10
        info = {'score': self._step_count * 10, 'progress': self._step_count / 10}
        return obs, reward, done, info


class FakeAdapter:
    """Stub adapter for metric extraction."""

    def __init__(self, completed=False):
        self._completed = completed

    def extract_metrics(self, info, episode_time):
        return StandardMetrics(
            progress=info.get('progress', 0.0),
            score=info.get('score', 0.0),
            completed=self._completed,
            time_elapsed=episode_time,
        )


class TestTimeRewardWrapper:
    def test_reset_returns_obs(self):
        env = FakeGameEnv()
        adapter = FakeAdapter()
        wrapped = TimeRewardWrapper(env, RewardConfig(), adapter)
        obs = wrapped.reset()
        assert obs.shape == (84, 84, 1)

    def test_step_returns_shaped_reward(self):
        env = FakeGameEnv()
        adapter = FakeAdapter()
        config = RewardConfig(time_penalty_per_second=0.0, idle_penalty_per_second=0.0)
        wrapped = TimeRewardWrapper(env, config, adapter)
        wrapped.reset()
        obs, reward, done, info = wrapped.step(0)
        # Base reward is 1.0, no time/idle penalty
        assert reward == pytest.approx(1.0, abs=0.1)

    def test_time_penalty_reduces_reward(self):
        env = FakeGameEnv()
        adapter = FakeAdapter()
        config = RewardConfig(
            time_penalty_per_second=100.0,  # Very large to be detectable
            idle_penalty_per_second=0.0,
        )
        wrapped = TimeRewardWrapper(env, config, adapter)
        wrapped.reset()
        time.sleep(0.05)  # Ensure some time passes
        _, reward, _, _ = wrapped.step(0)
        # Reward should be less than 1.0 due to time penalty
        assert reward < 1.0

    def test_completion_bonus_added(self):
        env = FakeGameEnv()
        adapter = FakeAdapter(completed=True)
        config = RewardConfig(
            time_penalty_per_second=0.0,
            idle_penalty_per_second=0.0,
            completion_bonus=50.0,
        )
        wrapped = TimeRewardWrapper(env, config, adapter)
        wrapped.reset()
        _, reward, _, _ = wrapped.step(0)
        # Base (1.0) + completion (50.0)
        assert reward >= 50.0

    def test_standard_metrics_in_info(self):
        env = FakeGameEnv()
        adapter = FakeAdapter()
        config = RewardConfig()
        wrapped = TimeRewardWrapper(env, config, adapter)
        wrapped.reset()
        _, _, _, info = wrapped.step(0)
        assert '_standard_metrics' in info
        assert isinstance(info['_standard_metrics'], StandardMetrics)

    def test_death_penalty_on_done_without_completion(self):
        """Agent dies (done=True, completed=False) -> death penalty applied."""
        env = FakeGameEnv()
        env._step_count = 9  # Next step will set done=True
        adapter = FakeAdapter(completed=False)
        config = RewardConfig(
            time_penalty_per_second=0.0,
            idle_penalty_per_second=0.0,
            death_penalty=-50.0,
        )
        wrapped = TimeRewardWrapper(env, config, adapter)
        wrapped.reset()
        env._step_count = 9  # Force done on next step
        _, reward, done, _ = wrapped.step(0)
        assert done is True
        # Base (1.0) + death (-50.0) = -49.0
        assert reward < 0
