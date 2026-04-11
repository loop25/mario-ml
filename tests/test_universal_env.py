"""Tests for universal environment factory."""
import pytest
import numpy as np

from src.environment.universal_env import create_env_from_adapter
from tests.test_base_adapter import StubAdapter


class TestUniversalEnvFactory:
    def test_creates_env(self):
        adapter = StubAdapter()
        env = create_env_from_adapter(adapter)
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        env.close()

    def test_env_reset_returns_obs(self):
        adapter = StubAdapter()
        env = create_env_from_adapter(adapter)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        env.close()

    def test_env_step_returns_4_tuple(self):
        adapter = StubAdapter()
        env = create_env_from_adapter(adapter)
        env.reset()
        result = env.step(0)
        assert len(result) == 4  # obs, reward, done, info
        env.close()

    def test_time_rewards_enabled_by_default(self):
        adapter = StubAdapter()
        env = create_env_from_adapter(adapter, use_time_rewards=True)
        env.reset()
        _, _, _, info = env.step(0)
        assert '_standard_metrics' in info
        env.close()

    def test_time_rewards_disabled(self):
        adapter = StubAdapter()
        env = create_env_from_adapter(adapter, use_time_rewards=False)
        env.reset()
        _, _, _, info = env.step(0)
        assert '_standard_metrics' not in info
        env.close()
