"""Tests for the Mario game adapter.

These tests verify the adapter interface without needing
the actual NES emulator — they test the adapter's identity,
config, and metrics extraction using mocked info dicts.
"""
import pytest

try:
    from games.retro.mario.adapter import MarioAdapter
    _HAS_MARIO = True
except ImportError:
    _HAS_MARIO = False

from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics, RewardConfig, ActionSpaceInfo

pytestmark = pytest.mark.skipif(
    not _HAS_MARIO,
    reason='gym-super-mario-bros not installed'
)


class TestMarioAdapter:
    def test_is_base_game_adapter(self):
        adapter = MarioAdapter()
        assert isinstance(adapter, BaseGameAdapter)

    def test_identity(self):
        adapter = MarioAdapter()
        assert adapter.name == 'Super Mario Bros'
        assert adapter.game_id == 'mario'
        assert adapter.category == 'platformer'
        assert 'Mario' in adapter.description or 'mario' in adapter.description.lower()

    def test_action_space_info(self):
        adapter = MarioAdapter()
        info = adapter.get_action_space_info()
        assert isinstance(info, ActionSpaceInfo)
        assert info.num_actions == 7  # Our simplified action space
        assert len(info.action_labels) == 7

    def test_observation_shape(self):
        adapter = MarioAdapter()
        shape = adapter.get_observation_shape()
        assert shape == (84, 84, 4)  # 84x84 with 4 stacked frames

    def test_extract_metrics_normal_play(self):
        adapter = MarioAdapter()
        info = {
            'x_pos': 500,
            'coins': 3,
            'flag_get': False,
            'life': 2,
        }
        metrics = adapter.extract_metrics(info, episode_time=30.0)
        assert isinstance(metrics, StandardMetrics)
        assert 0.0 <= metrics.progress <= 1.0
        assert metrics.score == 3  # coins
        assert metrics.completed is False
        assert metrics.time_elapsed == 30.0

    def test_extract_metrics_flag_reached(self):
        adapter = MarioAdapter()
        info = {'x_pos': 3000, 'coins': 10, 'flag_get': True, 'life': 2}
        metrics = adapter.extract_metrics(info, episode_time=60.0)
        assert metrics.completed is True

    def test_reward_config(self):
        adapter = MarioAdapter()
        rc = adapter.get_reward_config()
        assert isinstance(rc, RewardConfig)
        # Mario should have nonzero time penalty (action game)
        assert rc.time_penalty_per_second > 0

    def test_needs_sb3_compat(self):
        adapter = MarioAdapter()
        # Mario uses old gym API, needs SB3 compat
        assert adapter.needs_sb3_compat() is True

    def test_game_specific_options(self):
        adapter = MarioAdapter()
        opts = adapter.get_game_specific_options()
        assert 'world' in opts
        assert 'stage' in opts

    def test_has_curriculum(self):
        adapter = MarioAdapter()
        curriculum = adapter.get_curriculum()
        # Mario has multi-level curriculum
        assert curriculum is not None
