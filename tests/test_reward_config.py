"""Tests for reward config dataclasses."""
import pytest
from games.reward_config import StandardMetrics, RewardConfig, ActionSpaceInfo, TokenConfig


class TestStandardMetrics:
    def test_create_with_defaults(self):
        m = StandardMetrics(progress=0.5, score=100.0, completed=False, time_elapsed=10.0)
        assert m.progress == 0.5
        assert m.score == 100.0
        assert m.completed is False
        assert m.time_elapsed == 10.0


class TestRewardConfig:
    def test_default_values(self):
        rc = RewardConfig()
        assert rc.time_penalty_per_second == 0.01
        assert rc.completion_bonus == 100.0
        assert rc.death_penalty == -15.0
        assert rc.idle_penalty_per_second == 0.005
        assert rc.speed_bonus_multiplier == 1.5
        assert rc.par_time_seconds == 120.0

    def test_custom_values(self):
        rc = RewardConfig(time_penalty_per_second=0.0, completion_bonus=50.0)
        assert rc.time_penalty_per_second == 0.0
        assert rc.completion_bonus == 50.0


class TestActionSpaceInfo:
    def test_create(self):
        info = ActionSpaceInfo(num_actions=4, action_labels=['up', 'down', 'left', 'right'])
        assert info.num_actions == 4
        assert len(info.action_labels) == 4


class TestTokenConfig:
    def test_create(self):
        tc = TokenConfig(
            obs_resolution=(84, 84),
            obs_channels=1,
            action_vocab_size=7,
            game_token_id=42,
        )
        assert tc.obs_resolution == (84, 84)
        assert tc.obs_channels == 1
        assert tc.action_vocab_size == 7
        assert tc.game_token_id == 42
