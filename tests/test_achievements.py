"""Tests for achievement definitions and the check_achievements helper."""
import pytest
from src.achievements.definitions import (
    Achievement,
    AGENT_ACHIEVEMENTS,
    TRAINER_ACHIEVEMENTS,
    check_achievements,
)


class TestAchievementCounts:
    """Verify the achievement catalogues have enough entries."""

    def test_agent_achievements_exist(self):
        """At least 10 agent achievements are defined."""
        assert len(AGENT_ACHIEVEMENTS) >= 10

    def test_trainer_achievements_exist(self):
        """At least 8 trainer achievements are defined."""
        assert len(TRAINER_ACHIEVEMENTS) >= 8


class TestAchievementFields:
    """Every achievement must carry the required fields."""

    @pytest.mark.parametrize("ach", AGENT_ACHIEVEMENTS + TRAINER_ACHIEVEMENTS,
                             ids=lambda a: a.id)
    def test_achievement_has_required_fields(self, ach):
        """Each achievement has id, name, description, and a callable condition."""
        assert isinstance(ach.id, str) and ach.id
        assert isinstance(ach.name, str) and ach.name
        assert isinstance(ach.description, str) and ach.description
        assert callable(ach.condition)


class TestAgentConditions:
    """Verify individual agent achievement conditions."""

    def test_first_steps_triggers(self):
        """episodes=1 earns first_steps."""
        earned = check_achievements(AGENT_ACHIEVEMENTS, {"episodes": 1})
        ids = {a.id for a in earned}
        assert "first_steps" in ids

    def test_first_steps_does_not_trigger_at_zero(self):
        """episodes=0 does NOT earn first_steps."""
        earned = check_achievements(AGENT_ACHIEVEMENTS, {"episodes": 0})
        ids = {a.id for a in earned}
        assert "first_steps" not in ids

    def test_century_triggers(self):
        """episodes=100 earns century."""
        earned = check_achievements(AGENT_ACHIEVEMENTS, {"episodes": 100})
        ids = {a.id for a in earned}
        assert "century" in ids

    def test_first_win_triggers(self):
        """wins=1 earns first_win."""
        earned = check_achievements(AGENT_ACHIEVEMENTS, {"wins": 1})
        ids = {a.id for a in earned}
        assert "first_win" in ids


class TestTrainerConditions:
    """Verify individual trainer achievement conditions."""

    def test_getting_started_triggers(self):
        """total_sessions=1 earns getting_started."""
        earned = check_achievements(TRAINER_ACHIEVEMENTS, {"total_sessions": 1})
        ids = {a.id for a in earned}
        assert "getting_started" in ids

    def test_algo_collector_needs_all_six(self):
        """A set of 6 algorithms earns algo_collector."""
        stats = {"algorithms_used": {"ppo", "a2c", "dqn", "rainbow", "dt", "sac"}}
        earned = check_achievements(TRAINER_ACHIEVEMENTS, stats)
        ids = {a.id for a in earned}
        assert "algo_collector" in ids

    def test_algo_collector_does_not_trigger_partial(self):
        """A set of only 2 algorithms does NOT earn algo_collector."""
        stats = {"algorithms_used": {"ppo", "a2c"}}
        earned = check_achievements(TRAINER_ACHIEVEMENTS, stats)
        ids = {a.id for a in earned}
        assert "algo_collector" not in ids


class TestAlreadyEarned:
    """The already_earned parameter should prevent re-awarding."""

    def test_already_earned_excluded(self):
        """Achievements whose ids are in already_earned are skipped."""
        stats = {"episodes": 1000, "wins": 5}
        all_earned = check_achievements(AGENT_ACHIEVEMENTS, stats)
        all_ids = {a.id for a in all_earned}

        # Mark some as already earned and re-check.
        subset = check_achievements(AGENT_ACHIEVEMENTS, stats,
                                    already_earned=all_ids)
        assert subset == []
