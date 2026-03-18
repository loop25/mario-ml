"""Tests for the AchievementManager."""
import json
import os
import tempfile

import pytest

from src.achievements.event_bus import EventBus
from src.achievements.achievement_manager import AchievementManager


@pytest.fixture
def tmp_save_path(tmp_path):
    """Return a temporary JSON save path."""
    return str(tmp_path / "achievements.json")


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def manager(bus, tmp_save_path):
    return AchievementManager(bus, save_path=tmp_save_path)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestInitialState:
    def test_initial_state_empty(self, manager):
        """A fresh manager has no earned achievements and empty trainer stats."""
        assert manager.get_trainer_earned() == []
        assert manager.get_agent_earned("any_agent") == []
        stats = manager.get_trainer_stats()
        assert stats["total_sessions"] == 0
        assert len(stats["algorithms_used"]) == 0
        assert len(stats["games_played"]) == 0


class TestEpisodeComplete:
    def test_episode_complete_awards_first_steps(self, bus, manager):
        """Publishing episode_complete with episode=1 should earn first_steps."""
        manager.set_active_agent("agent_1")
        bus.publish({"type": "episode_complete", "reward": 5.0, "won": False})
        earned = manager.get_agent_earned("agent_1")
        assert "first_steps" in earned


class TestPersistence:
    def test_persistence_survives_reload(self, bus, tmp_save_path):
        """Achievements survive a save/load cycle."""
        mgr1 = AchievementManager(bus, save_path=tmp_save_path)
        mgr1.set_active_agent("agent_1")
        bus.publish({"type": "episode_complete", "reward": 1.0, "won": True})
        assert "first_steps" in mgr1.get_agent_earned("agent_1")
        mgr1.save()

        # Create a fresh manager from the same file (new bus to avoid double subscribe)
        bus2 = EventBus()
        mgr2 = AchievementManager(bus2, save_path=tmp_save_path)
        assert "first_steps" in mgr2.get_agent_earned("agent_1")

    def test_corrupted_json_starts_fresh(self, tmp_save_path):
        """A corrupted save file should not crash; manager starts fresh."""
        os.makedirs(os.path.dirname(tmp_save_path) or ".", exist_ok=True)
        with open(tmp_save_path, "w") as f:
            f.write("{corrupted json!@#$")
        bus = EventBus()
        mgr = AchievementManager(bus, save_path=tmp_save_path)
        assert mgr.get_trainer_earned() == []


class TestDuplicates:
    def test_duplicate_not_awarded(self, bus, manager):
        """The same achievement is not published twice."""
        events_received = []
        bus.subscribe("achievement_earned", lambda e: events_received.append(e))

        manager.set_active_agent("agent_1")
        # First episode — should earn first_steps
        bus.publish({"type": "episode_complete", "reward": 1.0, "won": False})
        # Second episode — should NOT re-earn first_steps
        bus.publish({"type": "episode_complete", "reward": 1.0, "won": False})

        first_steps_events = [
            e for e in events_received if e["achievement_id"] == "first_steps"
        ]
        assert len(first_steps_events) == 1


class TestTrainerTracking:
    def test_trainer_session_tracking(self, manager):
        """record_session updates trainer stats correctly."""
        manager.record_session("snake", "ppo", False)
        manager.record_session("chess", "a2c", True)
        stats = manager.get_trainer_stats()
        assert stats["total_sessions"] == 2
        assert "ppo" in stats["algorithms_used"]
        assert "a2c" in stats["algorithms_used"]
        assert "snake" in stats["games_played"]
        assert "chess" in stats["games_played"]
        assert stats["streamed_sessions"] == 1

    def test_trainer_achievement_triggers(self, manager):
        """record_session earns getting_started after one session."""
        manager.record_session("snake", "ppo", False)
        earned = manager.get_trainer_earned()
        assert "getting_started" in earned


class TestBusIntegration:
    def test_new_achievements_published_to_bus(self, bus, manager):
        """When an achievement is earned, an achievement_earned event fires."""
        received = []
        bus.subscribe("achievement_earned", lambda e: received.append(e))

        manager.set_active_agent("agent_1")
        bus.publish({"type": "episode_complete", "reward": 1.0, "won": True})

        # Should have first_steps and first_win
        ids = {e["achievement_id"] for e in received}
        assert "first_steps" in ids
        assert "first_win" in ids

        # Verify event shape
        for event in received:
            assert event["type"] == "achievement_earned"
            assert event["level"] in ("agent", "trainer")
            assert "achievement_name" in event
            assert "description" in event
