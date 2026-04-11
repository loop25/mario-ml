"""
Achievement Manager — glue between EventBus, achievement definitions, and persistence.

Subscribes to training events, tracks per-agent and global trainer stats,
checks achievement conditions, awards new achievements, persists to disk,
and publishes ``achievement_earned`` events back to the bus.
"""
import json
import os
from typing import Dict, List, Optional, Set

from src.achievements.definitions import (
    AGENT_ACHIEVEMENTS,
    TRAINER_ACHIEVEMENTS,
    check_achievements,
)
from src.achievements.event_bus import EventBus


class AchievementManager:
    """Tracks stats, awards achievements, and persists progress to JSON."""

    def __init__(self, event_bus: EventBus, save_path: str = "config/achievements.json"):
        self._bus = event_bus
        self._save_path = save_path
        self._active_agent: Optional[str] = None

        # Per-agent stats keyed by agent_id
        self._agent_stats: Dict[str, dict] = {}
        # Per-agent earned achievement ids
        self._agent_earned: Dict[str, Set[str]] = {}

        # Global trainer stats
        self._trainer_stats: dict = {
            "total_sessions": 0,
            "algorithms_used": set(),
            "games_played": set(),
            "overnight_sessions": 0,
            "tournaments_run": 0,
            "human_games": 0,
            "total_trajectories": 0,
            "streamed_sessions": 0,
        }
        self._trainer_earned: Set[str] = set()

        # Restore previous state
        self._load()

        # Subscribe to events
        self._bus.subscribe("episode_complete", self._on_episode_complete)
        self._bus.subscribe("session_complete", self._on_session_complete)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_active_agent(self, agent_id: str) -> None:
        """Set the agent whose stats are updated on episode events."""
        self._active_agent = agent_id
        if agent_id not in self._agent_stats:
            self._agent_stats[agent_id] = self._default_agent_stats()
        if agent_id not in self._agent_earned:
            self._agent_earned[agent_id] = set()

    def get_agent_earned(self, agent_id: str) -> List[str]:
        """Return list of earned achievement IDs for *agent_id*."""
        return list(self._agent_earned.get(agent_id, set()))

    def get_trainer_earned(self) -> List[str]:
        """Return list of earned trainer achievement IDs."""
        return list(self._trainer_earned)

    def get_trainer_stats(self) -> dict:
        """Return a copy of the current trainer stats."""
        return dict(self._trainer_stats)

    def record_session(self, game_id: str, algorithm: str, streamed: bool) -> None:
        """Record a training session for trainer-level achievements."""
        self._trainer_stats["total_sessions"] += 1
        self._trainer_stats["algorithms_used"].add(algorithm)
        self._trainer_stats["games_played"].add(game_id)
        if streamed:
            self._trainer_stats["streamed_sessions"] += 1
        self._check_trainer_achievements()
        self.save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist all state to the JSON save file."""
        data = {
            "agent_stats": self._agent_stats,
            "agent_earned": {
                aid: list(ids) for aid, ids in self._agent_earned.items()
            },
            "trainer_stats": self._serialize_trainer_stats(),
            "trainer_earned": list(self._trainer_earned),
        }
        os.makedirs(os.path.dirname(self._save_path) or ".", exist_ok=True)
        with open(self._save_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Restore state from JSON. Corrupted files are silently ignored."""
        if not os.path.exists(self._save_path):
            return
        try:
            with open(self._save_path, "r") as f:
                data = json.load(f)
            self._agent_stats = data.get("agent_stats", {})
            self._agent_earned = {
                aid: set(ids) for aid, ids in data.get("agent_earned", {}).items()
            }
            raw_trainer = data.get("trainer_stats", {})
            self._trainer_stats = {
                "total_sessions": raw_trainer.get("total_sessions", 0),
                "algorithms_used": set(raw_trainer.get("algorithms_used", [])),
                "games_played": set(raw_trainer.get("games_played", [])),
                "overnight_sessions": raw_trainer.get("overnight_sessions", 0),
                "tournaments_run": raw_trainer.get("tournaments_run", 0),
                "human_games": raw_trainer.get("human_games", 0),
                "total_trajectories": raw_trainer.get("total_trajectories", 0),
                "streamed_sessions": raw_trainer.get("streamed_sessions", 0),
            }
            self._trainer_earned = set(data.get("trainer_earned", []))
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            # Corrupted file — start fresh.
            pass

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_episode_complete(self, event: dict) -> None:
        """Handle an episode_complete event and update agent stats."""
        agent_id = self._active_agent
        if agent_id is None:
            return

        if agent_id not in self._agent_stats:
            self._agent_stats[agent_id] = self._default_agent_stats()
        if agent_id not in self._agent_earned:
            self._agent_earned[agent_id] = set()

        stats = self._agent_stats[agent_id]
        stats["episodes"] = stats.get("episodes", 0) + 1

        reward = event.get("reward", 0)
        stats["total_reward"] = stats.get("total_reward", 0) + reward
        stats["best_reward"] = max(stats.get("best_reward", float("-inf")), reward)

        won = event.get("won", False)
        if won:
            stats["wins"] = stats.get("wins", 0) + 1
            stats["win_streak"] = stats.get("win_streak", 0) + 1
        else:
            stats["losses"] = stats.get("losses", 0) + 1
            stats["win_streak"] = 0

        total_games = stats.get("wins", 0) + stats.get("losses", 0)
        if total_games > 0:
            stats["reward_ratio"] = stats["total_reward"] / total_games

        # Copy transient event fields the conditions might inspect
        for key in ("under_par", "perfect_game", "positive_games"):
            if key in event:
                stats[key] = event[key]

        self._check_agent_achievements(agent_id)
        self.save()

    def _on_session_complete(self, event: dict) -> None:
        """Handle a session_complete event and update trainer stats."""
        game_id = event.get("game_id", "")
        algorithm = event.get("algorithm", "")
        streamed = event.get("streamed", False)
        self.record_session(game_id, algorithm, streamed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_agent_achievements(self, agent_id: str) -> None:
        stats = self._agent_stats.get(agent_id, {})
        earned = self._agent_earned.get(agent_id, set())
        newly = check_achievements(AGENT_ACHIEVEMENTS, stats, already_earned=earned)
        for ach in newly:
            earned.add(ach.id)
            self._agent_earned[agent_id] = earned
            self._bus.publish({
                "type": "achievement_earned",
                "level": "agent",
                "achievement_id": ach.id,
                "achievement_name": ach.name,
                "description": ach.description,
            })

    def _check_trainer_achievements(self) -> None:
        newly = check_achievements(
            TRAINER_ACHIEVEMENTS, self._trainer_stats, already_earned=self._trainer_earned
        )
        for ach in newly:
            self._trainer_earned.add(ach.id)
            self._bus.publish({
                "type": "achievement_earned",
                "level": "trainer",
                "achievement_id": ach.id,
                "achievement_name": ach.name,
                "description": ach.description,
            })

    def _serialize_trainer_stats(self) -> dict:
        """Convert trainer stats for JSON (sets → lists)."""
        return {
            k: list(v) if isinstance(v, set) else v
            for k, v in self._trainer_stats.items()
        }

    @staticmethod
    def _default_agent_stats() -> dict:
        return {
            "episodes": 0,
            "wins": 0,
            "losses": 0,
            "win_streak": 0,
            "best_reward": 0,
            "total_reward": 0,
            "reward_ratio": 0.0,
        }
