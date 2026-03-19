"""Agent Personality Profile system.

Generates personality cards for trained agents based on their training
metrics.  Each profile captures core stats, five personality dimensions
(0-100 scale), a play-style classification, and earned achievement badges.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class AgentProfile:
    """Personality card for a single trained agent."""

    agent_id: str  # e.g. 'snake_ppo'
    game_id: str
    algorithm: str
    display_name: str  # Generated or user-set

    # Core stats
    total_episodes: int
    best_reward: float
    avg_reward: float
    training_time_hours: float

    # Personality dimensions (0-100 scale)
    consistency: int
    exploration: int
    speed: int
    resilience: int
    peak_performance: int

    # Classification
    play_style: str  # e.g. 'Aggressive', 'Cautious', 'Balanced', ...

    # Achievement badges earned
    badges: List[str] = field(default_factory=list)

    # Metadata
    created: str = ""
    checkpoint_path: str = ""


def _parse_elapsed(elapsed_str: str) -> float:
    """Convert an elapsed-time string to hours.

    Supported formats:
      - ``H:MM:SS``  (e.g. ``1:30:00``)
      - ``Xm Ys``    (e.g. ``1m 17s``)
      - ``Xh Ym``    (e.g. ``2h 5m``)
    Returns 0.0 on failure.
    """
    if not elapsed_str:
        return 0.0
    s = elapsed_str.strip()

    # H:MM:SS
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) + int(parts[1]) / 60 + int(parts[2]) / 3600
            if len(parts) == 2:
                return int(parts[0]) / 60 + int(parts[1]) / 3600
        except ValueError:
            return 0.0

    # Xm Ys / Xh Ym etc.
    total_seconds = 0.0
    import re

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*([hms])", s, re.IGNORECASE):
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "h":
            total_seconds += value * 3600
        elif unit == "m":
            total_seconds += value * 60
        else:
            total_seconds += value
    return total_seconds / 3600 if total_seconds else 0.0


def _clamp(value: float, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, int(round(value))))


def generate_profile(metadata_path: str) -> AgentProfile:
    """Read *metadata.json* and compute an :class:`AgentProfile`.

    Parameters
    ----------
    metadata_path:
        Absolute or relative path to a ``metadata.json`` file produced by
        one of the trainers.

    Returns
    -------
    AgentProfile
        A fully-populated personality profile.
    """
    with open(metadata_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    game_id = meta.get("game_id", "unknown")
    algorithm = meta.get("algorithm", "unknown").lower().replace("trainer", "")
    episodes = int(meta.get("episode", 0))
    best_reward = float(meta.get("best_reward", 0))
    avg_reward = float(meta.get("avg_reward", 0))
    reward_std = float(meta.get("reward_std", 0))
    first_reward = float(meta.get("first_reward", avg_reward))

    # Elapsed time
    elapsed_str = meta.get("elapsed_time", "")
    elapsed_secs = float(meta.get("elapsed_seconds", 0))
    if elapsed_str:
        hours = _parse_elapsed(elapsed_str)
    elif elapsed_secs:
        hours = elapsed_secs / 3600
    else:
        hours = 0.0

    agent_id = f"{game_id}_{algorithm}"
    display_name = f"{game_id.title()} {algorithm.upper()} Agent"

    # ── Personality dimensions ────────────────────────────────────────
    # Consistency: low reward variance → high consistency
    consistency = _clamp(
        100 - min(100, reward_std / max(1, abs(avg_reward)) * 100)
    )

    # Exploration: action diversity if available, else neutral
    unique_ratio = meta.get("unique_actions_ratio")
    if unique_ratio is not None:
        exploration = _clamp(float(unique_ratio) * 100)
    else:
        exploration = 50

    # Speed: episodes per hour
    if hours > 0:
        eps_per_hour = episodes / hours
        speed = _clamp(eps_per_hour / 100 * 100)
    else:
        speed = 50

    # Resilience: improvement rate
    improvement = (best_reward - first_reward) / max(1, episodes)
    resilience = _clamp(50 + min(50, improvement * 10))

    # Peak performance
    peak_performance = _clamp(
        best_reward / max(1, abs(avg_reward)) * 50
    )

    # ── Play style ────────────────────────────────────────────────────
    if consistency >= 80:
        play_style = "Consistent"
    elif exploration >= 70:
        play_style = "Explorer"
    elif speed >= 80:
        play_style = "Speedrunner"
    elif resilience >= 70:
        play_style = "Resilient"
    elif peak_performance >= 80:
        play_style = "Peak Performer"
    else:
        play_style = "Balanced"

    # ── Badges (from achievement manager, if available) ───────────────
    badges: List[str] = []
    try:
        from src.achievements.achievement_manager import AchievementManager

        mgr = AchievementManager()
        for ach in mgr.unlocked:
            badges.append(ach)
    except Exception:
        pass

    checkpoint_dir = os.path.dirname(os.path.abspath(metadata_path))
    created = meta.get("timestamp", datetime.now().isoformat())

    return AgentProfile(
        agent_id=agent_id,
        game_id=game_id,
        algorithm=algorithm,
        display_name=display_name,
        total_episodes=episodes,
        best_reward=best_reward,
        avg_reward=avg_reward,
        training_time_hours=round(hours, 2),
        consistency=consistency,
        exploration=exploration,
        speed=speed,
        resilience=resilience,
        peak_performance=peak_performance,
        play_style=play_style,
        badges=badges,
        created=created,
        checkpoint_path=checkpoint_dir,
    )


def scan_all_profiles(models_dir: str = "models") -> List[AgentProfile]:
    """Walk *models_dir*, generate a profile for every ``metadata.json``.

    Returns a list sorted by ``best_reward`` descending.
    """
    profiles: List[AgentProfile] = []
    if not os.path.isdir(models_dir):
        return profiles

    for dirpath, _dirnames, filenames in os.walk(models_dir):
        if "metadata.json" in filenames:
            meta_path = os.path.join(dirpath, "metadata.json")
            try:
                profiles.append(generate_profile(meta_path))
            except Exception:
                continue

    profiles.sort(key=lambda p: p.best_reward, reverse=True)
    return profiles
