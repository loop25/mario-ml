"""
Achievement definitions for the Game AI Training Studio.

Two tiers of achievements:
- **Agent achievements** track per-model milestones (episodes, wins, streaks).
- **Trainer achievements** track global user progression (sessions, games explored).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set


@dataclass
class Achievement:
    """A single achievement definition."""

    id: str
    name: str
    description: str
    condition: Callable[[dict], bool]
    icon: str = ""


# ---------------------------------------------------------------------------
# Agent achievements — earned per trained model
# ---------------------------------------------------------------------------

AGENT_ACHIEVEMENTS: List[Achievement] = [
    Achievement(
        id="first_steps",
        name="First Steps",
        description="Complete your first episode.",
        condition=lambda s: s.get("episodes", 0) >= 1,
        icon="👣",
    ),
    Achievement(
        id="century",
        name="Century",
        description="Complete 100 episodes.",
        condition=lambda s: s.get("episodes", 0) >= 100,
        icon="💯",
    ),
    Achievement(
        id="millennium",
        name="Millennium",
        description="Complete 1,000 episodes.",
        condition=lambda s: s.get("episodes", 0) >= 1000,
        icon="🏛️",
    ),
    Achievement(
        id="marathon",
        name="Marathon",
        description="Complete 10,000 episodes.",
        condition=lambda s: s.get("episodes", 0) >= 10000,
        icon="🏃",
    ),
    Achievement(
        id="first_win",
        name="First Win",
        description="Win your first game.",
        condition=lambda s: s.get("wins", 0) >= 1,
        icon="🏆",
    ),
    Achievement(
        id="streak_5",
        name="On Fire",
        description="Achieve a 5-game win streak.",
        condition=lambda s: s.get("win_streak", 0) >= 5,
        icon="🔥",
    ),
    Achievement(
        id="streak_10",
        name="Unstoppable",
        description="Achieve a 10-game win streak.",
        condition=lambda s: s.get("win_streak", 0) >= 10,
        icon="⚡",
    ),
    Achievement(
        id="streak_25",
        name="Legendary Streak",
        description="Achieve a 25-game win streak.",
        condition=lambda s: s.get("win_streak", 0) >= 25,
        icon="👑",
    ),
    Achievement(
        id="speed_demon",
        name="Speed Demon",
        description="Finish a game under par time.",
        condition=lambda s: bool(s.get("under_par", False)),
        icon="⏱️",
    ),
    Achievement(
        id="high_roller",
        name="High Roller",
        description="Achieve a reward ratio of 2.0 or higher.",
        condition=lambda s: s.get("reward_ratio", 0.0) >= 2.0,
        icon="🎰",
    ),
    Achievement(
        id="perfectionist",
        name="Perfectionist",
        description="Complete a perfect game with no mistakes.",
        condition=lambda s: bool(s.get("perfect_game", False)),
        icon="✨",
    ),
    Achievement(
        id="generalist",
        name="Generalist",
        description="Achieve positive scores in 3 or more games.",
        condition=lambda s: s.get("positive_games", 0) >= 3,
        icon="🎮",
    ),
]

# ---------------------------------------------------------------------------
# Trainer achievements — earned globally across all sessions
# ---------------------------------------------------------------------------

TRAINER_ACHIEVEMENTS: List[Achievement] = [
    Achievement(
        id="getting_started",
        name="Getting Started",
        description="Run your first training session.",
        condition=lambda s: s.get("total_sessions", 0) >= 1,
        icon="🚀",
    ),
    Achievement(
        id="algo_collector",
        name="Algorithm Collector",
        description="Train with all 6 supported algorithms.",
        condition=lambda s: len(s.get("algorithms_used", set())) >= 6,
        icon="🧪",
    ),
    Achievement(
        id="game_explorer",
        name="Game Explorer",
        description="Play at least 6 different games.",
        condition=lambda s: len(s.get("games_played", set())) >= 6,
        icon="🗺️",
    ),
    Achievement(
        id="night_owl",
        name="Night Owl",
        description="Run an overnight training session.",
        condition=lambda s: s.get("overnight_sessions", 0) >= 1,
        icon="🦉",
    ),
    Achievement(
        id="tournament_director",
        name="Tournament Director",
        description="Run a tournament.",
        condition=lambda s: s.get("tournaments_run", 0) >= 1,
        icon="🏟️",
    ),
    Achievement(
        id="human_touch",
        name="Human Touch",
        description="Play a game yourself against an AI agent.",
        condition=lambda s: s.get("human_games", 0) >= 1,
        icon="🕹️",
    ),
    Achievement(
        id="data_scientist",
        name="Data Scientist",
        description="Collect 1,000 training trajectories.",
        condition=lambda s: s.get("total_trajectories", 0) >= 1000,
        icon="📊",
    ),
    Achievement(
        id="streamer",
        name="Streamer",
        description="Stream a training session.",
        condition=lambda s: s.get("streamed_sessions", 0) >= 1,
        icon="📡",
    ),
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def check_achievements(
    definitions: List[Achievement],
    stats: dict,
    already_earned: Optional[Set[str]] = None,
) -> List[Achievement]:
    """Return achievements from *definitions* that are newly earned.

    Parameters
    ----------
    definitions:
        The list of Achievement objects to evaluate.
    stats:
        A dict of current statistics (keys depend on context).
    already_earned:
        Optional set of achievement ids to skip.

    Returns
    -------
    List of Achievement objects whose conditions are met and whose ids are
    not in *already_earned*.
    """
    earned = already_earned or set()
    newly_earned: List[Achievement] = []
    for ach in definitions:
        if ach.id in earned:
            continue
        try:
            if ach.condition(stats):
                newly_earned.append(ach)
        except Exception:
            # Malformed stats should never crash the system.
            pass
    return newly_earned
