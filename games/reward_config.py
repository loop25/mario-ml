"""
Shared dataclasses for the game adapter system.

These define the universal contracts between game adapters and the framework:
- StandardMetrics: what every game reports after each step
- RewardConfig: how the framework shapes rewards per game
- ActionSpaceInfo: describes the action space for the launcher/algorithms
- TokenConfig: how to tokenize obs/actions for the Decision Transformer
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class StandardMetrics:
    """Universal metrics that every game maps to."""
    progress: float        # 0.0 to 1.0, how far through the level/game
    score: float           # Game-specific score (coins, points, pieces)
    completed: bool        # Did the agent finish/win this episode?
    time_elapsed: float    # Seconds spent this episode


@dataclass
class RewardConfig:
    """Reward shaping parameters. The framework applies these automatically."""
    time_penalty_per_second: float = 0.01
    completion_bonus: float = 100.0
    death_penalty: float = -15.0
    idle_penalty_per_second: float = 0.005
    speed_bonus_multiplier: float = 1.5
    par_time_seconds: float = 120.0


@dataclass
class ActionSpaceInfo:
    """Describes the game's action space for the launcher and algorithms."""
    num_actions: int
    action_labels: List[str]


@dataclass
class TokenConfig:
    """How to tokenize this game for the Decision Transformer."""
    obs_resolution: Tuple[int, int]
    obs_channels: int
    action_vocab_size: int
    game_token_id: int
