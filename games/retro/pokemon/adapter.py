"""
Pokemon Red adapter for stable-retro.

Wraps the Game Boy Pokemon Red ROM via stable-retro. Requires:
  1. pip install stable-retro
  2. A legally obtained Pokemon Red ROM imported with:
     python -m retro.import /path/to/rom/directory

Pokemon is fundamentally different from platformers — it's an RPG
focused on exploration, collecting pokemon, and earning gym badges.
The reward signal emphasizes exploration (new map tiles visited)
rather than speed.

Metrics map:
  - badges (RAM): Number of gym badges earned → progress/score
  - pokemon_count (RAM): Pokemon in party → secondary metric
  - map_id + x/y (RAM): Current location → exploration tracking
"""
from typing import List, Tuple

from games.retro.base_retro_adapter import BaseRetroAdapter
from games.reward_config import (
    StandardMetrics,
    RewardConfig,
    ActionSpaceInfo,
)


# Game Boy: A, B, SELECT, START, UP, DOWN, LEFT, RIGHT
_POKEMON_NUM_ACTIONS = 8

_POKEMON_ACTION_LABELS = [
    'A', 'B', 'Select', 'Start',
    'Up', 'Down', 'Left', 'Right',
]

# Total badges in Pokemon Red
_TOTAL_BADGES = 8


class PokemonAdapter(BaseRetroAdapter):
    """Adapter for Pokemon Red (Game Boy) via stable-retro."""

    @property
    def name(self) -> str:
        return 'Pokemon Red'

    @property
    def game_id(self) -> str:
        return 'pokemon'

    @property
    def category(self) -> str:
        return 'rpg'

    @property
    def description(self) -> str:
        return 'Classic Game Boy RPG — explore, catch pokemon, and earn gym badges.'

    @property
    def rom_name(self) -> str:
        return 'PokemonRed-GameBoy'

    @property
    def frame_skip(self) -> int:
        # RPGs need less frame skip — menus and text matter
        return 1

    def get_action_space_info(self) -> ActionSpaceInfo:
        return ActionSpaceInfo(
            num_actions=_POKEMON_NUM_ACTIONS,
            action_labels=list(_POKEMON_ACTION_LABELS),
        )

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        badges = info.get('badges', 0)
        # Count bits set in badges byte (each bit = one badge)
        badge_count = bin(badges).count('1') if isinstance(badges, int) else 0
        progress = badge_count / _TOTAL_BADGES
        return StandardMetrics(
            progress=progress,
            score=float(badge_count),
            completed=badge_count >= _TOTAL_BADGES,
            time_elapsed=episode_time,
        )

    def get_reward_config(self) -> RewardConfig:
        return RewardConfig(
            time_penalty_per_second=0.0,    # RPG — no time pressure
            completion_bonus=200.0,          # Huge bonus for badges
            death_penalty=-5.0,              # Fainting is mild
            idle_penalty_per_second=0.001,   # Very gentle idle nudge
            speed_bonus_multiplier=0.0,      # No speed bonus for RPG
            par_time_seconds=3600.0,         # 1 hour per badge (generous)
        )

    def get_dashboard_config(self) -> dict:
        return {
            'graph_2_title': 'Badges Earned',
            'graph_2_metric': 'badges',
            'graph_2_info_key': 'badges',
            'status_metric_label': 'Badges',
            'status_metric_key': 'badges',
        }

    def get_completion_criteria(self) -> dict:
        return {
            'metric': 'badges',
            'threshold': 1,
            'window': 10,
            'description': 'Earn at least 1 gym badge',
        }

    def get_game_specific_options(self) -> dict:
        return {}
