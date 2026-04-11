"""
Pokemon Red adapter using PyBoy emulator.

Uses the PyBoy Game Boy emulator to run Pokemon Red with a rich
exploration-based reward signal. Does NOT require stable-retro.

Requirements:
  1. pip install pyboy
  2. A legally obtained Pokemon Red ROM file (PokemonRed.gb)
  3. A PyBoy save state file (init.state) to skip the game intro

The environment reads Game Boy RAM directly to extract game state
(player position, HP, badges, event flags) and rewards exploration
of new map tiles, earning badges, and healing at Pokemon Centers.

Adapted from PokemonRedExperiments v2 (MIT License):
  https://github.com/PWhiddy/PokemonRedExperiments
"""

import os
from typing import List, Tuple

try:
    import gymnasium as gym
except ImportError:
    import gym  # Legacy fallback

from games.retro.base_pyboy_adapter import BasePyBoyAdapter
from games.reward_config import (
    StandardMetrics,
    RewardConfig,
    ActionSpaceInfo,
)

# Total badges in Pokemon Red
_TOTAL_BADGES = 8


class PokemonAdapter(BasePyBoyAdapter):
    """Adapter for Pokemon Red via PyBoy emulator."""

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
        return 'Classic Game Boy RPG — explore Kanto, catch pokemon, earn gym badges. (Requires ROM)'

    def create_env(self, **kwargs) -> gym.Env:
        """Create the Pokemon Red environment.

        Kwargs:
            gb_path: Path to PokemonRed.gb ROM file.
            init_state: Path to init.state (PyBoy save state).
            headless: Run without display. Default True.
            simple_obs: Use image-only observations. Default True.
            action_freq: Emulator ticks per action. Default 24.
            max_steps: Steps per episode. Default 2048*80.
            explore_weight: Exploration reward weight. Default 1.0.
            reward_scale: Global reward scale. Default 0.5.
        """
        from games.retro.pokemon.red_gym_env import RedGymEnv
        return RedGymEnv(config=kwargs)

    def get_action_space_info(self) -> ActionSpaceInfo:
        return ActionSpaceInfo(
            num_actions=7,
            action_labels=['Down', 'Left', 'Right', 'Up', 'A', 'B', 'Start'],
        )

    def get_observation_shape(self) -> Tuple[int, ...]:
        # Default simple_obs mode: stacked grayscale frames
        return (72, 80, 3)

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        badges = info.get('badges', 0)
        # badges is already a count (not a bitmask) from our env
        badge_count = badges if isinstance(badges, int) else 0
        progress = badge_count / _TOTAL_BADGES
        return StandardMetrics(
            progress=progress,
            score=float(badge_count),
            completed=badge_count >= _TOTAL_BADGES,
            time_elapsed=episode_time,
        )

    def get_reward_config(self) -> RewardConfig:
        # The environment handles its own reward shaping internally,
        # so we set minimal framework-level shaping.
        return RewardConfig(
            time_penalty_per_second=0.0,
            completion_bonus=0.0,      # Env already rewards badges
            death_penalty=0.0,         # Env tracks deaths internally
            idle_penalty_per_second=0.0,
            speed_bonus_multiplier=0.0,
            par_time_seconds=3600.0,
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
        return {
            'gb_path': (str, 'PokemonRed.gb', 'Path to Pokemon Red ROM file'),
            'simple_obs': (bool, True, 'Use simple image observations (CNN compatible)'),
            'explore_weight': (float, 1.0, 'Exploration reward weight'),
        }

    # supported_algorithms, needs_sb3_compat, is_available, and
    # get_human_render_frame are inherited from BasePyBoyAdapter.
