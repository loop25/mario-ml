"""
Sonic the Hedgehog adapter for stable-retro.

Wraps the Sega Genesis Sonic ROM via stable-retro. Requires:
  1. pip install stable-retro
  2. A legally obtained Sonic ROM imported with:
     python -m retro.import /path/to/rom/directory

The agent learns to navigate Sonic through zones/acts, collecting
rings and reaching the end-of-level signpost.

Metrics map:
  - x (RAM): Sonic's horizontal position → progress
  - rings (RAM): Ring count → score
  - level_end_bonus (RAM): Level completion flag → completed
"""
from typing import List, Tuple

from games.retro.base_retro_adapter import BaseRetroAdapter
from games.reward_config import (
    StandardMetrics,
    RewardConfig,
    ActionSpaceInfo,
)


# Sonic Genesis action buttons: B, A, MODE, START, UP, DOWN, LEFT, RIGHT
# stable-retro Discrete mode gives 2^8 = 256 actions, but many are filtered.
# The meaningful ones for Sonic: LEFT, RIGHT, DOWN (crouch/roll), B/A (jump).
# We use the retro-provided discrete action set.
_SONIC_NUM_ACTIONS = 12  # Filtered discrete set from stable-retro

_SONIC_ACTION_LABELS = [
    'NOOP', 'Left', 'Right', 'Down',
    'Jump', 'Left+Jump', 'Right+Jump', 'Down+Jump',
    'Left+Down', 'Right+Down', 'Left+Down+Jump', 'Right+Down+Jump',
]

# Approximate max x position for Green Hill Zone Act 1
_SONIC_MAX_X = 9600


class SonicAdapter(BaseRetroAdapter):
    """Adapter for Sonic the Hedgehog (Genesis) via stable-retro."""

    @property
    def name(self) -> str:
        return 'Sonic the Hedgehog'

    @property
    def game_id(self) -> str:
        return 'sonic'

    @property
    def category(self) -> str:
        return 'platformer'

    @property
    def description(self) -> str:
        return 'Sega Genesis classic — race through zones at supersonic speed.'

    @property
    def rom_name(self) -> str:
        return 'SonicTheHedgehog-Genesis'

    @property
    def initial_state(self) -> str:
        return 'GreenHillZone.Act1'

    def get_action_space_info(self) -> ActionSpaceInfo:
        return ActionSpaceInfo(
            num_actions=_SONIC_NUM_ACTIONS,
            action_labels=list(_SONIC_ACTION_LABELS),
        )

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        x_pos = info.get('x', 0)
        progress = min(1.0, x_pos / _SONIC_MAX_X)
        return StandardMetrics(
            progress=progress,
            score=float(info.get('rings', 0)),
            completed=bool(info.get('level_end_bonus', 0) > 0),
            time_elapsed=episode_time,
        )

    def get_reward_config(self) -> RewardConfig:
        return RewardConfig(
            time_penalty_per_second=0.01,
            completion_bonus=100.0,
            death_penalty=-15.0,
            idle_penalty_per_second=0.005,
            speed_bonus_multiplier=2.0,  # Sonic is all about speed
            par_time_seconds=90.0,       # Faster par time than Mario
        )

    def get_dashboard_config(self) -> dict:
        return {
            'graph_2_title': 'Distance (x position)',
            'graph_2_metric': 'distance',
            'graph_2_info_key': 'x',
            'status_metric_label': 'Distance',
            'status_metric_key': 'distance',
        }

    def get_completion_criteria(self) -> dict:
        return {
            'metric': 'reward',
            'threshold': 500.0,
            'window': 50,
            'description': 'Avg reward > 500 over 50 episodes',
        }

    def get_game_specific_options(self) -> dict:
        return {
            'state': (str, 'GreenHillZone.Act1', 'Starting zone/act'),
        }
