"""Checkers adapter for the plugin registry."""
from typing import List, Tuple

try:
    import gymnasium as gym
except ImportError:
    import gym  # Legacy fallback

from games.base_adapter import BaseGameAdapter
from games.reward_config import (
    StandardMetrics,
    RewardConfig,
    ActionSpaceInfo,
)
from games.builtin.checkers.game import CheckersEnv, NUM_SQUARES


class CheckersAdapter(BaseGameAdapter):
    """Adapter for the built-in Checkers game."""

    @property
    def name(self) -> str:
        return 'Checkers'

    @property
    def game_id(self) -> str:
        return 'checkers'

    @property
    def category(self) -> str:
        return 'board'

    @property
    def description(self) -> str:
        return 'Classic checkers — capture all opponent pieces to win.'

    def create_env(self, opponent=None, **kwargs) -> gym.Env:
        return CheckersEnv(opponent=opponent, **kwargs)

    def get_action_space_info(self) -> ActionSpaceInfo:
        return ActionSpaceInfo(
            num_actions=NUM_SQUARES * NUM_SQUARES,
            action_labels=[f'{i}->{j}' for i in range(NUM_SQUARES)
                           for j in range(NUM_SQUARES)],
        )

    def get_observation_shape(self) -> Tuple[int, ...]:
        return (84, 84, 1)

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        p1 = info.get('p1_pieces', 0)
        p2 = info.get('p2_pieces', 0)
        total = p1 + p2 if (p1 + p2) > 0 else 1
        return StandardMetrics(
            progress=1.0 - (p2 / 12),
            score=float(1 if info.get('winner') == 1 else 0),
            completed=info.get('winner', 0) == 1,
            time_elapsed=episode_time,
        )

    def supported_algorithms(self) -> List[str]:
        return ['ppo', 'dqn', 'a2c', 'rainbow']

    def get_reward_config(self) -> RewardConfig:
        return RewardConfig(
            time_penalty_per_second=0.0,
            completion_bonus=50.0,
            death_penalty=-10.0,
            idle_penalty_per_second=0.0,
            speed_bonus_multiplier=0.0,
            par_time_seconds=600.0,
        )

    def get_dashboard_config(self) -> dict:
        return {
            'graph_2_title': 'Win Rate',
            'graph_2_metric': 'win_rate',
            'graph_2_info_key': 'winner',
            'graph_2_secondary_metric': 'opponent_win_rate',
            'graph_2_legend': ['Agent (Dark)', 'Opponent (Red)'],
            'status_metric_label': 'Win Rate',
            'status_metric_key': 'win_rate',
        }

    def get_completion_criteria(self) -> dict:
        return {
            'metric': 'win_rate',
            'threshold': 0.8,
            'window': 100,
            'description': 'Win rate > 80% over 100 games',
        }
