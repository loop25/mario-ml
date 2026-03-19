"""Snake game adapter for the plugin registry."""
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
from games.builtin.snake.game import SnakeEnv


class SnakeAdapter(BaseGameAdapter):
    """Adapter for the built-in Snake game."""

    @property
    def name(self) -> str:
        return 'Snake'

    @property
    def game_id(self) -> str:
        return 'snake'

    @property
    def category(self) -> str:
        return 'arcade'

    @property
    def description(self) -> str:
        return 'Classic snake game — eat food, grow longer, avoid walls.'

    def create_env(self, **kwargs) -> gym.Env:
        grid_size = kwargs.get('grid_size', 16)
        return SnakeEnv(grid_size=grid_size)

    def get_action_space_info(self) -> ActionSpaceInfo:
        return ActionSpaceInfo(
            num_actions=4,
            action_labels=['Up', 'Right', 'Down', 'Left'],
        )

    def get_observation_shape(self) -> Tuple[int, ...]:
        return (84, 84, 1)

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        snake_len = info.get('snake_length', 1)
        max_len = info.get('max_length', 256)
        return StandardMetrics(
            progress=snake_len / max_len,
            score=float(info.get('score', 0)),
            completed=(snake_len >= max_len),
            time_elapsed=episode_time,
        )

    def get_reward_config(self) -> RewardConfig:
        return RewardConfig(
            time_penalty_per_second=0.005,
            completion_bonus=100.0,
            death_penalty=-10.0,
            idle_penalty_per_second=0.003,
            speed_bonus_multiplier=1.0,
            par_time_seconds=60.0,
        )

    def supported_algorithms(self) -> List[str]:
        # NEAT requires small flat obs (13x13); Snake uses 84x84 images.
        return ['ppo', 'dqn', 'a2c', 'rainbow', 'dt']

    def get_dashboard_config(self) -> dict:
        return {
            'graph_2_title': 'Score (food eaten)',
            'graph_2_metric': 'score',
            'graph_2_info_key': 'score',
            'status_metric_label': 'Score',
            'status_metric_key': 'score',
        }

    def get_completion_criteria(self) -> dict:
        return {
            'metric': 'score',
            'threshold': 50.0,
            'window': 50,
            'description': 'Avg score > 50 over 50 episodes',
        }

    def get_game_specific_options(self) -> dict:
        return {
            'grid_size': (int, 16, 'Grid size (8, 16, or 32)'),
        }
