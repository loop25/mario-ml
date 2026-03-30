"""Dual Snake cooperative game adapter for the plugin registry."""
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
from games.builtin.dual_snake.game import DualSnakeEnv


class DualSnakeAdapter(BaseGameAdapter):
    """Adapter for the built-in cooperative Dual Snake game."""

    @property
    def name(self) -> str:
        return 'Dual Snake'

    @property
    def game_id(self) -> str:
        return 'dual_snake'

    @property
    def category(self) -> str:
        return 'cooperative'

    @property
    def description(self) -> str:
        return 'Two snakes cooperate on a shared board. Eat food together, avoid each other!'

    def create_env(self, **kwargs) -> gym.Env:
        grid_size = kwargs.get('grid_size', 16)
        return DualSnakeEnv(grid_size=grid_size)

    def get_action_space_info(self) -> ActionSpaceInfo:
        return ActionSpaceInfo(
            num_actions=16,
            action_labels=[
                f'{d1}+{d2}'
                for d1 in ['U', 'R', 'D', 'L']
                for d2 in ['U', 'R', 'D', 'L']
            ],
        )

    def get_observation_shape(self) -> Tuple[int, ...]:
        return (84, 84, 3)

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        s1_len = info.get('snake1_length', 1)
        s2_len = info.get('snake2_length', 1)
        max_len = info.get('max_length', 256)
        total_len = s1_len + s2_len
        score = float(info.get('score', 0))
        return StandardMetrics(
            progress=total_len / max_len,
            score=score,
            completed=(score >= 30),  # High cooperative score = completed
            time_elapsed=episode_time,
        )

    def get_reward_config(self) -> RewardConfig:
        return RewardConfig(
            time_penalty_per_second=0.001,
            completion_bonus=100.0,
            death_penalty=-2.0,
            idle_penalty_per_second=0.001,
            speed_bonus_multiplier=1.0,
            par_time_seconds=120.0,
        )

    def get_training_hints(self) -> dict:
        return {
            'ent_coef': 0.03,
            'gamma': 0.995,
        }

    def supported_algorithms(self) -> List[str]:
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
            'threshold': 30.0,
            'window': 50,
            'description': 'Avg score > 30 over 50 episodes',
        }

    def get_game_specific_options(self) -> dict:
        return {
            'grid_size': (int, 16, 'Grid size (8, 16, or 32)'),
        }
