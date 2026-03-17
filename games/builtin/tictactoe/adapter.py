"""Tic-Tac-Toe adapter for the plugin registry."""
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
from games.builtin.tictactoe.game import TicTacToeEnv


class TicTacToeAdapter(BaseGameAdapter):
    """Adapter for the built-in Tic-Tac-Toe game."""

    @property
    def name(self) -> str:
        return 'Tic-Tac-Toe'

    @property
    def game_id(self) -> str:
        return 'tictactoe'

    @property
    def category(self) -> str:
        return 'board'

    @property
    def description(self) -> str:
        return 'Classic 3x3 grid — get three in a row to win.'

    def create_env(self, **kwargs) -> gym.Env:
        return TicTacToeEnv(**kwargs)

    def get_action_space_info(self) -> ActionSpaceInfo:
        labels = [f'({r},{c})' for r in range(3) for c in range(3)]
        return ActionSpaceInfo(
            num_actions=9,
            action_labels=labels,
        )

    def get_observation_shape(self) -> Tuple[int, ...]:
        return (84, 84, 1)

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        return StandardMetrics(
            progress=info.get('moves_played', 0) / 9,
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
            par_time_seconds=60.0,
        )

    def get_dashboard_config(self) -> dict:
        return {
            'graph_2_title': 'Win Rate',
            'graph_2_metric': 'win_rate',
            'graph_2_info_key': 'winner',
            'graph_2_secondary_metric': 'opponent_win_rate',
            'graph_2_legend': ['Agent (X)', 'Opponent (O)'],
            'status_metric_label': 'Win Rate',
            'status_metric_key': 'win_rate',
        }

    def get_completion_criteria(self) -> dict:
        return {
            'metric': 'win_rate',
            'threshold': 0.95,
            'window': 100,
            'description': 'Win rate > 95% over 100 games',
        }
