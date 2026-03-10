"""Connect Four adapter for the plugin registry."""
from typing import List, Tuple

import gym

from games.base_adapter import BaseGameAdapter
from games.reward_config import (
    StandardMetrics,
    RewardConfig,
    ActionSpaceInfo,
)
from games.builtin.connect4.game import ConnectFourEnv


class ConnectFourAdapter(BaseGameAdapter):
    """Adapter for the built-in Connect Four game."""

    @property
    def name(self) -> str:
        return 'Connect Four'

    @property
    def game_id(self) -> str:
        return 'connect4'

    @property
    def category(self) -> str:
        return 'board'

    @property
    def description(self) -> str:
        return 'Drop pieces to get four in a row — vertical, horizontal, or diagonal.'

    def create_env(self, **kwargs) -> gym.Env:
        return ConnectFourEnv()

    def get_action_space_info(self) -> ActionSpaceInfo:
        return ActionSpaceInfo(
            num_actions=7,
            action_labels=[f'Col {i+1}' for i in range(7)],
        )

    def get_observation_shape(self) -> Tuple[int, ...]:
        return (84, 84, 1)

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        return StandardMetrics(
            progress=info.get('pieces_played', 0) / 42,
            score=float(1 if info.get('winner') == 1 else 0),
            completed=info.get('winner', 0) == 1,
            time_elapsed=episode_time,
        )

    def supported_algorithms(self) -> List[str]:
        # NEAT requires small flat obs (13x13); Connect Four uses 84x84 images.
        return ['ppo', 'dqn', 'a2c']

    def get_reward_config(self) -> RewardConfig:
        return RewardConfig(
            time_penalty_per_second=0.0,   # Board game — thinking is fine
            completion_bonus=50.0,
            death_penalty=-10.0,
            idle_penalty_per_second=0.0,
            speed_bonus_multiplier=0.0,
            par_time_seconds=300.0,
        )

    def get_dashboard_config(self) -> dict:
        return {
            'graph_2_title': 'Win Rate',
            'graph_2_metric': 'win_rate',
            'graph_2_info_key': 'winner',
            'graph_2_secondary_metric': 'opponent_win_rate',
            'graph_2_legend': ['Agent (Red)', 'Opponent (Yellow)'],
            'status_metric_label': 'Win Rate',
            'status_metric_key': 'win_rate',
        }

    def get_completion_criteria(self) -> dict:
        return {
            'metric': 'win_rate',
            'threshold': 0.9,
            'window': 100,
            'description': 'Win rate > 90% over 100 games',
        }
