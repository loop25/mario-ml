"""Chess adapter for the plugin registry."""
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


class ChessAdapter(BaseGameAdapter):
    """Adapter for the built-in Chess game.

    Requires `python-chess` to be installed (pip install python-chess).
    """

    @property
    def name(self) -> str:
        return 'Chess'

    @property
    def game_id(self) -> str:
        return 'chess'

    @property
    def category(self) -> str:
        return 'board'

    @property
    def description(self) -> str:
        return 'Classic chess — checkmate your opponent to win.'

    def create_env(self, **kwargs) -> gym.Env:
        from games.builtin.chess.game import ChessEnv
        return ChessEnv(**kwargs)

    def get_action_space_info(self) -> ActionSpaceInfo:
        from games.builtin.chess.game import NUM_SQUARES
        return ActionSpaceInfo(
            num_actions=NUM_SQUARES * NUM_SQUARES,
            action_labels=[
                f'{_square_name(i)}->{_square_name(j)}'
                for i in range(NUM_SQUARES)
                for j in range(NUM_SQUARES)
            ],
        )

    def get_observation_shape(self) -> Tuple[int, ...]:
        return (84, 84, 1)

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        return StandardMetrics(
            progress=min(1.0, info.get('moves_played', 0) / 200),
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
            'graph_2_legend': ['Agent (White)', 'Opponent (Black)'],
            'status_metric_label': 'Win Rate',
            'status_metric_key': 'win_rate',
        }

    def get_completion_criteria(self) -> dict:
        return {
            'metric': 'win_rate',
            'threshold': 0.7,
            'window': 100,
            'description': 'Win rate > 70% over 100 games',
        }

    def is_available(self) -> bool:
        """Check if python-chess is installed."""
        try:
            import chess  # noqa: F401
            return True
        except ImportError:
            return False


def _square_name(sq: int) -> str:
    """Convert square index (0-63) to algebraic notation (a1-h8)."""
    file_char = chr(ord('a') + sq % 8)
    rank_char = str(sq // 8 + 1)
    return file_char + rank_char
