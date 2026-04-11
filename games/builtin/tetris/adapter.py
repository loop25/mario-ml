"""Tetris game adapter for the plugin registry."""
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
from games.builtin.tetris.game import TetrisEnv


class TetrisAdapter(BaseGameAdapter):
    """Adapter for the built-in Tetris game."""

    @property
    def name(self) -> str:
        return 'Tetris'

    @property
    def game_id(self) -> str:
        return 'tetris'

    @property
    def category(self) -> str:
        return 'puzzle'

    @property
    def description(self) -> str:
        return 'Classic Tetris — clear lines by filling rows with falling pieces.'

    def create_env(self, **kwargs) -> gym.Env:
        return TetrisEnv(**kwargs)

    def get_action_space_info(self) -> ActionSpaceInfo:
        return ActionSpaceInfo(
            num_actions=7,
            action_labels=['Left', 'Right', 'Rot CW', 'Rot CCW', 'Drop', 'Soft Drop', 'Wait'],
        )

    def get_observation_shape(self) -> Tuple[int, ...]:
        return (84, 84, 1)

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        lines = info.get('lines_cleared', 0)
        # Target: 40 lines (marathon mode)
        target_lines = 40
        return StandardMetrics(
            progress=min(lines / target_lines, 1.0),
            score=float(info.get('score', 0)),
            completed=(lines >= target_lines),
            time_elapsed=episode_time,
        )

    def get_reward_config(self) -> RewardConfig:
        # Tetris is a survival/puzzle game — surviving longer IS success.
        # No time or idle penalties: the agent should be rewarded for lasting.
        # Death penalty is mild so the line-clear signal dominates learning.
        return RewardConfig(
            time_penalty_per_second=0.0,     # No time penalty — survival is good
            completion_bonus=100.0,
            death_penalty=-2.0,              # Mild — matches game.py's own -2.0
            idle_penalty_per_second=0.0,     # No idle penalty for puzzle games
            speed_bonus_multiplier=0.5,
            par_time_seconds=300.0,
        )

    def get_training_hints(self) -> dict:
        # Puzzle games need more exploration (higher entropy) and benefit
        # from longer rollouts to see the effect of line clears.
        return {
            'ent_coef': 0.05,          # 5x default — explore rotations/placements
            'learning_rate': 1.5e-4,   # Slightly lower for stability
            'n_steps': 1024,           # Longer rollouts capture line-clear episodes
            'gamma': 0.995,            # Value future line clears more
        }

    def supported_algorithms(self) -> List[str]:
        return ['ppo', 'dqn', 'a2c', 'rainbow', 'dt']

    def get_dashboard_config(self) -> dict:
        return {
            'graph_2_title': 'Lines Cleared',
            'graph_2_metric': 'lines_cleared',
            'graph_2_info_key': 'lines_cleared',
            'status_metric_label': 'Lines',
            'status_metric_key': 'lines_cleared',
        }

    def get_completion_criteria(self) -> dict:
        return {
            'metric': 'lines_cleared',
            'threshold': 20.0,
            'window': 50,
            'description': 'Avg lines > 20 over 50 episodes',
        }

    def get_game_specific_options(self) -> dict:
        return {
            'max_steps': (int, 10000, 'Max steps per episode'),
        }
