"""
Game Adapter Template.

The adapter tells the framework how to create, configure, and
measure your game. It bridges your game environment and the
training/visualization pipeline.

TODO:
    1. Update name, game_id, category, and description
    2. Import your CustomGameEnv in create_env()
    3. Customize get_reward_config() if you want shaped rewards
    4. Update get_action_space_info() with your action labels

See games/builtin/snake/adapter.py for a complete example.
"""
from typing import List, Tuple

try:
    import gymnasium as gym
except ImportError:
    import gym

from games.base_adapter import BaseGameAdapter
from games.reward_config import (
    ActionSpaceInfo,
    RewardConfig,
    StandardMetrics,
)


class CustomGameAdapter(BaseGameAdapter):
    """Adapter for your custom game."""

    # ---- Identity ----

    @property
    def name(self) -> str:
        return "My Custom Game"  # TODO: Change this

    @property
    def game_id(self) -> str:
        return "custom_game"  # TODO: Change this (must match folder name)

    @property
    def category(self) -> str:
        return "arcade"  # Options: platformer, puzzle, board, arcade, rpg

    @property
    def description(self) -> str:
        return "A custom game for the AI training platform."

    # ---- Environment ----

    def create_env(self, **kwargs) -> gym.Env:
        from .game import CustomGameEnv
        return CustomGameEnv(**kwargs)

    def supported_algorithms(self) -> List[str]:
        return ["ppo", "dqn", "a2c", "rainbow"]

    def get_observation_shape(self) -> Tuple[int, ...]:
        return (84, 84, 1)

    # ---- Metrics & Rewards ----

    def get_reward_config(self) -> RewardConfig:
        return RewardConfig(
            time_penalty_per_second=0.005,
            completion_bonus=100.0,
            death_penalty=-10.0,
            idle_penalty_per_second=0.003,
            speed_bonus_multiplier=1.0,
            par_time_seconds=60.0,
        )

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        return StandardMetrics(
            progress=info.get("score", 0.0) / 100.0,  # TODO: Adjust target
            score=float(info.get("score", 0)),
            completed=info.get("completed", False),
            time_elapsed=episode_time,
        )

    def get_action_space_info(self) -> ActionSpaceInfo:
        return ActionSpaceInfo(
            num_actions=4,
            action_labels=["Up", "Right", "Down", "Left"],  # TODO: Your labels
        )

    def get_dashboard_config(self) -> dict:
        return {
            "graph_2_title": "Score",
            "graph_2_metric": "score",
            "graph_2_info_key": "score",
            "status_metric_label": "Score",
            "status_metric_key": "score",
        }

    def get_completion_criteria(self) -> dict:
        return {
            "metric": "score",
            "threshold": 50.0,
            "window": 50,
            "description": "Avg score > 50 over 50 episodes",
        }
