"""
Game Adapter Template.

The adapter tells the framework how to create, configure, and
measure your game. It bridges your game environment and the
training/visualization pipeline.

TODO:
    1. Update name, game_id, category, and description
    2. Import your CustomGameEnv in create_env()
    3. Customize reward_config() if you want shaped rewards
    4. Update get_action_space_info() with your action labels

See games/builtin/snake/adapter.py for a complete example.
"""
try:
    import gymnasium as gym
except ImportError:
    import gym

from games.base_adapter import BaseGameAdapter
from games.reward_config import (
    ActionSpaceInfo,
    RewardConfig,
    StandardMetrics,
    TokenConfig,
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

    def supported_algorithms(self):
        return ["neat", "ppo", "dqn", "a2c", "rainbow"]

    # ---- Metrics & Rewards ----

    def reward_config(self) -> RewardConfig:
        return RewardConfig()

    def extract_metrics(self, info: dict) -> StandardMetrics:
        return StandardMetrics(
            reward=info.get("reward", 0.0),
            distance=info.get("score", 0.0),
            extra={},
        )

    def get_action_space_info(self) -> ActionSpaceInfo:
        return ActionSpaceInfo(
            num_actions=4,
            action_labels=["Up", "Right", "Down", "Left"],  # TODO: Your labels
        )

    def get_dashboard_config(self) -> dict:
        return {
            "graph_2_title": "Score",
            "graph_2_metric": "distance",
            "graph_2_info_key": "score",
            "status_metric_label": "Score",
            "status_metric_key": "distance",
        }

    def get_token_config(self) -> TokenConfig:
        return TokenConfig(game_token_id=99)  # TODO: Pick a unique ID
