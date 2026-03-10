"""
Abstract base class for all game adapters.

Every game in the platform (built-in, retro, or user-contributed)
must implement this interface. It defines how the framework
discovers, creates, wraps, and measures each game.

The adapter pattern keeps game-specific logic (env creation,
metrics extraction, reward tuning) encapsulated within each
game module, while the framework handles everything else
(training loops, visualization, checkpointing).
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import gym

from games.reward_config import (
    StandardMetrics,
    RewardConfig,
    ActionSpaceInfo,
    TokenConfig,
)


class BaseGameAdapter(ABC):
    """Abstract interface that every game must implement."""

    # ---- Identity (required properties) ----

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name, e.g. 'Super Mario Bros'."""

    @property
    @abstractmethod
    def game_id(self) -> str:
        """Unique identifier, e.g. 'mario'. Used as folder name."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Game category: 'platformer', 'puzzle', 'board', 'arcade', 'rpg'."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description for the launcher GUI."""

    # ---- Environment (required methods) ----

    @abstractmethod
    def create_env(self, **kwargs) -> gym.Env:
        """Create and return a ready-to-use Gym environment.

        Kwargs may include game-specific options (world, stage, difficulty).
        The returned env should NOT have reward shaping applied -- the
        framework handles that via get_reward_config().

        Returns:
            gym.Env with observations suitable for ML.
        """

    @abstractmethod
    def get_action_space_info(self) -> ActionSpaceInfo:
        """Describe the discrete action space."""

    @abstractmethod
    def get_observation_shape(self) -> Tuple[int, ...]:
        """Shape of preprocessed observations, e.g. (84, 84, 4)."""

    # ---- Metrics Translation (required) ----

    @abstractmethod
    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        """Map game-specific info dict to universal StandardMetrics.

        Args:
            info: The info dict from env.step().
            episode_time: Seconds elapsed in this episode.

        Returns:
            StandardMetrics with progress, score, completed, time_elapsed.
        """

    # ---- Reward Shaping (optional override) ----

    def get_reward_config(self) -> RewardConfig:
        """Return reward shaping parameters.

        Override to customize. Defaults tuned for action games.
        Board games should set time_penalty_per_second=0.
        """
        return RewardConfig()

    # ---- Optional Features ----

    def get_curriculum(self) -> Optional[object]:
        """Return a CurriculumManager for multi-level games.

        Single-screen games (Snake, Tetris) return None.
        Multi-level games (Mario, Sonic) return a curriculum.
        """
        return None

    def get_human_render_frame(self, env) -> Optional[np.ndarray]:
        """Return an RGB frame for dashboard display.

        Defaults to env.render(). Override if your game needs
        custom rendering for the dashboard.
        """
        try:
            return env.render(mode='rgb_array')
        except TypeError:
            return env.render()

    def get_game_specific_options(self) -> dict:
        """Return dict of option_name -> (type, default, description)
        for the launcher GUI to display game-specific settings.

        Example: {'world': (int, 1, 'World number 1-8')}
        """
        return {}

    # ---- For Generalist Agent (optional) ----

    def get_token_config(self) -> TokenConfig:
        """How to tokenize this game for the Decision Transformer."""
        obs_shape = self.get_observation_shape()
        return TokenConfig(
            obs_resolution=(obs_shape[0], obs_shape[1]),
            obs_channels=obs_shape[2] if len(obs_shape) > 2 else 1,
            action_vocab_size=self.get_action_space_info().num_actions,
            game_token_id=hash(self.game_id) % 1024,
        )

    # ---- Algorithm Compatibility (optional) ----

    def supported_algorithms(self) -> List[str]:
        """Return the list of algorithms this game supports.

        NEAT requires a small, flat observation (e.g. 13x13 = 169 inputs).
        Games with large image observations (84x84) should exclude 'neat'.

        Override to restrict. Default: all algorithms.
        """
        return ['neat', 'ppo', 'dqn', 'a2c']

    # ---- Dashboard Configuration (optional) ----

    def get_dashboard_config(self) -> dict:
        """Return dashboard display configuration for this game.

        The dashboard uses this to customize graph titles, status bar
        metrics, and other game-specific display elements.

        Keys:
            graph_2_title: Title for the second graph (top-right).
            graph_2_metric: Metric key in MetricsTracker for graph 2.
            graph_2_info_key: Key in env info dict to extract the value.
            status_metric_label: Label for the secondary status bar value.
            status_metric_key: Metric key for the secondary status value.

        Override to customize. Default is tuned for platformers (distance).
        """
        return {
            'graph_2_title': 'Distance (x position)',
            'graph_2_metric': 'distance',
            'graph_2_info_key': 'x_pos',
            'status_metric_label': 'Distance',
            'status_metric_key': 'distance',
        }

    def get_completion_criteria(self) -> dict:
        """Return criteria for detecting when training is "complete".

        The dashboard can show a mastery progress bar based on these
        criteria. Different games have different definitions of mastery.

        Keys:
            metric: Which metric to track ('reward', 'score', 'win_rate').
            threshold: Value that indicates mastery.
            window: Number of recent episodes to average over.
            description: Human-readable description of the goal.

        Override per game. Default: average reward > 500 over 50 episodes.
        """
        return {
            'metric': 'reward',
            'threshold': 500.0,
            'window': 50,
            'description': 'Avg reward > 500 over 50 episodes',
        }

    # ---- For SB3 Algorithms (optional) ----

    def needs_sb3_compat(self) -> bool:
        """Whether this game needs the SB3CompatWrapper.

        True for old-gym-API environments (like gym-super-mario-bros).
        False for gymnasium-native environments (built-in games).
        """
        return False
