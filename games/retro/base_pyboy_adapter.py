"""
Base adapter for PyBoy (Game Boy emulator) games.

Provides shared logic for creating Game Boy environments using PyBoy.
Subclasses define game-specific ROM paths, RAM addresses, reward
functions, and metrics extraction.

Unlike BaseRetroAdapter (which requires stable-retro and C++ builds),
PyBoy is a pure Python emulator that installs via pip. It supports
all original Game Boy (DMG) and Game Boy Color (GBC) ROMs.

Adding a new Game Boy game:
  1. Create games/retro/{game_id}/ folder
  2. Create adapter.py extending BasePyBoyAdapter
  3. Create {game}_env.py with a gymnasium.Env subclass
  4. Override create_env() to return your environment
  5. The registry auto-discovers it on next launch

Example games this supports:
  - Pokemon Red/Blue/Yellow
  - Legend of Zelda: Link's Awakening
  - Tetris (Game Boy)
  - Kirby's Dream Land
  - Any other .gb / .gbc ROM

Requirements:
  pip install pyboy scikit-image
"""

from abc import abstractmethod
from typing import List, Optional, Tuple

try:
    import gymnasium as gym
except ImportError:
    import gym

from games.base_adapter import BaseGameAdapter
from games.reward_config import RewardConfig

try:
    from pyboy import PyBoy
    HAS_PYBOY = True
except ImportError:
    HAS_PYBOY = False


class BasePyBoyAdapter(BaseGameAdapter):
    """Abstract base for all PyBoy (Game Boy) game adapters.

    Subclasses must define:
        - name, game_id, category, description (identity)
        - create_env(**kwargs): return a gymnasium.Env
        - get_action_space_info(), get_observation_shape()
        - extract_metrics()

    Optional overrides:
        - get_reward_config(), get_dashboard_config()
        - get_completion_criteria(), get_game_specific_options()
        - supported_algorithms()
    """

    # ---- Default Game Boy properties ----

    @property
    def native_resolution(self) -> Tuple[int, int]:
        """Native Game Boy screen resolution. (144, 160) for DMG."""
        return (144, 160)

    @property
    def downscale_factor(self) -> int:
        """Factor to downscale native resolution. Default 2 (-> 72x80)."""
        return 2

    @property
    def frame_stacks(self) -> int:
        """Number of frames to stack for temporal info. Default 3."""
        return 3

    # ---- Environment creation (override in subclasses) ----

    @abstractmethod
    def create_env(self, **kwargs) -> gym.Env:
        """Create the Game Boy environment.

        Must be implemented by each game. Typical pattern:
            from games.retro.{game_id}.{game}_env import {Game}Env
            return {Game}Env(config=kwargs)
        """

    # ---- Shared defaults ----

    def get_observation_shape(self) -> Tuple[int, ...]:
        """Default obs shape for downscaled Game Boy with frame stacking."""
        h, w = self.native_resolution
        f = self.downscale_factor
        return (h // f, w // f, self.frame_stacks)

    def needs_sb3_compat(self) -> bool:
        """PyBoy environments use gymnasium natively."""
        return False

    def supported_algorithms(self) -> List[str]:
        """Game Boy image obs — no NEAT (too large)."""
        return ['ppo', 'dqn', 'a2c', 'rainbow']

    def is_available(self) -> bool:
        """Check if PyBoy is installed."""
        return HAS_PYBOY

    def get_human_render_frame(self, env):
        """Return RGB frame for dashboard display."""
        try:
            return env.render(mode='rgb_array')
        except TypeError:
            return env.render()
