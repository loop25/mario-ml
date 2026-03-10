"""
Base adapter for stable-retro (emulated ROM) games.

Provides shared logic for creating retro environments with standard
preprocessing wrappers (frame skip, grayscale, resize, frame stack).
Subclasses only need to define identity properties, the ROM name,
and game-specific metrics extraction.

Requires `stable-retro` (pip install stable-retro) and the
appropriate ROM file imported via `python -m retro.import /path/to/rom`.

Architecture:
    Raw emulator output (e.g. 224x320 RGB)
    → SkipFrame (repeat action, accumulate reward)
    → GrayScaleObservation (RGB → 1 channel)
    → ResizeObservation (→ 84x84)
    → FrameStackObservation (stack N frames)
    → SB3CompatWrapper (old gym → gymnasium compat)
    → Final output: (84, 84, 4) uint8 array
"""
from abc import abstractmethod
from typing import List, Optional, Tuple

import gym

from games.base_adapter import BaseGameAdapter
from games.reward_config import RewardConfig

try:
    import retro
    HAS_RETRO = True
except ImportError:
    HAS_RETRO = False


class BaseRetroAdapter(BaseGameAdapter):
    """Abstract base for all retro/emulated ROM game adapters.

    Subclasses must define:
        - name, game_id, category, description (identity)
        - rom_name: The stable-retro game identifier
        - initial_state: Starting state/level
        - get_action_space_info(), extract_metrics()
    """

    # ---- Retro-specific properties (override in subclasses) ----

    @property
    @abstractmethod
    def rom_name(self) -> str:
        """stable-retro game identifier, e.g. 'SonicTheHedgehog-Genesis'."""

    @property
    def initial_state(self) -> Optional[str]:
        """Starting state/save-state name. None = default start."""
        return None

    @property
    def frame_skip(self) -> int:
        """Number of frames to skip (repeat action). Default 4."""
        return 4

    @property
    def frame_stack(self) -> int:
        """Number of frames to stack for temporal info. Default 4."""
        return 4

    @property
    def render_size(self) -> Tuple[int, int]:
        """Observation resize target. Default (84, 84)."""
        return (84, 84)

    # ---- Environment creation ----

    def create_env(self, **kwargs) -> gym.Env:
        """Create a retro environment with standard preprocessing.

        Kwargs:
            state (str): Override initial state/level.
            frame_skip (int): Override frame skip count.
            frame_stack (int): Override frame stack count.

        Raises:
            ImportError: If stable-retro is not installed.
            FileNotFoundError: If the ROM is not imported.
        """
        if not HAS_RETRO:
            raise ImportError(
                f"stable-retro is required for {self.name}. "
                "Install it with: pip install stable-retro"
            )

        state = kwargs.get('state', self.initial_state)
        skip = kwargs.get('frame_skip', self.frame_skip)
        stack = kwargs.get('frame_stack', self.frame_stack)

        # Create the base retro environment
        env_kwargs = {'game': self.rom_name}
        if state is not None:
            env_kwargs['state'] = state
        env_kwargs['use_restricted_actions'] = retro.Actions.DISCRETE

        try:
            env = retro.make(**env_kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"ROM not found for '{self.rom_name}'. "
                f"Import your ROM with: python -m retro.import /path/to/rom/directory"
            )

        # Apply standard preprocessing wrappers
        from src.environment.wrappers import (
            SkipFrame,
            GrayScaleObservation,
            ResizeObservation,
            FrameStackObservation,
        )

        if skip > 1:
            env = SkipFrame(env, skip=skip)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=self.render_size)
        if stack > 1:
            env = FrameStackObservation(env, num_stack=stack)

        return env

    def get_observation_shape(self) -> Tuple[int, ...]:
        h, w = self.render_size
        return (h, w, self.frame_stack)

    # ---- Shared retro defaults ----

    def needs_sb3_compat(self) -> bool:
        """Retro uses old gym API, needs SB3 wrapper."""
        return True

    def supported_algorithms(self) -> List[str]:
        """Retro games use 84x84 image obs — no NEAT."""
        return ['ppo', 'dqn', 'a2c', 'rainbow']

    def get_reward_config(self) -> RewardConfig:
        """Default reward config for retro platformers."""
        return RewardConfig(
            time_penalty_per_second=0.01,
            completion_bonus=100.0,
            death_penalty=-15.0,
            idle_penalty_per_second=0.005,
            speed_bonus_multiplier=1.5,
            par_time_seconds=120.0,
        )

    def is_available(self) -> bool:
        """Check if stable-retro is installed and ROM is available."""
        if not HAS_RETRO:
            return False
        try:
            # Check if the game ROM is imported
            return self.rom_name in retro.data.list_games()
        except Exception:
            return False
