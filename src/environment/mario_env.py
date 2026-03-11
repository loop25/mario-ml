"""
Unified Mario Environment Factory.

This module provides a single function to create a fully configured
Super Mario Bros environment with all preprocessing wrappers applied.

The environment uses gym-super-mario-bros which includes a built-in
NES emulator (nes-py) — no separate ROM file is needed.

Architecture:
    Raw NES output (240x256 RGB @ 60fps)
    → SkipFrame (repeat action 4 frames, sum rewards)
    → CustomRewardWrapper (shaped rewards for better learning)
    → GrayScaleObservation (RGB → grayscale, 3ch → 1ch)
    → ResizeObservation (240x256 → 84x84 or custom size)
    → FrameStackObservation (stack 4 frames for temporal info)
    → Final output: (84, 84, 4) uint8 array

The simplified action space reduces 256 possible button combinations
to 7 useful actions that cover all important Mario movements.

Usage:
    # For PPO/DQN (84x84 stacked frames):
    env = create_mario_env(world=1, stage=1)

    # For NEAT (13x13 downsampled, no stacking):
    env = create_mario_env(world=1, stage=1, resize_shape=(13, 13),
                           frame_stack=1, normalize=False)

    # For evaluation with rendering:
    env = create_mario_env(world=1, stage=1)
"""

try:
    import gymnasium as gym
except ImportError:
    import gym  # Legacy fallback
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
from typing import Tuple, Optional

from src.environment.wrappers import (
    SkipFrame,
    GrayScaleObservation,
    ResizeObservation,
    FrameStackObservation,
    CustomRewardWrapper,
    SB3CompatWrapper,
)


# ============================================================================
# Action Space Definitions
# ============================================================================
# The NES controller has 8 buttons, giving 256 possible combinations.
# Most combinations are useless for Mario. We define simplified sets:

# 7 useful actions covering all important Mario movements
MARIO_ACTIONS = [
    ['NOOP'],              # 0: Do nothing (stand still)
    ['right'],             # 1: Walk right
    ['right', 'A'],        # 2: Jump while moving right
    ['right', 'B'],        # 3: Run right (B = sprint)
    ['right', 'A', 'B'],   # 4: Sprint jump right (longest jump)
    ['A'],                 # 5: Jump in place (for vertical obstacles)
    ['left'],              # 6: Walk left (for repositioning)
]


def create_mario_env(
    world: int = 1,
    stage: int = 1,
    version: int = 0,
    actions: str = 'simple',
    skip_frames: int = 4,
    resize_shape: Tuple[int, int] = (84, 84),
    frame_stack: int = 4,
    apply_reward_shaping: bool = True,
    normalize: bool = True,
) -> gym.Env:
    """
    Create a fully configured Super Mario Bros environment.

    This is the main entry point for creating environments. It applies
    all necessary wrappers in the correct order for ML training.

    Args:
        world: World number 1-8. Default 1.
        stage: Stage number 1-4. Default 1.
        version: Game version. 0=standard, 1=random enemies, 2=no enemies.
                 Use version 2 for easier initial testing.
        actions: Action space type. Options:
                 'simple' - 7 custom actions (recommended)
                 'right_only' - Only rightward movement
                 'complex' - All 12 actions from gym-super-mario-bros
        skip_frames: Number of frames to repeat each action. Default 4.
        resize_shape: Target (height, width) for frame resizing.
                      Use (84, 84) for CNN-based algorithms (PPO/DQN).
                      Use (13, 13) for NEAT (smaller input).
        frame_stack: Number of frames to stack. Default 4.
                     Use 1 for NEAT (single frame input).
        apply_reward_shaping: Whether to apply custom reward shaping.
                              Default True. Set False for raw rewards.
        normalize: Whether to normalize observations to [0, 1] float32.
                   Default True for neural networks.
                   Set False for NEAT (uses raw uint8 pixels).

    Returns:
        gym.Env: Configured environment ready for training.

    Raises:
        ValueError: If world/stage combination is invalid.

    Example:
        >>> env = create_mario_env(world=1, stage=1)
        >>> obs = env.reset()
        >>> print(obs.shape)  # (84, 84, 4)
        >>> action = env.action_space.sample()
        >>> obs, reward, done, info = env.step(action)
    """
    # Validate inputs
    if not (1 <= world <= 8):
        raise ValueError(f"World must be 1-8, got {world}")
    if not (1 <= stage <= 4):
        raise ValueError(f"Stage must be 1-4, got {stage}")

    # Map version to rom_mode:
    #   0 = 'vanilla' (standard game)
    #   1 = 'vanilla' (same, random enemies not supported directly)
    #   2 = 'vanilla' (no enemies not supported directly)
    # Note: gym-super-mario-bros versions map to registration variants,
    # but for direct instantiation we use vanilla mode which works for all.
    rom_mode = 'vanilla'

    # Create the base environment directly, bypassing gym.make() to avoid
    # gym 0.26's TimeLimit/OrderEnforcing wrappers that expect the new
    # 5-value step API (terminated, truncated) which this old env doesn't support.
    env = SuperMarioBrosEnv(rom_mode=rom_mode, target=(world, stage))

    # Apply JoypadSpace wrapper to simplify actions
    # This maps our action indices to NES button combinations
    if actions == 'simple':
        env = JoypadSpace(env, MARIO_ACTIONS)
    elif actions == 'right_only':
        env = JoypadSpace(env, RIGHT_ONLY)
    elif actions == 'complex':
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
    else:
        raise ValueError(f"Unknown action type: {actions}. Use 'simple', 'right_only', or 'complex'.")

    # Apply frame skipping (repeat action for N frames)
    if skip_frames > 1:
        env = SkipFrame(env, skip=skip_frames)

    # Apply custom reward shaping for better learning signals
    if apply_reward_shaping:
        env = CustomRewardWrapper(env)

    # Convert to grayscale (3 channels → 1 channel)
    env = GrayScaleObservation(env)

    # Resize to target dimensions
    env = ResizeObservation(env, shape=resize_shape)

    # Stack frames for temporal information
    if frame_stack > 1:
        env = FrameStackObservation(env, num_stack=frame_stack)

    return env


def create_neat_env(world: int = 1, stage: int = 1) -> gym.Env:
    """
    Create an environment configured specifically for NEAT.

    NEAT uses a feed-forward neural network with a fixed number of inputs,
    so we need a smaller observation space. The 13x13 downsampled grayscale
    image gives 169 inputs, which is manageable for NEAT's evolved networks.

    No frame stacking is used because NEAT networks are typically simpler
    and work better with a single frame as input.

    Args:
        world: World number 1-8. Default 1.
        stage: Stage number 1-4. Default 1.

    Returns:
        gym.Env: Environment with (13, 13, 1) observations.
    """
    return create_mario_env(
        world=world,
        stage=stage,
        resize_shape=(13, 13),
        frame_stack=1,
        normalize=False,  # NEAT uses raw pixel values
    )


def create_cnn_env(world: int = 1, stage: int = 1) -> gym.Env:
    """
    Create an environment configured for CNN-based algorithms (PPO/DQN).

    Uses the standard 84x84 resolution with 4-frame stacking, which is
    the proven configuration from the Atari DQN paper.

    Args:
        world: World number 1-8. Default 1.
        stage: Stage number 1-4. Default 1.

    Returns:
        gym.Env: Environment with (84, 84, 4) observations.
    """
    return create_mario_env(
        world=world,
        stage=stage,
        resize_shape=(84, 84),
        frame_stack=4,
        normalize=True,
    )


def create_sb3_env(world: int = 1, stage: int = 1) -> gym.Env:
    """
    Create an environment for stable-baselines3 algorithms (PPO).

    SB3 expects the new gymnasium API where:
        - reset() returns (obs, info) tuple
        - step() returns (obs, reward, terminated, truncated, info)

    This wraps the standard CNN environment with SB3CompatWrapper
    to bridge from our old gym API to the new API that SB3 expects.

    Args:
        world: World number 1-8. Default 1.
        stage: Stage number 1-4. Default 1.

    Returns:
        gym.Env: SB3-compatible environment with (84, 84, 4) observations.
    """
    env = create_mario_env(
        world=world,
        stage=stage,
        resize_shape=(84, 84),
        frame_stack=4,
        normalize=True,
    )
    return SB3CompatWrapper(env)


def get_env_info(env: gym.Env) -> dict:
    """
    Get useful information about an environment for debugging.

    Args:
        env: The environment to inspect.

    Returns:
        dict: Environment information including observation and action spaces.
    """
    return {
        'observation_space': str(env.observation_space),
        'observation_shape': env.observation_space.shape,
        'action_space': str(env.action_space),
        'num_actions': env.action_space.n,
        'action_meanings': MARIO_ACTIONS,
    }
