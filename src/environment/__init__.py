"""
Environment module for Super Mario Bros ML.

Provides a unified game environment wrapper that handles:
- NES game initialization via gym-super-mario-bros
- Action space simplification (256 -> 7 useful actions)
- Frame preprocessing (grayscale, resize, normalization)
- Frame stacking for temporal information
- Custom reward shaping for better learning signals

Usage:
    from src.environment.mario_env import create_mario_env
    env = create_mario_env(world=1, stage=1, render_mode='rgb_array')
"""

try:
    from src.environment.mario_env import create_mario_env
    __all__ = ['create_mario_env']
except ImportError:
    # gym-super-mario-bros not installed; Mario env unavailable
    __all__ = []
