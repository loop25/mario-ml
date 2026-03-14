"""
Shared frame capture utility for all training algorithms.

Every algorithm (NEAT, PPO, DQN, A2C, Rainbow) needs to capture gameplay
frames for the dashboard display and video recording. The capture strategy
depends on the environment type:

    NES games (Super Mario Bros):
        env.unwrapped.screen  →  240×256 RGB numpy array

    Built-in games (Chess, Snake, Connect4, etc.):
        env.unwrapped.render(mode='rgb_array')  →  480×480 RGB numpy array
        (must call on unwrapped env — gymnasium wrappers strip mode arg)

    Fallback:
        The raw ML observation (84×84 grayscale) — last resort only.

This module provides a single function used by every algorithm to ensure
consistent, high-quality gameplay visuals across the board.
"""
import numpy as np
from typing import Optional


def capture_display_frame(
    env,
    fallback_obs: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Capture a display-quality frame from any environment.

    Tries three strategies in order of visual quality:

    1. NES screen buffer (direct pixel data, best for retro games)
    2. env.render(mode='rgb_array') (the standard Gym render path)
    3. Raw observation fallback (grayscale ML input — low quality)

    This function is intentionally defensive: it never raises, always
    returns *something* the dashboard can display (or None if truly
    nothing is available).

    Args:
        env: A gym environment, possibly wrapped in SB3CompatWrapper,
             DummyVecEnv, VecTransposeImage, or other wrappers.
             The function drills through wrapper layers automatically.
        fallback_obs: Optional raw observation to use as last resort.

    Returns:
        A numpy array suitable for dashboard display, or None.

    Example:
        >>> frame = capture_display_frame(env, fallback_obs=obs)
        >>> dashboard.update_visualization(frame=frame)
    """
    # --- Strategy 1: NES screen buffer ---
    # gym-super-mario-bros exposes the raw framebuffer on .unwrapped.screen
    try:
        screen = env.unwrapped.screen
        if screen is not None:
            return screen.copy()
    except AttributeError:
        pass

    # --- Strategy 2: Render on the *base* environment ---
    # Built-in games define render(mode='rgb_array') returning
    # high-quality RGB frames (typically 480×480).
    #
    # IMPORTANT: We call render() on env.unwrapped (the base GameEnv)
    # rather than on `env` itself because gymnasium's gym.Wrapper.render()
    # does NOT accept keyword arguments — calling env.render(mode='rgb_array')
    # on a wrapped env raises TypeError.  The base GameEnv classes still
    # use the legacy signature render(self, mode='rgb_array'), so calling
    # on unwrapped works correctly.
    try:
        base = env.unwrapped
        frame = base.render(mode='rgb_array')
        if frame is not None:
            return frame
    except Exception:
        pass

    # --- Strategy 2b: Gymnasium-native render (no mode arg) ---
    # If the env uses gymnasium's render_mode system, render() with
    # no arguments returns the frame based on how the env was created.
    try:
        frame = env.render()
        if frame is not None and isinstance(frame, np.ndarray):
            return frame
    except Exception:
        pass

    # --- Strategy 3: Raw observation fallback ---
    return fallback_obs


def capture_display_frame_from_vec_env(
    vec_env,
    fallback_obs: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Capture a display frame from an SB3 vectorized environment.

    SB3 wraps environments in DummyVecEnv → VecTransposeImage.
    This helper drills through the wrapper chain to find the
    actual game environment and captures a frame from it.

    Wrapper chain (outside → inside):
        VecTransposeImage → DummyVecEnv → SB3CompatWrapper → GameEnv

    Args:
        vec_env: An SB3 vectorized environment (VecTransposeImage
                 or DummyVecEnv).
        fallback_obs: Optional raw observation as last resort.

    Returns:
        A numpy array suitable for dashboard display, or None.
    """
    if vec_env is None:
        return fallback_obs

    # Drill through VecTransposeImage → DummyVecEnv
    dummy_env = vec_env
    if hasattr(vec_env, 'venv'):
        dummy_env = vec_env.venv

    # Access the first sub-environment in DummyVecEnv.envs[]
    try:
        actual_env = dummy_env.envs[0]
    except (AttributeError, IndexError):
        return fallback_obs

    return capture_display_frame(actual_env, fallback_obs)
