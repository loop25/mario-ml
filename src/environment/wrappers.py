"""
Environment wrappers for Super Mario Bros ML.

This module provides Gym-compatible wrappers that transform the raw NES
environment output into formats suitable for machine learning algorithms.

Wrapper pipeline (applied in order):
    1. SkipFrame: Repeat actions for N frames, sum rewards
    2. GrayScaleObservation: Convert RGB to grayscale
    3. ResizeObservation: Downscale frames to target size
    4. FrameStackObservation: Stack N consecutive frames

These wrappers follow the standard Gym wrapper pattern, where each
wrapper wraps the previous one, forming a chain of transformations.

References:
    - OpenAI Gym Wrappers: https://gymnasium.farama.org/api/wrappers/
    - Atari preprocessing: Mnih et al. (2015) "Human-level control through
      deep reinforcement learning"
"""

import numpy as np
try:
    import gymnasium as gym
    from gymnasium.spaces import Box
except ImportError:
    import gym  # Legacy fallback
    from gym.spaces import Box
import cv2
from collections import deque


def _strip_new_api_kwargs(kwargs: dict) -> dict:
    """
    Remove keyword arguments that belong to the new gymnasium API.

    shimmy (the gym→gymnasium compatibility layer used by stable-baselines3)
    calls reset(seed=..., options=...) which propagates down through our
    wrapper chain. The underlying gym-super-mario-bros JoypadSpace.reset()
    doesn't accept these kwargs and crashes with TypeError.

    This helper strips them so old-API environments work correctly.
    """
    kwargs.pop('seed', None)
    kwargs.pop('options', None)
    return kwargs


class SkipFrame(gym.Wrapper):
    """
    Skip N frames and repeat the same action, accumulating rewards.

    This is a standard technique in RL for Atari/NES games that:
    1. Reduces the number of decisions the agent needs to make
    2. Speeds up training by processing fewer frames
    3. Allows actions to have more visible effects before the next decision

    For Super Mario Bros, a skip of 4 means the agent makes a decision
    every 4th frame (~15 decisions per second at 60 FPS).

    Args:
        env (gym.Env): The environment to wrap.
        skip (int): Number of frames to skip. Default is 4.

    Example:
        >>> env = SkipFrame(env, skip=4)
        >>> obs, reward, done, info = env.step(action)  # action repeated 4 times
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def reset(self, **kwargs):
        """Reset with old-API compatibility (strip seed/options kwargs)."""
        kwargs = _strip_new_api_kwargs(kwargs)
        return self.env.reset(**kwargs)

    def step(self, action: int):
        """
        Repeat action for `skip` frames, accumulating reward.

        Args:
            action: The action to repeat.

        Returns:
            tuple: (observation, total_reward, done, info) where total_reward
                   is the sum of rewards across all skipped frames.
        """
        total_reward = 0.0
        done = False
        # Repeat the action and sum rewards
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.Wrapper):
    """
    Convert RGB observations to grayscale.

    Reduces the observation from 3 channels (RGB) to 1 channel,
    which significantly reduces the neural network input size
    while retaining the important structural information.

    Mario's gameplay is primarily about shapes and movement patterns,
    not colors, so grayscale works well for learning.

    Uses gym.Wrapper instead of ObservationWrapper for compatibility with
    gym-super-mario-bros (which uses the old gym reset API that returns
    just obs instead of (obs, info) tuple).

    Args:
        env (gym.Env): The environment to wrap.

    Note:
        The observation space changes from (H, W, 3) to (H, W, 1).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Get the original observation shape
        obs_shape = self.observation_space.shape[:2]
        # Update observation space to single channel
        self.observation_space = Box(
            low=0, high=255, shape=(*obs_shape, 1), dtype=np.uint8
        )

    def _apply_grayscale(self, observation: np.ndarray) -> np.ndarray:
        """
        Convert RGB observation to grayscale.

        Args:
            observation: RGB image array of shape (H, W, 3).

        Returns:
            Grayscale image array of shape (H, W, 1).
        """
        # Use OpenCV for efficient RGB->grayscale conversion
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # Add channel dimension: (H, W) -> (H, W, 1)
        return np.expand_dims(gray, axis=-1)

    def reset(self, **kwargs):
        """
        Reset environment and convert initial observation to grayscale.

        Handles both old gym API (returns obs) and new API (returns obs, info).

        Returns:
            Grayscale observation of shape (H, W, 1).
        """
        # Strip new-API kwargs that old gym envs don't understand
        kwargs = _strip_new_api_kwargs(kwargs)
        # gym-super-mario-bros uses old API (returns just obs)
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        return self._apply_grayscale(obs)

    def step(self, action):
        """Step environment and convert observation to grayscale."""
        obs, reward, done, info = self.env.step(action)
        return self._apply_grayscale(obs), reward, done, info


class ResizeObservation(gym.Wrapper):
    """
    Resize observations to a target shape.

    Downscaling the 240x256 NES output to a smaller size (typically 84x84)
    reduces the number of input features for the neural network while
    keeping enough visual detail for the agent to learn.

    The 84x84 size is a standard choice from the Atari DQN paper
    (Mnih et al., 2015) and works well for most retro game environments.

    Uses gym.Wrapper instead of ObservationWrapper for compatibility with
    gym-super-mario-bros (which uses the old gym reset API that returns
    just obs instead of (obs, info) tuple).

    Args:
        env (gym.Env): The environment to wrap.
        shape (tuple): Target (height, width) for the resized observation.
                       Default is (84, 84).

    Note:
        Uses bilinear interpolation (cv2.INTER_AREA) for downsampling,
        which produces the best results for reducing image size.
    """

    def __init__(self, env: gym.Env, shape: tuple = (84, 84)):
        super().__init__(env)
        self._shape = shape
        # Get number of channels from current observation space
        channels = self.observation_space.shape[-1]
        # Update observation space with new dimensions
        self.observation_space = Box(
            low=0, high=255, shape=(*shape, channels), dtype=np.uint8
        )

    def _apply_resize(self, observation: np.ndarray) -> np.ndarray:
        """
        Resize observation to target shape.

        Args:
            observation: Image array of shape (H, W, C).

        Returns:
            Resized image array of shape (target_h, target_w, C).
        """
        # cv2.resize expects (width, height), but our shape is (height, width)
        resized = cv2.resize(
            observation,
            (self._shape[1], self._shape[0]),
            interpolation=cv2.INTER_AREA
        )
        # If grayscale (2D after resize), add channel dimension back
        if resized.ndim == 2:
            resized = np.expand_dims(resized, axis=-1)
        return resized

    def reset(self, **kwargs):
        """
        Reset environment and resize initial observation.

        Handles both old gym API (returns obs) and new API (returns obs, info).

        Returns:
            Resized observation of shape (target_h, target_w, C).
        """
        # Strip new-API kwargs that old gym envs don't understand
        kwargs = _strip_new_api_kwargs(kwargs)
        # gym-super-mario-bros uses old API (returns just obs)
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        return self._apply_resize(obs)

    def step(self, action):
        """Step environment and resize observation."""
        obs, reward, done, info = self.env.step(action)
        return self._apply_resize(obs), reward, done, info


class FrameStackObservation(gym.Wrapper):
    """
    Stack N consecutive frames as a single observation.

    Frame stacking gives the agent temporal information (motion, velocity)
    from a single observation. Without stacking, the agent sees a static
    image and cannot determine which direction enemies or Mario are moving.

    Typically 4 frames are stacked, providing information about the last
    ~0.27 seconds of gameplay (at 15 decisions/sec with frame skip of 4).

    The stacked frames are concatenated along the channel dimension,
    producing an observation of shape (H, W, N) where N is the stack size.

    Uses gym.Wrapper instead of ObservationWrapper for compatibility with
    gym-super-mario-bros (which uses the old gym reset API).

    Args:
        env (gym.Env): The environment to wrap.
        num_stack (int): Number of frames to stack. Default is 4.

    Example:
        With grayscale 84x84 input and num_stack=4:
        - Input per frame: (84, 84, 1)
        - Stacked output: (84, 84, 4)
    """

    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self._num_stack = num_stack
        # Initialize frame buffer with zeros
        self._frames = deque(maxlen=num_stack)
        # Get single frame shape (without channel dim)
        old_shape = self.observation_space.shape
        # New observation has num_stack channels instead of 1
        self.observation_space = Box(
            low=0, high=255,
            shape=(old_shape[0], old_shape[1], num_stack),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        """
        Reset environment and fill frame stack with initial observation.

        Returns:
            Stacked observation of shape (H, W, num_stack).
        """
        # Strip new-API kwargs that old gym envs don't understand
        kwargs = _strip_new_api_kwargs(kwargs)
        # gym-super-mario-bros returns just obs (old API)
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        # Fill the frame buffer with copies of the first frame
        for _ in range(self._num_stack):
            self._frames.append(obs)
        return self._get_observation()

    def step(self, action):
        """Step environment and add frame to stack."""
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        """
        Concatenate stacked frames along the channel axis.

        Returns:
            np.ndarray: Stacked observation of shape (H, W, num_stack).
        """
        # Concatenate frames along the last axis (channels)
        # Each frame is (H, W, 1), result is (H, W, num_stack)
        return np.concatenate(list(self._frames), axis=-1)


class CustomRewardWrapper(gym.Wrapper):
    """
    Custom reward shaping for better learning signals.

    The default Super Mario Bros reward is based on x-position change,
    which is sparse and doesn't provide enough learning signal. This
    wrapper adds additional reward components:

    Reward components:
        - Progress: Reward proportional to rightward movement (+)
        - Death penalty: Large negative reward for losing a life (-)
        - Coin bonus: Reward for collecting coins (+)
        - Level complete: Large bonus for reaching the flag (+)
        - Time penalty: Small penalty per step to encourage speed (-)

    The shaped reward helps all three algorithms (NEAT, PPO, DQN) learn
    faster by providing more informative feedback.

    Args:
        env (gym.Env): The environment to wrap.
        death_penalty (float): Penalty for dying. Default -15.
        coin_reward (float): Reward per coin collected. Default 5.
        flag_reward (float): Reward for completing level. Default 100.
        time_penalty (float): Penalty per step. Default -0.01.
    """

    def __init__(
        self,
        env: gym.Env,
        death_penalty: float = -15.0,
        coin_reward: float = 5.0,
        flag_reward: float = 100.0,
        time_penalty: float = -0.01,
    ):
        super().__init__(env)
        self._death_penalty = death_penalty
        self._coin_reward = coin_reward
        self._flag_reward = flag_reward
        self._time_penalty = time_penalty
        # Track previous state for delta calculations
        self._prev_x_pos = 0
        self._prev_coins = 0
        self._prev_life = 2  # Mario starts with 2 lives
        self._stage_completions = 0  # Total flag captures this session

    def reset(self, **kwargs):
        """Reset environment and tracking variables."""
        kwargs = _strip_new_api_kwargs(kwargs)
        obs = self.env.reset(**kwargs)
        self._prev_x_pos = 0
        self._prev_coins = 0
        self._prev_life = 2
        return obs

    def step(self, action: int):
        """
        Execute action and compute shaped reward.

        Args:
            action: The action to execute.

        Returns:
            tuple: (observation, shaped_reward, done, info)
        """
        obs, reward, done, info = self.env.step(action)

        # Start with the environment's base reward (x-position delta)
        shaped_reward = reward

        # Death penalty: large negative reward when Mario loses a life
        current_life = info.get('life', 2)
        if current_life < self._prev_life:
            shaped_reward += self._death_penalty
        self._prev_life = current_life

        # Coin collection bonus
        current_coins = info.get('coins', 0)
        coins_collected = current_coins - self._prev_coins
        if coins_collected > 0:
            shaped_reward += coins_collected * self._coin_reward
        self._prev_coins = current_coins

        # Level completion bonus (flag reached).
        # Also mark stage_completed in info so trainers can track it.
        stage_completed = info.get('flag_get', False)
        if stage_completed:
            shaped_reward += self._flag_reward
            self._stage_completions += 1
        info['stage_completed'] = stage_completed
        info['stage_completions'] = self._stage_completions

        # Small time penalty to encourage faster completion
        shaped_reward += self._time_penalty

        return obs, shaped_reward, done, info


def _make_sb3_compat_wrapper_class():
    """
    Factory that creates SB3CompatWrapper inheriting from gymnasium.Env.

    We use a factory function so `import gymnasium` only happens when
    SB3CompatWrapper is actually needed (PPO training), not on every import
    of the wrappers module (which would break NEAT/DQN that don't need gymnasium).
    """
    import gymnasium

    class SB3CompatWrapper(gymnasium.Env):
        """
        Compatibility wrapper bridging old gym API to stable-baselines3.

        stable-baselines3 (via DummyVecEnv) expects the new gymnasium API:
            - reset() returns (obs, info)
            - step() returns (obs, reward, terminated, truncated, info)

        Our environment chain uses the old gym API:
            - reset() returns obs
            - step() returns (obs, reward, done, info)

        This wrapper sits at the outermost layer and translates between
        the two APIs. It inherits from gymnasium.Env so SB3's _patch_env()
        recognizes it and does NOT double-wrap it with shimmy.

        Args:
            env (gym.Env): The environment to wrap (our full wrapper chain).
        """

        def __init__(self, env):
            super().__init__()
            self.env = env
            # Convert spaces from old gym → gymnasium so SB3 works correctly
            self.observation_space = gymnasium.spaces.Box(
                low=env.observation_space.low,
                high=env.observation_space.high,
                shape=env.observation_space.shape,
                dtype=env.observation_space.dtype,
            )
            self.action_space = gymnasium.spaces.Discrete(
                n=env.action_space.n,
            )
            self.metadata = getattr(env, 'metadata', {})
            self.render_mode = None
            self.reward_range = getattr(env, 'reward_range', (-float('inf'), float('inf')))
            self.spec = None

        @property
        def unwrapped(self):
            """Access the underlying unwrapped env (for raw NES screen capture)."""
            return self.env.unwrapped

        def reset(self, seed=None, options=None, **kwargs):
            """
            Reset and return (obs, info) tuple as gymnasium expects.

            The seed/options params are accepted but not passed down —
            the old gym env doesn't understand them.
            """
            obs = self.env.reset()
            return obs, {}

        def step(self, action):
            """
            Step and return 5-tuple as gymnasium expects.

            Converts old gym's (obs, reward, done, info) to gymnasium's
            (obs, reward, terminated, truncated, info).
            """
            obs, reward, done, info = self.env.step(action)
            terminated = done
            truncated = False
            return obs, reward, terminated, truncated, info

        def render(self, *args, **kwargs):
            """Render the environment (passes through mode parameter)."""
            return self.env.render(*args, **kwargs)

        def close(self):
            """Close the environment."""
            return self.env.close()

    return SB3CompatWrapper


def SB3CompatWrapper(env):
    """
    Create an SB3-compatible wrapper around an old gym environment.

    This is a convenience function that creates the wrapper class
    (which inherits from gymnasium.Env) and instantiates it.
    """
    cls = _make_sb3_compat_wrapper_class()
    return cls(env)
