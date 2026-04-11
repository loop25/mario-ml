"""
Custom Game Environment Template.

Implement your game as a Gymnasium environment. The framework will
call reset() to start a new episode and step(action) each frame.

TODO:
    1. Define your observation_space and action_space in __init__
    2. Implement reset() to initialize a new episode
    3. Implement step() to advance the game by one action
    4. Implement _render_frame() to return an RGB numpy array

See games/builtin/snake/game.py for a complete example.
"""
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from gymnasium.spaces import Box, Discrete


class CustomGameEnv(gym.Env):
    """Custom game environment.

    Replace this docstring with a description of your game.
    """

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, render_size: int = 84):
        super().__init__()
        self.render_size = render_size

        # TODO: Define your action space (number of possible actions)
        self.action_space = Discrete(4)

        # TODO: Define your observation space (image shape)
        self.observation_space = Box(
            low=0, high=255,
            shape=(render_size, render_size, 1),
            dtype=np.uint8,
        )

    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation.

        TODO: Initialize your game state here.
        """
        super().reset(seed=seed)
        observation = np.zeros(
            (self.render_size, self.render_size, 1), dtype=np.uint8
        )
        return observation, {}

    def step(self, action):
        """Execute one step in the environment.

        TODO: Update game state based on the action.

        Returns:
            observation: Current game frame
            reward: Reward for this step
            terminated: True if episode ended (win/lose)
            truncated: True if episode was cut short (timeout)
            info: Extra information dict
        """
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        observation = self._render_frame()
        return observation, reward, terminated, truncated, info

    def _render_frame(self) -> np.ndarray:
        """Render the current game state as an image.

        TODO: Draw your game state onto a numpy array.

        Returns:
            np.ndarray of shape (render_size, render_size, 1), dtype uint8.
        """
        frame = np.zeros(
            (self.render_size, self.render_size, 1), dtype=np.uint8
        )
        return frame

    def render(self):
        """Return RGB frame for visualization."""
        return self._render_frame()
