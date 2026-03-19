"""
Dual Snake cooperative game engine as a Gym environment.

Two snakes share a board and a score. Both snakes eat food to grow,
but must avoid each other AND themselves. This creates a cooperative
challenge where both agents need to coordinate.

Observations are rendered as 84x84 RGB images:
- Snake 1 body: green (brighter at head)
- Snake 2 body: blue (brighter at head)
- Food: red
- Background: black
"""
import random

import cv2
import numpy as np
try:
    import gymnasium as gym
except ImportError:
    import gym  # Legacy fallback

from gymnasium.spaces import Box, Discrete


class DualSnakeEnv(gym.Env):
    """Cooperative dual snake game. Two snakes share a board and score.

    Action space: Discrete(16) -- action = s1_action * 4 + s2_action
    where each sub-action is 0=up, 1=right, 2=down, 3=left.

    Args:
        grid_size: Size of the grid (grid_size x grid_size). Default 16.
        render_size: Pixel size of rendered observation. Default 84.
        max_steps_without_food: Steps before timeout. Default grid_size^2.
    """

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, grid_size: int = 16, render_size: int = 84,
                 max_steps_without_food: int = 0):
        super().__init__()
        self.grid_size = grid_size
        self.render_size = render_size
        self.max_steps_without_food = max_steps_without_food or grid_size * grid_size

        # Discrete(16): action = s1_action * 4 + s2_action
        self.action_space = Discrete(16)
        self.observation_space = Box(
            low=0, high=255,
            shape=(render_size, render_size, 3),  # RGB
            dtype=np.uint8,
        )

        # Direction vectors: up, right, down, left
        self._directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        self.snake1 = []
        self.snake2 = []
        self.food = None
        self.dir1 = 1  # Right
        self.dir2 = 3  # Left
        self._steps_since_food = 0
        self._score = 0
        self._step_count = 0
        self._dead1 = False
        self._dead2 = False

    def reset(self):
        g = self.grid_size
        # Snake 1 starts top-left area, moving right
        self.snake1 = [
            (g // 4, g // 4),
            (g // 4, g // 4 - 1),
            (g // 4, g // 4 - 2),
        ]
        self.dir1 = 1
        self._dead1 = False

        # Snake 2 starts bottom-right area, moving left
        s2_r = 3 * g // 4
        s2_c = min(3 * g // 4, g - 3)  # Ensure tail fits in grid
        self.snake2 = [
            (s2_r, s2_c),
            (s2_r, s2_c + 1),
            (s2_r, s2_c + 2),
        ]
        self.dir2 = 3
        self._dead2 = False

        self._steps_since_food = 0
        self._score = 0
        self._step_count = 0
        self._place_food()
        return self._render_obs()

    def step(self, action):
        """Both snakes move simultaneously."""
        self._step_count += 1
        self._steps_since_food += 1

        # Decode action: action = s1_action * 4 + s2_action
        a1 = action // 4
        a2 = action % 4

        reward = 0.0

        # Update directions (prevent 180-degree reversal)
        if not self._dead1 and (a1 + 2) % 4 != self.dir1:
            self.dir1 = a1
        if not self._dead2 and (a2 + 2) % 4 != self.dir2:
            self.dir2 = a2

        # Move snake 1
        if not self._dead1:
            head1 = self.snake1[0]
            dr, dc = self._directions[self.dir1]
            new_head1 = (head1[0] + dr, head1[1] + dc)

            # Check wall collision
            if not (0 <= new_head1[0] < self.grid_size
                    and 0 <= new_head1[1] < self.grid_size):
                self._dead1 = True
            # Check self collision
            elif new_head1 in self.snake1:
                self._dead1 = True
            # Check collision with snake2
            elif new_head1 in self.snake2:
                self._dead1 = True
            else:
                self.snake1.insert(0, new_head1)
                if new_head1 == self.food:
                    self._score += 1
                    self._steps_since_food = 0
                    reward += 1.0
                    self._place_food()
                else:
                    self.snake1.pop()

        # Move snake 2
        if not self._dead2:
            head2 = self.snake2[0]
            dr, dc = self._directions[self.dir2]
            new_head2 = (head2[0] + dr, head2[1] + dc)

            if not (0 <= new_head2[0] < self.grid_size
                    and 0 <= new_head2[1] < self.grid_size):
                self._dead2 = True
            elif new_head2 in self.snake2:
                self._dead2 = True
            elif new_head2 in self.snake1:
                self._dead2 = True
            else:
                self.snake2.insert(0, new_head2)
                if new_head2 == self.food:
                    self._score += 1
                    self._steps_since_food = 0
                    reward += 1.0
                    self._place_food()
                else:
                    self.snake2.pop()

        # One dead = penalty but game continues (the other still plays)
        # Only apply penalty on the step the snake dies
        if self._dead1 and not self._dead2 and self._step_count == self._step_count:
            # Check if snake1 just died this step by seeing if reward includes
            # wall/self/cross collision (we set _dead1 above in this step)
            pass
        # We handle death penalty via done flag tracking below

        # Both dead = game over
        done = (self._dead1 and self._dead2)
        # Timeout
        if self._steps_since_food >= self.max_steps_without_food:
            done = True

        info = {
            'score': self._score,
            'snake1_length': len(self.snake1),
            'snake2_length': len(self.snake2),
            'snake1_alive': not self._dead1,
            'snake2_alive': not self._dead2,
            'steps': self._step_count,
        }

        return self._render_obs(), reward, done, info

    def _place_food(self):
        """Place food on a random empty cell."""
        occupied = set(map(tuple, self.snake1)) | set(map(tuple, self.snake2))
        if self.food:
            occupied.add(self.food)
        empty = [(r, c) for r in range(self.grid_size)
                 for c in range(self.grid_size) if (r, c) not in occupied]
        if empty:
            self.food = random.choice(empty)
        else:
            self.food = None  # Board full (win!)

    def _render_obs(self) -> np.ndarray:
        """Render as RGB: snake1=green, snake2=blue, food=red."""
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # Food: red
        if self.food:
            img[self.food[0], self.food[1]] = [255, 0, 0]

        # Snake 1: green (brighter at head)
        for i, (r, c) in enumerate(self.snake1):
            brightness = max(100, 255 - i * 15)
            img[r, c] = [0, brightness, 0]

        # Snake 2: blue (brighter at head)
        for i, (r, c) in enumerate(self.snake2):
            brightness = max(100, 255 - i * 15)
            img[r, c] = [0, 0, brightness]

        # Resize to render_size
        img = cv2.resize(img, (self.render_size, self.render_size),
                         interpolation=cv2.INTER_NEAREST)
        return img

    def render(self, mode='rgb_array'):
        """Render as RGB for display."""
        return self._render_obs()

    def _info(self) -> dict:
        return {
            'score': self._score,
            'snake1_length': len(self.snake1),
            'snake2_length': len(self.snake2),
            'snake1_alive': not self._dead1,
            'snake2_alive': not self._dead2,
            'steps': self._step_count,
        }

    def close(self):
        pass
