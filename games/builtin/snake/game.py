"""
Snake game engine as a Gym environment.

A self-contained implementation of the classic Snake game that
follows the Gym interface. The snake moves on a grid, eats food
to grow, and dies if it hits a wall or itself.

Observations are rendered as 84x84 grayscale images:
- Snake body: white (255)
- Food: gray (180)
- Background: black (0)
"""
import random

import cv2
import numpy as np
try:
    import gymnasium as gym
except ImportError:
    import gym  # Legacy fallback

from gymnasium.spaces import Box, Discrete


class SnakeEnv(gym.Env):
    """Snake game as a Gym environment.

    Args:
        grid_size: Size of the grid (grid_size x grid_size). Default 16.
        max_steps_without_food: Steps before timeout. Default grid_size^2.
        render_size: Pixel size of rendered observation. Default 84.
    """

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, grid_size: int = 16, max_steps_without_food: int = 0,
                 render_size: int = 84):
        super().__init__()
        self.grid_size = grid_size
        self.render_size = render_size
        self.max_steps_without_food = max_steps_without_food or grid_size * grid_size

        # 4 actions: 0=up, 1=right, 2=down, 3=left
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=0, high=255,
            shape=(render_size, render_size, 1),
            dtype=np.uint8,
        )

        # Direction vectors: up, right, down, left
        self._directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        self.snake = []
        self.food = None
        self.direction = 1  # Start moving right
        self._steps_since_food = 0
        self._score = 0

    def reset(self):
        # Place snake in center with 3 segments (standard Snake gameplay)
        center = self.grid_size // 2
        self.snake = [
            (center, center),         # Head
            (center, center - 1),     # Body (one left of head)
            (center, center - 2),     # Tail (two left of head)
        ]
        self.direction = 1  # Right
        self._steps_since_food = 0
        self._score = 0
        self._place_food()
        return self._render_obs()

    def step(self, action):
        # Prevent 180-degree turns (can't reverse into yourself)
        opposite = {0: 2, 1: 3, 2: 0, 3: 1}
        if action != opposite.get(self.direction, -1):
            self.direction = action

        # Move head
        head_r, head_c = self.snake[0]
        dr, dc = self._directions[self.direction]
        new_head = (head_r + dr, head_c + dc)

        # Check wall collision
        r, c = new_head
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return self._render_obs(), -1.0, True, self._info()

        # Check self collision
        if new_head in self.snake:
            return self._render_obs(), -1.0, True, self._info()

        # Move
        self.snake.insert(0, new_head)

        # Check food
        reward = 0.0
        if new_head == self.food:
            self._score += 1
            self._steps_since_food = 0
            self._place_food()
            reward = 1.0
        else:
            self.snake.pop()  # Remove tail (no growth)
            self._steps_since_food += 1

        # Timeout check
        done = self._steps_since_food >= self.max_steps_without_food

        return self._render_obs(), reward, done, self._info()

    def _place_food(self):
        """Place food on a random empty cell."""
        empty = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in self.snake:
                    empty.append((r, c))
        if empty:
            self.food = random.choice(empty)
        else:
            self.food = None  # Board full (win condition)

    def _render_obs(self) -> np.ndarray:
        """Render grid as an 84x84 grayscale image."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Draw snake
        for segment in self.snake:
            grid[segment[0], segment[1]] = 255

        # Draw food
        if self.food:
            grid[self.food[0], self.food[1]] = 180

        # Resize to render_size x render_size
        obs = cv2.resize(grid, (self.render_size, self.render_size),
                         interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(obs, axis=-1)

    def _render_rgb(self) -> np.ndarray:
        """Render a colorful version for dashboard/stream display.

        Green snake with lighter head, red food, dark grid with
        subtle gridlines — much more visually appealing for viewers.
        """
        cell_px = max(2, self.render_size // self.grid_size)
        img_size = cell_px * self.grid_size
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Background: very dark green-gray
        img[:] = (20, 25, 20)

        # Subtle gridlines
        for i in range(1, self.grid_size):
            pos = i * cell_px
            img[pos, :] = (35, 40, 35)
            img[:, pos] = (35, 40, 35)

        # Draw snake body (green gradient — darker tail, brighter head)
        n = len(self.snake)
        for idx, segment in enumerate(self.snake):
            r, c = segment
            y, x = r * cell_px, c * cell_px
            # Gradient from tail (dark green) to head (bright green)
            brightness = 0.4 + 0.6 * (1 - idx / max(n, 1))
            g = int(200 * brightness)
            b_val = int(60 * brightness)
            color = (0, g, b_val)
            img[y + 1:y + cell_px - 1, x + 1:x + cell_px - 1] = color

        # Snake head: brightest green with white eye
        if self.snake:
            hr, hc = self.snake[0]
            hy, hx = hr * cell_px, hc * cell_px
            img[hy + 1:hy + cell_px - 1, hx + 1:hx + cell_px - 1] = (40, 230, 80)
            # Eye (small white dot)
            eye_y = hy + cell_px // 3
            eye_x = hx + cell_px // 3
            if cell_px > 4:
                img[eye_y:eye_y + 2, eye_x:eye_x + 2] = (255, 255, 255)

        # Food: bright red with slight glow
        if self.food:
            fr, fc = self.food
            fy, fx = fr * cell_px, fc * cell_px
            img[fy + 1:fy + cell_px - 1, fx + 1:fx + cell_px - 1] = (220, 50, 50)
            # Glow effect (lighter border)
            if cell_px > 4:
                img[fy, fx + 1:fx + cell_px - 1] = (180, 40, 40)
                img[fy + cell_px - 1, fx + 1:fx + cell_px - 1] = (180, 40, 40)
                img[fy + 1:fy + cell_px - 1, fx] = (180, 40, 40)
                img[fy + 1:fy + cell_px - 1, fx + cell_px - 1] = (180, 40, 40)

        # Resize to match render_size
        if img_size != self.render_size:
            img = cv2.resize(img, (self.render_size, self.render_size),
                             interpolation=cv2.INTER_NEAREST)
        return img

    def render(self, mode='rgb_array'):
        """Render as RGB for dashboard display."""
        return self._render_rgb()

    def _info(self) -> dict:
        return {
            'score': self._score,
            'snake_length': len(self.snake),
            'max_length': self.grid_size * self.grid_size,
        }

    def close(self):
        pass
