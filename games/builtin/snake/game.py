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

    # High-res rendering for the dashboard (independent of ML obs size)
    DISPLAY_SIZE = 480

    def _render_rgb(self) -> np.ndarray:
        """Render a colorful version for dashboard/stream display.

        Renders at 480px for crisp visuals. Green gradient snake with
        rounded segments, red apple with glow, score overlay, and
        subtle gridlines on a dark field.
        """
        size = self.DISPLAY_SIZE
        cell = size // self.grid_size  # 30px per cell at 480/16
        img_size = cell * self.grid_size
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Background: dark green-tinted field
        img[:] = (18, 22, 18)

        # Subtle gridlines
        for i in range(1, self.grid_size):
            pos = i * cell
            cv2.line(img, (pos, 0), (pos, img_size), (30, 38, 30), 1)
            cv2.line(img, (0, pos), (img_size, pos), (30, 38, 30), 1)

        # Draw snake body with gradient and rounded segments
        n = len(self.snake)
        for idx, segment in enumerate(reversed(self.snake)):
            # Draw tail-to-head so head draws last (on top)
            seg_idx = n - 1 - idx  # 0 = head, n-1 = tail
            r, c = segment
            cy_center = r * cell + cell // 2
            cx_center = c * cell + cell // 2
            # Gradient: tail is dark, head is bright
            t = 1.0 - seg_idx / max(n, 1)  # 0=tail, 1=head
            g_val = int(80 + 160 * t)
            b_val = int(30 + 50 * t)
            color = (10, g_val, b_val)
            radius = cell // 2 - 2
            cv2.circle(img, (cx_center, cy_center), radius, color, -1,
                       lineType=cv2.LINE_AA)
            # Subtle border
            cv2.circle(img, (cx_center, cy_center), radius,
                       (10, min(255, g_val + 30), b_val + 10), 1,
                       lineType=cv2.LINE_AA)

        # Snake head with eyes and distinct color
        if self.snake:
            hr, hc = self.snake[0]
            hcy = hr * cell + cell // 2
            hcx = hc * cell + cell // 2
            head_r = cell // 2 - 1
            cv2.circle(img, (hcx, hcy), head_r, (30, 220, 70), -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(img, (hcx, hcy), head_r, (50, 255, 100), 1,
                       lineType=cv2.LINE_AA)
            # Eyes (direction-aware)
            dr, dc = self._directions[self.direction]
            eye_offset = cell // 5
            for side in (-1, 1):
                # Perpendicular offset for two eyes
                ex = hcx + dc * eye_offset + (-dr) * side * (cell // 5)
                ey = hcy + dr * eye_offset + dc * side * (cell // 5)
                cv2.circle(img, (ex, ey), max(2, cell // 8),
                           (255, 255, 255), -1, lineType=cv2.LINE_AA)
                # Pupil
                cv2.circle(img, (ex + dc, ey + dr), max(1, cell // 14),
                           (20, 20, 20), -1, lineType=cv2.LINE_AA)

        # Food: bright red apple with glow
        if self.food:
            fr, fc = self.food
            fcy = fr * cell + cell // 2
            fcx = fc * cell + cell // 2
            food_r = cell // 2 - 2
            # Outer glow
            cv2.circle(img, (fcx, fcy), food_r + 3, (60, 15, 15), -1,
                       lineType=cv2.LINE_AA)
            # Main apple
            cv2.circle(img, (fcx, fcy), food_r, (220, 40, 40), -1,
                       lineType=cv2.LINE_AA)
            # Highlight
            cv2.circle(img, (fcx - food_r // 3, fcy - food_r // 3),
                       max(2, food_r // 3), (255, 130, 130), -1,
                       lineType=cv2.LINE_AA)

        # Score overlay (top-right) showing snake length
        score_text = f"Length: {n}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, score_text, (img_size - 145, 22), font, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, score_text, (img_size - 145, 22), font, 0.55,
                    (100, 255, 140), 1, cv2.LINE_AA)

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
