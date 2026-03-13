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

    def _segment_color(self, seg_idx: int, n: int):
        """Color for a snake segment. Head=bright lime, tail=darker green."""
        t = 1.0 - seg_idx / max(n, 1)  # 0=tail end, 1=head end
        g = int(120 + 135 * t)          # 120..255
        r = int(30 + 80 * t)            # 30..110
        return (r, g, 20)

    def _render_rgb(self) -> np.ndarray:
        """Render a classic-style snake game for dashboard/stream.

        High-contrast design: bright green snake on a black background
        with visible grid lines — instantly recognizable as Snake.
        Connected body via filled rectangles between segments.
        """
        size = self.DISPLAY_SIZE
        cell = size // self.grid_size  # 30px per cell at 480/16
        img_size = cell * self.grid_size
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Pure black background — classic snake look, high contrast
        img[:] = (10, 10, 10)

        # Visible grid lines (dark gray on black — subtle but clear)
        for i in range(1, self.grid_size):
            pos = i * cell
            cv2.line(img, (pos, 0), (pos, img_size), (35, 35, 35), 1)
            cv2.line(img, (0, pos), (img_size, pos), (35, 35, 35), 1)

        n = len(self.snake)
        half = cell // 2
        # Body thickness: nearly fills the cell for visibility
        thick = cell // 2 - 2  # half-width of the body band

        # --- Draw connected body: rects between segments + circles ---
        # Tail-to-head order so brighter head colors draw on top.
        for i in range(n - 1, 0, -1):
            r1, c1 = self.snake[i]
            r2, c2 = self.snake[i - 1]
            cx1, cy1 = c1 * cell + half, r1 * cell + half
            cx2, cy2 = c2 * cell + half, r2 * cell + half
            color = self._segment_color(i - 1, n)
            if r1 == r2:
                min_x, max_x = min(cx1, cx2), max(cx1, cx2)
                cv2.rectangle(img, (min_x, cy1 - thick),
                              (max_x, cy1 + thick), color, -1)
            elif c1 == c2:
                min_y, max_y = min(cy1, cy2), max(cy1, cy2)
                cv2.rectangle(img, (cx1 - thick, min_y),
                              (cx1 + thick, max_y), color, -1)

        # Rounded joints at every segment
        for i in range(n - 1, -1, -1):
            r, c = self.snake[i]
            cx, cy = c * cell + half, r * cell + half
            color = self._segment_color(i, n)
            cv2.circle(img, (cx, cy), thick, color, -1, cv2.LINE_AA)

        # Thin dark outline along body edges for definition
        for i in range(n - 1, -1, -1):
            r, c = self.snake[i]
            cx, cy = c * cell + half, r * cell + half
            cv2.circle(img, (cx, cy), thick, (20, 50, 10), 1, cv2.LINE_AA)

        # --- Head: distinctly larger and brighter ---
        if self.snake:
            hr, hc = self.snake[0]
            hcx, hcy = hc * cell + half, hr * cell + half
            head_r = thick + 3
            # Bright lime-green head stands out from body
            cv2.circle(img, (hcx, hcy), head_r, (60, 255, 60), -1,
                       cv2.LINE_AA)
            cv2.circle(img, (hcx, hcy), head_r, (30, 120, 30), 2,
                       cv2.LINE_AA)
            # Highlight
            cv2.circle(img, (hcx - head_r // 3, hcy - head_r // 3),
                       max(2, head_r // 2), (140, 255, 140), -1,
                       cv2.LINE_AA)

            dr, dc = self._directions[self.direction]
            eye_off = cell // 4
            eye_r = max(3, cell // 6)
            pupil_r = max(2, cell // 10)
            for side in (-1, 1):
                ex = hcx + dc * eye_off + (-dr) * side * (cell // 4)
                ey = hcy + dr * eye_off + dc * side * (cell // 4)
                cv2.circle(img, (ex, ey), eye_r,
                           (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (ex + dc * 2, ey + dr * 2), pupil_r,
                           (10, 10, 10), -1, cv2.LINE_AA)

            # Red forked tongue
            tongue_len = cell // 3
            base_x = hcx + dc * head_r
            base_y = hcy + dr * head_r
            tip_x = base_x + dc * tongue_len
            tip_y = base_y + dr * tongue_len
            cv2.line(img, (base_x, base_y), (tip_x, tip_y),
                     (220, 50, 50), 2, cv2.LINE_AA)
            fork = cell // 5
            for side in (-1, 1):
                fx = tip_x + dc * fork + (-dr) * side * fork
                fy = tip_y + dr * fork + dc * side * fork
                cv2.line(img, (tip_x, tip_y), (fx, fy),
                         (220, 50, 50), 2, cv2.LINE_AA)

        # --- Food: bright red apple (high contrast on black) ---
        if self.food:
            fr, fc = self.food
            fcy = fr * cell + half
            fcx = fc * cell + half
            food_r = cell // 2 - 2
            # Glow ring
            cv2.circle(img, (fcx, fcy), food_r + 3, (80, 15, 15), -1,
                       cv2.LINE_AA)
            # Apple body
            cv2.circle(img, (fcx, fcy), food_r, (230, 40, 40), -1,
                       cv2.LINE_AA)
            # Specular highlight
            cv2.circle(img, (fcx - food_r // 3, fcy - food_r // 3),
                       max(2, food_r // 3), (255, 150, 150), -1,
                       cv2.LINE_AA)
            # Stem
            stem_top = fcy - food_r - 4
            cv2.line(img, (fcx, fcy - food_r + 2), (fcx + 1, stem_top),
                     (90, 60, 30), 2, cv2.LINE_AA)
            # Leaf
            leaf_pts = np.array([
                [fcx + 2, stem_top],
                [fcx + food_r // 2 + 4, stem_top - 4],
                [fcx + 3, stem_top + 3],
            ], dtype=np.int32)
            cv2.fillConvexPoly(img, leaf_pts, (50, 180, 60),
                               lineType=cv2.LINE_AA)

        # Bright green border (like a classic game frame)
        cv2.rectangle(img, (0, 0), (img_size - 1, img_size - 1),
                      (30, 140, 30), 3)

        # Score overlay (top-right) — white text for visibility
        score_text = f"Score: {self._score}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_w = cv2.getTextSize(score_text, font, 0.6, 2)[0][0]
        tx = img_size - text_w - 12
        # Background box for readability
        cv2.rectangle(img, (tx - 4, 2), (img_size - 4, 28),
                      (10, 10, 10), -1)
        cv2.putText(img, score_text, (tx, 22), font, 0.6,
                    (60, 255, 60), 2, cv2.LINE_AA)

        # Length indicator (top-left)
        len_text = f"Len: {n}"
        cv2.rectangle(img, (4, 2), (90, 28), (10, 10, 10), -1)
        cv2.putText(img, len_text, (8, 22), font, 0.6,
                    (60, 255, 60), 2, cv2.LINE_AA)

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
