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
        self._trail = []       # last ~8 positions the head vacated
        self._dead = False      # set True on collision for death flash
        self._step_count = 0    # global tick for animation timing

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
        self._trail = []
        self._dead = False
        self._step_count = 0
        self._place_food()
        return self._render_obs()

    def step(self, action):
        self._step_count += 1

        # Prevent 180-degree turns (can't reverse into yourself)
        opposite = {0: 2, 1: 3, 2: 0, 3: 1}
        if action != opposite.get(self.direction, -1):
            self.direction = action

        # Track trail (head position before moving)
        self._trail.append(self.snake[0])
        if len(self._trail) > 8:
            self._trail = self._trail[-8:]

        # Move head
        head_r, head_c = self.snake[0]
        dr, dc = self._directions[self.direction]
        new_head = (head_r + dr, head_c + dc)

        # Check wall collision
        r, c = new_head
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            self._dead = True
            return self._render_obs(), -1.0, True, self._info()

        # Check self collision
        if new_head in self.snake:
            self._dead = True
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

            # Tiny proximity signal: reward moving closer to food,
            # penalize moving away. Keeps total shaping << food reward.
            old_dist = abs(head_r - self.food[0]) + abs(head_c - self.food[1])
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = (old_dist - new_dist) * 0.005  # ~±0.005 per step

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
        """Render a polished, stream-quality snake game for dashboard.

        Features: gradient background, checkerboard grid, snake glow,
        pulsing food rings, trail ghost marks, gradient HUD banner,
        and red death flash.
        """
        size = self.DISPLAY_SIZE
        cell = size // self.grid_size  # 30px per cell at 480/16
        img_size = cell * self.grid_size
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # --- 1. Gradient background: dark green, lighter toward center ---
        cx_bg = img_size / 2.0
        cy_bg = img_size / 2.0
        max_dist = (cx_bg ** 2 + cy_bg ** 2) ** 0.5
        # Build row of distances from center for each pixel row
        ys = np.arange(img_size, dtype=np.float32) - cy_bg
        xs = np.arange(img_size, dtype=np.float32) - cx_bg
        dist_sq = ys[:, None] ** 2 + xs[None, :] ** 2
        dist = np.sqrt(dist_sq)
        t = dist / max_dist  # 0 at center, 1 at corners
        # Background: center=(14, 22, 12), edges=(6, 8, 6)
        img[:, :, 0] = (14 - 8 * t).astype(np.uint8)   # B
        img[:, :, 1] = (22 - 14 * t).astype(np.uint8)   # G
        img[:, :, 2] = (12 - 6 * t).astype(np.uint8)    # R

        # --- 2. Checkerboard grid (alternating dark squares) ---
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row + col) % 2 == 0:
                    x0 = col * cell
                    y0 = row * cell
                    # Slightly lighter square overlay
                    overlay = img[y0:y0 + cell, x0:x0 + cell]
                    img[y0:y0 + cell, x0:x0 + cell] = np.clip(
                        overlay.astype(np.int16) + 6, 0, 255
                    ).astype(np.uint8)

        n = len(self.snake)
        half = cell // 2
        thick = cell // 2 - 2  # half-width of the body band

        # --- 5. Trail effect: faint marks on recently vacated cells ---
        snake_set = set(self.snake)
        for ti, pos in enumerate(self._trail):
            if pos in snake_set:
                continue  # don't draw trail under current body
            tr, tc = pos
            tcx, tcy = tc * cell + half, tr * cell + half
            age = len(self._trail) - ti  # 1=newest, len=oldest
            alpha = max(0.05, 0.25 - age * 0.025)
            glow_c = (int(15 * alpha * 4), int(50 * alpha * 4),
                      int(10 * alpha * 4))
            cv2.circle(img, (tcx, tcy), thick - 2, glow_c, -1,
                       cv2.LINE_AA)

        # --- 3. Snake body glow (subtle bloom behind body) ---
        for i in range(n - 1, -1, -1):
            r, c = self.snake[i]
            cx_s, cy_s = c * cell + half, r * cell + half
            seg_c = self._segment_color(i, n)
            glow_color = (seg_c[0] // 5, seg_c[1] // 5, seg_c[2] // 5)
            cv2.circle(img, (cx_s, cy_s), thick + 4, glow_color, -1,
                       cv2.LINE_AA)

        # --- Draw connected body: rects between segments + circles ---
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
            cx_s, cy_s = c * cell + half, r * cell + half
            color = self._segment_color(i, n)
            cv2.circle(img, (cx_s, cy_s), thick, color, -1, cv2.LINE_AA)

        # Thin dark outline along body edges for definition
        for i in range(n - 1, -1, -1):
            r, c = self.snake[i]
            cx_s, cy_s = c * cell + half, r * cell + half
            cv2.circle(img, (cx_s, cy_s), thick, (20, 50, 10), 1,
                       cv2.LINE_AA)

        # --- Head: distinctly larger and brighter ---
        if self.snake:
            hr, hc = self.snake[0]
            hcx, hcy = hc * cell + half, hr * cell + half
            head_r = thick + 3
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

        # --- 4. Food: pulsing concentric rings + apple ---
        if self.food:
            fr, fc = self.food
            fcy = fr * cell + half
            fcx = fc * cell + half
            food_r = cell // 2 - 2

            # Pulsing concentric rings (animation based on step count)
            pulse = (self._step_count % 20) / 20.0  # 0..1 cycle
            for ring_i in range(3):
                ring_phase = (pulse + ring_i * 0.33) % 1.0
                ring_radius = int(food_r + 4 + ring_phase * cell * 0.6)
                ring_alpha = max(0.0, 0.5 - ring_phase * 0.5)
                ring_color = (int(80 * ring_alpha), int(12 * ring_alpha),
                              int(12 * ring_alpha))
                cv2.circle(img, (fcx, fcy), ring_radius, ring_color, 1,
                           cv2.LINE_AA)

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

        # --- 7. Border: green normally, red flash on death ---
        if self._dead:
            border_color = (40, 40, 220)  # BGR red
        else:
            border_color = (30, 140, 30)
        cv2.rectangle(img, (0, 0), (img_size - 1, img_size - 1),
                      border_color, 3)

        # --- 6. Gradient HUD banner at top ---
        banner_h = 32
        banner = img[0:banner_h, :, :].copy()
        # Semi-transparent dark gradient overlay
        for by in range(banner_h):
            alpha = 0.85 - 0.3 * (by / banner_h)  # stronger at top
            banner[by] = (banner[by].astype(np.float32) * (1 - alpha)
                          + np.array([12, 14, 8], dtype=np.float32)
                          * alpha * 255 / 14).astype(np.uint8)
        img[0:banner_h, :, :] = banner
        # Thin separator line under banner
        cv2.line(img, (0, banner_h - 1), (img_size, banner_h - 1),
                 (30, 100, 30), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thick = 2
        title_color = (50, 220, 50)
        score_color = (60, 255, 60)

        # "SNAKE" title left
        cv2.putText(img, "SNAKE", (10, 23), font, font_scale,
                    title_color, font_thick, cv2.LINE_AA)

        # "Score: X" center
        score_text = f"Score: {self._score}"
        stw = cv2.getTextSize(score_text, font, font_scale, font_thick)[0][0]
        cv2.putText(img, score_text, ((img_size - stw) // 2, 23), font,
                    font_scale, score_color, font_thick, cv2.LINE_AA)

        # "Length: Y" right
        len_text = f"Length: {n}"
        ltw = cv2.getTextSize(len_text, font, font_scale, font_thick)[0][0]
        cv2.putText(img, len_text, (img_size - ltw - 10, 23), font,
                    font_scale, score_color, font_thick, cv2.LINE_AA)

        return img

    def render(self, mode='rgb_array'):
        """Render as RGB for dashboard display."""
        return self._render_rgb()

    # ------------------------------------------------------------------
    # Swarm / ghost rendering
    # ------------------------------------------------------------------

    # Default palette: agent 0 = lime green, then HSV-spaced hues
    _SWARM_COLORS = [
        (60, 255, 60),    # 0  lime green (matches default snake)
        (0, 220, 255),    # 1  cyan
        (255, 60, 220),   # 2  magenta
        (255, 180, 30),   # 3  orange
        (160, 80, 255),   # 4  purple
        (255, 255, 60),   # 5  yellow
        (60, 180, 255),   # 6  sky blue
        (255, 100, 100),  # 7  coral
    ]

    @classmethod
    def _swarm_color(cls, agent_idx: int) -> tuple:
        """Return the color for a given agent index."""
        import colorsys
        if agent_idx < len(cls._SWARM_COLORS):
            return cls._SWARM_COLORS[agent_idx]
        # Fall back to HSV wheel for large swarms
        hue = (0.3 + agent_idx * 0.618033988749895) % 1.0  # golden ratio
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        return (int(r * 255), int(g * 255), int(b * 255))

    @staticmethod
    def _dim(color: tuple, factor: float = 0.35) -> tuple:
        return (int(color[0] * factor),
                int(color[1] * factor),
                int(color[2] * factor))

    def render_swarm(
        self,
        all_snakes: list,
        all_foods: list,
        scores: list = None,
    ) -> np.ndarray:
        """Render multiple snakes on the same grid (ghost / swarm mode).

        Each snake gets a unique color from the HSV palette.  Dead snakes
        (empty body lists) are silently skipped.

        Args:
            all_snakes: List of snake body lists.  Each body is a list of
                        ``(row, col)`` tuples ordered head-first.
            all_foods:  List of food positions ``(row, col)`` or ``None``,
                        one per snake.
            scores:     Optional list of scores (ints).  The *best* score
                        is shown in the overlay.

        Returns:
            An RGB numpy array of shape ``(img_size, img_size, 3)``.
        """
        size = self.DISPLAY_SIZE
        cell = size // self.grid_size
        img_size = cell * self.grid_size
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Black background
        img[:] = (10, 10, 10)

        # Grid lines
        for i in range(1, self.grid_size):
            pos = i * cell
            cv2.line(img, (pos, 0), (pos, img_size), (35, 35, 35), 1)
            cv2.line(img, (0, pos), (img_size, pos), (35, 35, 35), 1)

        half = cell // 2
        thick = cell // 2 - 2

        num_agents = len(all_snakes)

        # --- Draw each snake ---
        for agent_idx in range(num_agents):
            snake = all_snakes[agent_idx]
            if not snake:
                continue  # dead / empty — skip

            color = self._swarm_color(agent_idx)
            n = len(snake)

            # Segment color gradient (bright head, dimmer tail)
            def seg_color(seg_i):
                t = 1.0 - seg_i / max(n, 1)
                return (
                    int(color[0] * (0.45 + 0.55 * t)),
                    int(color[1] * (0.45 + 0.55 * t)),
                    int(color[2] * (0.45 + 0.55 * t)),
                )

            # Connected body rectangles (tail to head)
            for i in range(n - 1, 0, -1):
                r1, c1 = snake[i]
                r2, c2 = snake[i - 1]
                cx1, cy1 = c1 * cell + half, r1 * cell + half
                cx2, cy2 = c2 * cell + half, r2 * cell + half
                sc = seg_color(i - 1)
                if r1 == r2:
                    mn, mx = min(cx1, cx2), max(cx1, cx2)
                    cv2.rectangle(img, (mn, cy1 - thick),
                                  (mx, cy1 + thick), sc, -1)
                elif c1 == c2:
                    mn, mx = min(cy1, cy2), max(cy1, cy2)
                    cv2.rectangle(img, (cx1 - thick, mn),
                                  (cx1 + thick, mx), sc, -1)

            # Rounded joints
            for i in range(n - 1, -1, -1):
                r, c = snake[i]
                cx, cy = c * cell + half, r * cell + half
                cv2.circle(img, (cx, cy), thick, seg_color(i), -1,
                           cv2.LINE_AA)

            # Outline for definition
            outline = self._dim(color, 0.25)
            for i in range(n - 1, -1, -1):
                r, c = snake[i]
                cx, cy = c * cell + half, r * cell + half
                cv2.circle(img, (cx, cy), thick, outline, 1, cv2.LINE_AA)

            # Head circle (slightly larger, brighter)
            hr, hc = snake[0]
            hcx, hcy = hc * cell + half, hr * cell + half
            head_r = thick + 3
            cv2.circle(img, (hcx, hcy), head_r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (hcx, hcy), head_r, self._dim(color, 0.5),
                       2, cv2.LINE_AA)

            # Eyes (use direction from agent 0 for simplicity; others
            # get a default right-facing look)
            direction = self.direction if agent_idx == 0 else 1
            dr, dc = self._directions[direction]
            eye_off = cell // 4
            eye_r = max(3, cell // 6)
            pupil_r = max(2, cell // 10)
            for side in (-1, 1):
                ex = hcx + dc * eye_off + (-dr) * side * (cell // 4)
                ey = hcy + dr * eye_off + dc * side * (cell // 4)
                cv2.circle(img, (ex, ey), eye_r, (255, 255, 255), -1,
                           cv2.LINE_AA)
                cv2.circle(img, (ex + dc * 2, ey + dr * 2), pupil_r,
                           (10, 10, 10), -1, cv2.LINE_AA)

        # --- Draw food for each agent (dimmer version of agent color) ---
        for agent_idx in range(num_agents):
            if agent_idx >= len(all_foods):
                continue
            food = all_foods[agent_idx]
            if food is None:
                continue
            fc_color = self._dim(self._swarm_color(agent_idx), 0.55)
            fr, fcc = food
            fcy = fr * cell + half
            fcx = fcc * cell + half
            food_r = cell // 2 - 2
            cv2.circle(img, (fcx, fcy), food_r, fc_color, -1, cv2.LINE_AA)
            # Small bright center dot
            cv2.circle(img, (fcx, fcy), max(2, food_r // 3),
                       self._swarm_color(agent_idx), -1, cv2.LINE_AA)

        # Border
        cv2.rectangle(img, (0, 0), (img_size - 1, img_size - 1),
                      (30, 140, 30), 3)

        # Score overlay — show best score
        best_score = 0
        if scores:
            best_score = max(scores)
        score_text = f"Best: {best_score}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_w = cv2.getTextSize(score_text, font, 0.6, 2)[0][0]
        tx = img_size - text_w - 12
        cv2.rectangle(img, (tx - 4, 2), (img_size - 4, 28),
                      (10, 10, 10), -1)
        cv2.putText(img, score_text, (tx, 22), font, 0.6,
                    (60, 255, 60), 2, cv2.LINE_AA)

        # Agent count
        alive = sum(1 for s in all_snakes if s)
        count_text = f"Agents: {alive}/{num_agents}"
        cv2.rectangle(img, (4, 2), (160, 28), (10, 10, 10), -1)
        cv2.putText(img, count_text, (8, 22), font, 0.6,
                    (60, 255, 60), 2, cv2.LINE_AA)

        return img

    def _info(self) -> dict:
        return {
            'score': self._score,
            'snake_length': len(self.snake),
            'max_length': self.grid_size * self.grid_size,
        }

    def close(self):
        pass
