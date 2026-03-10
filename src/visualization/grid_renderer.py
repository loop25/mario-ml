"""
Grid Renderer for Multi-Environment Display.

Manages N GameRenderer instances arranged in a grid layout within the
dashboard's game panel area. Each cell shows a different environment running
simultaneously (different genomes for NEAT, different environments for
PPO/DQN).

Layout auto-calculates grid dimensions:
    N=1  → 1x1 (640x700 — same as single game)
    N=2  → 2x1 (320x700 each)
    N=4  → 2x2 (320x350 each)
    N=6  → 3x2 (213x350 each)
    N=8  → 3x3 (213x233 each, one cell empty)
    N=16 → 4x4 (160x175 each)

When num_envs=1, uses a single GameRenderer with zero overhead
(identical to the original dashboard behavior).

Usage:
    grid = GridRenderer(width=640, height=700, num_envs=4)
    grid.render_frames(screen, frames=[f1, f2, f3, f4],
                       panel_x=0, panel_y=50)
"""

import math
import pygame
import numpy as np
from typing import List, Optional, Tuple

from src.visualization.game_renderer import GameRenderer


class GridRenderer:
    """
    Renders multiple game frames in a grid layout.

    Creates N GameRenderer instances, each sized to fit within one cell
    of the grid. The grid auto-calculates rows and columns based on N.

    Args:
        width: Total panel width in pixels.
        height: Total panel height in pixels.
        num_envs: Number of environments to display. Default 1.

    Attributes:
        num_envs: Number of game cells.
        cols: Number of grid columns.
        rows: Number of grid rows.
        renderers: List of GameRenderer instances.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 700,
        num_envs: int = 1,
    ):
        self.width = width
        self.height = height
        self.num_envs = num_envs

        # Calculate grid dimensions
        if num_envs <= 1:
            self.cols = 1
            self.rows = 1
        else:
            self.cols = math.ceil(math.sqrt(num_envs))
            self.rows = math.ceil(num_envs / self.cols)

        # Calculate cell size
        self.cell_width = width // self.cols
        self.cell_height = height // self.rows

        # Create a GameRenderer for each cell
        self.renderers: List[GameRenderer] = []
        for _ in range(num_envs):
            renderer = GameRenderer(
                width=self.cell_width,
                height=self.cell_height,
            )
            self.renderers.append(renderer)

        # Track last frames for re-rendering
        self.last_frames: List[Optional[np.ndarray]] = [None] * num_envs

        # Font for cell labels (small index number in corner)
        self._label_font: Optional[pygame.font.Font] = None

    def _get_label_font(self) -> pygame.font.Font:
        """Get or create the cell label font."""
        if self._label_font is None:
            self._label_font = pygame.font.SysFont('Consolas', 10)
        return self._label_font

    def render_frames(
        self,
        screen: pygame.Surface,
        frames: List[Optional[np.ndarray]],
        panel_x: int = 0,
        panel_y: int = 0,
    ) -> None:
        """
        Render multiple game frames in the grid layout.

        Each frame is rendered in its corresponding grid cell.
        If a frame is None, the previous frame for that cell is
        re-rendered (or a placeholder is shown).

        Args:
            screen: The pygame surface to draw on.
            frames: List of game frames (numpy arrays), one per env.
                    Can be shorter than num_envs — missing entries
                    use last known frame.
            panel_x: X position of the panel on screen.
            panel_y: Y position of the panel on screen.
        """
        for i in range(self.num_envs):
            # Calculate grid position for cell i
            col = i % self.cols
            row = i // self.cols
            cell_x = panel_x + col * self.cell_width
            cell_y = panel_y + row * self.cell_height

            # Get frame for this cell
            frame = frames[i] if i < len(frames) else None
            if frame is not None:
                self.last_frames[i] = frame
            else:
                frame = self.last_frames[i]

            # Render the frame
            if frame is not None:
                self.renderers[i].render_frame(
                    screen, frame,
                    panel_x=cell_x,
                    panel_y=cell_y,
                )
            else:
                # No frame yet — draw placeholder
                self._draw_placeholder(screen, cell_x, cell_y, i)

            # Draw grid lines between cells (subtle separators)
            if self.num_envs > 1:
                self._draw_cell_border(screen, cell_x, cell_y)

                # Small env index label in top-left of each cell
                if self.num_envs > 2:
                    self._draw_cell_label(screen, cell_x, cell_y, i)

    def render_single_frame(
        self,
        screen: pygame.Surface,
        frame: Optional[np.ndarray],
        env_index: int,
        panel_x: int = 0,
        panel_y: int = 0,
    ) -> None:
        """
        Render a single frame in one grid cell.

        Useful when frames arrive asynchronously (e.g., from
        multiprocessing workers). Updates only the specified cell.

        Args:
            screen: The pygame surface to draw on.
            frame: Game frame as numpy array.
            env_index: Which cell to render in (0-indexed).
            panel_x: X position of the panel on screen.
            panel_y: Y position of the panel on screen.
        """
        if env_index < 0 or env_index >= self.num_envs:
            return

        if frame is not None:
            self.last_frames[env_index] = frame

        frame = frame or self.last_frames[env_index]

        col = env_index % self.cols
        row = env_index // self.cols
        cell_x = panel_x + col * self.cell_width
        cell_y = panel_y + row * self.cell_height

        if frame is not None:
            self.renderers[env_index].render_frame(
                screen, frame,
                panel_x=cell_x,
                panel_y=cell_y,
            )

    def _draw_placeholder(
        self,
        screen: pygame.Surface,
        x: int,
        y: int,
        index: int,
    ) -> None:
        """Draw a placeholder for a cell with no frame yet."""
        rect = pygame.Rect(x, y, self.cell_width, self.cell_height)
        pygame.draw.rect(screen, (20, 20, 30), rect)
        font = self._get_label_font()
        text = font.render(f'Env {index + 1} — waiting...', True, (100, 100, 120))
        text_rect = text.get_rect(center=rect.center)
        screen.blit(text, text_rect)

    def _draw_cell_border(
        self,
        screen: pygame.Surface,
        x: int,
        y: int,
    ) -> None:
        """Draw subtle border lines around a grid cell."""
        border_color = (42, 42, 74)  # Same as dashboard BORDER_COLOR
        # Right edge
        pygame.draw.line(
            screen, border_color,
            (x + self.cell_width - 1, y),
            (x + self.cell_width - 1, y + self.cell_height),
            1,
        )
        # Bottom edge
        pygame.draw.line(
            screen, border_color,
            (x, y + self.cell_height - 1),
            (x + self.cell_width, y + self.cell_height - 1),
            1,
        )

    def _draw_cell_label(
        self,
        screen: pygame.Surface,
        x: int,
        y: int,
        index: int,
    ) -> None:
        """Draw a small index label in the top-left corner of a cell."""
        font = self._get_label_font()
        label = font.render(f'#{index + 1}', True, (160, 160, 180))

        # Semi-transparent background for readability
        bg = pygame.Surface((label.get_width() + 4, label.get_height() + 2), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 140))
        screen.blit(bg, (x + 2, y + 2))
        screen.blit(label, (x + 4, y + 3))

    def resize(self, width: int, height: int) -> None:
        """
        Resize the grid to new panel dimensions.

        Recalculates cell sizes and resizes each child GameRenderer.
        Preserves last_frames so the display doesn't flicker.

        Args:
            width: New total panel width in pixels.
            height: New total panel height in pixels.
        """
        self.width = width
        self.height = height

        # Recalculate cell size
        self.cell_width = width // self.cols
        self.cell_height = height // self.rows

        # Resize each child renderer
        for renderer in self.renderers:
            renderer.resize(self.cell_width, self.cell_height)

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        """
        Get the last frame from the first renderer.

        This property provides backward compatibility with code that
        expects a single last_frame attribute (like dashboard.py).

        Returns:
            The most recent frame from env 0, or None.
        """
        return self.last_frames[0] if self.last_frames else None
