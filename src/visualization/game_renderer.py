"""
Game Renderer for ML Training Dashboard.

Handles rendering game frames to a pygame surface. The renderer
scales the raw game output to fit the dashboard's game panel
while maintaining the correct aspect ratio.

Features:
    - Pre-allocated surfaces for zero-allocation rendering
    - Smooth upscaling using hardware-accelerated transform.scale
    - Optional overlay text (episode info, fps counter)
    - Frame rate independent of training speed
    - Efficient surface blitting for minimal performance impact

The game panel occupies the left side of the dashboard window.

Usage:
    renderer = GameRenderer(width=640, height=600)
    renderer.render_frame(screen, frame, x_offset=0, y_offset=50)
"""

import pygame
import numpy as np
from typing import Optional, Tuple


class GameRenderer:
    """
    Renders NES game frames to a pygame surface region.

    Uses pre-allocated surfaces to avoid per-frame memory allocation,
    which is the single biggest performance bottleneck in the dashboard.

    Args:
        width: Width of the game panel in pixels.
        height: Height of the game panel in pixels.
        bg_color: Background color for the game panel. Default dark gray.

    Attributes:
        width: Panel width.
        height: Panel height.
        last_frame: Most recent frame rendered (for re-drawing).
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 600,
        bg_color: Tuple[int, int, int] = (20, 20, 30),
    ):
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.last_frame: Optional[np.ndarray] = None

        # Pre-calculate the scaled size maintaining NES aspect ratio
        # NES native resolution: 256x240 (width x height)
        nes_aspect = 256 / 240
        panel_aspect = width / height

        if panel_aspect > nes_aspect:
            # Panel is wider than NES aspect: fit to height
            self._scaled_height = height - 20  # padding
            self._scaled_width = int(self._scaled_height * nes_aspect)
        else:
            # Panel is taller than NES aspect: fit to width
            self._scaled_width = width - 20  # padding
            self._scaled_height = int(self._scaled_width / nes_aspect)

        # Center the game display within the panel
        self._x_offset = (width - self._scaled_width) // 2
        self._y_offset = (height - self._scaled_height) // 2

        # Pre-allocate reusable surfaces to avoid per-frame allocation.
        # These are created lazily on first frame (need to know frame size).
        self._raw_surface: Optional[pygame.Surface] = None
        self._raw_size: Optional[Tuple[int, int]] = None
        self._scaled_surface = pygame.Surface(
            (self._scaled_width, self._scaled_height)
        )

        # Cache for overlay fonts (avoid re-creating every frame)
        self._overlay_fonts: dict = {}

    def resize(self, width: int, height: int) -> None:
        """
        Resize the renderer to new panel dimensions.

        Recalculates NES aspect ratio scaling and recreates the
        pre-allocated scaled surface. Preserves last_frame so
        the display doesn't flicker after a resize.

        Args:
            width: New panel width in pixels.
            height: New panel height in pixels.
        """
        self.width = width
        self.height = height

        # Recalculate scaled size maintaining NES aspect ratio
        nes_aspect = 256 / 240
        panel_aspect = width / height

        if panel_aspect > nes_aspect:
            self._scaled_height = height - 20
            self._scaled_width = int(self._scaled_height * nes_aspect)
        else:
            self._scaled_width = width - 20
            self._scaled_height = int(self._scaled_width / nes_aspect)

        # Ensure minimum size (avoid zero-size surfaces)
        self._scaled_width = max(self._scaled_width, 16)
        self._scaled_height = max(self._scaled_height, 16)

        # Recenter
        self._x_offset = (width - self._scaled_width) // 2
        self._y_offset = (height - self._scaled_height) // 2

        # Recreate scaled surface at new dimensions
        self._scaled_surface = pygame.Surface(
            (self._scaled_width, self._scaled_height)
        )

        # Reset raw surface so it gets re-created on next frame
        self._raw_surface = None
        self._raw_size = None

        # Clear font cache (sizes may need to change)
        self._overlay_fonts.clear()

    def _get_overlay_font(self, font_size: int) -> pygame.font.Font:
        """Get or create a cached font for overlay text."""
        if font_size not in self._overlay_fonts:
            self._overlay_fonts[font_size] = pygame.font.SysFont(
                'Consolas', font_size,
            )
        return self._overlay_fonts[font_size]

    def render_frame(
        self,
        screen: pygame.Surface,
        frame: np.ndarray,
        panel_x: int = 0,
        panel_y: int = 0,
    ) -> None:
        """
        Render a game frame to the screen at the specified panel position.

        Handles both RGB and grayscale frames. Uses pre-allocated surfaces
        for maximum performance — no memory allocation per frame.

        Args:
            screen: The pygame surface to draw on.
            frame: Game frame as numpy array. Can be:
                   - RGB: shape (H, W, 3)
                   - Grayscale: shape (H, W) or (H, W, 1)
                   - Stacked frames: shape (H, W, N) — uses last frame
            panel_x: X position of the panel on screen.
            panel_y: Y position of the panel on screen.
        """
        if frame is None:
            return

        self.last_frame = frame

        # Handle different frame formats
        if frame.ndim == 2:
            # Grayscale (H, W) -> RGB (H, W, 3)
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.ndim == 3 and frame.shape[-1] == 1:
            # Grayscale with channel dim (H, W, 1) -> RGB (H, W, 3)
            frame = np.concatenate([frame] * 3, axis=-1)
        elif frame.ndim == 3 and frame.shape[-1] > 3:
            # Stacked frames (H, W, N) -> use last frame as grayscale
            last_channel = frame[:, :, -1]
            frame = np.stack([last_channel] * 3, axis=-1)

        # Ensure uint8 format
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # Draw background for the panel area
        panel_rect = pygame.Rect(panel_x, panel_y, self.width, self.height)
        pygame.draw.rect(screen, self.bg_color, panel_rect)

        try:
            h, w = frame.shape[:2]
            frame_size = (w, h)

            # Create or resize the raw surface only when frame dimensions change.
            # This avoids the expensive make_surface() call every frame.
            if self._raw_surface is None or self._raw_size != frame_size:
                self._raw_surface = pygame.Surface(frame_size)
                self._raw_size = frame_size

            # Blit numpy pixel data directly into the pre-allocated surface.
            # surfarray.blit_array is much faster than make_surface because
            # it reuses the existing surface instead of allocating a new one.
            # It expects (W, H, 3) — width as first axis.
            pygame.surfarray.blit_array(
                self._raw_surface, frame.transpose((1, 0, 2))
            )

            # Scale to fit the panel.
            # transform.scale with a destination surface avoids allocation.
            pygame.transform.scale(
                self._raw_surface,
                (self._scaled_width, self._scaled_height),
                self._scaled_surface,
            )

            # Blit to screen at the correct position within the panel
            screen.blit(
                self._scaled_surface,
                (panel_x + self._x_offset, panel_y + self._y_offset),
            )
        except Exception:
            # Fallback: draw a placeholder if frame rendering fails
            placeholder_rect = pygame.Rect(
                panel_x + self._x_offset,
                panel_y + self._y_offset,
                self._scaled_width,
                self._scaled_height,
            )
            pygame.draw.rect(screen, (40, 40, 50), placeholder_rect)
            font = self._get_overlay_font(20)
            text = font.render('Waiting for frame...', True, (150, 150, 150))
            text_rect = text.get_rect(center=placeholder_rect.center)
            screen.blit(text, text_rect)

    def render_overlay_text(
        self,
        screen: pygame.Surface,
        text: str,
        panel_x: int = 0,
        panel_y: int = 0,
        position: str = 'top_left',
        color: Tuple[int, int, int] = (255, 255, 255),
        font_size: int = 14,
    ) -> None:
        """
        Render overlay text on top of the game display.

        Uses cached fonts for performance.

        Args:
            screen: The pygame surface to draw on.
            text: Text string to display.
            panel_x: X position of the panel.
            panel_y: Y position of the panel.
            position: Where to place text: 'top_left', 'top_right',
                      'bottom_left', 'bottom_right'.
            color: RGB color tuple for the text.
            font_size: Font size in pixels.
        """
        font = self._get_overlay_font(font_size)
        text_surface = font.render(text, True, color)

        # Calculate position based on anchor
        padding = 8
        if position == 'top_left':
            pos = (panel_x + padding, panel_y + padding)
        elif position == 'top_right':
            pos = (
                panel_x + self.width - text_surface.get_width() - padding,
                panel_y + padding,
            )
        elif position == 'bottom_left':
            pos = (
                panel_x + padding,
                panel_y + self.height - text_surface.get_height() - padding,
            )
        elif position == 'bottom_right':
            pos = (
                panel_x + self.width - text_surface.get_width() - padding,
                panel_y + self.height - text_surface.get_height() - padding,
            )
        else:
            pos = (panel_x + padding, panel_y + padding)

        # Draw semi-transparent background for readability
        bg_rect = pygame.Rect(
            pos[0] - 2, pos[1] - 2,
            text_surface.get_width() + 4,
            text_surface.get_height() + 4,
        )
        bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 160))  # Semi-transparent black
        screen.blit(bg_surface, bg_rect)

        # Draw the text
        screen.blit(text_surface, pos)
