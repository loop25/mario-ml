"""
Overlay Manager for Stream-Ready Frame Compositing.

Composites the dashboard display into a standard streaming resolution
(1920x1080) with optional overlays for stream-specific elements.

This is a stub module prepared for future streaming integration.
Currently, the pygame dashboard window can be captured directly by
OBS Studio via Window Capture, which works for most streaming setups.

Future enhancements:
    - Custom stream overlays (subscriber count, chat, etc.)
    - Webcam picture-in-picture integration
    - Custom branding/watermark support
    - Animated transitions between algorithms

Usage (future):
    overlay = OverlayManager(resolution=(1920, 1080))
    overlay.set_dashboard_surface(dashboard.screen)
    stream_frame = overlay.compose()
"""

import numpy as np
from typing import Tuple, Optional


class OverlayManager:
    """
    Composites dashboard and overlay elements for streaming.

    Stub implementation — currently just passes through the dashboard
    surface. Future versions will add overlay compositing.

    Args:
        resolution: Output resolution (width, height). Default 1920x1080.
    """

    def __init__(self, resolution: Tuple[int, int] = (1920, 1080)):
        self.resolution = resolution
        self._dashboard_surface = None

    def set_dashboard_surface(self, surface) -> None:
        """
        Set the dashboard pygame surface to composite.

        Args:
            surface: pygame.Surface from the Dashboard.
        """
        self._dashboard_surface = surface

    def compose(self) -> Optional[np.ndarray]:
        """
        Compose the final stream frame.

        Currently returns the dashboard surface as-is.
        Future: Add overlays, resize, add branding.

        Returns:
            np.ndarray: Composited frame, or None if no surface set.
        """
        # TODO: Implement full compositing pipeline
        # For now, this is a pass-through stub
        if self._dashboard_surface is None:
            return None

        try:
            import pygame
            frame = pygame.surfarray.array3d(self._dashboard_surface)
            return np.transpose(frame, (1, 0, 2))  # (H, W, 3)
        except ImportError:
            return None
