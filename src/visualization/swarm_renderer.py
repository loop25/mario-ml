"""
Swarm Renderer for Multi-Agent Ghost Mode.

Composites multiple game frames into a single view, showing all agents
playing simultaneously in the same game space. This creates a "ghost mode"
effect where you can see how different agents (genomes, policies) behave
in the same environment.

Rendering strategies by game type:
    - grid:         All agents overlaid on a shared grid with unique colors
                    (Snake, TicTacToe). Uses alpha blending with per-agent tints.
    - sidescroller: Agent sprites composited onto a shared background using
                    x_pos offsets from info dicts (Mario, Sonic).
    - board:        Falls back to tiled grid mode via GridRenderer
                    (Chess, Connect4, Checkers — swarm doesn't make sense).
    - generic:      Alpha-blending composite of raw frames with color tints.

Usage:
    swarm = SwarmRenderer(width=640, height=700, num_envs=8,
                          game_type='grid')
    swarm.render_frames(screen, frames, panel_x=0, panel_y=50,
                        infos=[info1, info2, ...])
"""

import colorsys
import math

import numpy as np
import pygame
from typing import List, Optional, Tuple

from src.visualization.game_renderer import GameRenderer
from src.visualization.grid_renderer import GridRenderer


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def generate_agent_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate *n* visually distinct RGB colors using the HSV wheel.

    Agent 0 always gets lime green (the default snake color).
    Remaining agents are spaced evenly around the hue wheel with
    full saturation and value so they pop on a dark background.

    Returns:
        List of (R, G, B) tuples with values in 0..255.
    """
    if n <= 0:
        return []

    colors: List[Tuple[int, int, int]] = [(60, 255, 60)]  # agent 0 — lime

    for i in range(1, n):
        hue = (0.55 + (i - 1) / max(n - 1, 1) * 0.9) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))

    return colors


def dim_color(
    color: Tuple[int, int, int],
    factor: float = 0.35,
) -> Tuple[int, int, int]:
    """Return a dimmer version of *color* (for food markers, etc.)."""
    return (
        int(color[0] * factor),
        int(color[1] * factor),
        int(color[2] * factor),
    )


# Game types that should fall back to tiled grid mode
_BOARD_GAME_TYPES = {'board', 'chess', 'connect4', 'checkers'}


class SwarmRenderer:
    """Composites multiple environment frames into one shared view.

    For grid-based and side-scroller games, agents are rendered on the
    same canvas with unique color tints. For board games (where a swarm
    view doesn't make sense), it falls back to a standard tiled grid.

    The public interface mirrors :class:`GridRenderer` so the two can be
    swapped transparently by the dashboard.

    Args:
        width:     Total panel width in pixels.
        height:    Total panel height in pixels.
        num_envs:  Number of parallel environments / agents.
        game_type: One of ``'grid'``, ``'sidescroller'``, ``'board'``,
                   or ``'generic'`` (default).
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 700,
        num_envs: int = 1,
        game_type: str = 'generic',
    ):
        self.width = width
        self.height = height
        self.num_envs = max(num_envs, 1)
        self.game_type = game_type.lower()

        # Agent colors (generated once, reused every frame)
        self.agent_colors = generate_agent_colors(self.num_envs)

        # For board games, delegate entirely to a GridRenderer
        self._grid_fallback: Optional[GridRenderer] = None
        if self.game_type in _BOARD_GAME_TYPES:
            self._grid_fallback = GridRenderer(
                width=width, height=height, num_envs=num_envs,
            )

        # Single GameRenderer for blitting the final composite frame
        self._renderer = GameRenderer(width=width, height=height)

        # Cached composite frame (numpy RGB)
        self._last_composite: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_frames(
        self,
        screen: pygame.Surface,
        frames: List[Optional[np.ndarray]],
        panel_x: int = 0,
        panel_y: int = 0,
        infos: Optional[List[dict]] = None,
    ) -> None:
        """Render all agent frames as a single composite image.

        Args:
            screen:  Pygame surface to draw on.
            frames:  List of RGB numpy arrays, one per env.
            panel_x: X offset of the game panel.
            panel_y: Y offset of the game panel.
            infos:   Optional list of info dicts (used for x_pos in
                     side-scroller mode).
        """
        # Board-game fallback — just use a tiled grid
        if self._grid_fallback is not None:
            self._grid_fallback.render_frames(screen, frames, panel_x, panel_y)
            return

        composite = self._composite_frames(frames, infos)
        if composite is not None:
            self._last_composite = composite
            self._renderer.render_frame(screen, composite, panel_x, panel_y)

    def resize(self, width: int, height: int) -> None:
        """Resize the renderer to new dimensions."""
        self.width = width
        self.height = height
        self._renderer.resize(width, height)
        if self._grid_fallback is not None:
            self._grid_fallback.resize(width, height)

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        """Most recent composite frame (backward-compat with GridRenderer)."""
        return self._last_composite

    # ------------------------------------------------------------------
    # Compositing strategies
    # ------------------------------------------------------------------

    def _composite_frames(
        self,
        frames: List[Optional[np.ndarray]],
        infos: Optional[List[dict]] = None,
    ) -> Optional[np.ndarray]:
        """Select and apply the right compositing strategy."""
        # Filter out None frames
        valid: List[Tuple[int, np.ndarray]] = []
        for i, f in enumerate(frames):
            if f is not None:
                valid.append((i, self._ensure_rgb(f)))
        if not valid:
            return self._last_composite  # nothing new — keep last

        if self.game_type == 'sidescroller':
            return self._composite_sidescroller(valid, infos)
        else:
            # 'grid' and 'generic' both use alpha-blend tinting
            return self._composite_alpha_blend(valid)

    @staticmethod
    def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
        """Normalise any frame format to (H, W, 3) uint8 RGB."""
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.ndim == 3 and frame.shape[-1] == 1:
            frame = np.concatenate([frame] * 3, axis=-1)
        elif frame.ndim == 3 and frame.shape[-1] > 3:
            ch = frame[:, :, -1]
            frame = np.stack([ch] * 3, axis=-1)
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        return frame

    # -- Alpha-blend composite (grid / generic) -------------------------

    def _composite_alpha_blend(
        self,
        valid: List[Tuple[int, np.ndarray]],
    ) -> np.ndarray:
        """Overlay non-base frames on top of frame 0 with color tints.

        Frame 0 (the "best" agent) is rendered at full opacity.
        Subsequent frames contribute their non-black pixels with an
        alpha of 0.45, tinted to their assigned agent color so you
        can distinguish who is who.
        """
        # Use the first valid frame as the base canvas
        base_idx, base_frame = valid[0]
        canvas = base_frame.astype(np.float32)

        ghost_alpha = 0.45

        for idx, frame in valid[1:]:
            # Mask: pixels that aren't near-black (sum of channels > 30)
            brightness = frame.astype(np.float32).sum(axis=-1)
            mask = brightness > 30.0  # shape (H, W)

            if not mask.any():
                continue

            # Tint the ghost frame toward the agent color
            color = self.agent_colors[idx] if idx < len(self.agent_colors) \
                else (200, 200, 200)
            tint = np.array(color, dtype=np.float32) / 255.0

            tinted = frame.astype(np.float32) * 0.5 + \
                frame.astype(np.float32) * tint[None, None, :] * 0.5

            # Alpha-blend onto canvas where mask is True
            mask3 = mask[:, :, None]
            canvas = canvas * (1.0 - ghost_alpha * mask3) + \
                tinted * (ghost_alpha * mask3)

        return np.clip(canvas, 0, 255).astype(np.uint8)

    # -- Side-scroller composite ----------------------------------------

    def _composite_sidescroller(
        self,
        valid: List[Tuple[int, np.ndarray]],
        infos: Optional[List[dict]],
    ) -> np.ndarray:
        """Composite side-scroller agents onto a shared background.

        Uses x_pos from info dicts to offset each agent's sprite
        relative to the base agent's position, then alpha-blends
        the non-background pixels.
        """
        base_idx, base_frame = valid[0]
        canvas = base_frame.copy().astype(np.float32)
        h, w = canvas.shape[:2]

        base_x = 0.0
        if infos and base_idx < len(infos) and infos[base_idx]:
            base_x = float(infos[base_idx].get('x_pos', 0))

        ghost_alpha = 0.4

        for idx, frame in valid[1:]:
            agent_x = 0.0
            if infos and idx < len(infos) and infos[idx]:
                agent_x = float(infos[idx].get('x_pos', 0))

            # Pixel offset (positive = agent is ahead of base)
            dx = int(agent_x - base_x)

            # Build shifted version of the frame
            shifted = np.zeros_like(canvas)
            if dx >= 0:
                src_end = min(w - dx, w)
                shifted[:, dx:dx + src_end] = frame[:, :src_end].astype(
                    np.float32)
            else:
                adx = -dx
                src_end = min(w - adx, w)
                shifted[:, :src_end] = frame[:, adx:adx + src_end].astype(
                    np.float32)

            brightness = shifted.sum(axis=-1)
            mask = brightness > 30.0

            if not mask.any():
                continue

            color = self.agent_colors[idx] if idx < len(self.agent_colors) \
                else (200, 200, 200)
            tint = np.array(color, dtype=np.float32) / 255.0

            tinted = shifted * 0.6 + shifted * tint[None, None, :] * 0.4
            mask3 = mask[:, :, None]
            canvas = canvas * (1.0 - ghost_alpha * mask3) + \
                tinted * (ghost_alpha * mask3)

        return np.clip(canvas, 0, 255).astype(np.uint8)
