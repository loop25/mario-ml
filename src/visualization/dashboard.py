"""
Main Visualization Dashboard for Super Mario Bros ML.

The dashboard is a resizable pygame window that displays:
    - Left panel (~46% width): Live game rendering showing Mario playing
    - Right panel (~54% width): 4 real-time updating performance graphs
    - Bottom bar: Current stats and best performance
    - Top bar: Title and algorithm name

The window can be maximized or dragged to any size. All panels,
fonts, and graphs scale proportionally. The layout preserves the
same proportions at every size.

The dashboard is designed to be OBS Window Capture compatible
for streaming. The pygame surface can also be captured for
video recording.

Usage:
    dashboard = Dashboard(algorithm='neat')
    dashboard.update(frame=game_frame, metrics={'reward': 250, ...})
    if not dashboard.handle_events():
        break  # User closed the window
    dashboard.close()
"""

import pygame
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple

from src.visualization.game_renderer import GameRenderer
from src.visualization.grid_renderer import GridRenderer
from src.visualization.graph_panel import GraphPanel
from src.visualization.metrics_tracker import MetricsTracker


# ============================================================================
# Dashboard Layout Ratios & Defaults
# ============================================================================
# Ratios are derived from the original 1400x800 design.
DEFAULT_WIDTH = 1400
DEFAULT_HEIGHT = 800
MIN_WIDTH = 800
MIN_HEIGHT = 500
GAME_PANEL_RATIO = 640 / 1400    # ~0.457 — game panel share of width
TOP_BAR_RATIO = 50 / 800         # 0.0625 — top bar share of height
BOTTOM_BAR_RATIO = 50 / 800      # 0.0625 — bottom bar share of height
MIN_BAR_HEIGHT = 32
MAX_BAR_HEIGHT = 60
RESIZE_DEBOUNCE_MS = 150         # ms to wait before expensive rebuild

# Colors (fixed — don't change with window size)
BG_COLOR = (26, 26, 46)         # Dark navy background (#1a1a2e)
TOP_BAR_COLOR = (15, 15, 30)    # Darker top bar
BOTTOM_BAR_COLOR = (15, 15, 30) # Darker bottom bar
BORDER_COLOR = (42, 42, 74)     # Subtle border lines
ACCENT_COLOR = (0, 255, 136)    # Neon green accent (#00ff88)
TEXT_COLOR = (224, 224, 224)     # Light gray text
DIM_TEXT_COLOR = (140, 140, 160) # Dimmed text for labels


class Dashboard:
    """
    Main training visualization dashboard.

    Combines game rendering, performance graphs, and status information
    in a single resizable pygame window. Call update() each frame to
    refresh the display.

    The window supports maximize, drag-resize, and Windows snap layouts.
    All panels and fonts scale proportionally.

    Args:
        algorithm: Algorithm name ('neat', 'ppo', 'dqn').
                   Affects graph labels and title display.
        num_envs: Number of parallel environments to display.
                  When > 1, the game panel shows a grid of games.
                  Default 1.
        graph_update_interval: How often to update graphs (in episodes).
                               Lower values = smoother graphs but slower.
                               Default 1.
        fps_cap: Maximum frames per second for the display.
                 Training speed is independent of this cap.
                 Default 60.

    Attributes:
        is_paused: Whether training display is paused.
        metrics: MetricsTracker instance for data storage.
        num_envs: Number of parallel environments displayed.

    Example:
        dashboard = Dashboard(algorithm='neat')
        while training:
            dashboard.update(frame=env_frame, metrics={
                'episode': ep, 'reward': total_reward,
                'distance': x_pos, 'complexity': nodes + conns,
                'action_distribution': action_counts,
            })
            if not dashboard.handle_events():
                break
        dashboard.close()
    """

    def __init__(
        self,
        algorithm: str = 'neat',
        num_envs: int = 1,
        graph_update_interval: int = 1,
        fps_cap: int = 60,
        recorder=None,
    ):
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption(f'Mario ML Dashboard - {algorithm.upper()}')

        # Configuration (stored before layout calc so _rebuild uses them)
        self.algorithm = algorithm
        self.num_envs = num_envs
        self.graph_update_interval = graph_update_interval
        self.fps_cap = fps_cap
        self.recorder = recorder
        self.metrics = MetricsTracker()

        # Compute initial layout dimensions from default size
        self.window_width = DEFAULT_WIDTH
        self.window_height = DEFAULT_HEIGHT
        self.top_bar_height = 50
        self.bottom_bar_height = 50
        self.game_panel_width = 640
        self.graph_panel_width = 760
        self.content_height = 700
        self._recalculate_layout(DEFAULT_WIDTH, DEFAULT_HEIGHT)

        # Create the main window (RESIZABLE for maximize/drag support)
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height),
            pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.RESIZABLE,
        )

        # Create sub-components at initial size
        if num_envs > 1:
            self.game_renderer = GridRenderer(
                width=self.game_panel_width,
                height=self.content_height,
                num_envs=num_envs,
            )
        else:
            self.game_renderer = GameRenderer(
                width=self.game_panel_width,
                height=self.content_height,
            )

        self.graph_panel = GraphPanel(
            width=self.graph_panel_width,
            height=self.content_height,
            algorithm=algorithm,
        )

        # State tracking
        self.is_paused = False
        self._last_graph_update = 0
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._display_fps = 0
        self._cached_graph_surface = None
        self.clock = pygame.time.Clock()

        # Resize debounce state
        self._pending_resize = None   # (width, height) waiting to be applied
        self._resize_timer = 0        # Timestamp of last resize event

        # NEAT genome progress tracking (shown in status bar during generation)
        self._genome_progress = ''  # e.g. "23/50"
        self._genome_reward = 0.0   # reward of the last evaluated genome

        # Initialize fonts
        self._title_font = None
        self._status_font = None
        self._label_font = None
        self._rebuild_fonts()

        # Render initial graph panel (empty graphs with titles)
        self._cached_graph_surface = self.graph_panel.render()

        # Draw initial frame
        self._draw_background()
        pygame.display.flip()

    # ========================================================================
    # Layout & Resize
    # ========================================================================

    def _recalculate_layout(self, width: int, height: int) -> None:
        """
        Recompute all layout dimensions from the given window size.

        Uses fixed ratios derived from the original 1400x800 design
        so the layout looks the same at every size.

        Args:
            width: Window width in pixels.
            height: Window height in pixels.
        """
        self.window_width = max(width, MIN_WIDTH)
        self.window_height = max(height, MIN_HEIGHT)

        # Bar heights scale with window but are clamped
        self.top_bar_height = int(max(
            MIN_BAR_HEIGHT,
            min(MAX_BAR_HEIGHT, self.window_height * TOP_BAR_RATIO),
        ))
        self.bottom_bar_height = int(max(
            MIN_BAR_HEIGHT,
            min(MAX_BAR_HEIGHT, self.window_height * BOTTOM_BAR_RATIO),
        ))

        # Panel widths from ratio
        self.game_panel_width = int(self.window_width * GAME_PANEL_RATIO)
        self.graph_panel_width = self.window_width - self.game_panel_width

        # Content area between bars
        self.content_height = (
            self.window_height - self.top_bar_height - self.bottom_bar_height
        )

    def _rebuild_components(self) -> None:
        """
        Resize sub-renderers to match current layout dimensions.

        Called after the debounce timer fires (not on every resize event).
        The expensive part is graph_panel.resize() which recreates the
        matplotlib figure.
        """
        self.game_renderer.resize(self.game_panel_width, self.content_height)

        self.graph_panel.resize(self.graph_panel_width, self.content_height)
        if self.metrics.episode_count > 0:
            self.graph_panel.update_data(self.metrics)
        self._cached_graph_surface = self.graph_panel.render()

    def _rebuild_fonts(self) -> None:
        """Rebuild fonts scaled to current window size."""
        scale = self.window_height / DEFAULT_HEIGHT  # 1.0 at 800px

        title_size = int(max(16, min(32, 22 * scale)))
        status_size = int(max(12, min(22, 16 * scale)))
        label_size = int(max(10, min(16, 12 * scale)))

        self._title_font = pygame.font.SysFont('Consolas', title_size, bold=True)
        self._status_font = pygame.font.SysFont('Consolas', status_size)
        self._label_font = pygame.font.SysFont('Consolas', label_size)

    # ========================================================================
    # Rendering — graph surface helper
    # ========================================================================

    def _blit_graph_surface(self) -> None:
        """Blit the cached graph surface, scaling if needed during resize."""
        if self._cached_graph_surface is None:
            return

        cached_w = self._cached_graph_surface.get_width()
        cached_h = self._cached_graph_surface.get_height()

        if cached_w != self.graph_panel_width or cached_h != self.content_height:
            # During resize debounce: smoothscale old graph to fit
            target_w = max(16, self.graph_panel_width)
            target_h = max(16, self.content_height)
            scaled = pygame.transform.smoothscale(
                self._cached_graph_surface,
                (target_w, target_h),
            )
            self.screen.blit(scaled, (self.game_panel_width, self.top_bar_height))
        else:
            self.screen.blit(
                self._cached_graph_surface,
                (self.game_panel_width, self.top_bar_height),
            )

    # ========================================================================
    # Main Update Methods
    # ========================================================================

    def update(
        self,
        frame: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update the dashboard with new data.

        This is the main method called each training step. It:
        1. Records metrics to the tracker
        2. Renders the game frame (if provided)
        3. Updates graphs (at configured interval)
        4. Draws status bars
        5. Flips the display buffer

        Args:
            frame: Current game frame as numpy array (RGB or grayscale).
                   Pass None to keep the last rendered frame.
            metrics: Dictionary of metric values to record. Common keys:
                     - 'episode' / 'generation': Current episode number
                     - 'reward': Episode total reward
                     - 'distance': Mario's x position
                     - 'loss': Training loss value
                     - 'complexity': Network complexity (NEAT)
                     - 'action_distribution': List of action counts
                     - 'epsilon': Exploration rate (DQN)
        """
        # Separate genome-progress metrics from graph-triggering metrics.
        is_graph_update = False
        if metrics:
            if 'genome_progress' in metrics:
                self._genome_progress = metrics.pop('genome_progress')
            if 'genome_reward' in metrics:
                self._genome_reward = metrics.pop('genome_reward')
            if metrics:
                self.metrics.record(**metrics)
                is_graph_update = True

        # Draw the background (clears previous frame)
        self._draw_background()

        # Render game frame on the left panel
        if isinstance(self.game_renderer, GridRenderer):
            if frame is not None:
                self.game_renderer.render_frames(
                    self.screen, [frame],
                    panel_x=0, panel_y=self.top_bar_height,
                )
            elif self.game_renderer.last_frames:
                self.game_renderer.render_frames(
                    self.screen, self.game_renderer.last_frames,
                    panel_x=0, panel_y=self.top_bar_height,
                )
        else:
            if frame is not None:
                self.game_renderer.render_frame(
                    self.screen, frame,
                    panel_x=0, panel_y=self.top_bar_height,
                )
            elif self.game_renderer.last_frame is not None:
                self.game_renderer.render_frame(
                    self.screen, self.game_renderer.last_frame,
                    panel_x=0, panel_y=self.top_bar_height,
                )

        # Update and render graphs (only when real metrics arrive)
        if is_graph_update:
            current_episode = self.metrics.episode_count
            if current_episode - self._last_graph_update >= self.graph_update_interval:
                self.graph_panel.update_data(self.metrics)
                self._last_graph_update = current_episode
                self._cached_graph_surface = self.graph_panel.render()

        # Blit the cached graph surface (with smoothscale fallback)
        self._blit_graph_surface()

        # Draw the top title bar
        self._draw_title_bar()

        # Draw the bottom status bar
        self._draw_status_bar()

        # Draw divider lines
        self._draw_dividers()

        # Overlay game info text on the game panel
        # (GridRenderer doesn't have render_overlay_text — info is in
        # the status bar and cell labels instead)
        if not isinstance(self.game_renderer, GridRenderer):
            ep_label = 'Gen' if self.algorithm == 'neat' else 'Ep'
            episode_text = f'{ep_label}: {self.metrics.episode_count}'
            self.game_renderer.render_overlay_text(
                self.screen, episode_text,
                panel_x=0, panel_y=self.top_bar_height,
                position='top_left',
            )

            # Update FPS counter
            self._update_fps()
            fps_text = f'FPS: {self._display_fps}'
            self.game_renderer.render_overlay_text(
                self.screen, fps_text,
                panel_x=0, panel_y=self.top_bar_height,
                position='top_right', font_size=12,
            )
        else:
            self._update_fps()

        # Pause indicator
        if self.is_paused:
            self._draw_pause_overlay()

        # Capture frame for video recording (before flip)
        if self.recorder is not None:
            self.recorder.capture_frame(self.screen)

        # Flip the display buffer
        pygame.display.flip()

        # Cap frame rate
        if is_graph_update:
            self.clock.tick(self.fps_cap)
        else:
            self.clock.tick(0)

    def update_grid(
        self,
        frames: Optional[list] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update the dashboard with multiple game frames (grid mode).

        Similar to update() but accepts a list of frames for the
        multi-env grid display. Only works when num_envs > 1.

        Args:
            frames: List of game frames (numpy arrays), one per env.
                    Can be None or shorter than num_envs.
            metrics: Dictionary of metric values to record.
        """
        # Record metrics (same as single-env update)
        is_graph_update = False
        if metrics:
            if 'genome_progress' in metrics:
                self._genome_progress = metrics.pop('genome_progress')
            if 'genome_reward' in metrics:
                self._genome_reward = metrics.pop('genome_reward')
            if metrics:
                self.metrics.record(**metrics)
                is_graph_update = True

        # Draw the background
        self._draw_background()

        # Render game frames in the grid
        if frames is not None and isinstance(self.game_renderer, GridRenderer):
            self.game_renderer.render_frames(
                self.screen, frames,
                panel_x=0, panel_y=self.top_bar_height,
            )
        elif frames is not None and len(frames) > 0:
            # Fallback for single renderer: use first frame
            self.game_renderer.render_frame(
                self.screen, frames[0],
                panel_x=0, panel_y=self.top_bar_height,
            )
        elif self.game_renderer.last_frame is not None:
            # Re-render last known frames
            if isinstance(self.game_renderer, GridRenderer):
                self.game_renderer.render_frames(
                    self.screen, self.game_renderer.last_frames,
                    panel_x=0, panel_y=self.top_bar_height,
                )
            else:
                self.game_renderer.render_frame(
                    self.screen, self.game_renderer.last_frame,
                    panel_x=0, panel_y=self.top_bar_height,
                )

        # Update and render graphs
        if is_graph_update:
            current_episode = self.metrics.episode_count
            if current_episode - self._last_graph_update >= self.graph_update_interval:
                self.graph_panel.update_data(self.metrics)
                self._last_graph_update = current_episode
                self._cached_graph_surface = self.graph_panel.render()

        self._blit_graph_surface()

        # Draw bars and dividers
        self._draw_title_bar()
        self._draw_status_bar()
        self._draw_dividers()

        # Overlay info (only for single GameRenderer — grid has the status bar)
        if not isinstance(self.game_renderer, GridRenderer):
            ep_label = 'Gen' if self.algorithm == 'neat' else 'Ep'
            episode_text = f'{ep_label}: {self.metrics.episode_count}'
            self.game_renderer.render_overlay_text(
                self.screen, episode_text,
                panel_x=0, panel_y=self.top_bar_height,
                position='top_left',
            )

        # FPS
        self._update_fps()
        if not isinstance(self.game_renderer, GridRenderer):
            fps_text = f'FPS: {self._display_fps}'
            self.game_renderer.render_overlay_text(
                self.screen, fps_text,
                panel_x=0, panel_y=self.top_bar_height,
                position='top_right', font_size=12,
            )

        if self.is_paused:
            self._draw_pause_overlay()

        # Capture frame for video recording (before flip)
        if self.recorder is not None:
            self.recorder.capture_frame(self.screen)

        pygame.display.flip()

        if is_graph_update:
            self.clock.tick(self.fps_cap)
        else:
            self.clock.tick(0)

    # ========================================================================
    # Event Handling
    # ========================================================================

    def handle_events(self) -> bool:
        """
        Process pygame events (window close, resize, keyboard shortcuts).

        Keyboard controls:
            - ESC or window close: Quit (returns False)
            - SPACE: Toggle pause
            - S: Take screenshot

        Resize handling:
            - VIDEORESIZE events recalculate layout immediately (cheap)
            - Expensive component rebuild is debounced (150ms delay)
            - During debounce, graphs are smoothscaled to fit

        Returns:
            bool: True to continue, False to quit.
        """
        # Check if a debounced resize is ready to apply
        if self._pending_resize is not None:
            elapsed_ms = pygame.time.get_ticks() - self._resize_timer
            if elapsed_ms >= RESIZE_DEBOUNCE_MS:
                self._pending_resize = None
                self._rebuild_components()
                self._rebuild_fonts()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.VIDEORESIZE:
                # Clamp to minimum size
                w = max(event.w, MIN_WIDTH)
                h = max(event.h, MIN_HEIGHT)

                # Skip if size didn't actually change (prevents loops)
                if w == self.window_width and h == self.window_height:
                    continue

                # Resize the pygame display (cheap)
                self.screen = pygame.display.set_mode(
                    (w, h),
                    pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.RESIZABLE,
                )

                # Recalculate layout immediately (cheap — just math)
                self._recalculate_layout(w, h)

                # Defer expensive component rebuild (debounce)
                self._pending_resize = (w, h)
                self._resize_timer = pygame.time.get_ticks()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                elif event.key == pygame.K_s:
                    # Save screenshot
                    timestamp = int(time.time())
                    filename = f'recordings/screenshot_{timestamp}.png'
                    try:
                        pygame.image.save(self.screen, filename)
                        print(f'Screenshot saved: {filename}')
                    except Exception as e:
                        print(f'Screenshot failed: {e}')
        return True

    def get_surface(self) -> pygame.Surface:
        """
        Get the current screen surface for recording/streaming.

        Returns:
            pygame.Surface: The current dashboard display.
        """
        return self.screen.copy()

    # ========================================================================
    # Drawing Helpers
    # ========================================================================

    def _draw_background(self) -> None:
        """Fill the entire window with the background color."""
        self.screen.fill(BG_COLOR)

    def _draw_title_bar(self) -> None:
        """Draw the top title bar with algorithm name and elapsed time."""
        # Background
        title_rect = pygame.Rect(0, 0, self.window_width, self.top_bar_height)
        pygame.draw.rect(self.screen, TOP_BAR_COLOR, title_rect)

        # Title text (centered)
        algo_names = {'neat': 'NEAT', 'ppo': 'PPO', 'dqn': 'DQN'}
        algo_display = algo_names.get(self.algorithm, self.algorithm.upper())
        title_text = self._title_font.render(
            f'Mario ML Dashboard  -  {algo_display}',
            True, ACCENT_COLOR,
        )
        title_x = (self.window_width - title_text.get_width()) // 2
        title_y = (self.top_bar_height - title_text.get_height()) // 2
        self.screen.blit(title_text, (title_x, title_y))

        # Elapsed time on the right
        elapsed = self.metrics.get_elapsed_time_str()
        time_text = self._label_font.render(
            f'Time: {elapsed}', True, DIM_TEXT_COLOR,
        )
        time_y = (self.top_bar_height - time_text.get_height()) // 2
        self.screen.blit(
            time_text,
            (self.window_width - time_text.get_width() - 15, time_y),
        )

    def _draw_status_bar(self) -> None:
        """Draw the bottom status bar with current and best metrics."""
        y = self.window_height - self.bottom_bar_height
        bar_rect = pygame.Rect(0, y, self.window_width, self.bottom_bar_height)
        pygame.draw.rect(self.screen, BOTTOM_BAR_COLOR, bar_rect)

        # Build status text from current metrics
        episode = self.metrics.episode_count
        reward = self.metrics.get_latest('reward')
        distance = self.metrics.get_latest('distance')
        best_reward = self.metrics.get_best('reward')
        best_distance = self.metrics.get_best('distance')

        # Current stats (left side)
        ep_label = 'Gen' if self.algorithm == 'neat' else 'Ep'
        current_text = (
            f'{ep_label}: {episode}  |  '
            f'Reward: {reward:.0f}  |  '
            f'Distance: {distance:.0f}'
        )

        # Algorithm-specific extras
        if self.algorithm == 'dqn':
            epsilon = self.metrics.get_latest('epsilon', 1.0)
            current_text += f'  |  Epsilon: {epsilon:.3f}'
        elif self.algorithm == 'neat':
            complexity = self.metrics.get_latest('complexity')
            current_text += f'  |  Complexity: {complexity:.0f}'
            if self._genome_progress:
                current_text += f'  |  Genome: {self._genome_progress}'

        current_surface = self._status_font.render(
            current_text, True, TEXT_COLOR,
        )
        text_y = y + (self.bottom_bar_height - current_surface.get_height()) // 2
        self.screen.blit(current_surface, (15, text_y))

        # Best stats (right side)
        best_text = f'Best Reward: {best_reward:.0f}  |  Best Dist: {best_distance:.0f}'
        best_surface = self._status_font.render(
            best_text, True, ACCENT_COLOR,
        )
        self.screen.blit(
            best_surface,
            (self.window_width - best_surface.get_width() - 15, text_y),
        )

    def _draw_dividers(self) -> None:
        """Draw subtle divider lines between panels."""
        # Vertical divider between game and graphs
        pygame.draw.line(
            self.screen, BORDER_COLOR,
            (self.game_panel_width, self.top_bar_height),
            (self.game_panel_width, self.window_height - self.bottom_bar_height),
            1,
        )
        # Horizontal divider at top bar
        pygame.draw.line(
            self.screen, BORDER_COLOR,
            (0, self.top_bar_height),
            (self.window_width, self.top_bar_height),
            1,
        )
        # Horizontal divider at bottom bar
        pygame.draw.line(
            self.screen, BORDER_COLOR,
            (0, self.window_height - self.bottom_bar_height),
            (self.window_width, self.window_height - self.bottom_bar_height),
            1,
        )

    def _draw_pause_overlay(self) -> None:
        """Draw a semi-transparent pause overlay."""
        overlay = pygame.Surface(
            (self.window_width, self.window_height), pygame.SRCALPHA,
        )
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))

        pause_text = self._title_font.render(
            'PAUSED - Press SPACE to resume', True, ACCENT_COLOR,
        )
        text_x = (self.window_width - pause_text.get_width()) // 2
        text_y = (self.window_height - pause_text.get_height()) // 2
        self.screen.blit(pause_text, (text_x, text_y))

    def _update_fps(self) -> None:
        """Calculate and update the displayed FPS value."""
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._display_fps = int(self._frame_count / elapsed)
            self._frame_count = 0
            self._last_fps_time = now

    def close(self) -> None:
        """Clean up pygame and matplotlib resources."""
        self.graph_panel.cleanup()
        pygame.quit()
