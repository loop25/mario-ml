"""
Graph Panel for ML Training Dashboard.

Renders real-time updating matplotlib graphs to pygame surfaces.
Uses matplotlib's Agg backend (non-interactive) to render plots to
raw pixel buffers, which are then displayed via pygame.

The panel shows 4 graphs in a 2x2 grid:
    - Top Left: Reward over episodes/generations (with rolling average)
    - Top Right: Game-specific metric (distance, score, win rate, etc.)
    - Bottom Left: Loss or network complexity (algorithm-dependent)
    - Bottom Right: Action distribution (bar chart)

Design goals:
    - Visually appealing with custom color scheme
    - Smooth trend lines over raw data points
    - Efficient updates (only redraw when new data arrives)
    - Configurable for different algorithms

Usage:
    panel = GraphPanel(width=700, height=600)
    panel.update_data(tracker)
    surface = panel.render()
    screen.blit(surface, (700, 50))
"""

import matplotlib
# Use non-interactive Agg backend (renders to memory, not a window)
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pygame
from typing import List, Optional, Dict

from src.visualization.metrics_tracker import MetricsTracker


# ============================================================================
# Color Scheme - Retro gaming inspired, visually appealing for streams
# ============================================================================
COLORS = {
    'bg': '#1a1a2e',          # Dark navy background
    'panel_bg': '#16213e',     # Slightly lighter panel background
    'grid': '#2a2a4a',         # Subtle grid lines
    'reward': '#e94560',       # Vibrant red-pink for reward
    'reward_avg': '#ff6b6b',   # Lighter red for rolling average
    'distance': '#0f3460',     # Deep blue for distance
    'distance_fill': '#533483', # Purple fill for distance area
    'loss': '#e94560',         # Red for loss
    'complexity': '#00b4d8',   # Cyan for network complexity
    'actions': [               # Rainbow palette for action bars
        '#e94560',  # Red - NOOP
        '#ff6b6b',  # Light red - Right
        '#ffa500',  # Orange - Jump Right
        '#ffd700',  # Gold - Run Right
        '#00ff88',  # Green - Run Jump Right
        '#00b4d8',  # Cyan - Jump
        '#533483',  # Purple - Left
    ],
    'text': '#e0e0e0',        # Light gray text
    'accent': '#00ff88',       # Neon green accent
}

# Default action labels (generic fallback; overridden per game via adapter)
DEFAULT_ACTION_LABELS = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']


class GraphPanel:
    """
    Manages 4 real-time graphs rendered as a pygame surface.

    Creates a matplotlib figure with 4 subplots, updates them with
    new data from MetricsTracker, and renders to a pygame surface.

    Args:
        width: Panel width in pixels. Default 700.
        height: Panel height in pixels. Default 600.
        algorithm: Algorithm name ('neat', 'ppo', 'dqn') to customize
                   graph labels. Default 'neat'.
        action_labels: List of action names for the bar chart.
        dashboard_config: Dict with graph_2_title, graph_2_metric, etc.

    Attributes:
        fig: Matplotlib figure containing all subplots.
        axes: Dictionary of subplot axes.
        surface: Most recent pygame surface render.
    """

    def __init__(
        self,
        width: int = 700,
        height: int = 600,
        algorithm: str = 'neat',
        action_labels: Optional[List[str]] = None,
        dashboard_config: Optional[Dict] = None,
    ):
        self.width = width
        self.height = height
        self.algorithm = algorithm
        self.action_labels = action_labels or list(DEFAULT_ACTION_LABELS)
        self.dashboard_config = dashboard_config or {
            'graph_2_title': 'Distance (x position)',
            'graph_2_metric': 'distance',
        }
        self._needs_update = True

        # Convert pixel dimensions to inches for matplotlib
        # Using 100 DPI for a good balance of quality and performance
        self.dpi = 100
        fig_width = width / self.dpi
        fig_height = height / self.dpi

        # Create matplotlib figure with dark background
        self.fig, axes_array = plt.subplots(
            2, 2,
            figsize=(fig_width, fig_height),
            dpi=self.dpi,
            facecolor=COLORS['bg'],
        )

        # Store axes with descriptive names
        self.axes = {
            'reward': axes_array[0, 0],
            'distance': axes_array[0, 1],
            'loss': axes_array[1, 0],
            'actions': axes_array[1, 1],
        }

        # Apply dark theme to all axes
        for ax in self.axes.values():
            ax.set_facecolor(COLORS['panel_bg'])
            ax.tick_params(colors=COLORS['text'], labelsize=8)
            ax.spines['bottom'].set_color(COLORS['grid'])
            ax.spines['top'].set_color(COLORS['grid'])
            ax.spines['left'].set_color(COLORS['grid'])
            ax.spines['right'].set_color(COLORS['grid'])
            ax.grid(True, alpha=0.2, color=COLORS['grid'])

        # Set up initial titles and labels
        self._setup_labels()

        # Tight layout to maximize graph area
        self.fig.tight_layout(pad=1.5)

        # Create the Agg canvas for rendering
        self.canvas = FigureCanvasAgg(self.fig)

        # Cache the pygame surface
        self.surface: Optional[pygame.Surface] = None

    def _setup_labels(self) -> None:
        """Configure axis titles and labels based on algorithm type."""
        title_props = {'color': COLORS['text'], 'fontsize': 10, 'fontweight': 'bold'}
        label_props = {'color': COLORS['text'], 'fontsize': 8}

        # Reward graph
        x_label = 'Generation' if self.algorithm == 'neat' else 'Episode'
        self.axes['reward'].set_title('Reward', **title_props)
        self.axes['reward'].set_xlabel(x_label, **label_props)
        self.axes['reward'].set_ylabel('Reward', **label_props)

        # Graph 2: game-specific metric (distance, score, win rate, etc.)
        graph_2_title = self.dashboard_config.get('graph_2_title', 'Distance (x position)')
        self.axes['distance'].set_title(graph_2_title, **title_props)
        self.axes['distance'].set_xlabel(x_label, **label_props)
        self.axes['distance'].set_ylabel(graph_2_title.split('(')[0].strip(), **label_props)

        # Loss / Complexity graph (changes based on algorithm)
        if self.algorithm == 'neat':
            self.axes['loss'].set_title('Network Complexity', **title_props)
            self.axes['loss'].set_xlabel('Generation', **label_props)
            self.axes['loss'].set_ylabel('Nodes + Connections', **label_props)
        else:
            self.axes['loss'].set_title('Training Loss', **title_props)
            self.axes['loss'].set_xlabel('Episode', **label_props)
            self.axes['loss'].set_ylabel('Loss', **label_props)

        # Action distribution
        self.axes['actions'].set_title('Action Distribution', **title_props)
        self.axes['actions'].set_ylabel('Count', **label_props)

    def update_data(self, tracker: MetricsTracker) -> None:
        """
        Update all graphs with latest data from metrics tracker.

        Clears existing plots and redraws with current data.
        Only updates if new data has been recorded since last update.

        Args:
            tracker: MetricsTracker instance containing training metrics.
        """
        # --- Reward Graph ---
        ax = self.axes['reward']
        ax.clear()
        ax.set_facecolor(COLORS['panel_bg'])
        ax.grid(True, alpha=0.2, color=COLORS['grid'])

        rewards = tracker.get_values('reward')
        if rewards:
            episodes = list(range(1, len(rewards) + 1))
            # Raw data as faint scatter points
            ax.scatter(
                episodes, rewards,
                color=COLORS['reward'], alpha=0.3, s=8, zorder=2,
            )
            # Rolling average as a smooth line
            avg = tracker.get_rolling_average('reward', window=10)
            ax.plot(
                episodes, avg,
                color=COLORS['reward_avg'], linewidth=2, zorder=3,
                label=f'Avg (10)',
            )
            ax.legend(loc='upper left', fontsize=7, facecolor=COLORS['panel_bg'],
                     edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
            # Ensure Y-axis shows a meaningful range even when rewards are flat
            r_min, r_max = min(rewards), max(rewards)
            if r_max - r_min < 0.5:
                mid = (r_min + r_max) / 2
                ax.set_ylim(mid - 1.0, mid + 1.0)

        x_label = 'Generation' if self.algorithm == 'neat' else 'Episode'
        ax.set_title('Reward', color=COLORS['text'], fontsize=10, fontweight='bold')
        ax.set_xlabel(x_label, color=COLORS['text'], fontsize=8)
        ax.tick_params(colors=COLORS['text'], labelsize=7)

        # --- Graph 2: Game-Specific Metric (Distance / Score / Win Rate) ---
        ax = self.axes['distance']
        ax.clear()
        ax.set_facecolor(COLORS['panel_bg'])
        ax.grid(True, alpha=0.2, color=COLORS['grid'])

        graph_2_metric = self.dashboard_config.get('graph_2_metric', 'distance')
        graph_2_title = self.dashboard_config.get('graph_2_title', 'Distance (x position)')
        secondary_metric = self.dashboard_config.get('graph_2_secondary_metric')
        legend_labels = self.dashboard_config.get('graph_2_legend')

        values = tracker.get_values(graph_2_metric)
        if values:
            episodes = list(range(1, len(values) + 1))

            if secondary_metric:
                # Dual-line mode (e.g. agent vs opponent win rate)
                avg_vals = tracker.get_rolling_average(graph_2_metric, window=10)
                label_1 = legend_labels[0] if legend_labels else graph_2_metric
                ax.plot(
                    episodes, avg_vals,
                    color=COLORS['accent'], linewidth=2, zorder=3,
                    label=label_1,
                )

                sec_values = tracker.get_values(secondary_metric)
                if sec_values:
                    sec_episodes = list(range(1, len(sec_values) + 1))
                    sec_avg = tracker.get_rolling_average(secondary_metric, window=10)
                    label_2 = legend_labels[1] if legend_labels and len(legend_labels) > 1 else secondary_metric
                    ax.plot(
                        sec_episodes, sec_avg,
                        color=COLORS['reward'], linewidth=2, zorder=3,
                        label=label_2,
                    )

                ax.legend(
                    loc='upper left', fontsize=7,
                    facecolor=COLORS['panel_bg'],
                    edgecolor=COLORS['grid'],
                    labelcolor=COLORS['text'],
                )
            else:
                # Single-metric mode with area fill
                ax.fill_between(
                    episodes, values,
                    color=COLORS['distance_fill'], alpha=0.3, zorder=2,
                )
                avg_vals = tracker.get_rolling_average(graph_2_metric, window=10)
                ax.plot(
                    episodes, avg_vals,
                    color=COLORS['accent'], linewidth=2, zorder=3,
                )
                # Best value marker
                if values:
                    best_idx = values.index(max(values))
                    ax.scatter(
                        [best_idx + 1], [max(values)],
                        color=COLORS['accent'], s=50, zorder=4,
                        marker='*', edgecolors='white', linewidths=0.5,
                    )

        ax.set_title(graph_2_title, color=COLORS['text'],
                     fontsize=10, fontweight='bold')
        ax.set_xlabel(x_label, color=COLORS['text'], fontsize=8)
        ax.tick_params(colors=COLORS['text'], labelsize=7)

        # --- Loss / Complexity Graph ---
        ax = self.axes['loss']
        ax.clear()
        ax.set_facecolor(COLORS['panel_bg'])
        ax.grid(True, alpha=0.2, color=COLORS['grid'])

        if self.algorithm == 'neat':
            # Show network complexity for NEAT
            complexity = tracker.get_values('complexity')
            if complexity:
                episodes = list(range(1, len(complexity) + 1))
                ax.plot(
                    episodes, complexity,
                    color=COLORS['complexity'], linewidth=2, zorder=3,
                )
                ax.fill_between(
                    episodes, complexity,
                    color=COLORS['complexity'], alpha=0.15, zorder=2,
                )
            ax.set_title('Network Complexity', color=COLORS['text'],
                         fontsize=10, fontweight='bold')
            ax.set_ylabel('Nodes + Connections', color=COLORS['text'], fontsize=8)
        else:
            # Show training loss for PPO/DQN
            losses = tracker.get_values('loss')
            if losses:
                # Check if all losses are zero (training hasn't started)
                non_zero = [l for l in losses if l > 0]
                if non_zero:
                    episodes = list(range(1, len(losses) + 1))
                    ax.plot(
                        episodes, losses,
                        color=COLORS['loss'], linewidth=1, alpha=0.5, zorder=2,
                    )
                    # Smooth loss curve
                    avg_loss = tracker.get_rolling_average('loss', window=20)
                    ax.plot(
                        episodes, avg_loss,
                        color=COLORS['reward_avg'], linewidth=2, zorder=3,
                    )
                else:
                    ax.text(
                        0.5, 0.5, 'Filling replay buffer...\nTraining starts soon',
                        transform=ax.transAxes, ha='center', va='center',
                        color=COLORS['text'], fontsize=9, alpha=0.6,
                    )
            else:
                ax.text(
                    0.5, 0.5, 'Waiting for data...',
                    transform=ax.transAxes, ha='center', va='center',
                    color=COLORS['text'], fontsize=9, alpha=0.6,
                )
            ax.set_title('Training Loss', color=COLORS['text'],
                         fontsize=10, fontweight='bold')
            ax.set_ylabel('Loss', color=COLORS['text'], fontsize=8)

        ax.set_xlabel(x_label, color=COLORS['text'], fontsize=8)
        ax.tick_params(colors=COLORS['text'], labelsize=7)

        # --- Action Distribution Bar Chart ---
        ax = self.axes['actions']
        ax.clear()
        ax.set_facecolor(COLORS['panel_bg'])
        ax.grid(True, alpha=0.15, color=COLORS['grid'], axis='y')

        action_counts = tracker.get_values('action_distribution')
        has_actions = False
        if action_counts and len(action_counts) > 0:
            # Get the latest action distribution (list of counts per action)
            latest = action_counts[-1]
            if isinstance(latest, (list, np.ndarray)) and len(latest) > 0:
                has_actions = True
                n_actions = len(latest)
                all_labels = self.action_labels[:n_actions]

                # For large action spaces (board games: 1024, 4096 actions),
                # showing every bar makes the chart unreadable.  Aggregate
                # to the top-N most-used actions + an "Other" bucket.
                MAX_DISPLAY = 12
                if n_actions > MAX_DISPLAY:
                    arr = np.array(latest, dtype=np.float64)
                    top_idx = np.argsort(arr)[::-1][:MAX_DISPLAY]
                    top_idx_sorted = np.sort(top_idx)  # keep original order
                    display_vals = arr[top_idx_sorted].tolist()
                    display_labels = [all_labels[i] if i < len(all_labels)
                                      else f'A{i}' for i in top_idx_sorted]
                    other_sum = arr.sum() - sum(display_vals)
                    if other_sum > 0:
                        display_vals.append(other_sum)
                        display_labels.append('Other')
                    n_display = len(display_vals)
                else:
                    display_vals = list(latest)
                    display_labels = [l[:8] for l in all_labels]
                    n_display = n_actions

                # Extend the action color palette if needed (cycle through)
                bar_colors = (COLORS['actions'] * ((n_display // len(COLORS['actions'])) + 1))[:n_display]
                # Use wider bars for games with few actions (fills the chart)
                bar_width = max(0.4, min(0.85, 5.0 / max(n_display, 1)))
                bars = ax.bar(
                    range(n_display), display_vals,
                    width=bar_width,
                    color=bar_colors,
                    edgecolor='white', linewidth=0.5,
                    alpha=0.9, zorder=3,
                )
                # Add percentage labels on top of bars
                total = sum(display_vals) if sum(display_vals) > 0 else 1
                for bar, count in zip(bars, display_vals):
                    pct = count / total * 100
                    if pct >= 1:  # Only label bars with ≥1% to avoid clutter
                        label_y = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2, label_y,
                            f'{pct:.0f}%', ha='center', va='bottom',
                            color=COLORS['text'], fontsize=7, fontweight='bold',
                        )
                ax.set_xticks(range(n_display))
                truncated = [l[:8] for l in display_labels]
                rotation = 35 if n_display > 5 else 0
                ax.set_xticklabels(truncated, rotation=rotation, fontsize=7,
                                   ha='right' if rotation else 'center')
                # Ensure bars are always visible: set a minimum Y range
                max_val = max(display_vals) if max(display_vals) > 0 else 1
                ax.set_ylim(0, max_val * 1.25)  # 25% headroom for labels

        if not has_actions:
            ax.text(
                0.5, 0.5, 'Waiting for data...',
                transform=ax.transAxes, ha='center', va='center',
                color=COLORS['text'], fontsize=9, alpha=0.6,
            )

        ax.set_title('Action Distribution', color=COLORS['text'],
                     fontsize=10, fontweight='bold')
        ax.tick_params(colors=COLORS['text'], labelsize=7)

        self._needs_update = True

    def render(self) -> pygame.Surface:
        """
        Render the matplotlib figure to a pygame surface.

        Draws the figure to the Agg canvas, extracts the raw RGB buffer,
        and converts it to a pygame surface.

        Returns:
            pygame.Surface: The rendered graph panel as a surface.
        """
        # Draw the matplotlib figure to the canvas buffer
        self.canvas.draw()

        # Extract raw RGB data from the canvas
        raw_data = self.canvas.buffer_rgba()
        size = self.canvas.get_width_height()

        # Create pygame surface from the raw buffer
        # matplotlib gives RGBA, pygame expects RGB
        surf = pygame.image.frombuffer(raw_data, size, 'RGBA')
        self.surface = surf
        self._needs_update = False
        return surf

    def set_algorithm(self, algorithm: str) -> None:
        """
        Change the algorithm type, updating graph labels accordingly.

        Args:
            algorithm: Algorithm name ('neat', 'ppo', 'dqn').
        """
        self.algorithm = algorithm
        self._setup_labels()

    def resize(self, width: int, height: int) -> None:
        """
        Resize the graph panel by recreating the matplotlib figure.

        Closes the old figure and creates a new one at the target
        dimensions. Re-applies dark theme styling to all axes.

        Args:
            width: New panel width in pixels.
            height: New panel height in pixels.
        """
        self.width = width
        self.height = height
        fig_width = width / self.dpi
        fig_height = height / self.dpi

        # Close old figure to prevent memory leak
        plt.close(self.fig)

        # Recreate figure with new dimensions
        self.fig, axes_array = plt.subplots(
            2, 2,
            figsize=(fig_width, fig_height),
            dpi=self.dpi,
            facecolor=COLORS['bg'],
        )

        self.axes = {
            'reward': axes_array[0, 0],
            'distance': axes_array[0, 1],
            'loss': axes_array[1, 0],
            'actions': axes_array[1, 1],
        }

        # Re-apply dark theme to all axes
        for ax in self.axes.values():
            ax.set_facecolor(COLORS['panel_bg'])
            ax.tick_params(colors=COLORS['text'], labelsize=8)
            ax.spines['bottom'].set_color(COLORS['grid'])
            ax.spines['top'].set_color(COLORS['grid'])
            ax.spines['left'].set_color(COLORS['grid'])
            ax.spines['right'].set_color(COLORS['grid'])
            ax.grid(True, alpha=0.2, color=COLORS['grid'])

        self._setup_labels()
        self.fig.tight_layout(pad=1.5)
        self.canvas = FigureCanvasAgg(self.fig)
        self.surface = None
        self._needs_update = True

    def cleanup(self) -> None:
        """Release matplotlib resources."""
        plt.close(self.fig)
