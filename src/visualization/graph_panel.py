"""
Graph Panel for Super Mario Bros ML Dashboard.

Renders real-time updating matplotlib graphs to pygame surfaces.
Uses matplotlib's Agg backend (non-interactive) to render plots to
raw pixel buffers, which are then displayed via pygame.

The panel shows 4 graphs in a 2x2 grid:
    - Top Left: Reward over episodes/generations (with rolling average)
    - Top Right: Distance progression (how far Mario gets)
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

# Action labels for the bar chart
ACTION_LABELS = ['NOOP', 'Right', 'JmpR', 'RunR', 'RJmpR', 'Jump', 'Left']


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
    ):
        self.width = width
        self.height = height
        self.algorithm = algorithm
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

        # Distance graph
        self.axes['distance'].set_title('Distance (x position)', **title_props)
        self.axes['distance'].set_xlabel(x_label, **label_props)
        self.axes['distance'].set_ylabel('Distance', **label_props)

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

        x_label = 'Generation' if self.algorithm == 'neat' else 'Episode'
        ax.set_title('Reward', color=COLORS['text'], fontsize=10, fontweight='bold')
        ax.set_xlabel(x_label, color=COLORS['text'], fontsize=8)
        ax.tick_params(colors=COLORS['text'], labelsize=7)

        # --- Distance Graph ---
        ax = self.axes['distance']
        ax.clear()
        ax.set_facecolor(COLORS['panel_bg'])
        ax.grid(True, alpha=0.2, color=COLORS['grid'])

        distances = tracker.get_values('distance')
        if distances:
            episodes = list(range(1, len(distances) + 1))
            # Area fill for visual impact
            ax.fill_between(
                episodes, distances,
                color=COLORS['distance_fill'], alpha=0.3, zorder=2,
            )
            # Main line
            avg_dist = tracker.get_rolling_average('distance', window=10)
            ax.plot(
                episodes, avg_dist,
                color=COLORS['accent'], linewidth=2, zorder=3,
            )
            # Best distance marker
            if distances:
                best_idx = distances.index(max(distances))
                ax.scatter(
                    [best_idx + 1], [max(distances)],
                    color=COLORS['accent'], s=50, zorder=4,
                    marker='*', edgecolors='white', linewidths=0.5,
                )

        ax.set_title('Distance (x position)', color=COLORS['text'],
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

        action_counts = tracker.get_values('action_distribution')
        has_actions = False
        if action_counts and len(action_counts) > 0:
            # Get the latest action distribution (list of counts per action)
            latest = action_counts[-1]
            if isinstance(latest, (list, np.ndarray)) and len(latest) > 0:
                has_actions = True
                bars = ax.bar(
                    range(len(latest)), latest,
                    color=COLORS['actions'][:len(latest)],
                    edgecolor='none', alpha=0.85, zorder=3,
                )
                # Add value labels on top of bars
                total = sum(latest) if sum(latest) > 0 else 1
                for bar, count in zip(bars, latest):
                    pct = count / total * 100
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{pct:.0f}%', ha='center', va='bottom',
                        color=COLORS['text'], fontsize=6,
                    )
                ax.set_xticks(range(len(latest)))
                ax.set_xticklabels(
                    ACTION_LABELS[:len(latest)],
                    rotation=30, fontsize=6,
                )

        if not has_actions:
            ax.text(
                0.5, 0.5, 'Waiting for data...',
                transform=ax.transAxes, ha='center', va='center',
                color=COLORS['text'], fontsize=9, alpha=0.6,
            )

        ax.set_title('Action Distribution', color=COLORS['text'],
                     fontsize=10, fontweight='bold')
        ax.tick_params(colors=COLORS['text'], labelsize=7)

        # Re-apply tight layout after clearing and redrawing
        self.fig.tight_layout(pad=1.5)
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
