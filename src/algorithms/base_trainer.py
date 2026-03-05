"""
Base Trainer Abstract Class for Super Mario Bros ML.

Defines the common interface that all algorithm trainers (NEAT, PPO, DQN)
must implement. This ensures consistent behavior for:
    - Training loops
    - Model evaluation
    - Checkpoint saving and loading
    - Dashboard visualization integration
    - Graceful shutdown with auto-save

The BaseTrainer also provides shared functionality:
    - Ctrl+C signal handling (saves model before exit)
    - Automatic periodic checkpointing
    - Metadata JSON export alongside model files
    - Elapsed time tracking

Usage:
    class MyTrainer(BaseTrainer):
        def train(self, num_episodes):
            ...
        def evaluate(self, num_episodes):
            ...
        def save_checkpoint(self, path):
            ...
        def load_checkpoint(self, path):
            ...
"""

import os
import sys
import json
import time
import signal
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

try:
    import pygame
except ImportError:
    pygame = None

from src.visualization.dashboard import Dashboard


class BaseTrainer(ABC):
    """
    Abstract base class for all ML algorithm trainers.

    Provides the common interface and shared utility methods that
    NEAT, PPO, and DQN trainers all use.

    Args:
        env: The Mario environment instance (pre-wrapped).
        config: Dictionary of hyperparameters for the algorithm.
        visualizer: Optional Dashboard instance for live visualization.
        save_dir: Directory for saving checkpoints. Default 'models/'.
        log_dir: Directory for saving logs. Default 'logs/'.

    Attributes:
        env: The game environment.
        config: Algorithm hyperparameters.
        visualizer: Dashboard for live display (may be None).
        episode_count: Number of episodes/generations completed.
        best_reward: Best reward achieved during training.
        is_training: Whether training is currently active.

    Subclasses must implement:
        - train(): Main training loop
        - evaluate(): Run the trained model without learning
        - save_checkpoint(): Persist the model to disk
        - load_checkpoint(): Restore a model from disk
    """

    def __init__(
        self,
        env,
        config: Dict[str, Any],
        visualizer: Optional[Dashboard] = None,
        save_dir: str = 'models',
        log_dir: str = 'logs',
    ):
        self.env = env
        self.config = config
        self.visualizer = visualizer
        self.save_dir = save_dir
        self.log_dir = log_dir

        # Training state
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.best_distance = 0
        self.is_training = False
        self.start_time = time.time()
        self._dashboard_closed = False  # Prevents double-save on window close

        # Checkpoint settings (can be overridden in config)
        self.checkpoint_interval = config.get('save_freq', 50)

        # Register graceful shutdown handler
        # This ensures models are saved when the user presses Ctrl+C
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    @abstractmethod
    def train(self, num_episodes: int) -> None:
        """
        Main training loop.

        Implementations should:
        1. Run episodes/generations
        2. Update the model based on experience
        3. Call self.update_visualization() with metrics
        4. Call self._auto_checkpoint() periodically
        5. Save the final model when training completes

        Args:
            num_episodes: Number of episodes or generations to train for.
        """
        pass

    @abstractmethod
    def evaluate(self, num_episodes: int = 5) -> float:
        """
        Evaluate the current model without training.

        Runs the model in the environment and returns the average reward.
        Should render frames to the dashboard if available.

        Args:
            num_episodes: Number of episodes to evaluate over.

        Returns:
            float: Average reward across evaluation episodes.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """
        Save the current model state to disk.

        Implementations should save everything needed to resume training
        or run evaluation: model weights, optimizer state, episode count, etc.

        Args:
            path: File path to save the checkpoint.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """
        Load a model state from disk.

        Implementations should restore all state needed for training
        or evaluation.

        Args:
            path: File path to load the checkpoint from.

        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
        """
        pass

    def update_visualization(
        self,
        frame: Optional[Any] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send data to the dashboard for display.

        Also handles dashboard events (window close, pause).
        Returns False if the user has closed the dashboard.

        Args:
            frame: Current game frame (numpy array) for display.
            metrics: Dictionary of metric values to display.

        Returns:
            bool: True to continue, False if dashboard was closed.
        """
        if self.visualizer is None:
            return True

        # If dashboard was already closed, don't try to use it again
        if self._dashboard_closed:
            return False

        # Update the dashboard display
        try:
            self.visualizer.update(frame=frame, metrics=metrics)
        except pygame.error:
            # Pygame already shut down
            self._dashboard_closed = True
            return False

        # Check for user events (close window, pause, etc.)
        if not self.visualizer.handle_events():
            # User closed the window — trigger graceful shutdown (once)
            self._dashboard_closed = True
            print('\nDashboard closed. Saving model...')
            self._save_on_exit()
            return False

        # Handle pause state
        while self.visualizer.is_paused:
            if not self.visualizer.handle_events():
                self._dashboard_closed = True
                print('\nDashboard closed. Saving model...')
                self._save_on_exit()
                return False
            self.visualizer.update()  # Keep rendering while paused
            time.sleep(0.1)

        return True

    def update_visualization_grid(
        self,
        frames: Optional[list] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send multiple game frames to the dashboard grid display.

        Used when running multiple environments in parallel. Falls back
        to single-frame update_visualization if the dashboard doesn't
        support grid mode.

        Args:
            frames: List of game frames (numpy arrays), one per env.
            metrics: Dictionary of metric values to display.

        Returns:
            bool: True to continue, False if dashboard was closed.
        """
        if self.visualizer is None:
            return True

        if self._dashboard_closed:
            return False

        try:
            if hasattr(self.visualizer, 'update_grid'):
                self.visualizer.update_grid(frames=frames, metrics=metrics)
            elif frames and len(frames) > 0:
                # Fallback: use first frame
                self.visualizer.update(frame=frames[0], metrics=metrics)
            else:
                self.visualizer.update(frame=None, metrics=metrics)
        except pygame.error:
            self._dashboard_closed = True
            return False

        # Check for user events
        if not self.visualizer.handle_events():
            self._dashboard_closed = True
            print('\nDashboard closed. Saving model...')
            self._save_on_exit()
            return False

        # Handle pause
        while self.visualizer.is_paused:
            if not self.visualizer.handle_events():
                self._dashboard_closed = True
                print('\nDashboard closed. Saving model...')
                self._save_on_exit()
                return False
            self.visualizer.update()
            time.sleep(0.1)

        return True

    def _auto_checkpoint(self, episode: int) -> None:
        """
        Automatically save checkpoint at configured intervals.

        Args:
            episode: Current episode/generation number.
        """
        if episode > 0 and episode % self.checkpoint_interval == 0:
            algo_name = self.__class__.__name__.lower().replace('trainer', '')
            checkpoint_dir = os.path.join(self.save_dir, algo_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f'checkpoint_ep_{episode}')
            print(f'\n  Auto-saving checkpoint at episode {episode}...')
            self.save_checkpoint(path)
            self._save_metadata(checkpoint_dir, episode)

    def _save_metadata(self, directory: str, episode: int) -> None:
        """
        Save training metadata as JSON alongside model files.

        Includes: episode count, best metrics, elapsed time, config hash.

        Args:
            directory: Directory to save metadata in.
            episode: Current episode number.
        """
        metadata = {
            'algorithm': self.__class__.__name__,
            'episode': int(episode),
            'best_reward': float(self.best_reward),
            'best_distance': int(self.best_distance),
            'elapsed_seconds': time.time() - self.start_time,
            'elapsed_time': self._format_time(time.time() - self.start_time),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {k: str(v) for k, v in self.config.items()},
        }

        metadata_path = os.path.join(directory, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _handle_shutdown(self, signum, frame) -> None:
        """
        Handle Ctrl+C gracefully by saving the model before exiting.

        This signal handler catches SIGINT (Ctrl+C) and saves the
        current model state before terminating. This prevents loss
        of training progress when stopping training manually.
        """
        print('\n\nCtrl+C detected! Saving model before exit...')
        self._save_on_exit()
        # Restore original handler and re-raise
        signal.signal(signal.SIGINT, self._original_sigint)
        sys.exit(0)

    def _save_on_exit(self) -> None:
        """Save model and metadata on training exit (graceful or forced).

        Uses a guard flag to prevent saving multiple times (e.g., if
        both the dashboard-close handler and the end-of-training code
        both call this method).
        """
        if getattr(self, '_already_saved', False):
            return
        self._already_saved = True
        try:
            algo_name = self.__class__.__name__.lower().replace('trainer', '')
            save_path = os.path.join(self.save_dir, algo_name)
            os.makedirs(save_path, exist_ok=True)

            final_path = os.path.join(save_path, 'final')
            self.save_checkpoint(final_path)
            self._save_metadata(save_path, self.episode_count)

            # Also export metrics
            if self.visualizer:
                metrics_path = os.path.join(self.log_dir, algo_name, 'metrics.json')
                os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
                self.visualizer.metrics.export_json(metrics_path)

            print(f'  Model saved to: {save_path}/')
            print(f'  Episodes completed: {self.episode_count}')
            print(f'  Best reward: {self.best_reward:.1f}')
        except Exception as e:
            print(f'  Warning: Failed to save model: {e}')

    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Format seconds into a human-readable string.

        Args:
            seconds: Number of seconds.

        Returns:
            Formatted string like "1h 23m 45s".
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f'{hours}h {minutes}m {secs}s'
        elif minutes > 0:
            return f'{minutes}m {secs}s'
        else:
            return f'{secs}s'
