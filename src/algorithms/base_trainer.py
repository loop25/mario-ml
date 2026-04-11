"""
Base Trainer Abstract Class.

Defines the common interface that all algorithm trainers (NEAT, PPO, DQN, A2C)
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
import threading
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
        env: The game environment instance (pre-wrapped).
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
        event_bus=None,
    ):
        self.env = env
        self.config = config
        self.visualizer = visualizer
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.event_bus = event_bus

        # Training state
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.best_distance = 0
        self.is_training = False
        self.start_time = time.time()
        self._dashboard_closed = False  # Prevents double-save on window close

        # Pause/resume state
        self._training_paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        # Episode callbacks — used by curriculum learning and other
        # cross-cutting concerns that need per-episode notifications.
        self._episode_callbacks = []

        # Game identity — set by main.py after trainer creation so
        # metadata.json records which game produced this checkpoint.
        self.game_id = None

        # Dashboard config — set by main.py after trainer creation.
        # Tells callbacks which info-dict keys to extract and what
        # metric names to report to the dashboard.
        self.dashboard_config = None
        self.num_actions = 7  # Default; overridden per game

        # Checkpoint settings (can be overridden in config)
        self.checkpoint_interval = config.get('save_freq', 50)

        # ── Mastery Detection (auto-stop when fully trained) ─────────
        # Set via set_completion_criteria(). When the rolling metric average
        # exceeds the threshold over `window` episodes, training auto-stops
        # and the model is saved.
        self._completion_criteria = None
        self._completion_metric_history = []
        self.mastered = False  # True when completion criteria met

        # ── DT Trajectory Collection ────────────────────────────────
        # When set, episode data is automatically saved to the experience
        # store for later Decision Transformer training.
        self._experience_store = None
        self._tokenizer = None
        self._episode_obs_buffer = []
        self._episode_act_buffer = []
        self._episode_rew_buffer = []
        self._collect_for_dt = False

        # Register graceful shutdown handler
        # This ensures models are saved when the user presses Ctrl+C
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # On Windows, also handle SIGBREAK so that CTRL_BREAK_EVENT
        # from the launcher triggers graceful shutdown instead of
        # crashing Intel MKL/Fortran runtime (forrtl error 200).
        if sys.platform == 'win32' and hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._handle_shutdown)

    @property
    def training_paused(self):
        """Whether training is currently paused."""
        return self._training_paused

    @training_paused.setter
    def training_paused(self, value: bool):
        self._training_paused = value
        if value:
            self._pause_event.clear()
            print('\n⏸  Training PAUSED (press SPACE to resume)')
        else:
            self._pause_event.set()
            print('\n▶  Training RESUMED')

    def check_pause(self) -> bool:
        """
        Call this at the top of each training step/episode.

        Blocks the calling thread while training_paused is True.
        Returns True to continue training, False if dashboard was closed.
        """
        if not self._training_paused:
            return True

        # Block until resumed (check every 0.1s for dashboard close)
        while self._training_paused:
            if self._pause_event.wait(timeout=0.1):
                return True
            # While waiting, keep processing dashboard events
            if self.visualizer and not self.visualizer.handle_events():
                self._dashboard_closed = True
                print('\nDashboard closed. Saving model...')
                self.visualizer.close()
                self._save_on_exit()
                return False
            if self.visualizer:
                self.visualizer.update()
        return True

    def add_episode_callback(self, callback) -> None:
        """
        Register a callback to be invoked after every episode completes.

        Callbacks receive keyword arguments:
            reward (float): Total episode reward.
            distance (int): Max x-position reached.
            completed (bool): Whether the stage was completed.

        Used by CurriculumManager to receive per-episode data for
        sliding-window advancement decisions.

        Args:
            callback: Callable(reward, distance, completed) -> None.
        """
        self._episode_callbacks.append(callback)

    # ── DT Trajectory Collection Helpers ───────────────────────────

    def enable_dt_collection(self, experience_store, game_adapter):
        """Enable automatic trajectory collection for the Decision Transformer.

        When enabled, each training episode is tokenized and saved to the
        experience store. This allows specialist agents (PPO/DQN/etc.) to
        automatically build up the data the DT needs.

        Args:
            experience_store: ExperienceStore instance.
            game_adapter: The game's adapter (for TokenConfig).
        """
        from src.experience.tokenizer import Tokenizer
        self._experience_store = experience_store
        self._tokenizer = Tokenizer(game_adapter.get_token_config())
        self._collect_for_dt = True
        print(f'  DT collection enabled → {experience_store.store_dir}')

    def _dt_record_step(self, obs, action, reward):
        """Record a single step for DT trajectory collection."""
        if not self._collect_for_dt:
            return
        self._episode_obs_buffer.append(obs)
        self._episode_act_buffer.append(int(action))
        self._episode_rew_buffer.append(float(reward))

    def _dt_finalize_episode(self):
        """Finalize and save the current episode to the experience store."""
        if not self._collect_for_dt or not self._episode_act_buffer:
            self._episode_obs_buffer.clear()
            self._episode_act_buffer.clear()
            self._episode_rew_buffer.clear()
            return
        try:
            trajectory = self._tokenizer.tokenize_episode(
                observations=self._episode_obs_buffer,
                actions=self._episode_act_buffer,
                rewards=self._episode_rew_buffer,
            )
            self._experience_store.add_trajectory(trajectory)
        except Exception as e:
            print(f'  Warning: DT trajectory save failed: {e}')
        self._episode_obs_buffer.clear()
        self._episode_act_buffer.clear()
        self._episode_rew_buffer.clear()

    def _fire_episode_complete(
        self,
        reward: float,
        distance: int = 0,
        completed: bool = False,
    ) -> None:
        """
        Notify all registered callbacks that an episode has finished.

        Called by each trainer subclass at the point where an episode
        (or genome evaluation, for NEAT) completes.

        Args:
            reward: Total episode reward.
            distance: Max x-position reached this episode.
            completed: Whether the stage was cleared.
        """
        for cb in self._episode_callbacks:
            try:
                cb(reward=reward, distance=distance, completed=completed)
            except Exception as e:
                print(f'  Warning: episode callback error: {e}')

    # ── Event Bus Publishing ──────────────────────────────────────

    def publish_event(self, event: dict) -> None:
        """Publish a training event to the event bus if available."""
        if self.event_bus:
            # Auto-inject game_id and episode count for context
            event.setdefault('game_id', self.game_id)
            event.setdefault('episode', self.episode_count)
            self.event_bus.publish(event)

    def publish_episode_complete(self, reward: float, info: dict = None) -> None:
        """Publish an episode_complete event with standard fields."""
        event = {
            'type': 'episode_complete',
            'reward': reward,
            'episode': self.episode_count,
            'game_id': self.game_id,
            'best_reward': self.best_reward,
            'elapsed': time.time() - self.start_time,
        }
        if info:
            event['info'] = info
        self.publish_event(event)

    def publish_new_best(self, reward: float) -> None:
        """Publish event when a new best reward is achieved."""
        self.publish_event({
            'type': 'new_best_reward',
            'reward': reward,
            'episode': self.episode_count,
        })

    def publish_training_start(self) -> None:
        """Publish event when training begins."""
        self.publish_event({
            'type': 'training_start',
            'algorithm': self.__class__.__name__,
        })

    def publish_training_end(self, total_episodes: int) -> None:
        """Publish event when training ends."""
        self.publish_event({
            'type': 'training_end',
            'total_episodes': total_episodes,
            'best_reward': self.best_reward,
            'elapsed': time.time() - self.start_time,
        })

    # ── Mastery Detection ──────────────────────────────────────────

    def set_completion_criteria(self, criteria: dict) -> None:
        """Configure auto-stop when the agent masters the game.

        Args:
            criteria: Dict with keys:
                metric (str): Info-dict key to track (e.g. 'score', 'win_rate')
                threshold (float): Average must exceed this to be "mastered"
                window (int): Number of episodes for the rolling average
                description (str): Human-readable description
        """
        self._completion_criteria = criteria
        self._completion_metric_history = []
        self.mastered = False

    def check_mastery(self, info: dict) -> bool:
        """Check if the agent has mastered the game based on completion criteria.

        Call this after each episode with the info dict. Returns True if the
        rolling average of the tracked metric exceeds the threshold.

        When mastery is detected:
        - self.mastered is set to True
        - A 'game_mastered' event is published
        - The dashboard shows a mastery notification
        - Training should stop (checked by caller)
        """
        if not self._completion_criteria or self.mastered:
            return self.mastered

        metric_key = self._completion_criteria['metric']

        # For win_rate, compute it from the winner field
        if metric_key == 'win_rate':
            value = 1.0 if info.get('winner') == 1 else 0.0
        else:
            value = float(info.get(metric_key, 0))

        self._completion_metric_history.append(value)

        window = self._completion_criteria.get('window', 50)
        threshold = self._completion_criteria['threshold']

        if len(self._completion_metric_history) < window:
            return False

        # Rolling average over the last `window` episodes
        recent = self._completion_metric_history[-window:]
        avg = sum(recent) / len(recent)

        if avg >= threshold:
            self.mastered = True
            desc = self._completion_criteria.get('description', f'{metric_key} >= {threshold}')
            print(f'\n{"="*60}')
            print(f'  GAME MASTERED! {desc}')
            print(f'  Rolling avg: {avg:.3f} (threshold: {threshold})')
            print(f'  After {self.episode_count} episodes')
            print(f'{"="*60}\n')

            self.publish_event({
                'type': 'game_mastered',
                'metric': metric_key,
                'average': avg,
                'threshold': threshold,
                'window': window,
            })

            # Save checkpoint immediately upon mastery
            try:
                algo_name = self.__class__.__name__.replace('Trainer', '').lower()
                mastery_path = os.path.join(self.save_dir, algo_name, 'mastered')
                self.save_checkpoint(mastery_path)
                print(f'  Mastered model saved to: {mastery_path}')
            except Exception as e:
                print(f'  Warning: Could not save mastery checkpoint: {e}')

            return True

        return False

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
            # User closed the window — close it immediately so it
            # doesn't appear frozen while the model saves.
            self._dashboard_closed = True
            print('\nDashboard closed. Saving model...')
            self.visualizer.close()
            self._save_on_exit()
            return False

        # Sync pause state from dashboard to trainer
        if self.visualizer.is_paused != self._training_paused:
            self.training_paused = self.visualizer.is_paused

        return True

    def update_visualization_grid(
        self,
        frames: Optional[list] = None,
        metrics: Optional[Dict[str, Any]] = None,
        infos: Optional[list] = None,
    ) -> bool:
        """
        Send multiple game frames to the dashboard grid/swarm display.

        Used when running multiple environments in parallel. Falls back
        to single-frame update_visualization if the dashboard doesn't
        support grid mode. When in swarm mode, ``infos`` is forwarded
        so the SwarmRenderer can use x_pos offsets for side-scrollers.

        Args:
            frames: List of game frames (numpy arrays), one per env.
            metrics: Dictionary of metric values to display.
            infos: Optional info dicts from each env (for swarm renderer).

        Returns:
            bool: True to continue, False if dashboard was closed.
        """
        if self.visualizer is None:
            return True

        if self._dashboard_closed:
            return False

        try:
            if hasattr(self.visualizer, 'update_grid'):
                self.visualizer.update_grid(
                    frames=frames, metrics=metrics, infos=infos,
                )
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
            self.visualizer.close()
            self._save_on_exit()
            return False

        # Sync pause state from dashboard to trainer
        if self.visualizer.is_paused != self._training_paused:
            self.training_paused = self.visualizer.is_paused

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
            'game_id': self.game_id or 'mario',
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
        Handle Ctrl+C / SIGBREAK gracefully.

        Sets _dashboard_closed so the training loop exits on its next
        iteration check, saves the model immediately for safety, then
        calls sys.exit(0) to unwind to main.py's finally block where
        the stream manager and dashboard are cleaned up.
        """
        print('\n\nShutdown signal received! Saving model...')
        self._dashboard_closed = True
        self._save_on_exit()
        # Close the dashboard window immediately so it doesn't appear
        # frozen while the finally block runs stream cleanup.
        if self.visualizer:
            try:
                self.visualizer.close()
            except Exception:
                pass
        # Restore original handler and exit — the finally block in
        # main.py will call stream_manager.stop() and env.close().
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

                # Save timestamped run log for cross-game comparison
                game_id = getattr(self, 'game_id', 'unknown')
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                run_dir = os.path.join(self.log_dir, 'runs')
                os.makedirs(run_dir, exist_ok=True)
                run_path = os.path.join(
                    run_dir, f'{game_id}_{algo_name}_{timestamp}.json'
                )
                self.visualizer.metrics.export_json(run_path)

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
