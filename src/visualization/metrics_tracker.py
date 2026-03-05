"""
Metrics Tracker for Super Mario Bros ML.

Collects, stores, and provides access to training metrics across all
algorithms. Metrics are stored in memory for real-time graph updates
and can be exported to JSON for external tools (like stream overlays).

The tracker maintains separate metric histories and computes rolling
averages for smoother graph visualization.

Usage:
    tracker = MetricsTracker()
    tracker.record(episode=1, reward=250.0, distance=500, loss=0.05)
    avg_reward = tracker.get_rolling_average('reward', window=10)
    tracker.export_json('logs/metrics.json')
"""

import json
import os
import time
from collections import defaultdict
from typing import Dict, Any, List, Optional

import numpy as np


class MetricsTracker:
    """
    Tracks and stores training metrics with rolling average support.

    Stores all metric values in lists indexed by episode/generation,
    allowing for efficient graph updates and statistical analysis.

    Attributes:
        metrics: Dictionary mapping metric names to lists of values.
        best: Dictionary tracking the best value seen for each metric.
        start_time: Timestamp when tracking began.

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.record(episode=1, reward=100, distance=300)
        >>> tracker.record(episode=2, reward=200, distance=500)
        >>> print(tracker.get_latest('reward'))  # 200
        >>> print(tracker.get_best('reward'))     # 200
    """

    def __init__(self):
        """Initialize empty metrics tracker."""
        # Main storage: metric_name -> list of values
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        # Best values seen for each metric
        self.best: Dict[str, float] = {}
        # Track which metrics should be minimized (like loss)
        self._minimize_metrics = {'loss', 'policy_loss', 'value_loss', 'epsilon'}
        # Timestamp for elapsed time tracking
        self.start_time = time.time()
        # Episode/generation counter
        self.episode_count = 0

    def record(self, **kwargs) -> None:
        """
        Record one or more metric values.

        Each keyword argument becomes a metric entry. Special keys:
        - 'episode' or 'generation': Updates the internal counter
        - All other keys are stored as metric values

        Args:
            **kwargs: Metric name-value pairs to record.

        Example:
            tracker.record(episode=5, reward=350.0, distance=1200,
                          loss=0.023, epsilon=0.85)
        """
        # Update episode counter if provided
        if 'episode' in kwargs:
            self.episode_count = kwargs.pop('episode')
        if 'generation' in kwargs:
            self.episode_count = kwargs.pop('generation')

        # Record each metric value
        for name, value in kwargs.items():
            if value is not None:
                # Store list/array values (e.g. action_distribution) as-is
                # so the graph panel can display them directly.
                if isinstance(value, (list, tuple)):
                    self.metrics[name].append(list(value))
                    continue
                if isinstance(value, np.ndarray):
                    self.metrics[name].append(value.tolist())
                    continue
                if isinstance(value, dict):
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                self.metrics[name].append(numeric)
                # Update best value tracking
                self._update_best(name, numeric)

    def _update_best(self, name: str, value: float) -> None:
        """
        Update the best-seen value for a metric.

        For loss-type metrics (in _minimize_metrics), 'best' means lowest.
        For reward-type metrics, 'best' means highest.

        Args:
            name: Metric name.
            value: New value to compare against current best.
        """
        if name not in self.best:
            self.best[name] = value
        elif name in self._minimize_metrics:
            self.best[name] = min(self.best[name], value)
        else:
            self.best[name] = max(self.best[name], value)

    def get_values(self, name: str) -> List[float]:
        """
        Get all recorded values for a metric.

        Args:
            name: Metric name.

        Returns:
            List of all values, or empty list if metric not found.
        """
        return self.metrics.get(name, [])

    def get_latest(self, name: str, default: float = 0.0) -> float:
        """
        Get the most recently recorded value for a metric.

        Args:
            name: Metric name.
            default: Value to return if no data exists.

        Returns:
            Most recent value, or default if no data.
        """
        values = self.metrics.get(name, [])
        return values[-1] if values else default

    def get_best(self, name: str, default: float = 0.0) -> float:
        """
        Get the best value ever recorded for a metric.

        Args:
            name: Metric name.
            default: Value to return if no data exists.

        Returns:
            Best value seen, or default if no data.
        """
        return self.best.get(name, default)

    def get_rolling_average(
        self, name: str, window: int = 10
    ) -> List[float]:
        """
        Compute a rolling average over the metric's history.

        Used for smoothing noisy graphs in the dashboard. The rolling
        average at each point is the mean of the last `window` values.

        Args:
            name: Metric name.
            window: Number of values to average over.

        Returns:
            List of rolling average values (same length as original).
            Early values use a smaller window when not enough history exists.
        """
        values = self.metrics.get(name, [])
        if not values:
            return []

        averages = []
        for i in range(len(values)):
            # Use available values when window isn't full yet
            start = max(0, i - window + 1)
            window_values = values[start:i + 1]
            averages.append(sum(window_values) / len(window_values))
        return averages

    def get_elapsed_time(self) -> float:
        """
        Get elapsed time since tracking started.

        Returns:
            Elapsed time in seconds.
        """
        return time.time() - self.start_time

    def get_elapsed_time_str(self) -> str:
        """
        Get elapsed time as a formatted string.

        Returns:
            Time string in "HH:MM:SS" format.
        """
        elapsed = int(self.get_elapsed_time())
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tracked metrics.

        Returns:
            Dictionary with latest values, best values, and counts.
        """
        summary = {
            'episode': self.episode_count,
            'elapsed_time': self.get_elapsed_time_str(),
            'metrics': {},
        }
        for name in self.metrics:
            summary['metrics'][name] = {
                'latest': self.get_latest(name),
                'best': self.get_best(name),
                'count': len(self.metrics[name]),
            }
        return summary

    def export_json(self, filepath: str) -> None:
        """
        Export all metrics to a JSON file.

        Useful for external tools like stream overlays (StreamElements)
        or post-training analysis.

        Args:
            filepath: Path to save the JSON file.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            'episode': self.episode_count,
            'elapsed_time': self.get_elapsed_time_str(),
            'elapsed_seconds': self.get_elapsed_time(),
            'best': self.best,
            'latest': {name: self.get_latest(name) for name in self.metrics},
            'history': {name: values for name, values in self.metrics.items()},
        }

        class NumpyEncoder(json.JSONEncoder):
            """Handle numpy types that aren't JSON serializable."""
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    def reset(self) -> None:
        """Clear all metrics and reset counters."""
        self.metrics = defaultdict(list)
        self.best = {}
        self.episode_count = 0
        self.start_time = time.time()
