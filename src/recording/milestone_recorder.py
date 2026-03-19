"""
Records training milestone markers for replay navigation.

The MilestoneRecorder subscribes to EventBus events (new_best_reward,
achievement_earned, episode_complete) and writes timestamp/frame markers
into a JSON sidecar file alongside the video recording.  Replay tools
can read the sidecar to let users jump to interesting moments.
"""

import json
import os
import time


class MilestoneRecorder:
    """Records training milestone markers for replay navigation."""

    def __init__(self, event_bus=None, output_dir='recordings'):
        self._bus = event_bus
        self._output_dir = output_dir
        self._markers = []           # List of marker dicts
        self._start_time = None      # Set when recording starts
        self._frame_count = 0        # Updated externally
        self._recording_active = False

        if event_bus:
            event_bus.subscribe('new_best_reward', self._on_new_best)
            event_bus.subscribe('achievement_earned', self._on_achievement)
            event_bus.subscribe('episode_complete', self._on_episode_milestone)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_recording(self):
        """Called when video recording begins."""
        self._start_time = time.time()
        self._markers = []
        self._frame_count = 0
        self._recording_active = True

    def stop_recording(self, recording_filename=None):
        """Called when recording ends. Saves markers to JSON sidecar file."""
        self._recording_active = False
        if recording_filename and self._markers:
            self._save_markers(recording_filename)

    def update_frame(self, frame_number):
        """Called each frame to track current position."""
        self._frame_count = frame_number

    def add_marker(self, marker_type, value=None, description=''):
        """Manually add a marker at the current position."""
        if not self._recording_active or self._start_time is None:
            return
        marker = {
            'frame': self._frame_count,
            'time': time.time() - self._start_time,
            'type': marker_type,
            'description': description,
        }
        if value is not None:
            marker['value'] = value
        self._markers.append(marker)

    def get_markers(self):
        """Return a copy of the current markers list."""
        return list(self._markers)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_new_best(self, event):
        self.add_marker(
            'new_best_reward',
            value=event.get('reward'),
            description=f"New best reward: {event.get('reward', '?')}",
        )

    def _on_achievement(self, event):
        self.add_marker(
            'achievement',
            value=event.get('achievement_name', ''),
            description=f"Achievement: {event.get('achievement_name', '?')}",
        )

    def _on_episode_milestone(self, event):
        episode = event.get('episode', 0)
        # Only mark round-number episodes (every 10 up to 100, every 500, every 1000)
        if episode > 0 and (
            episode % 1000 == 0
            or episode % 500 == 0
            or (episode <= 100 and episode % 10 == 0)
        ):
            self.add_marker(
                'episode_milestone',
                value=episode,
                description=f"Episode {episode}",
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_markers(self, recording_filename):
        """Save markers as JSON sidecar: recording.mp4 -> recording_markers.json"""
        base = os.path.splitext(recording_filename)[0]
        markers_path = f"{base}_markers.json"
        os.makedirs(os.path.dirname(markers_path) or '.', exist_ok=True)
        data = {
            'recording': os.path.basename(recording_filename),
            'total_markers': len(self._markers),
            'duration': time.time() - self._start_time if self._start_time else 0,
            'markers': self._markers,
        }
        with open(markers_path, 'w') as f:
            json.dump(data, f, indent=2)
