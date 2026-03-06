"""
Overlay Manager for stream-ready frame compositing.

Composites the dashboard display with streaming overlays:
- LIVE indicator badge (pulsing red dot)
- Algorithm and stage info
- Training stats ticker
"""
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class OverlayManager:
    """
    Composites dashboard frames with stream overlays.

    Args:
        resolution: Output resolution (width, height). Default 1280x720.
    """

    def __init__(self, resolution: Tuple[int, int] = (1280, 720)):
        self.width, self.height = resolution
        self._live_pulse_start = time.time()

    def compose(
        self,
        frame: np.ndarray,
        is_live: bool = False,
        algorithm: Optional[str] = None,
        stage: Optional[str] = None,
        episode: Optional[int] = None,
        reward: Optional[float] = None,
        elapsed_time: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compose the final stream frame with overlays.

        Args:
            frame: Input frame (H, W, 3) uint8 RGB from dashboard.
            is_live: Whether currently streaming (shows LIVE badge).
            algorithm: Current algorithm name for overlay.
            stage: Current world-stage string (e.g., '1-1').
            episode: Current episode number.
            reward: Current/best reward.
            elapsed_time: Training elapsed time in seconds.

        Returns:
            Composited frame at target resolution (H, W, 3) uint8.
        """
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        else:
            frame = frame.copy()

        # Convert RGB to BGR for OpenCV drawing, then back
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if is_live:
            self._draw_live_badge(frame_bgr)

        if algorithm:
            self._draw_algo_info(frame_bgr, algorithm, stage)

        if episode is not None or reward is not None:
            self._draw_stats_ticker(frame_bgr, episode, reward, elapsed_time)

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _draw_live_badge(self, frame_bgr: np.ndarray) -> None:
        elapsed = time.time() - self._live_pulse_start
        pulse = abs((elapsed % 2.0) - 1.0)
        red_intensity = int(180 + 75 * pulse)
        cv2.circle(frame_bgr, (20, 20), 8, (0, 0, red_intensity), -1)
        cv2.putText(frame_bgr, 'LIVE', (35, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _draw_algo_info(self, frame_bgr: np.ndarray, algorithm: str, stage: Optional[str]) -> None:
        text = algorithm.upper()
        if stage:
            text += f'  |  World {stage}'
        x = self.width - 300
        y = 27
        cv2.putText(frame_bgr, text, (x + 1, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame_bgr, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_stats_ticker(self, frame_bgr: np.ndarray, episode: Optional[int],
                           reward: Optional[float], elapsed_time: Optional[float]) -> None:
        parts = []
        if episode is not None:
            parts.append(f'EP: {episode}')
        if reward is not None:
            parts.append(f'Reward: {reward:.0f}')
        if elapsed_time is not None:
            mins = int(elapsed_time // 60)
            secs = int(elapsed_time % 60)
            parts.append(f'Time: {mins}:{secs:02d}')
        text = '   |   '.join(parts)
        y = self.height - 15

        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (0, self.height - 35), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)
        cv2.putText(frame_bgr, text, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
