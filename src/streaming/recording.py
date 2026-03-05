"""
Video Recording Module for Super Mario Bros ML.

Records the training dashboard as an MP4 video file using OpenCV's
VideoWriter. This captures everything visible in the pygame window:
game display, graphs, and status bar.

The recording runs in a separate thread to minimize impact on training
performance. Frames are captured from the pygame surface.

Future enhancement: Add audio recording from the NES emulator.

Usage:
    recorder = Recorder(output_dir='recordings')
    recorder.start(filename='neat_training.mp4')

    # During training loop:
    recorder.capture_frame(pygame_surface)

    # When done:
    recorder.stop()
"""

import os
import time
import threading
import numpy as np
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


class Recorder:
    """
    Records pygame dashboard frames to MP4 video.

    Captures frames from the pygame display surface and writes them
    to a video file using OpenCV's VideoWriter with H.264 encoding.

    Args:
        output_dir: Directory to save video files. Default 'recordings'.
        fps: Output video frame rate. Default 30.
        resolution: Output video resolution (width, height).
                    Default (1400, 800) matching the dashboard.

    Attributes:
        is_recording: Whether recording is currently active.
        frame_count: Number of frames recorded so far.
        output_path: Path to the current output video file.

    Example:
        recorder = Recorder(output_dir='recordings')
        recorder.start()
        for frame in training_loop:
            recorder.capture_frame(dashboard.screen)
        recorder.stop()
        print(f'Video saved to: {recorder.output_path}')
    """

    def __init__(
        self,
        output_dir: str = 'recordings',
        fps: int = 30,
        resolution: tuple = (1400, 800),
    ):
        self.output_dir = output_dir
        self.fps = fps
        self.resolution = resolution
        self.is_recording = False
        self.frame_count = 0
        self.output_path: Optional[str] = None

        # Video writer (initialized on start)
        self._writer = None
        self._lock = threading.Lock()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        if not HAS_CV2:
            print('Warning: opencv-python not installed. Recording disabled.')

    def start(self, filename: Optional[str] = None) -> str:
        """
        Start recording.

        Args:
            filename: Output filename. If None, generates a timestamped name.

        Returns:
            str: Path to the output video file.
        """
        if not HAS_CV2:
            print('Recording not available (opencv-python not installed)')
            return ''

        if self.is_recording:
            print('Already recording!')
            return self.output_path or ''

        # Generate filename if not provided
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'training_{timestamp}.mp4'

        self.output_path = os.path.join(self.output_dir, filename)

        # Create VideoWriter with mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            self.resolution,
        )

        if not self._writer.isOpened():
            print(f'Error: Could not create video file: {self.output_path}')
            return ''

        self.is_recording = True
        self.frame_count = 0
        print(f'Recording started: {self.output_path}')
        return self.output_path

    def capture_frame(self, surface) -> None:
        """
        Capture a single frame from a pygame surface.

        Converts the pygame surface to a numpy array and writes
        it to the video file. Thread-safe.

        Args:
            surface: pygame.Surface to capture.
        """
        if not self.is_recording or self._writer is None:
            return

        if not HAS_PYGAME:
            return

        with self._lock:
            try:
                # Convert pygame surface to numpy array
                frame = pygame.surfarray.array3d(surface)
                # pygame gives (W, H, 3) in RGB, OpenCV needs (H, W, 3) in BGR
                frame = np.transpose(frame, (1, 0, 2))  # (H, W, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Resize if needed
                if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
                    frame = cv2.resize(frame, self.resolution)

                self._writer.write(frame)
                self.frame_count += 1
            except Exception as e:
                print(f'Recording frame error: {e}')

    def stop(self) -> None:
        """
        Stop recording and finalize the video file.

        Releases the VideoWriter and prints recording stats.
        """
        if not self.is_recording:
            return

        self.is_recording = False

        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None

        if self.output_path and os.path.exists(self.output_path):
            file_size = os.path.getsize(self.output_path)
            size_mb = file_size / (1024 * 1024)
            duration = self.frame_count / self.fps if self.fps > 0 else 0
            print(f'Recording saved: {self.output_path}')
            print(f'  Frames: {self.frame_count}')
            print(f'  Duration: {duration:.1f}s')
            print(f'  Size: {size_mb:.1f} MB')
