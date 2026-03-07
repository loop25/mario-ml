"""
Live streaming manager via ffmpeg RTMP.

Sends raw video frames and audio to Twitch and/or YouTube simultaneously
using ffmpeg's tee muxer. Requires ffmpeg installed and on PATH.
"""
import os
import shutil
import subprocess
import threading
import time
from typing import Optional

import numpy as np


class StreamManager:
    """
    Manages an ffmpeg subprocess for live RTMP streaming.

    Args:
        twitch_key: Twitch stream key (None to skip Twitch).
        youtube_key: YouTube stream key (None to skip YouTube).
        width: Output video width. Default 1280.
        height: Output video height. Default 720.
        fps: Output frame rate. Default 30.
        video_bitrate: Video bitrate string. Default '4500k'.
        audio_file: Path to audio file for stream (None for silent).
    """

    TWITCH_RTMP = 'rtmp://live.twitch.tv/app'
    YOUTUBE_RTMP = 'rtmp://a.rtmp.youtube.com/live2'

    def __init__(
        self,
        twitch_key: Optional[str] = None,
        youtube_key: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        video_bitrate: str = '4500k',
        audio_file: Optional[str] = None,
    ):
        self.twitch_key = twitch_key
        self.youtube_key = youtube_key
        self.width = width
        self.height = height
        self.fps = fps
        self.video_bitrate = video_bitrate
        self.audio_file = audio_file

        self._process: Optional[subprocess.Popen] = None
        self._health_thread: Optional[threading.Thread] = None
        self.is_streaming = False
        self.error_message: Optional[str] = None
        self._frame_count = 0
        self._start_time = 0.0

    def _build_ffmpeg_command(self) -> list:
        cmd = ['ffmpeg', '-y']

        # Video input: raw RGB frames from pipe
        cmd += [
            '-f', 'rawvideo',
            '-pixel_format', 'rgb24',
            '-video_size', f'{self.width}x{self.height}',
            '-framerate', str(self.fps),
            '-i', 'pipe:0',
        ]

        # Audio input
        if self.audio_file and os.path.isfile(self.audio_file):
            cmd += ['-stream_loop', '-1', '-i', self.audio_file]
        else:
            cmd += ['-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100']

        # Video encoding
        cmd += [
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-tune', 'zerolatency',
            '-b:v', self.video_bitrate,
            '-maxrate', self.video_bitrate,
            '-bufsize', str(int(self.video_bitrate.replace('k', '')) * 2) + 'k',
            '-pix_fmt', 'yuv420p',
            '-g', str(self.fps * 2),
        ]

        # Audio encoding
        cmd += ['-c:a', 'aac', '-b:a', '128k', '-ar', '44100']
        cmd += ['-shortest']

        # Build output destinations
        destinations = []
        if self.twitch_key:
            destinations.append(f'[f=flv]{self.TWITCH_RTMP}/{self.twitch_key}')
        if self.youtube_key:
            destinations.append(f'[f=flv]{self.YOUTUBE_RTMP}/{self.youtube_key}')

        if len(destinations) == 1:
            cmd += ['-f', 'flv', destinations[0].split(']')[1]]
        else:
            cmd += ['-f', 'tee', '|'.join(destinations)]

        return cmd

    def start(self) -> bool:
        if self.is_streaming:
            return True

        if not self.twitch_key and not self.youtube_key:
            self.error_message = 'No stream keys configured'
            return False

        if not shutil.which('ffmpeg'):
            self.error_message = (
                'ffmpeg not found. Install from https://ffmpeg.org/download.html '
                'and ensure it is on your system PATH.'
            )
            print(f'[Stream] ERROR: {self.error_message}')
            return False

        cmd = self._build_ffmpeg_command()
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
        except OSError as e:
            self.error_message = f'Failed to start ffmpeg: {e}'
            print(f'[Stream] ERROR: {self.error_message}')
            return False

        self.is_streaming = True
        self._frame_count = 0
        self._start_time = time.time()
        self.error_message = None

        self._health_thread = threading.Thread(target=self._monitor_health, daemon=True)
        self._health_thread.start()

        destinations = []
        if self.twitch_key:
            destinations.append('Twitch')
        if self.youtube_key:
            destinations.append('YouTube')
        print(f'[Stream] LIVE on {" + ".join(destinations)} ({self.width}x{self.height} @ {self.fps}fps)')
        return True

    def stop(self) -> None:
        if not self.is_streaming:
            return
        self.is_streaming = False
        if self._process:
            try:
                self._process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        elapsed = time.time() - self._start_time
        print(f'[Stream] Stopped after {elapsed:.0f}s, {self._frame_count} frames sent')

    def send_frame(self, frame: np.ndarray) -> None:
        if not self.is_streaming or self._process is None:
            return
        try:
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                import cv2
                frame = cv2.resize(frame, (self.width, self.height))
            self._process.stdin.write(frame.tobytes())
            self._frame_count += 1
        except (BrokenPipeError, OSError):
            self.is_streaming = False
            self.error_message = 'Stream connection lost'
            print(f'[Stream] ERROR: {self.error_message}')

    def _monitor_health(self) -> None:
        if not self._process:
            return
        try:
            for line in iter(self._process.stderr.readline, b''):
                text = line.decode('utf-8', errors='replace').strip()
                if 'error' in text.lower() or 'failed' in text.lower():
                    self.error_message = text
                    print(f'[Stream] WARNING: {text}')
                if not self.is_streaming:
                    break
        except (ValueError, OSError):
            pass

    @property
    def stream_uptime(self) -> float:
        if self.is_streaming:
            return time.time() - self._start_time
        return 0.0

    @property
    def status_text(self) -> str:
        if self.is_streaming:
            uptime_min = int(self.stream_uptime / 60)
            return f'LIVE {uptime_min}m | {self._frame_count} frames'
        if self.error_message:
            return f'ERROR: {self.error_message}'
        return 'OFF'
