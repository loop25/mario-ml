"""
Live streaming manager via ffmpeg RTMP.

Sends raw video frames and audio to Twitch and/or YouTube simultaneously
using ffmpeg's tee muxer. Requires ffmpeg installed and on PATH.
"""
import atexit
import os
import re
import shutil
import subprocess
import tempfile
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
        audio_files: Optional[list] = None,
    ):
        self.twitch_key = twitch_key
        self.youtube_key = youtube_key
        self.width = width
        self.height = height
        self.fps = fps
        self.video_bitrate = video_bitrate
        # Support single file or playlist
        if audio_files:
            self.audio_files = [f for f in audio_files if os.path.isfile(f)]
        elif audio_file and os.path.isfile(audio_file):
            self.audio_files = [audio_file]
        else:
            self.audio_files = []
        # For backward compat
        self.audio_file = self.audio_files[0] if self.audio_files else None
        self._concat_file: Optional[str] = None

        self._process: Optional[subprocess.Popen] = None
        self._health_thread: Optional[threading.Thread] = None
        self.is_streaming = False
        self.error_message: Optional[str] = None
        self._frame_count = 0
        self._start_time = 0.0
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5  # seconds
        self._auto_reconnect = True
        self._last_frame: Optional[np.ndarray] = None

    def _build_ffmpeg_command(self) -> list:
        cmd = ['ffmpeg', '-y']

        # Video input: raw RGB frames from pipe
        # -use_wallclock_as_timestamps: timestamps based on wall clock,
        #   not frame count — critical when frames arrive slower than fps
        # -thread_queue_size: large buffer so slow frame delivery doesn't
        #   cause the audio queue to overflow and crash
        cmd += [
            '-thread_queue_size', '512',
            '-use_wallclock_as_timestamps', '1',
            '-f', 'rawvideo',
            '-pixel_format', 'rgb24',
            '-video_size', f'{self.width}x{self.height}',
            '-framerate', str(self.fps),
            '-i', 'pipe:0',
        ]

        # Audio input — use concat demuxer for playlists, single loop
        # for one file, or silent generator if no audio files at all.
        if len(self.audio_files) > 1:
            # Build a temporary concat playlist file for ffmpeg
            self._concat_file = self._create_concat_file()
            cmd += ['-thread_queue_size', '512',
                    '-f', 'concat', '-safe', '0',
                    '-stream_loop', '-1', '-i', self._concat_file]
        elif self.audio_file and os.path.isfile(self.audio_file):
            cmd += ['-thread_queue_size', '512',
                    '-stream_loop', '-1', '-i', self.audio_file]
        else:
            cmd += ['-f', 'lavfi', '-i',
                    'anullsrc=channel_layout=stereo:sample_rate=44100']

        # Video encoding
        cmd += [
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-tune', 'zerolatency',
            '-b:v', self.video_bitrate,
            '-maxrate', self.video_bitrate,
            '-bufsize', self._calc_bufsize(self.video_bitrate),
            '-pix_fmt', 'yuv420p',
            '-g', str(self.fps * 2),
            '-vsync', 'cfr',
        ]

        # Audio encoding — do NOT use -shortest (it kills the stream
        # when audio buffer fills up faster than video frames arrive)
        cmd += ['-c:a', 'aac', '-b:a', '128k', '-ar', '44100']

        # Build output destinations
        # When streaming to a single destination, use plain flv output.
        # When streaming to multiple, use ffmpeg's tee muxer.
        # On Windows the tee muxer's pipe-separated URL string must be
        # carefully constructed — each destination is [f=flv]<url> and
        # they're joined with '|'.  If that still fails (common on
        # Windows due to shell escaping), fall back to running separate
        # ffmpeg output args instead of tee.
        twitch_url = f'{self.TWITCH_RTMP}/{self.twitch_key}' if self.twitch_key else None
        youtube_url = f'{self.YOUTUBE_RTMP}/{self.youtube_key}' if self.youtube_key else None

        urls = [u for u in [twitch_url, youtube_url] if u]

        if len(urls) == 1:
            cmd += ['-f', 'flv', urls[0]]
        elif len(urls) == 2:
            # Use two separate -f flv outputs instead of tee muxer —
            # tee muxer has known issues with pipe chars on Windows.
            cmd += ['-f', 'flv', urls[0], '-f', 'flv', urls[1]]

        return cmd

    @staticmethod
    def _calc_bufsize(bitrate: str) -> str:
        """
        Calculate bufsize as 2x the video bitrate.

        Handles formats like '4500k', '4500K', '4.5M', '4500' (plain numeric).
        Returns an ffmpeg-compatible bitrate string (e.g. '9000k').

        Args:
            bitrate: Video bitrate string (e.g. '4500k').

        Returns:
            Bufsize string at 2x the input bitrate.
        """
        match = re.match(r'^([\d.]+)\s*([kKmM]?)$', bitrate.strip())
        if not match:
            # Fallback: pass bitrate through unchanged (ffmpeg will validate)
            return bitrate
        value = float(match.group(1))
        suffix = match.group(2).lower()
        if suffix == 'm':
            # Convert megabits to kilobits for consistency
            value_k = int(value * 1000)
        elif suffix == 'k' or suffix == '':
            value_k = int(value)
        else:
            value_k = int(value)
        return f'{value_k * 2}k'

    def _create_concat_file(self) -> str:
        """Create a temporary ffmpeg concat playlist file.

        ffmpeg's concat demuxer reads a text file listing audio files:
            file '/path/to/track1.mp3'
            file '/path/to/track2.ogg'

        Combined with -stream_loop -1, the entire playlist repeats
        infinitely — giving the stream continuous shuffled music.

        Returns:
            Path to the temporary concat playlist file.
        """
        fd, path = tempfile.mkstemp(suffix='.txt', prefix='stream_audio_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for audio_path in self.audio_files:
                # ffmpeg concat requires forward slashes and single-quote
                # escaping — on Windows, backslashes must be replaced.
                safe_path = audio_path.replace('\\', '/')
                f.write(f"file '{safe_path}'\n")
        return path

    def _cleanup_concat_file(self) -> None:
        """Remove the temporary concat playlist file if it exists."""
        if self._concat_file and os.path.isfile(self._concat_file):
            try:
                os.remove(self._concat_file)
            except OSError:
                pass
            self._concat_file = None

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

        # Register atexit handler so ffmpeg is killed even if the Python
        # process is terminated abruptly (e.g. launcher force-kill).
        atexit.register(self._atexit_cleanup)

        destinations = []
        if self.twitch_key:
            destinations.append('Twitch')
        if self.youtube_key:
            destinations.append('YouTube')
        print(f'[Stream] LIVE on {" + ".join(destinations)} ({self.width}x{self.height} @ {self.fps}fps)')
        return True

    def stop(self) -> None:
        if not self.is_streaming and self._process is None:
            return
        self.is_streaming = False
        self._kill_ffmpeg()
        self._cleanup_concat_file()

        elapsed = time.time() - self._start_time
        print(f'[Stream] Stopped after {elapsed:.0f}s, {self._frame_count} frames sent')

    def _kill_ffmpeg(self) -> None:
        """Terminate the ffmpeg subprocess, with escalation."""
        proc = self._process
        if proc is None:
            return
        self._process = None

        # Close stdin to signal ffmpeg to flush and exit
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

        # Give ffmpeg a moment to exit gracefully
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # Escalate: terminate, then kill
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass

    def _atexit_cleanup(self) -> None:
        """Last-resort cleanup registered via atexit.

        If the Python process is terminating and ffmpeg is still running,
        kill it immediately so it doesn't become an orphan process that
        keeps streaming silence/frozen frames forever.
        """
        if self._process is not None:
            try:
                self._process.kill()
            except OSError:
                pass
            self._process = None
            self.is_streaming = False
        self._cleanup_concat_file()

    def send_frame(self, frame: np.ndarray) -> None:
        if not self.is_streaming or self._process is None:
            # Try auto-reconnect if we were streaming and lost connection
            if self._auto_reconnect and self._last_frame is not None:
                self._try_reconnect()
            return
        try:
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                import cv2
                frame = cv2.resize(frame, (self.width, self.height))
            self._last_frame = frame
            self._process.stdin.write(frame.tobytes())
            self._frame_count += 1
            self._reconnect_attempts = 0  # Reset on successful send
        except (BrokenPipeError, OSError):
            self.is_streaming = False
            self.error_message = 'Stream connection lost'
            print(f'[Stream] WARNING: Connection lost, will auto-reconnect...')
            self._try_reconnect()

    def _try_reconnect(self) -> None:
        """Attempt to reconnect the stream after a dropped connection."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            print(f'[Stream] ERROR: Failed to reconnect after '
                  f'{self._max_reconnect_attempts} attempts. Stream stopped.')
            self._auto_reconnect = False
            return

        self._reconnect_attempts += 1
        print(f'[Stream] Reconnect attempt {self._reconnect_attempts}/'
              f'{self._max_reconnect_attempts} in {self._reconnect_delay}s...')

        # Kill old process
        self._kill_ffmpeg()

        time.sleep(self._reconnect_delay)

        # Restart ffmpeg
        cmd = self._build_ffmpeg_command()
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
            self.is_streaming = True
            self.error_message = None

            # Restart health monitor
            self._health_thread = threading.Thread(
                target=self._monitor_health, daemon=True)
            self._health_thread.start()

            print(f'[Stream] Reconnected successfully!')
        except OSError as e:
            print(f'[Stream] Reconnect failed: {e}')

    def _monitor_health(self) -> None:
        if not self._process:
            return
        try:
            for line in iter(self._process.stderr.readline, b''):
                text = line.decode('utf-8', errors='replace').strip()
                if not text:
                    continue
                text_lower = text.lower()
                # Ignore normal ffmpeg progress lines
                if text.startswith('frame=') or text.startswith('size='):
                    continue
                if 'error' in text_lower or 'failed' in text_lower:
                    # Distinguish fatal errors from transient warnings
                    if 'conversion failed' in text_lower:
                        self.error_message = 'Stream connection lost'
                        self.is_streaming = False
                        print(f'[Stream] ERROR: {self.error_message}')
                    else:
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
