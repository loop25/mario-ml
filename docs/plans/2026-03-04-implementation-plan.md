# Mario-ML Streaming, Music & Optimization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add simultaneous Twitch+YouTube streaming, background music, true pause/resume, curriculum-based whole-game training, and performance optimizations to the mario-ml project.

**Architecture:** ffmpeg RTMP pipeline for multi-platform streaming with audio muxing, pygame.mixer for local music playback, training-thread pause/resume in BaseTrainer, and a CurriculumManager for structured stage progression with revisitation. The launcher GUI gets new sections for stream keys, music config, and whole-game mode.

**Tech Stack:** Python 3.11, pygame (display + audio), ffmpeg (streaming + recording), PyYAML (config), existing NEAT/PPO/DQN trainers.

**Note on testing:** This project has no existing test infrastructure (no pytest, no test directory). Tests in this plan use pytest with mocks to avoid requiring the NES emulator / pygame display. Create `tests/` directory and a minimal `pytest.ini` first.

---

## Task 1: Set Up Test Infrastructure

**Files:**
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
```

**Step 2: Create tests directory and conftest**

```python
# tests/__init__.py
# (empty)
```

```python
# tests/conftest.py
"""Shared fixtures for mario-ml tests."""
import pytest


@pytest.fixture
def tmp_music_dir(tmp_path):
    """Create a temporary directory with a fake music file."""
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    # Create a minimal valid WAV file (44 bytes — header only, 0 samples)
    wav_header = (
        b'RIFF' + (36).to_bytes(4, 'little') +
        b'WAVE' +
        b'fmt ' + (16).to_bytes(4, 'little') +
        (1).to_bytes(2, 'little') +    # PCM
        (1).to_bytes(2, 'little') +    # mono
        (22050).to_bytes(4, 'little') + # sample rate
        (22050).to_bytes(4, 'little') + # byte rate
        (1).to_bytes(2, 'little') +    # block align
        (8).to_bytes(2, 'little') +    # bits per sample
        b'data' + (0).to_bytes(4, 'little')
    )
    (music_dir / "test_track.wav").write_bytes(wav_header)
    return music_dir


@pytest.fixture
def sample_config():
    """Minimal training config dict."""
    return {
        'save_freq': 50,
        'learning_rate': 1e-4,
    }
```

**Step 3: Install pytest**

Run: `pip install pytest`

**Step 4: Verify test setup**

Run: `python -m pytest --co`
Expected: "no tests ran" (collected 0 items)

**Step 5: Commit**

```bash
git add pytest.ini tests/
git commit -m "chore: add pytest test infrastructure"
```

---

## Task 2: True Pause/Resume — BaseTrainer

The current pause at `base_trainer.py:209-216` only blocks `update_visualization()` — which means the training loop in DQN/PPO keeps running while the display says "paused." We need to add a method trainers call in their own loops.

**Files:**
- Modify: `src/algorithms/base_trainer.py:74-104` (\_\_init\_\_), `:167-218` (update_visualization)
- Create: `tests/test_pause_resume.py`

**Step 1: Write failing test**

```python
# tests/test_pause_resume.py
"""Tests for true pause/resume in BaseTrainer."""
import time
import threading
import pytest
from unittest.mock import MagicMock, patch


class FakeTrainer:
    """Minimal concrete trainer for testing pause logic."""

    def __init__(self):
        # Replicate BaseTrainer pause state
        self._training_paused = False
        self._pause_lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially
        self.steps_taken = 0

    @property
    def training_paused(self):
        return self._training_paused

    @training_paused.setter
    def training_paused(self, value):
        self._training_paused = value
        if value:
            self._pause_event.clear()
        else:
            self._pause_event.set()

    def check_pause(self, timeout=5.0):
        """Block if training is paused. Returns True to continue, False to stop."""
        if not self._training_paused:
            return True
        return self._pause_event.wait(timeout=timeout)

    def fake_training_loop(self, max_steps=100):
        """Simulates a training loop that checks for pause."""
        for _ in range(max_steps):
            if not self.check_pause(timeout=0.5):
                break
            self.steps_taken += 1
            time.sleep(0.01)


def test_not_paused_by_default():
    t = FakeTrainer()
    assert not t.training_paused
    assert t.check_pause() is True


def test_pause_blocks_training():
    t = FakeTrainer()
    t.training_paused = True

    # check_pause should block and return False after timeout
    start = time.time()
    result = t.check_pause(timeout=0.2)
    elapsed = time.time() - start

    assert result is False
    assert elapsed >= 0.15  # Blocked for ~0.2s


def test_resume_unblocks_training():
    t = FakeTrainer()
    t.training_paused = True

    def resume_after_delay():
        time.sleep(0.1)
        t.training_paused = False

    thread = threading.Thread(target=resume_after_delay)
    thread.start()

    result = t.check_pause(timeout=2.0)
    assert result is True
    thread.join()


def test_pause_stops_step_accumulation():
    t = FakeTrainer()

    # Start training in background
    train_thread = threading.Thread(target=t.fake_training_loop, args=(1000,))
    train_thread.start()

    time.sleep(0.1)
    steps_before_pause = t.steps_taken

    # Pause
    t.training_paused = True
    time.sleep(0.2)
    steps_while_paused = t.steps_taken

    # Steps should have stopped accumulating (allow +1 for in-flight step)
    assert steps_while_paused - steps_before_pause <= 1

    # Resume
    t.training_paused = False
    time.sleep(0.1)

    assert t.steps_taken > steps_while_paused
    train_thread.join(timeout=5)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pause_resume.py -v`
Expected: All 4 tests PASS (they test the FakeTrainer — this validates the pattern we'll apply)

**Step 3: Implement in BaseTrainer**

In `src/algorithms/base_trainer.py`, add to `__init__` (after line 95):

```python
        # Pause/resume state
        self._training_paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused = event is set
```

Add import at top of file:

```python
import threading
```

Add property and method after `__init__` (before the abstract methods around line 105):

```python
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
        Returns True to continue training, False if dashboard was closed
        during pause.
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
                self._save_on_exit()
                return False
            if self.visualizer:
                self.visualizer.update()
        return True
```

Replace the pause handling in `update_visualization` (lines 208-216) with:

```python
        # Handle pause state — set the flag, actual blocking happens in check_pause()
        if self.visualizer.is_paused != self._training_paused:
            self.training_paused = self.visualizer.is_paused

        return True
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_pause_resume.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/algorithms/base_trainer.py tests/test_pause_resume.py
git commit -m "feat: add true pause/resume to BaseTrainer with threading.Event"
```

---

## Task 3: Wire Pause Into Each Trainer

**Files:**
- Modify: `src/algorithms/neat/neat_trainer.py` (~line 400 in `_play_episode`)
- Modify: `src/algorithms/dqn/dqn_trainer.py` (~line 370 in step loop)
- Modify: `src/algorithms/ppo/ppo_trainer.py` (~line 99 in DashboardCallback._on_step)

**Step 1: Add check_pause to NEAT trainer**

In `neat_trainer.py`, inside `_play_episode()`, add at the start of the while loop (around line 397, right after `while not done:`):

```python
            # Check for pause
            if not self.check_pause():
                return total_reward, max_distance, action_counts, last_frame, False
```

**Step 2: Add check_pause to DQN trainer**

In `dqn_trainer.py`, inside the step loop of `_train_single_env()` (around line 370, at the top of `for step in range(max_steps_per_episode):`):

```python
                # Check for pause
                if not self.check_pause():
                    return
```

**Step 3: Add check_pause to PPO trainer**

In `ppo_trainer.py`, inside `DashboardCallback._on_step()` (around line 100), the callback has `self.trainer` reference. Add:

```python
        # Check for pause (blocks this thread until resumed)
        if not self.trainer.check_pause():
            return False  # Abort training
```

**Step 4: Test manually**

Run: `python main.py --algorithm dqn --visualize --episodes 10`
Press SPACE during training — training steps should freeze.
Press SPACE again — training resumes.

**Step 5: Commit**

```bash
git add src/algorithms/neat/neat_trainer.py src/algorithms/dqn/dqn_trainer.py src/algorithms/ppo/ppo_trainer.py
git commit -m "feat: wire check_pause() into NEAT, DQN, and PPO training loops"
```

---

## Task 4: Music Manager

**Files:**
- Create: `src/audio/__init__.py`
- Create: `src/audio/music_manager.py`
- Create: `assets/music/README.txt`
- Create: `tests/test_music_manager.py`

**Step 1: Write failing test**

```python
# tests/test_music_manager.py
"""Tests for MusicManager."""
import os
import pytest
from unittest.mock import patch, MagicMock


def test_music_manager_init_no_files(tmp_path):
    """MusicManager initializes gracefully with empty directory."""
    with patch('pygame.mixer') as mock_mixer:
        mock_mixer.music = MagicMock()
        from src.audio.music_manager import MusicManager
        mm = MusicManager(str(tmp_path))
        assert mm.playlist == []
        assert mm.volume == 0.5


def test_music_manager_finds_tracks(tmp_music_dir):
    """MusicManager discovers music files in directory."""
    with patch('pygame.mixer') as mock_mixer:
        mock_mixer.music = MagicMock()
        from src.audio.music_manager import MusicManager
        mm = MusicManager(str(tmp_music_dir))
        assert len(mm.playlist) == 1
        assert 'test_track.wav' in mm.playlist[0]


def test_volume_clamp():
    """Volume stays in 0.0-1.0 range."""
    with patch('pygame.mixer') as mock_mixer:
        mock_mixer.music = MagicMock()
        from src.audio.music_manager import MusicManager
        mm = MusicManager(".")
        mm.set_volume(1.5)
        assert mm.volume == 1.0
        mm.set_volume(-0.5)
        assert mm.volume == 0.0


def test_mute_toggle():
    """Mute preserves previous volume."""
    with patch('pygame.mixer') as mock_mixer:
        mock_mixer.music = MagicMock()
        from src.audio.music_manager import MusicManager
        mm = MusicManager(".")
        mm.set_volume(0.7)
        mm.toggle_mute()
        assert mm.is_muted is True
        mm.toggle_mute()
        assert mm.is_muted is False
        assert mm.volume == 0.7
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_music_manager.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'src.audio')

**Step 3: Implement MusicManager**

```python
# src/audio/__init__.py
"""Audio subsystem for background music during training."""
from .music_manager import MusicManager

__all__ = ['MusicManager']
```

```python
# src/audio/music_manager.py
"""
Background music manager for training sessions.

Uses pygame.mixer.music for playlist-based playback.
Supports .ogg, .mp3, and .wav files dropped into a music directory.
"""
import os
import random
from typing import List, Optional

import pygame


SUPPORTED_EXTENSIONS = {'.ogg', '.mp3', '.wav', '.flac'}


class MusicManager:
    """
    Manages background music playback during training.

    Scans a directory for audio files, shuffles them into a playlist,
    and loops playback continuously. Supports pause, resume, volume
    control, and mute toggle.

    Args:
        music_dir: Path to directory containing music files.
        volume: Initial volume (0.0 to 1.0). Default 0.5.
    """

    def __init__(self, music_dir: str, volume: float = 0.5):
        self.music_dir = music_dir
        self._volume = max(0.0, min(1.0, volume))
        self._pre_mute_volume = self._volume
        self.is_muted = False
        self.is_playing = False
        self.current_track: Optional[str] = None
        self._track_index = 0

        # Discover music files
        self.playlist: List[str] = self._scan_directory()
        if self.playlist:
            random.shuffle(self.playlist)

        # Initialize mixer if not already done
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            pygame.mixer.music.set_volume(self._volume)
        except pygame.error:
            pass  # Mixer init may fail in headless/test environments

    def _scan_directory(self) -> List[str]:
        """Find all supported audio files in the music directory."""
        if not os.path.isdir(self.music_dir):
            return []
        tracks = []
        for f in sorted(os.listdir(self.music_dir)):
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                tracks.append(os.path.join(self.music_dir, f))
        return tracks

    def play(self) -> None:
        """Start playing the playlist from the current track."""
        if not self.playlist:
            return
        try:
            self.current_track = self.playlist[self._track_index]
            pygame.mixer.music.load(self.current_track)
            pygame.mixer.music.play()
            # Set up end event to advance playlist
            pygame.mixer.music.set_endevent(pygame.USEREVENT + 99)
            self.is_playing = True
        except pygame.error as e:
            print(f'Music playback error: {e}')

    def stop(self) -> None:
        """Stop music playback."""
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        self.is_playing = False

    def pause(self) -> None:
        """Pause music (matches training pause)."""
        if self.is_playing:
            try:
                pygame.mixer.music.pause()
            except pygame.error:
                pass

    def resume(self) -> None:
        """Resume music (matches training resume)."""
        if self.is_playing:
            try:
                pygame.mixer.music.unpause()
            except pygame.error:
                pass

    def next_track(self) -> None:
        """Advance to the next track in the playlist."""
        if not self.playlist:
            return
        self._track_index = (self._track_index + 1) % len(self.playlist)
        if self.is_playing:
            self.play()

    def tick(self) -> None:
        """
        Call each frame. Advances playlist when a track ends.

        Check for the USEREVENT+99 end event set in play().
        """
        # This is handled via pygame event loop in dashboard
        pass

    def handle_music_end_event(self) -> None:
        """Called when pygame fires the music-end event."""
        self.next_track()

    @property
    def volume(self) -> float:
        return self._volume

    def set_volume(self, vol: float) -> None:
        """Set volume (clamped to 0.0-1.0)."""
        self._volume = max(0.0, min(1.0, vol))
        if not self.is_muted:
            try:
                pygame.mixer.music.set_volume(self._volume)
            except pygame.error:
                pass

    def volume_up(self, step: float = 0.05) -> None:
        """Increase volume by step."""
        self.set_volume(self._volume + step)

    def volume_down(self, step: float = 0.05) -> None:
        """Decrease volume by step."""
        self.set_volume(self._volume - step)

    def toggle_mute(self) -> None:
        """Toggle mute on/off, preserving volume level."""
        if self.is_muted:
            self.is_muted = False
            self.set_volume(self._pre_mute_volume)
        else:
            self._pre_mute_volume = self._volume
            self.is_muted = True
            try:
                pygame.mixer.music.set_volume(0.0)
            except pygame.error:
                pass

    @property
    def current_track_name(self) -> str:
        """Human-readable name of the current track."""
        if self.current_track:
            return os.path.splitext(os.path.basename(self.current_track))[0]
        return "No music"

    @property
    def track_count(self) -> int:
        return len(self.playlist)
```

```text
# assets/music/README.txt
Drop your music files here!

Supported formats: .ogg, .mp3, .wav, .flac

The AI training dashboard will shuffle and loop through all tracks
in this folder as background music during training sessions.

Tips:
- .ogg files have the best compatibility with pygame
- Keep file sizes reasonable (< 10MB each recommended)
- Instrumental/lo-fi music works great for training streams
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_music_manager.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/audio/ assets/music/README.txt tests/test_music_manager.py
git commit -m "feat: add MusicManager with playlist, volume control, and mute"
```

---

## Task 5: Integrate Music Into Dashboard

**Files:**
- Modify: `src/visualization/dashboard.py:95-125` (\_\_init\_\_), `:368-442` (update), `:568-582` (keyboard events)
- Modify: `main.py:72-159` (argparse), `:284-352` (training setup)

**Step 1: Add music_manager param to Dashboard.__init__**

In `dashboard.py`, add to `__init__` parameters (around line 95):

```python
    def __init__(
        self,
        algorithm: str,
        num_envs: int = 1,
        graph_update_interval: int = 5,
        fps_cap: int = 60,
        recorder=None,
        music_manager=None,    # <-- ADD THIS
    ):
```

Store it:
```python
        self.music_manager = music_manager
```

**Step 2: Add keyboard controls for music**

In `dashboard.py`, in the `handle_events` method (around line 568), after the `K_s` screenshot handler:

```python
                elif event.key == pygame.K_m:
                    if self.music_manager:
                        self.music_manager.toggle_mute()
                elif event.key == pygame.K_UP:
                    if self.music_manager:
                        self.music_manager.volume_up()
                elif event.key == pygame.K_DOWN:
                    if self.music_manager:
                        self.music_manager.volume_down()
```

Also add music end event handling in the event loop:

```python
            elif event.type == pygame.USEREVENT + 99:
                if self.music_manager:
                    self.music_manager.handle_music_end_event()
```

**Step 3: Add music status to dashboard status bar**

In the `update` method, when drawing the status bar, add music info. Find where the status bar text is drawn and append:

```python
        # Music status (draw on status bar)
        if self.music_manager and self.music_manager.is_playing:
            vol_pct = int(self.music_manager.volume * 100)
            mute_str = " [MUTED]" if self.music_manager.is_muted else ""
            music_text = f"♪ {self.music_manager.current_track_name}  Vol: {vol_pct}%{mute_str}"
            # Render on status bar area
```

**Step 4: Add --music CLI arg to main.py**

In `main.py` argparse section (after line 157):

```python
    # Music
    parser.add_argument(
        '--music',
        type=str,
        default=None,
        help='Path to music directory for background playback (default: assets/music/)',
    )
```

In the training setup section (around line 270, before dashboard creation):

```python
    # Music manager
    music_manager = None
    music_dir = args.music or os.path.join(PROJECT_ROOT, 'assets', 'music')
    if os.path.isdir(music_dir):
        from src.audio.music_manager import MusicManager
        music_manager = MusicManager(music_dir)
        if music_manager.track_count > 0:
            print(f'♪ Loaded {music_manager.track_count} music tracks from {music_dir}')
        else:
            music_manager = None
```

Pass it to Dashboard:
```python
    dashboard = Dashboard(
        algorithm=args.algorithm,
        ...,
        music_manager=music_manager,
    )
```

Start music after dashboard is ready (after line 307):
```python
    if music_manager:
        music_manager.play()
```

**Step 5: Test manually**

Drop an .ogg file in `assets/music/`, then:
Run: `python main.py --algorithm dqn --visualize --episodes 5`
Expected: Music plays, M mutes, Up/Down changes volume.

**Step 6: Commit**

```bash
git add src/visualization/dashboard.py main.py
git commit -m "feat: integrate music manager into dashboard with keyboard controls"
```

---

## Task 6: Stream Manager (ffmpeg RTMP)

**Files:**
- Create: `src/streaming/stream_manager.py`
- Create: `tests/test_stream_manager.py`

**Step 1: Write failing test**

```python
# tests/test_stream_manager.py
"""Tests for StreamManager."""
import pytest
from unittest.mock import patch, MagicMock, call


def test_stream_manager_init():
    from src.streaming.stream_manager import StreamManager
    sm = StreamManager(
        twitch_key="test_twitch_key",
        youtube_key="test_youtube_key",
    )
    assert sm.twitch_key == "test_twitch_key"
    assert sm.youtube_key == "test_youtube_key"
    assert sm.is_streaming is False


def test_stream_manager_no_ffmpeg():
    """Gracefully handles missing ffmpeg."""
    from src.streaming.stream_manager import StreamManager
    sm = StreamManager(twitch_key="key")
    with patch('shutil.which', return_value=None):
        success = sm.start()
        assert success is False
        assert sm.is_streaming is False


def test_build_ffmpeg_command_twitch_only():
    from src.streaming.stream_manager import StreamManager
    sm = StreamManager(twitch_key="abc123")
    cmd = sm._build_ffmpeg_command()
    assert 'rtmp://live.twitch.tv/app/abc123' in ' '.join(cmd)
    assert 'youtube' not in ' '.join(cmd).lower()


def test_build_ffmpeg_command_both():
    from src.streaming.stream_manager import StreamManager
    sm = StreamManager(twitch_key="tw_key", youtube_key="yt_key")
    cmd = sm._build_ffmpeg_command()
    cmd_str = ' '.join(cmd)
    assert 'rtmp://live.twitch.tv/app/tw_key' in cmd_str
    assert 'rtmp://a.rtmp.youtube.com/live2/yt_key' in cmd_str


def test_send_frame_when_not_streaming():
    """send_frame is a no-op when not streaming."""
    import numpy as np
    from src.streaming.stream_manager import StreamManager
    sm = StreamManager(twitch_key="key")
    # Should not raise
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    sm.send_frame(frame)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stream_manager.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Implement StreamManager**

```python
# src/streaming/stream_manager.py
"""
Live streaming manager via ffmpeg RTMP.

Sends raw video frames and audio to Twitch and/or YouTube simultaneously
using ffmpeg's tee muxer. Requires ffmpeg to be installed and on PATH.

Usage:
    sm = StreamManager(twitch_key="your_key", youtube_key="your_key")
    sm.start()
    for frame in training_frames:
        sm.send_frame(frame)  # numpy array (H, W, 3) uint8
    sm.stop()
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

    Streams to one or both of Twitch and YouTube simultaneously.
    Video is piped as raw RGB frames; audio from a looping music file.

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
        """Build the ffmpeg command for streaming."""
        cmd = ['ffmpeg', '-y']

        # Video input: raw RGB frames from pipe
        cmd += [
            '-f', 'rawvideo',
            '-pixel_format', 'rgb24',
            '-video_size', f'{self.width}x{self.height}',
            '-framerate', str(self.fps),
            '-i', 'pipe:0',
        ]

        # Audio input: loop a music file, or generate silence
        if self.audio_file and os.path.isfile(self.audio_file):
            cmd += ['-stream_loop', '-1', '-i', self.audio_file]
        else:
            # Generate silent audio
            cmd += [
                '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100',
            ]

        # Video encoding
        cmd += [
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-tune', 'zerolatency',
            '-b:v', self.video_bitrate,
            '-maxrate', self.video_bitrate,
            '-bufsize', str(int(self.video_bitrate.replace('k', '')) * 2) + 'k',
            '-pix_fmt', 'yuv420p',
            '-g', str(self.fps * 2),  # Keyframe every 2 seconds
        ]

        # Audio encoding
        cmd += ['-c:a', 'aac', '-b:a', '128k', '-ar', '44100']

        # Shortest flag to stop when video ends
        cmd += ['-shortest']

        # Build output destinations
        destinations = []
        if self.twitch_key:
            destinations.append(
                f'[f=flv]{self.TWITCH_RTMP}/{self.twitch_key}'
            )
        if self.youtube_key:
            destinations.append(
                f'[f=flv]{self.YOUTUBE_RTMP}/{self.youtube_key}'
            )

        if len(destinations) == 1:
            cmd += ['-f', 'flv', destinations[0].split(']')[1]]
        else:
            cmd += ['-f', 'tee', '|'.join(destinations)]

        return cmd

    def start(self) -> bool:
        """
        Start the streaming subprocess.

        Returns:
            True if started successfully, False on error.
        """
        if self.is_streaming:
            return True

        if not self.twitch_key and not self.youtube_key:
            self.error_message = 'No stream keys configured'
            return False

        # Check ffmpeg is available
        if not shutil.which('ffmpeg'):
            self.error_message = (
                'ffmpeg not found. Install it from https://ffmpeg.org/download.html '
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

        # Start health monitor thread
        self._health_thread = threading.Thread(
            target=self._monitor_health, daemon=True
        )
        self._health_thread.start()

        destinations = []
        if self.twitch_key:
            destinations.append('Twitch')
        if self.youtube_key:
            destinations.append('YouTube')
        print(f'[Stream] LIVE on {" + ".join(destinations)} '
              f'({self.width}x{self.height} @ {self.fps}fps)')
        return True

    def stop(self) -> None:
        """Stop streaming and close the ffmpeg process."""
        if not self.is_streaming:
            return

        self.is_streaming = False
        if self._process:
            try:
                self._process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        elapsed = time.time() - self._start_time
        print(f'[Stream] Stopped after {elapsed:.0f}s, '
              f'{self._frame_count} frames sent')

    def send_frame(self, frame: np.ndarray) -> None:
        """
        Send a single video frame to the stream.

        Args:
            frame: numpy array of shape (height, width, 3), dtype uint8, RGB.
                   Will be resized if dimensions don't match stream settings.
        """
        if not self.is_streaming or self._process is None:
            return

        try:
            # Resize frame if needed
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
        """Background thread that reads ffmpeg stderr for errors."""
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
            pass  # Process closed

    @property
    def stream_uptime(self) -> float:
        """Seconds since stream started."""
        if self.is_streaming:
            return time.time() - self._start_time
        return 0.0

    @property
    def status_text(self) -> str:
        """Human-readable stream status for dashboard overlay."""
        if self.is_streaming:
            uptime_min = int(self.stream_uptime / 60)
            return f'LIVE {uptime_min}m | {self._frame_count} frames'
        if self.error_message:
            return f'ERROR: {self.error_message}'
        return 'OFF'
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_stream_manager.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/streaming/stream_manager.py tests/test_stream_manager.py
git commit -m "feat: add StreamManager for ffmpeg RTMP streaming to Twitch/YouTube"
```

---

## Task 7: Overlay Manager Rewrite

**Files:**
- Rewrite: `src/streaming/overlay_manager.py`
- Create: `tests/test_overlay_manager.py`

**Step 1: Write failing test**

```python
# tests/test_overlay_manager.py
"""Tests for OverlayManager."""
import numpy as np
import pytest


def test_compose_adds_live_badge():
    from src.streaming.overlay_manager import OverlayManager
    om = OverlayManager(resolution=(1280, 720))
    # Create a black frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = om.compose(frame, is_live=True, algorithm='dqn', stage='1-1')
    assert result.shape == (720, 1280, 3)
    # Top-left corner should not be all black (LIVE badge drawn there)
    top_left = result[5:25, 5:60]
    assert top_left.sum() > 0


def test_compose_without_live():
    from src.streaming.overlay_manager import OverlayManager
    om = OverlayManager(resolution=(1280, 720))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = om.compose(frame, is_live=False)
    assert result.shape == (720, 1280, 3)


def test_compose_resizes_input():
    from src.streaming.overlay_manager import OverlayManager
    om = OverlayManager(resolution=(1280, 720))
    # Input frame at different resolution
    frame = np.zeros((800, 1400, 3), dtype=np.uint8)
    result = om.compose(frame, is_live=False)
    assert result.shape == (720, 1280, 3)
```

**Step 2: Run tests, verify fail**

Run: `python -m pytest tests/test_overlay_manager.py -v`
Expected: FAIL

**Step 3: Rewrite overlay_manager.py**

```python
# src/streaming/overlay_manager.py
"""
Overlay Manager for stream-ready frame compositing.

Composites the dashboard display with streaming overlays:
- LIVE indicator badge
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
        # Resize to target resolution if needed
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        else:
            frame = frame.copy()  # Don't mutate input

        # Convert RGB to BGR for OpenCV drawing, then back
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if is_live:
            self._draw_live_badge(frame_bgr)

        if algorithm:
            self._draw_algo_info(frame_bgr, algorithm, stage)

        if episode is not None or reward is not None:
            self._draw_stats_ticker(frame_bgr, episode, reward, elapsed_time)

        # Convert back to RGB
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _draw_live_badge(self, frame_bgr: np.ndarray) -> None:
        """Draw a LIVE indicator in the top-left corner."""
        # Pulsing red dot
        elapsed = time.time() - self._live_pulse_start
        pulse = abs((elapsed % 2.0) - 1.0)  # 0->1->0 over 2 seconds
        red_intensity = int(180 + 75 * pulse)

        # Red circle
        cv2.circle(frame_bgr, (20, 20), 8, (0, 0, red_intensity), -1)

        # "LIVE" text
        cv2.putText(
            frame_bgr, 'LIVE', (35, 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    def _draw_algo_info(
        self, frame_bgr: np.ndarray, algorithm: str, stage: Optional[str]
    ) -> None:
        """Draw algorithm and stage info in top-right area."""
        text = algorithm.upper()
        if stage:
            text += f'  |  World {stage}'

        # Draw with shadow for readability
        x = self.width - 300
        y = 27
        cv2.putText(frame_bgr, text, (x + 1, y + 1),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame_bgr, text, (x, y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_stats_ticker(
        self,
        frame_bgr: np.ndarray,
        episode: Optional[int],
        reward: Optional[float],
        elapsed_time: Optional[float],
    ) -> None:
        """Draw training stats at bottom of frame."""
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

        # Semi-transparent bar at bottom
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (0, self.height - 35),
                      (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)

        cv2.putText(frame_bgr, text, (15, y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_overlay_manager.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/streaming/overlay_manager.py tests/test_overlay_manager.py
git commit -m "feat: rewrite OverlayManager with LIVE badge, algo info, and stats ticker"
```

---

## Task 8: Curriculum Learning Manager

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/curriculum.py`
- Create: `tests/test_curriculum.py`

**Step 1: Write failing test**

```python
# tests/test_curriculum.py
"""Tests for CurriculumManager."""
import pytest
import json


def test_initial_stage():
    from src.training.curriculum import CurriculumManager
    cm = CurriculumManager()
    assert cm.current_stage == (1, 1)


def test_advance_requires_threshold():
    from src.training.curriculum import CurriculumManager
    cm = CurriculumManager(advance_threshold=100.0, advance_window=3)
    # Report rewards below threshold
    cm.report_episode(reward=50.0, distance=500, completed=False)
    cm.report_episode(reward=60.0, distance=600, completed=False)
    cm.report_episode(reward=70.0, distance=700, completed=False)
    assert cm.should_advance() is False


def test_advance_on_threshold_met():
    from src.training.curriculum import CurriculumManager
    cm = CurriculumManager(advance_threshold=100.0, advance_window=3)
    cm.report_episode(reward=150.0, distance=2000, completed=True)
    cm.report_episode(reward=120.0, distance=2000, completed=True)
    cm.report_episode(reward=130.0, distance=2000, completed=True)
    assert cm.should_advance() is True


def test_advance_moves_to_next_stage():
    from src.training.curriculum import CurriculumManager
    cm = CurriculumManager()
    cm.advance()
    assert cm.current_stage == (1, 2)


def test_advance_wraps_world():
    from src.training.curriculum import CurriculumManager
    cm = CurriculumManager(start_world=1, start_stage=4)
    cm.advance()
    assert cm.current_stage == (2, 1)


def test_revisitation_schedule():
    from src.training.curriculum import CurriculumManager
    cm = CurriculumManager(revisit_interval=5)
    # Complete first stage
    cm.advance()  # Now on 1-2
    # After 5 episodes, should suggest revisit
    for i in range(5):
        cm.report_episode(reward=50, distance=500, completed=False)
    stage = cm.get_revisit_stage()
    assert stage == (1, 1)  # Revisit the completed stage


def test_save_load_state(tmp_path):
    from src.training.curriculum import CurriculumManager
    cm = CurriculumManager()
    cm.advance()
    cm.advance()
    path = str(tmp_path / 'curriculum.json')
    cm.save_state(path)

    cm2 = CurriculumManager()
    cm2.load_state(path)
    assert cm2.current_stage == cm.current_stage
    assert cm2.completed_stages == cm.completed_stages
```

**Step 2: Run tests, verify fail**

Run: `python -m pytest tests/test_curriculum.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Implement CurriculumManager**

```python
# src/training/__init__.py
"""Training utilities and managers."""
from .curriculum import CurriculumManager

__all__ = ['CurriculumManager']
```

```python
# src/training/curriculum.py
"""
Curriculum learning manager for whole-game Mario training.

Manages structured stage progression with:
- Performance-based advancement thresholds
- Periodic revisitation of earlier stages to prevent forgetting
- Per-stage metrics tracking
- State persistence for pause/resume across sessions
"""
import json
import random
from collections import deque
from typing import Dict, List, Optional, Tuple


# All 32 stages in Super Mario Bros
ALL_STAGES = [
    (w, s) for w in range(1, 9) for s in range(1, 5)
]


class CurriculumManager:
    """
    Manages stage progression with curriculum learning.

    Args:
        start_world: Starting world (1-8). Default 1.
        start_stage: Starting stage (1-4). Default 1.
        advance_threshold: Average reward needed to advance. Default 200.
        advance_window: Number of recent episodes to average. Default 10.
        revisit_interval: Episodes between revisitation checks. Default 20.
        completion_count: Times stage must be completed to advance. Default 3.
    """

    def __init__(
        self,
        start_world: int = 1,
        start_stage: int = 1,
        advance_threshold: float = 200.0,
        advance_window: int = 10,
        revisit_interval: int = 20,
        completion_count: int = 3,
    ):
        self._current_world = start_world
        self._current_stage = start_stage
        self.advance_threshold = advance_threshold
        self.advance_window = advance_window
        self.revisit_interval = revisit_interval
        self.completion_count = completion_count

        # Per-stage tracking
        self.stage_rewards: Dict[Tuple[int, int], deque] = {}
        self.stage_completions: Dict[Tuple[int, int], int] = {}
        self.completed_stages: List[Tuple[int, int]] = []

        # Episode counter for revisitation
        self._episode_counter = 0
        self._is_revisiting = False
        self._revisit_stage: Optional[Tuple[int, int]] = None

    @property
    def current_stage(self) -> Tuple[int, int]:
        """Current (world, stage) tuple."""
        if self._is_revisiting and self._revisit_stage:
            return self._revisit_stage
        return (self._current_world, self._current_stage)

    @property
    def progress(self) -> float:
        """Progress through all 32 stages (0.0 to 1.0)."""
        idx = ALL_STAGES.index((self._current_world, self._current_stage))
        return idx / (len(ALL_STAGES) - 1)

    def report_episode(
        self,
        reward: float,
        distance: int = 0,
        completed: bool = False,
    ) -> None:
        """
        Report an episode result for the current stage.

        Args:
            reward: Total episode reward.
            distance: Max distance reached.
            completed: Whether the stage flag was reached.
        """
        stage = (self._current_world, self._current_stage)

        if stage not in self.stage_rewards:
            self.stage_rewards[stage] = deque(maxlen=self.advance_window)
            self.stage_completions[stage] = 0

        self.stage_rewards[stage].append(reward)
        if completed:
            self.stage_completions[stage] += 1

        self._episode_counter += 1

        # End revisitation after one episode
        if self._is_revisiting:
            self._is_revisiting = False
            self._revisit_stage = None

    def should_advance(self) -> bool:
        """Check if current stage meets advancement criteria."""
        stage = (self._current_world, self._current_stage)
        rewards = self.stage_rewards.get(stage, deque())

        if len(rewards) < self.advance_window:
            return False

        avg_reward = sum(rewards) / len(rewards)
        completions = self.stage_completions.get(stage, 0)

        return (avg_reward >= self.advance_threshold or
                completions >= self.completion_count)

    def advance(self) -> Optional[Tuple[int, int]]:
        """
        Advance to the next stage.

        Returns:
            New (world, stage) tuple, or None if at final stage (8-4).
        """
        # Record current as completed
        current = (self._current_world, self._current_stage)
        if current not in self.completed_stages:
            self.completed_stages.append(current)

        # Find next stage
        try:
            idx = ALL_STAGES.index(current)
        except ValueError:
            return None

        if idx >= len(ALL_STAGES) - 1:
            return None  # At 8-4

        next_stage = ALL_STAGES[idx + 1]
        self._current_world, self._current_stage = next_stage
        self._episode_counter = 0
        return next_stage

    def get_revisit_stage(self) -> Optional[Tuple[int, int]]:
        """
        Get a previously-completed stage to revisit.

        Returns:
            (world, stage) to revisit, or None if no completed stages.
        """
        if not self.completed_stages:
            return None

        if self._episode_counter >= self.revisit_interval:
            return random.choice(self.completed_stages)
        return None

    def start_revisit(self, stage: Tuple[int, int]) -> None:
        """Begin a revisitation episode on a completed stage."""
        self._is_revisiting = True
        self._revisit_stage = stage

    def save_state(self, path: str) -> None:
        """Save curriculum state to JSON file."""
        state = {
            'current_world': self._current_world,
            'current_stage': self._current_stage,
            'completed_stages': self.completed_stages,
            'stage_completions': {
                f'{w}-{s}': c
                for (w, s), c in self.stage_completions.items()
            },
            'episode_counter': self._episode_counter,
            'advance_threshold': self.advance_threshold,
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str) -> None:
        """Load curriculum state from JSON file."""
        with open(path, 'r') as f:
            state = json.load(f)

        self._current_world = state['current_world']
        self._current_stage = state['current_stage']
        self.completed_stages = [tuple(s) for s in state['completed_stages']]
        self.stage_completions = {
            tuple(int(x) for x in k.split('-')): v
            for k, v in state['stage_completions'].items()
        }
        self._episode_counter = state.get('episode_counter', 0)
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_curriculum.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/ tests/test_curriculum.py
git commit -m "feat: add CurriculumManager for whole-game training with revisitation"
```

---

## Task 9: Integrate Streaming + Overlay Into Main Pipeline

**Files:**
- Modify: `main.py` (argparse + training setup + loop)
- Modify: `config/streaming_config.yaml`
- Modify: `src/visualization/dashboard.py` (stream status indicator)

**Step 1: Add streaming args to main.py argparse**

After the `--music` argument (added in Task 5), add:

```python
    # Streaming
    parser.add_argument(
        '--stream-twitch',
        type=str,
        default=None,
        help='Twitch stream key for live streaming',
    )
    parser.add_argument(
        '--stream-youtube',
        type=str,
        default=None,
        help='YouTube stream key for live streaming',
    )
```

**Step 2: Add streaming config to streaming_config.yaml**

Append to `config/streaming_config.yaml`:

```yaml
# --- Streaming Settings ---
streaming:
  # Output resolution for stream (independent of dashboard)
  resolution: [1280, 720]
  # Stream frame rate
  fps: 30
  # Video bitrate (4500k recommended for 720p on Twitch)
  video_bitrate: "4500k"
  # Stream keys (set via CLI args or here)
  twitch_key: null
  youtube_key: null
```

**Step 3: Initialize StreamManager and OverlayManager in main.py**

In the training setup section of main.py (after music setup, before the training loop):

```python
    # Streaming
    stream_manager = None
    overlay_manager = None
    twitch_key = args.stream_twitch
    youtube_key = args.stream_youtube

    if twitch_key or youtube_key:
        from src.streaming.stream_manager import StreamManager
        from src.streaming.overlay_manager import OverlayManager

        # Get first music file for audio stream (if available)
        audio_file = None
        if music_manager and music_manager.playlist:
            audio_file = music_manager.playlist[0]

        stream_manager = StreamManager(
            twitch_key=twitch_key,
            youtube_key=youtube_key,
            audio_file=audio_file,
        )
        overlay_manager = OverlayManager(resolution=(1280, 720))

        if stream_manager.start():
            print('[Stream] Streaming started successfully')
        else:
            print(f'[Stream] Failed to start: {stream_manager.error_message}')
            stream_manager = None
```

**Step 4: Send frames to stream in dashboard update cycle**

In `dashboard.py`, modify the `update` method to send frames to stream after rendering. Add `stream_manager` and `overlay_manager` as optional params to Dashboard.__init__:

```python
        self.stream_manager = stream_manager
        self.overlay_manager = overlay_manager
```

At the end of the `update` method, after all rendering:

```python
        # Send frame to stream if active
        if self.stream_manager and self.stream_manager.is_streaming:
            surface = self.screen.copy()
            frame = pygame.surfarray.array3d(surface)
            frame = np.transpose(frame, (1, 0, 2))  # (H, W, 3)

            if self.overlay_manager:
                frame = self.overlay_manager.compose(
                    frame,
                    is_live=True,
                    algorithm=self.algorithm,
                )

            self.stream_manager.send_frame(frame)
```

**Step 5: Clean up stream on exit**

In `main.py`, in the finally block at the end:

```python
    finally:
        if stream_manager:
            stream_manager.stop()
```

**Step 6: Test manually**

Run: `python main.py --algorithm dqn --visualize --episodes 3 --stream-twitch test_key`
Expected: See "[Stream] ERROR: ..." (unless ffmpeg + valid key), but training runs fine.

**Step 7: Commit**

```bash
git add main.py config/streaming_config.yaml src/visualization/dashboard.py
git commit -m "feat: integrate streaming pipeline into main training loop"
```

---

## Task 10: Launcher GUI — Streaming & Music Controls

**Files:**
- Modify: `launcher.py` (add streaming section, music section, pass CLI args)

**Step 1: Add tkinter variables in __init__**

In the launcher's `__init__` (where other tk.StringVar/BooleanVar are initialized), add:

```python
        # Streaming
        self.twitch_key_var = tk.StringVar()
        self.youtube_key_var = tk.StringVar()
        self.stream_var = tk.BooleanVar(value=False)

        # Music
        self.music_var = tk.BooleanVar(value=True)
        self.music_dir_var = tk.StringVar(
            value=os.path.join(PROJECT_ROOT, 'assets', 'music')
        )
```

**Step 2: Build streaming UI section**

Add a new method `_build_streaming_section` (call it in the main build flow, after options section):

```python
    def _build_streaming_section(self):
        """Build the streaming configuration section."""
        section = tk.LabelFrame(
            self.root,
            text="  Streaming  ",
            font=("Segoe UI", 10),
            fg=TEXT_DIM,
            bg=BG_DARK,
            bd=1,
            relief="groove",
            highlightbackground=BORDER_COLOR,
            padx=15,
            pady=8,
        )
        section.pack(fill="x", padx=25, pady=8)

        # Enable streaming checkbox
        stream_cb = tk.Checkbutton(
            section,
            text="Enable Live Streaming",
            variable=self.stream_var,
            font=("Segoe UI", 10),
            fg=ACCENT_RED,
            bg=BG_DARK,
            selectcolor=BG_MEDIUM,
            activebackground=BG_DARK,
            activeforeground=ACCENT_RED,
            cursor="hand2",
        )
        stream_cb.pack(anchor="w")

        # Twitch key
        tk.Label(section, text="Twitch Stream Key:", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w", pady=(5, 0))
        twitch_entry = tk.Entry(
            section, textvariable=self.twitch_key_var, show="*",
            font=("Segoe UI", 10), bg=BG_MEDIUM, fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY, relief="flat",
        )
        twitch_entry.pack(fill="x", pady=2)

        # YouTube key
        tk.Label(section, text="YouTube Stream Key:", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w", pady=(5, 0))
        youtube_entry = tk.Entry(
            section, textvariable=self.youtube_key_var, show="*",
            font=("Segoe UI", 10), bg=BG_MEDIUM, fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY, relief="flat",
        )
        youtube_entry.pack(fill="x", pady=2)
```

**Step 3: Build music UI section**

```python
    def _build_music_section(self):
        """Build the music configuration section."""
        section = tk.LabelFrame(
            self.root,
            text="  Music  ",
            font=("Segoe UI", 10),
            fg=TEXT_DIM,
            bg=BG_DARK,
            bd=1,
            relief="groove",
            highlightbackground=BORDER_COLOR,
            padx=15,
            pady=8,
        )
        section.pack(fill="x", padx=25, pady=8)

        music_cb = tk.Checkbutton(
            section,
            text="Play Background Music",
            variable=self.music_var,
            font=("Segoe UI", 10),
            fg=TEXT_PRIMARY,
            bg=BG_DARK,
            selectcolor=BG_MEDIUM,
            activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY,
            cursor="hand2",
        )
        music_cb.pack(anchor="w")
```

**Step 4: Pass streaming and music args in _start_training**

In the `_start_training` method (around line 690), before launching the subprocess:

```python
        # Add streaming flags
        if self.stream_var.get():
            twitch_key = self.twitch_key_var.get().strip()
            youtube_key = self.youtube_key_var.get().strip()
            if twitch_key:
                cmd.extend(["--stream-twitch", twitch_key])
            if youtube_key:
                cmd.extend(["--stream-youtube", youtube_key])

        # Add music flag
        if self.music_var.get():
            music_dir = self.music_dir_var.get().strip()
            if music_dir and os.path.isdir(music_dir):
                cmd.extend(["--music", music_dir])
```

**Step 5: Test manually**

Run: `python launcher.py`
Expected: See Streaming and Music sections in the GUI. Stream keys are masked.

**Step 6: Commit**

```bash
git add launcher.py
git commit -m "feat: add streaming and music controls to launcher GUI"
```

---

## Task 11: Integrate Curriculum Learning Into Main

**Files:**
- Modify: `main.py` (add --curriculum flag, integrate CurriculumManager into training loop)

**Step 1: Add CLI argument**

```python
    parser.add_argument(
        '--curriculum',
        action='store_true',
        help='Enable curriculum learning for whole-game training (all 32 stages)',
    )
```

**Step 2: Replace stage progression logic in training loop**

In the training loop (around line 317), when `--curriculum` is set, use CurriculumManager instead of the simple `next_world_stage()`:

```python
    # Initialize curriculum manager if enabled
    curriculum = None
    if args.curriculum:
        from src.training.curriculum import CurriculumManager
        curriculum = CurriculumManager(
            start_world=args.world,
            start_stage=args.stage,
        )
        print(f'Curriculum learning enabled: training across all stages')
        print(f'Starting at World {args.world}-{args.stage}')
```

Then in the training while loop, after each training call completes, check curriculum:

```python
                # Curriculum-based stage advancement
                if curriculum:
                    if curriculum.should_advance():
                        result = curriculum.advance()
                        if result is None:
                            print('\nAll 32 stages complete!')
                            break
                        next_w, next_s = result
                        print(f'\nCurriculum advancing: -> World {next_w}-{next_s}')
                        # Create new environment for next stage
                        # ... (environment recreation logic)
                        current_world, current_stage = next_w, next_s
                        continue  # Start training on new stage

                    # Check for revisitation
                    revisit = curriculum.get_revisit_stage()
                    if revisit:
                        print(f'\nRevisiting World {revisit[0]}-{revisit[1]}...')
                        curriculum.start_revisit(revisit)
                        # ... (temporary environment swap)
                elif not args.next_stage:
                    break
```

**Step 3: Add --curriculum to launcher**

In `launcher.py`, add a checkbox variable and pass `--curriculum` flag:

```python
        self.curriculum_var = tk.BooleanVar(value=False)
```

Add checkbox in options section and pass in _start_training:

```python
        if self.curriculum_var.get():
            cmd.append("--curriculum")
```

**Step 4: Commit**

```bash
git add main.py launcher.py
git commit -m "feat: integrate curriculum learning into training loop and launcher"
```

---

## Task 12: Performance Optimizations

**Files:**
- Modify: `src/visualization/dashboard.py` (metrics store cap, graph caching)
- Modify: `src/algorithms/dqn/dqn_trainer.py` (gradient clipping, epsilon decay)

**Step 1: Cap metrics store**

In `dashboard.py`, find where metrics are appended to lists (the metrics tracker). Add a max length:

```python
MAX_METRICS_HISTORY = 10000

# When appending metrics, use deque or trim:
if len(self.rewards_history) > MAX_METRICS_HISTORY:
    self.rewards_history = self.rewards_history[-MAX_METRICS_HISTORY:]
```

**Step 2: Optimize DQN gradient clipping**

In `dqn_trainer.py`, find the gradient clipping call and change from 10.0 to 1.0:

```python
# Change: torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
# To:     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
```

**Step 3: Add configurable epsilon decay**

In `config/dqn_config.yaml`, the epsilon_decay is 0.995. Document that users can adjust it:

```yaml
  # Epsilon decay rate per episode (0.995 = slow exploration, 0.99 = faster)
  epsilon_decay: 0.995
```

**Step 4: Commit**

```bash
git add src/visualization/dashboard.py src/algorithms/dqn/dqn_trainer.py config/dqn_config.yaml
git commit -m "perf: cap metrics history, tighten DQN gradient clipping"
```

---

## Task 13: Update Requirements & Documentation

**Files:**
- Modify: `requirements.txt`
- Update: `config/streaming_config.yaml` (final version)

**Step 1: No new pip dependencies needed**

pygame.mixer is already in pygame. ffmpeg is a system dependency, not a pip package. OpenCV is already in requirements. No changes needed to requirements.txt.

**Step 2: Add ffmpeg install note**

Create or update a section in any existing README about ffmpeg:

```text
## Streaming Requirements

For live streaming to Twitch/YouTube, install ffmpeg:
- Windows: Download from https://ffmpeg.org/download.html and add to PATH
- Or use: winget install ffmpeg
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "docs: add ffmpeg install instructions and finalize streaming config"
```

---

## Summary of All Tasks

| Task | Component | Description |
|------|-----------|-------------|
| 1 | Infrastructure | Set up pytest test infrastructure |
| 2 | Pause/Resume | True pause/resume in BaseTrainer |
| 3 | Pause/Resume | Wire check_pause() into all 3 trainers |
| 4 | Music | MusicManager with playlist and controls |
| 5 | Music | Integrate music into Dashboard + main.py |
| 6 | Streaming | StreamManager with ffmpeg RTMP |
| 7 | Streaming | OverlayManager rewrite with LIVE badge |
| 8 | Training | CurriculumManager for whole-game training |
| 9 | Integration | Wire streaming into main pipeline |
| 10 | GUI | Launcher streaming + music controls |
| 11 | Integration | Wire curriculum learning into main |
| 12 | Performance | Metrics cap, DQN optimizations |
| 13 | Docs | Requirements and ffmpeg instructions |
