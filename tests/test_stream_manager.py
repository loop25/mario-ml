"""Tests for StreamManager."""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


def test_stream_manager_init():
    from src.streaming.stream_manager import StreamManager
    sm = StreamManager(twitch_key="test_twitch_key", youtube_key="test_youtube_key")
    assert sm.twitch_key == "test_twitch_key"
    assert sm.youtube_key == "test_youtube_key"
    assert sm.is_streaming is False


def test_stream_manager_no_ffmpeg():
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
    cmd_str = ' '.join(cmd)
    assert 'rtmp://live.twitch.tv/app/abc123' in cmd_str


def test_build_ffmpeg_command_both():
    from src.streaming.stream_manager import StreamManager
    sm = StreamManager(twitch_key="tw_key", youtube_key="yt_key")
    cmd = sm._build_ffmpeg_command()
    cmd_str = ' '.join(cmd)
    assert 'rtmp://live.twitch.tv/app/tw_key' in cmd_str
    assert 'rtmp://a.rtmp.youtube.com/live2/yt_key' in cmd_str


def test_send_frame_when_not_streaming():
    from src.streaming.stream_manager import StreamManager
    sm = StreamManager(twitch_key="key")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    sm.send_frame(frame)  # Should not raise


def test_no_keys_returns_false():
    from src.streaming.stream_manager import StreamManager
    sm = StreamManager()
    assert sm.start() is False
