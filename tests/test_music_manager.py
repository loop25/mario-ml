"""Tests for MusicManager."""
import os
import pytest
from unittest.mock import patch, MagicMock


def test_music_manager_init_no_files(tmp_path):
    with patch('pygame.mixer') as mock_mixer:
        mock_mixer.music = MagicMock()
        mock_mixer.get_init = MagicMock(return_value=True)
        from src.audio.music_manager import MusicManager
        mm = MusicManager(str(tmp_path))
        assert mm.playlist == []
        assert mm.volume == 0.5


def test_music_manager_finds_tracks(tmp_music_dir):
    with patch('pygame.mixer') as mock_mixer:
        mock_mixer.music = MagicMock()
        mock_mixer.get_init = MagicMock(return_value=True)
        from src.audio.music_manager import MusicManager
        mm = MusicManager(str(tmp_music_dir))
        assert len(mm.playlist) == 1
        assert 'test_track.wav' in mm.playlist[0]


def test_volume_clamp():
    with patch('pygame.mixer') as mock_mixer:
        mock_mixer.music = MagicMock()
        mock_mixer.get_init = MagicMock(return_value=True)
        from src.audio.music_manager import MusicManager
        mm = MusicManager(".")
        mm.set_volume(1.5)
        assert mm.volume == 1.0
        mm.set_volume(-0.5)
        assert mm.volume == 0.0


def test_mute_toggle():
    with patch('pygame.mixer') as mock_mixer:
        mock_mixer.music = MagicMock()
        mock_mixer.get_init = MagicMock(return_value=True)
        from src.audio.music_manager import MusicManager
        mm = MusicManager(".")
        mm.set_volume(0.7)
        mm.toggle_mute()
        assert mm.is_muted is True
        mm.toggle_mute()
        assert mm.is_muted is False
        assert mm.volume == 0.7
