"""Shared fixtures for mario-ml tests."""
import pytest


@pytest.fixture
def tmp_music_dir(tmp_path):
    """Create a temporary directory with a fake music file."""
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    # Create a minimal valid WAV file (44 bytes - header only, 0 samples)
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
