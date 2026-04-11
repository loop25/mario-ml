"""Tests for the highlight reel generator (metadata/selection logic only)."""

import json
import os
import tempfile

import pytest

from src.recording.highlight_reel import HighlightReel


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def reel():
    return HighlightReel(clip_duration=5.0, max_clips=10)


@pytest.fixture
def sample_markers():
    """A realistic set of markers covering all three types."""
    return [
        {'frame': 100, 'time': 3.3, 'type': 'episode_milestone',
         'description': 'Episode 10', 'value': 10},
        {'frame': 500, 'time': 16.7, 'type': 'new_best_reward',
         'description': 'New best reward: 42.5', 'value': 42.5},
        {'frame': 800, 'time': 26.7, 'type': 'achievement',
         'description': 'Achievement: first_clear', 'value': 'first_clear'},
        {'frame': 1200, 'time': 40.0, 'type': 'episode_milestone',
         'description': 'Episode 50', 'value': 50},
        {'frame': 1500, 'time': 50.0, 'type': 'new_best_reward',
         'description': 'New best reward: 85.0', 'value': 85.0},
    ]


@pytest.fixture
def markers_file(sample_markers):
    """Write sample markers to a temp JSON file and return the path."""
    data = {
        'recording': 'test_recording.mp4',
        'total_markers': len(sample_markers),
        'duration': 60.0,
        'markers': sample_markers,
    }
    fd, path = tempfile.mkstemp(suffix='_markers.json')
    with os.fdopen(fd, 'w') as f:
        json.dump(data, f)
    yield path
    os.unlink(path)


# ── Tests ─────────────────────────────────────────────────────────────

def test_load_markers(reel, markers_file, sample_markers):
    """load_markers returns the list of marker dicts from JSON."""
    loaded = reel.load_markers(markers_file)
    assert isinstance(loaded, list)
    assert len(loaded) == len(sample_markers)
    assert loaded[0]['type'] == 'episode_milestone'


def test_select_highlights_priority(reel, sample_markers):
    """new_best_reward markers are ranked above episode_milestone."""
    selected = reel.select_highlights(sample_markers)
    types = [m['type'] for m in selected]
    # Both new_best_reward markers should be present
    assert types.count('new_best_reward') == 2
    assert types.count('achievement') == 1
    assert types.count('episode_milestone') == 2


def test_select_highlights_max_clips(sample_markers):
    """Respects the max_clips limit, keeping highest-priority markers."""
    reel = HighlightReel(max_clips=2)
    selected = reel.select_highlights(sample_markers)
    assert len(selected) == 2
    # Only the two new_best_reward markers (highest priority) should survive
    for m in selected:
        assert m['type'] == 'new_best_reward'


def test_select_highlights_sorted_by_time(reel, sample_markers):
    """Selected highlights are sorted chronologically by time."""
    selected = reel.select_highlights(sample_markers)
    times = [m['time'] for m in selected]
    assert times == sorted(times)


def test_generate_reel_metadata(reel, markers_file):
    """generate_reel_metadata returns a dict with clips and source info."""
    meta = reel.generate_reel_metadata('recording.mp4', markers_file)
    assert meta['source_recording'] == 'recording.mp4'
    assert meta['source_markers'] == markers_file
    assert meta['total_clips'] == len(meta['clips'])
    assert meta['total_clips'] > 0


def test_clip_timing(reel, markers_file):
    """Clip start/end are centered around the marker time."""
    meta = reel.generate_reel_metadata('recording.mp4', markers_file)
    for clip in meta['clips']:
        center = (clip['start_time'] + clip['end_time']) / 2
        # The center should be at the marker time, unless clamped at 0
        if clip['start_time'] > 0:
            assert abs(center - (clip['end_time'] - reel.clip_duration / 2)) < 0.01
        assert clip['end_time'] - clip['start_time'] == pytest.approx(
            reel.clip_duration, abs=0.5)


def test_empty_markers(reel):
    """No markers produces an empty clips list."""
    fd, path = tempfile.mkstemp(suffix='_markers.json')
    with os.fdopen(fd, 'w') as f:
        json.dump({'markers': []}, f)
    try:
        meta = reel.generate_reel_metadata('recording.mp4', path)
        assert meta['clips'] == []
        assert meta['total_clips'] == 0
    finally:
        os.unlink(path)


def test_reel_metadata_structure(reel, markers_file):
    """Verify all required keys are present in the metadata dict."""
    meta = reel.generate_reel_metadata('recording.mp4', markers_file)
    required_keys = {
        'source_recording', 'source_markers', 'total_clips',
        'clip_duration', 'clips',
    }
    assert required_keys.issubset(meta.keys())

    # Each clip should have the expected keys
    clip_keys = {'start_time', 'end_time', 'marker_type', 'description', 'value'}
    for clip in meta['clips']:
        assert clip_keys.issubset(clip.keys())
