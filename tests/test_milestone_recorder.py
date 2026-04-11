"""Tests for the MilestoneRecorder."""

import json
import os
import tempfile

import pytest

from src.achievements.event_bus import EventBus
from src.recording.milestone_recorder import MilestoneRecorder


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def recorder(bus):
    return MilestoneRecorder(event_bus=bus)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_start_and_stop(recorder):
    """start_recording / stop_recording toggle recording state."""
    assert recorder._recording_active is False

    recorder.start_recording()
    assert recorder._recording_active is True
    assert recorder._start_time is not None

    recorder.stop_recording()
    assert recorder._recording_active is False


def test_add_marker(recorder):
    """add_marker creates a marker dict with the correct fields."""
    recorder.start_recording()
    recorder.update_frame(42)
    recorder.add_marker('test_type', value=99, description='hello')

    markers = recorder.get_markers()
    assert len(markers) == 1
    m = markers[0]
    assert m['frame'] == 42
    assert m['type'] == 'test_type'
    assert m['value'] == 99
    assert m['description'] == 'hello'
    assert 'time' in m


def test_markers_not_added_when_inactive(recorder):
    """add_marker does nothing before start_recording is called."""
    recorder.add_marker('ignored', description='should not appear')
    assert recorder.get_markers() == []


def test_new_best_event(bus, recorder):
    """Publishing new_best_reward creates a marker."""
    recorder.start_recording()
    recorder.update_frame(10)
    bus.publish({'type': 'new_best_reward', 'reward': 500.0})

    markers = recorder.get_markers()
    assert len(markers) == 1
    assert markers[0]['type'] == 'new_best_reward'
    assert markers[0]['value'] == 500.0
    assert '500' in markers[0]['description']


def test_achievement_event(bus, recorder):
    """Publishing achievement_earned creates a marker."""
    recorder.start_recording()
    bus.publish({'type': 'achievement_earned', 'achievement_name': 'first_win'})

    markers = recorder.get_markers()
    assert len(markers) == 1
    assert markers[0]['type'] == 'achievement'
    assert markers[0]['value'] == 'first_win'
    assert 'first_win' in markers[0]['description']


def test_episode_milestone_at_round_numbers(bus, recorder):
    """Round-number episodes create markers; odd episodes do not."""
    recorder.start_recording()

    # Episode 100 should create a marker (<=100 and divisible by 10)
    bus.publish({'type': 'episode_complete', 'episode': 100})
    assert len(recorder.get_markers()) == 1

    # Episode 37 should NOT create a marker
    bus.publish({'type': 'episode_complete', 'episode': 37})
    assert len(recorder.get_markers()) == 1  # still 1

    # Episode 500 should create a marker
    bus.publish({'type': 'episode_complete', 'episode': 500})
    assert len(recorder.get_markers()) == 2

    # Episode 1000 should create a marker
    bus.publish({'type': 'episode_complete', 'episode': 1000})
    assert len(recorder.get_markers()) == 3


def test_save_markers(bus, recorder):
    """stop_recording with a filename creates a JSON sidecar file."""
    recorder.start_recording()
    recorder.add_marker('test')

    with tempfile.TemporaryDirectory() as tmpdir:
        rec_path = os.path.join(tmpdir, 'recording.mp4')
        recorder.stop_recording(recording_filename=rec_path)

        sidecar = os.path.join(tmpdir, 'recording_markers.json')
        assert os.path.isfile(sidecar)


def test_sidecar_content(bus, recorder):
    """JSON sidecar contains correct top-level keys and marker structure."""
    recorder.start_recording()
    recorder.update_frame(5)
    recorder.add_marker('check', value=1, description='d')

    with tempfile.TemporaryDirectory() as tmpdir:
        rec_path = os.path.join(tmpdir, 'vid.mp4')
        recorder.stop_recording(recording_filename=rec_path)

        sidecar = os.path.join(tmpdir, 'vid_markers.json')
        with open(sidecar) as f:
            data = json.load(f)

        assert data['recording'] == 'vid.mp4'
        assert data['total_markers'] == 1
        assert 'duration' in data
        assert len(data['markers']) == 1
        m = data['markers'][0]
        assert m['frame'] == 5
        assert m['type'] == 'check'
        assert m['value'] == 1


def test_get_markers(recorder):
    """get_markers returns a copy, not the internal list."""
    recorder.start_recording()
    recorder.add_marker('a')

    markers = recorder.get_markers()
    markers.append({'fake': True})  # mutate the copy

    assert len(recorder.get_markers()) == 1  # original unchanged
