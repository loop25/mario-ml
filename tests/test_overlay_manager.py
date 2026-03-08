"""Tests for OverlayManager."""
import numpy as np
import pytest

cv2 = pytest.importorskip('cv2', reason='cv2 (opencv-python) not installed')


def test_compose_adds_live_badge():
    from src.streaming.overlay_manager import OverlayManager
    om = OverlayManager(resolution=(1280, 720))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = om.compose(frame, is_live=True, algorithm='dqn', stage='1-1')
    assert result.shape == (720, 1280, 3)
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
    frame = np.zeros((800, 1400, 3), dtype=np.uint8)
    result = om.compose(frame, is_live=False)
    assert result.shape == (720, 1280, 3)
