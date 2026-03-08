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


def test_advance_on_completion_count():
    from src.training.curriculum import CurriculumManager
    cm = CurriculumManager(advance_threshold=9999, completion_count=3, advance_window=3)
    cm.report_episode(reward=50, distance=500, completed=True)
    cm.report_episode(reward=50, distance=500, completed=True)
    cm.report_episode(reward=50, distance=500, completed=True)
    assert cm.should_advance() is True


def test_revisitation_schedule():
    from src.training.curriculum import CurriculumManager
    cm = CurriculumManager(revisit_interval=5)
    cm.advance()
    for i in range(5):
        cm.report_episode(reward=50, distance=500, completed=False)
    stage = cm.get_revisit_stage()
    assert stage == (1, 1)


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


def test_progress():
    from src.training.curriculum import CurriculumManager
    cm = CurriculumManager()
    assert cm.progress == 0.0
    cm = CurriculumManager(start_world=8, start_stage=4)
    assert cm.progress == 1.0


def test_all_stages_complete():
    from src.training.curriculum import CurriculumManager
    cm = CurriculumManager(start_world=8, start_stage=4)
    result = cm.advance()
    assert result is None
