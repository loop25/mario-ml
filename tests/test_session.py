"""Tests for src/scheduler/session.py — dataclasses, factories, serialization."""

from datetime import datetime

from src.scheduler.session import (
    BUILT_IN_GAMES,
    SchedulePreset,
    TrainingSession,
    create_session,
    dt_generalist_run,
    overnight_all_games,
    session_from_dict,
    session_to_dict,
)


def test_create_session_defaults():
    s = create_session("snake", "ppo")
    assert s.game_id == "snake"
    assert s.algorithm == "ppo"
    assert s.status == "pending"
    assert len(s.session_id) > 0


def test_create_session_custom():
    s = create_session("chess", "dqn", episodes=5000, stream=True)
    assert s.episodes == 5000
    assert s.stream is True
    assert s.game_id == "chess"
    assert s.algorithm == "dqn"


def test_session_to_dict_roundtrip():
    original = create_session(
        "tetris",
        "a2c",
        episodes=2000,
        device="cuda",
        opponent_type="minimax",
        opponent_config={"depth": 3},
        notes="test run",
    )
    d = session_to_dict(original)
    restored = session_from_dict(d)

    assert restored.game_id == original.game_id
    assert restored.algorithm == original.algorithm
    assert restored.episodes == original.episodes
    assert restored.device == original.device
    assert restored.opponent_type == original.opponent_type
    assert restored.opponent_config == original.opponent_config
    assert restored.notes == original.notes
    assert restored.session_id == original.session_id
    assert restored.status == original.status


def test_datetime_serialization():
    now = datetime(2026, 3, 18, 14, 30, 0)
    s = create_session("mario", "ppo", scheduled_start=now)
    d = session_to_dict(s)

    # datetime stored as ISO string
    assert isinstance(d["scheduled_start"], str)

    restored = session_from_dict(d)
    assert restored.scheduled_start == now


def test_overnight_preset():
    sessions = overnight_all_games(episodes_per_game=500)
    assert len(sessions) == len(BUILT_IN_GAMES)  # 7
    game_ids = {s.game_id for s in sessions}
    assert game_ids == set(BUILT_IN_GAMES)
    assert all(s.algorithm == "ppo" for s in sessions)
    assert all(s.episodes == 500 for s in sessions)


def test_dt_generalist_preset():
    sessions = dt_generalist_run(episodes_per_game=100)
    # 7 collection sessions + 1 DT session
    assert len(sessions) == len(BUILT_IN_GAMES) + 1
    assert sessions[-1].algorithm == "dt"
    assert sessions[-1].game_id == "all"
    # all preceding sessions are PPO collection runs
    for s in sessions[:-1]:
        assert s.algorithm == "ppo"
        assert s.episodes == 100


def test_schedule_preset_dataclass():
    sessions = overnight_all_games()
    preset = SchedulePreset(
        name="overnight",
        description="Train all games overnight with PPO",
        sessions=sessions,
    )
    assert preset.name == "overnight"
    assert preset.description == "Train all games overnight with PPO"
    assert isinstance(preset.sessions, list)
    assert len(preset.sessions) == len(BUILT_IN_GAMES)
