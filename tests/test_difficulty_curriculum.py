"""Tests for the DifficultyCurriculum auto-adjusting opponent system."""

import pytest

from src.achievements.event_bus import EventBus
from src.opponents.difficulty_curriculum import DifficultyCurriculum, DIFFICULTY_TIERS


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def curriculum(bus):
    return DifficultyCurriculum(game_id='tictactoe', event_bus=bus, window_size=50)


@pytest.fixture
def curriculum_no_bus():
    return DifficultyCurriculum(game_id='tictactoe', window_size=50)


# ------------------------------------------------------------------
# Basic state
# ------------------------------------------------------------------

def test_initial_tier_is_zero(curriculum):
    """Starts at tier 0 (Random)."""
    assert curriculum.current_tier == 0
    assert curriculum.current_tier_name == 'Random'


def test_initial_win_rate_zero(curriculum):
    """No results -> win rate 0.0."""
    assert curriculum.win_rate == 0.0


# ------------------------------------------------------------------
# Recording results
# ------------------------------------------------------------------

def test_record_win(curriculum):
    """record_result(1.0) adds a win to results."""
    curriculum.record_result(1.0)
    assert len(curriculum._results) == 1
    assert curriculum._results[0] == 1


# ------------------------------------------------------------------
# Promotion / demotion
# ------------------------------------------------------------------

def test_promotion_at_threshold(curriculum):
    """Record 50 wins -> promotes to tier 1."""
    for _ in range(50):
        curriculum.record_result(1.0)
    assert curriculum.current_tier == 1
    assert curriculum.current_tier_name == 'Minimax Easy'
    assert curriculum._total_promotions == 1


def test_no_promotion_below_threshold(curriculum):
    """Record 30 wins + 20 losses -> stays at tier 0."""
    for _ in range(30):
        curriculum.record_result(1.0)
    for _ in range(20):
        curriculum.record_result(-1.0)
    assert curriculum.current_tier == 0


def test_demotion_at_threshold(curriculum):
    """Promote to tier 1, then record 50 losses -> demotes to tier 0."""
    # Promote first
    for _ in range(50):
        curriculum.record_result(1.0)
    assert curriculum.current_tier == 1

    # Now lose 50 games at tier 1
    for _ in range(50):
        curriculum.record_result(-1.0)
    assert curriculum.current_tier == 0
    assert curriculum._total_demotions == 1


def test_no_demotion_at_tier_zero(curriculum):
    """Record 50 losses at tier 0 -> stays at 0."""
    for _ in range(50):
        curriculum.record_result(-1.0)
    assert curriculum.current_tier == 0
    assert curriculum._total_demotions == 0


def test_max_tier_cap(curriculum):
    """Promote through all tiers, stays at max."""
    max_tier = curriculum.max_tier
    for _tier in range(max_tier + 2):  # Try to go beyond max
        for _ in range(50):
            curriculum.record_result(1.0)
    assert curriculum.current_tier == max_tier
    assert curriculum.current_tier_name == DIFFICULTY_TIERS[max_tier]['name']


# ------------------------------------------------------------------
# Opponent creation
# ------------------------------------------------------------------

def test_create_opponent_random(curriculum):
    """Tier 0 creates RandomOpponent."""
    from src.opponents import RandomOpponent
    opp = curriculum.create_opponent()
    assert isinstance(opp, RandomOpponent)


def test_create_opponent_minimax(curriculum):
    """Tier 1 creates MinimaxOpponent."""
    from src.opponents import MinimaxOpponent
    # Promote to tier 1
    for _ in range(50):
        curriculum.record_result(1.0)
    opp = curriculum.create_opponent()
    assert isinstance(opp, MinimaxOpponent)
    assert opp.depth == 1


# ------------------------------------------------------------------
# EventBus integration
# ------------------------------------------------------------------

def test_event_bus_records_results(curriculum, bus):
    """Publish episode_complete with reward, verify results updated."""
    bus.publish({
        'type': 'episode_complete',
        'reward': 1.0,
        'game_id': 'tictactoe',
    })
    assert len(curriculum._results) == 1
    assert curriculum._results[0] == 1


def test_event_bus_ignores_other_games(curriculum, bus):
    """Events for other games should be ignored."""
    bus.publish({
        'type': 'episode_complete',
        'reward': 1.0,
        'game_id': 'chess',
    })
    assert len(curriculum._results) == 0


def test_promotion_event_published(bus):
    """On promotion, difficulty_changed event fires."""
    events = []
    bus.subscribe('difficulty_changed', lambda e: events.append(e))
    cur = DifficultyCurriculum(game_id='tictactoe', event_bus=bus, window_size=50)
    for _ in range(50):
        cur.record_result(1.0)
    assert len(events) == 1
    assert events[0]['direction'] == 'promoted'
    assert events[0]['new_tier'] == 1
    assert events[0]['tier_name'] == 'Minimax Easy'


# ------------------------------------------------------------------
# Stats and window
# ------------------------------------------------------------------

def test_get_stats(curriculum):
    """Verify all stats fields present."""
    curriculum.record_result(1.0)
    stats = curriculum.get_stats()
    expected_keys = {
        'current_tier', 'tier_name', 'win_rate', 'progress',
        'total_promotions', 'total_demotions', 'games_in_window',
        'window_size',
    }
    assert set(stats.keys()) == expected_keys
    assert stats['window_size'] == 50
    assert stats['games_in_window'] == 1


def test_sliding_window(curriculum_no_bus):
    """Record more than window_size results, only last N kept."""
    cur = curriculum_no_bus
    for _ in range(60):
        cur.record_result(1.0)
    # After 50 wins it promoted and cleared, then 10 more wins recorded
    assert len(cur._results) == 10


def test_progress_property(curriculum):
    """At 50% win rate with 0.70 promote threshold -> progress ~0.71."""
    # Record 25 wins and 25 losses (but we need the window full first)
    for _ in range(25):
        curriculum.record_result(1.0)
    for _ in range(25):
        curriculum.record_result(-1.0)
    # win_rate = 25/50 = 0.5, progress = 0.5 / 0.7 ~ 0.714
    assert abs(curriculum.progress - (0.5 / 0.7)) < 0.01
