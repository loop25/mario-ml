# Feature Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 13 features to the Game AI Training Studio: scheduling, achievements, tournaments, pluggable opponents, recording milestones, difficulty curriculum, agent profiles, NEAT visualizer, chat bot, highlight reels, model zoo, cooperative games, and NL commands.

**Architecture:** Three foundational systems (Event Bus, Opponent Manager, Session Scheduler) provide shared infrastructure. Features are built in dependency order across Phases 5-8. Each phase is independently testable and committable.

**Tech Stack:** Python 3.13, pygame, tkinter, OpenCV, numpy, python-chess, stable-baselines3, neat-python, ffmpeg (streaming). No new external dependencies except `twitchio` for Twitch IRC (P9 only).

**Design doc:** `docs/plans/2026-03-17-feature-expansion-design.md`

---

## Phase 5: Foundations

### Task 1: Event Bus

**Files:**
- Create: `src/achievements/__init__.py`
- Create: `src/achievements/event_bus.py`
- Test: `tests/test_event_bus.py`

**Step 1: Write the failing test**

```python
# tests/test_event_bus.py
"""Tests for the event bus pub/sub system."""
from src.achievements.event_bus import EventBus


class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe('test_event', lambda e: received.append(e))
        bus.publish({'type': 'test_event', 'value': 42})
        assert len(received) == 1
        assert received[0]['value'] == 42

    def test_multiple_subscribers(self):
        bus = EventBus()
        a, b = [], []
        bus.subscribe('evt', lambda e: a.append(e))
        bus.subscribe('evt', lambda e: b.append(e))
        bus.publish({'type': 'evt', 'data': 'hello'})
        assert len(a) == 1
        assert len(b) == 1

    def test_no_crosstalk(self):
        bus = EventBus()
        received = []
        bus.subscribe('alpha', lambda e: received.append(e))
        bus.publish({'type': 'beta', 'data': 1})
        assert len(received) == 0

    def test_wildcard_subscriber(self):
        bus = EventBus()
        received = []
        bus.subscribe('*', lambda e: received.append(e))
        bus.publish({'type': 'anything', 'v': 1})
        bus.publish({'type': 'else', 'v': 2})
        assert len(received) == 2

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        cb = lambda e: received.append(e)
        bus.subscribe('evt', cb)
        bus.unsubscribe('evt', cb)
        bus.publish({'type': 'evt'})
        assert len(received) == 0

    def test_publish_without_subscribers(self):
        bus = EventBus()
        bus.publish({'type': 'orphan'})  # Should not raise

    def test_subscriber_exception_does_not_crash(self):
        bus = EventBus()
        ok = []
        bus.subscribe('evt', lambda e: 1/0)  # Will raise ZeroDivisionError
        bus.subscribe('evt', lambda e: ok.append(e))
        bus.publish({'type': 'evt'})
        assert len(ok) == 1  # Second subscriber still called
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_event_bus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.achievements'`

**Step 3: Write minimal implementation**

```python
# src/achievements/__init__.py
"""Achievement system: event bus, achievement tracking, agent profiles."""

# src/achievements/event_bus.py
"""
Lightweight publish/subscribe event bus for training events.

Events are dicts with a 'type' key and arbitrary data. Subscribers
register for specific event types or '*' for all events. Subscriber
exceptions are caught and logged to prevent one bad handler from
breaking the pipeline.

Usage:
    bus = EventBus()
    bus.subscribe('new_best_reward', lambda e: print(e['reward']))
    bus.publish({'type': 'new_best_reward', 'reward': 142.5})
"""
from collections import defaultdict
from typing import Callable, Dict, List


class EventBus:
    """Thread-safe publish/subscribe event hub."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Register a callback for an event type. Use '*' for all events."""
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Remove a callback for an event type."""
        subs = self._subscribers.get(event_type, [])
        if callback in subs:
            subs.remove(callback)

    def publish(self, event: dict) -> None:
        """Publish an event to all matching subscribers.

        Args:
            event: Dict with at least a 'type' key.
        """
        event_type = event.get('type', '')

        # Notify type-specific subscribers
        for cb in self._subscribers.get(event_type, []):
            try:
                cb(event)
            except Exception as e:
                print(f'  [EventBus] Subscriber error on {event_type}: {e}')

        # Notify wildcard subscribers
        if event_type != '*':
            for cb in self._subscribers.get('*', []):
                try:
                    cb(event)
                except Exception as e:
                    print(f'  [EventBus] Wildcard subscriber error: {e}')
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_event_bus.py -v`
Expected: All 7 tests PASS

**Step 5: Run full test suite to verify no regressions**

Run: `python -m pytest tests/ --tb=short -q`
Expected: 493+ passed

**Step 6: Commit**

```bash
git add src/achievements/__init__.py src/achievements/event_bus.py tests/test_event_bus.py
git commit -m "feat: add event bus pub/sub system (Phase 5 foundation)"
```

---

### Task 2: Achievement Definitions

**Files:**
- Create: `src/achievements/definitions.py`
- Test: `tests/test_achievements.py`

**Step 1: Write the failing test**

```python
# tests/test_achievements.py
"""Tests for achievement definitions and condition checking."""
from src.achievements.definitions import (
    Achievement, AGENT_ACHIEVEMENTS, TRAINER_ACHIEVEMENTS,
    check_achievements,
)


class TestAchievementDefinitions:
    def test_agent_achievements_exist(self):
        assert len(AGENT_ACHIEVEMENTS) >= 10

    def test_trainer_achievements_exist(self):
        assert len(TRAINER_ACHIEVEMENTS) >= 8

    def test_achievement_has_required_fields(self):
        for a in AGENT_ACHIEVEMENTS + TRAINER_ACHIEVEMENTS:
            assert a.id, f'Achievement missing id'
            assert a.name, f'Achievement missing name'
            assert a.description, f'Achievement missing description'
            assert callable(a.condition), f'{a.id} condition not callable'

    def test_first_steps_triggers(self):
        earned = check_achievements(
            AGENT_ACHIEVEMENTS, {'episodes': 1}
        )
        assert 'first_steps' in [a.id for a in earned]

    def test_first_steps_does_not_trigger_at_zero(self):
        earned = check_achievements(
            AGENT_ACHIEVEMENTS, {'episodes': 0}
        )
        assert 'first_steps' not in [a.id for a in earned]

    def test_century_triggers(self):
        earned = check_achievements(
            AGENT_ACHIEVEMENTS, {'episodes': 100}
        )
        assert 'century' in [a.id for a in earned]

    def test_first_win_triggers(self):
        earned = check_achievements(
            AGENT_ACHIEVEMENTS, {'episodes': 10, 'wins': 1}
        )
        assert 'first_win' in [a.id for a in earned]

    def test_getting_started_triggers(self):
        earned = check_achievements(
            TRAINER_ACHIEVEMENTS, {'total_sessions': 1}
        )
        assert 'getting_started' in [a.id for a in earned]

    def test_algo_collector_needs_all_six(self):
        earned = check_achievements(
            TRAINER_ACHIEVEMENTS,
            {'algorithms_used': {'ppo', 'dqn', 'a2c', 'rainbow', 'neat', 'dt'}},
        )
        assert 'algo_collector' in [a.id for a in earned]

    def test_algo_collector_does_not_trigger_partial(self):
        earned = check_achievements(
            TRAINER_ACHIEVEMENTS,
            {'algorithms_used': {'ppo', 'dqn'}},
        )
        assert 'algo_collector' not in [a.id for a in earned]

    def test_already_earned_excluded(self):
        earned = check_achievements(
            AGENT_ACHIEVEMENTS,
            {'episodes': 100},
            already_earned={'first_steps', 'century'},
        )
        ids = [a.id for a in earned]
        assert 'first_steps' not in ids
        assert 'century' not in ids
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_achievements.py -v`
Expected: FAIL — ImportError

**Step 3: Write minimal implementation**

```python
# src/achievements/definitions.py
"""
Achievement definitions for agent-level and trainer-level progression.

Each Achievement has an id, display name, description, and a condition
function that takes a stats dict and returns True if the achievement
should be awarded.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set


@dataclass
class Achievement:
    id: str
    name: str
    description: str
    condition: Callable[[dict], bool]
    icon: str = ''  # Future: emoji or asset path


# ── Agent Achievements (per model checkpoint) ───────────────────────

AGENT_ACHIEVEMENTS = [
    Achievement('first_steps', 'First Steps', 'Complete 1 episode',
                condition=lambda s: s.get('episodes', 0) >= 1),
    Achievement('century', 'Century', 'Complete 100 episodes',
                condition=lambda s: s.get('episodes', 0) >= 100),
    Achievement('millennium', 'Millennium', 'Complete 1,000 episodes',
                condition=lambda s: s.get('episodes', 0) >= 1000),
    Achievement('marathon', 'Marathon', 'Complete 10,000 episodes',
                condition=lambda s: s.get('episodes', 0) >= 10000),
    Achievement('first_win', 'First Win', 'Win a board game',
                condition=lambda s: s.get('wins', 0) >= 1),
    Achievement('streak_5', 'Hot Streak', '5 consecutive wins',
                condition=lambda s: s.get('win_streak', 0) >= 5),
    Achievement('streak_10', 'On Fire', '10 consecutive wins',
                condition=lambda s: s.get('win_streak', 0) >= 10),
    Achievement('streak_25', 'Unstoppable', '25 consecutive wins',
                condition=lambda s: s.get('win_streak', 0) >= 25),
    Achievement('speed_demon', 'Speed Demon', 'Clear under par time',
                condition=lambda s: s.get('under_par', False)),
    Achievement('high_roller', 'High Roller', 'Reward > 2x average',
                condition=lambda s: s.get('reward_ratio', 0) >= 2.0),
    Achievement('perfectionist', 'Perfectionist', 'Win with zero invalid moves',
                condition=lambda s: s.get('perfect_game', False)),
    Achievement('generalist', 'Generalist', 'Positive avg reward in 3+ games (DT)',
                condition=lambda s: s.get('positive_games', 0) >= 3),
]

# ── Trainer Achievements (global progression) ───────────────────────

TRAINER_ACHIEVEMENTS = [
    Achievement('getting_started', 'Getting Started', 'Train your first agent',
                condition=lambda s: s.get('total_sessions', 0) >= 1),
    Achievement('algo_collector', 'Algorithm Collector', 'Use all 6 algorithms',
                condition=lambda s: len(s.get('algorithms_used', set())) >= 6),
    Achievement('game_explorer', 'Game Explorer', 'Train on all 6 built-in games',
                condition=lambda s: len(s.get('games_played', set())) >= 6),
    Achievement('night_owl', 'Night Owl', 'Complete a scheduled overnight session',
                condition=lambda s: s.get('overnight_sessions', 0) >= 1),
    Achievement('tournament_director', 'Tournament Director', 'Run a tournament',
                condition=lambda s: s.get('tournaments_run', 0) >= 1),
    Achievement('human_touch', 'Human Touch', 'Play against your trained agent',
                condition=lambda s: s.get('human_games', 0) >= 1),
    Achievement('data_scientist', 'Data Scientist', 'Collect 1000 DT trajectories',
                condition=lambda s: s.get('total_trajectories', 0) >= 1000),
    Achievement('streamer', 'Streamer', 'Stream a training session',
                condition=lambda s: s.get('streamed_sessions', 0) >= 1),
]


def check_achievements(
    definitions: List[Achievement],
    stats: dict,
    already_earned: Optional[Set[str]] = None,
) -> List[Achievement]:
    """Check which achievements have been newly earned.

    Args:
        definitions: List of Achievement definitions to check.
        stats: Current stats dict to evaluate conditions against.
        already_earned: Set of achievement IDs already awarded (skipped).

    Returns:
        List of newly earned Achievement objects.
    """
    already = already_earned or set()
    newly_earned = []
    for ach in definitions:
        if ach.id in already:
            continue
        try:
            if ach.condition(stats):
                newly_earned.append(ach)
        except Exception:
            pass  # Malformed stats — skip silently
    return newly_earned
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_achievements.py -v`
Expected: All 11 tests PASS

**Step 5: Run full suite, commit**

```bash
python -m pytest tests/ --tb=short -q
git add src/achievements/definitions.py tests/test_achievements.py
git commit -m "feat: add achievement definitions — agent + trainer level"
```

---

### Task 3: Achievement Manager (Persistence + Event Integration)

**Files:**
- Create: `src/achievements/achievement_manager.py`
- Test: `tests/test_achievement_manager.py`

**Step 1: Write the failing test**

```python
# tests/test_achievement_manager.py
"""Tests for achievement manager persistence and event integration."""
import json
import os
import tempfile
from src.achievements.achievement_manager import AchievementManager
from src.achievements.event_bus import EventBus


class TestAchievementManager:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, 'achievements.json')
        self.bus = EventBus()
        self.mgr = AchievementManager(self.bus, save_path=self.path)

    def test_initial_state_empty(self):
        assert self.mgr.get_trainer_earned() == []
        assert self.mgr.get_agent_earned('test_model') == []

    def test_episode_complete_awards_first_steps(self):
        self.mgr.set_active_agent('test_model')
        self.bus.publish({
            'type': 'episode_complete',
            'episode': 1, 'reward': 10.0,
        })
        earned = self.mgr.get_agent_earned('test_model')
        assert 'first_steps' in earned

    def test_persistence_survives_reload(self):
        self.mgr.set_active_agent('test_model')
        self.bus.publish({
            'type': 'episode_complete',
            'episode': 1, 'reward': 10.0,
        })
        self.mgr.save()
        # Reload from disk
        mgr2 = AchievementManager(EventBus(), save_path=self.path)
        assert 'first_steps' in mgr2.get_agent_earned('test_model')

    def test_duplicate_not_awarded(self):
        self.mgr.set_active_agent('test_model')
        popups = []
        self.bus.subscribe('achievement_earned', lambda e: popups.append(e))
        self.bus.publish({'type': 'episode_complete', 'episode': 1, 'reward': 5})
        self.bus.publish({'type': 'episode_complete', 'episode': 2, 'reward': 5})
        first_steps_popups = [p for p in popups if p['achievement_id'] == 'first_steps']
        assert len(first_steps_popups) == 1

    def test_trainer_session_tracking(self):
        self.mgr.record_session(game_id='snake', algorithm='ppo', streamed=False)
        stats = self.mgr.get_trainer_stats()
        assert stats['total_sessions'] == 1
        assert 'ppo' in stats['algorithms_used']
        assert 'snake' in stats['games_played']

    def test_trainer_achievement_triggers(self):
        self.mgr.record_session(game_id='snake', algorithm='ppo', streamed=False)
        earned = self.mgr.get_trainer_earned()
        assert 'getting_started' in earned

    def test_new_achievements_published_to_bus(self):
        events = []
        self.bus.subscribe('achievement_earned', lambda e: events.append(e))
        self.mgr.set_active_agent('model_a')
        self.bus.publish({'type': 'episode_complete', 'episode': 1, 'reward': 5})
        assert len(events) >= 1
        assert events[0]['type'] == 'achievement_earned'
        assert events[0]['achievement_id'] == 'first_steps'
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_achievement_manager.py -v`
Expected: FAIL — ImportError

**Step 3: Write minimal implementation**

```python
# src/achievements/achievement_manager.py
"""
Achievement Manager — tracks, awards, and persists achievements.

Subscribes to the EventBus to monitor training events. When an
achievement condition is met, awards it, persists to disk, and
publishes an 'achievement_earned' event for the overlay/chat bot.
"""
import json
import os
from typing import Dict, List, Optional, Set

from src.achievements.event_bus import EventBus
from src.achievements.definitions import (
    AGENT_ACHIEVEMENTS, TRAINER_ACHIEVEMENTS, check_achievements,
)


class AchievementManager:
    """Tracks and persists achievements for agents and the trainer.

    Args:
        event_bus: EventBus to subscribe to and publish on.
        save_path: Path to persist achievements JSON.
    """

    def __init__(
        self,
        event_bus: EventBus,
        save_path: str = 'config/achievements.json',
    ):
        self._bus = event_bus
        self._save_path = save_path
        self._active_agent: Optional[str] = None

        # State
        self._trainer_stats: Dict = {
            'total_sessions': 0,
            'algorithms_used': set(),
            'games_played': set(),
            'overnight_sessions': 0,
            'tournaments_run': 0,
            'human_games': 0,
            'total_trajectories': 0,
            'streamed_sessions': 0,
        }
        self._trainer_earned: Set[str] = set()
        self._agents: Dict[str, Dict] = {}  # agent_id -> {earned, stats}

        self._load()

        # Subscribe to training events
        event_bus.subscribe('episode_complete', self._on_episode_complete)
        event_bus.subscribe('session_complete', self._on_session_complete)

    # ── Public API ───────────────────────────────────────────────────

    def set_active_agent(self, agent_id: str) -> None:
        """Set the current agent being trained."""
        self._active_agent = agent_id
        if agent_id not in self._agents:
            self._agents[agent_id] = {
                'earned': set(),
                'stats': {
                    'episodes': 0, 'wins': 0, 'losses': 0,
                    'win_streak': 0, 'current_streak': 0,
                    'best_reward': float('-inf'),
                    'total_reward': 0.0,
                },
            }

    def get_agent_earned(self, agent_id: str) -> List[str]:
        """Get list of earned achievement IDs for an agent."""
        agent = self._agents.get(agent_id, {})
        return list(agent.get('earned', set()))

    def get_trainer_earned(self) -> List[str]:
        """Get list of earned trainer achievement IDs."""
        return list(self._trainer_earned)

    def get_trainer_stats(self) -> Dict:
        """Get current trainer stats."""
        return {
            **self._trainer_stats,
            'algorithms_used': set(self._trainer_stats['algorithms_used']),
            'games_played': set(self._trainer_stats['games_played']),
        }

    def record_session(
        self, game_id: str, algorithm: str, streamed: bool = False,
    ) -> None:
        """Record a completed training session for trainer achievements."""
        self._trainer_stats['total_sessions'] += 1
        self._trainer_stats['algorithms_used'].add(algorithm)
        self._trainer_stats['games_played'].add(game_id)
        if streamed:
            self._trainer_stats['streamed_sessions'] += 1
        self._check_trainer_achievements()
        self.save()

    # ── Event Handlers ───────────────────────────────────────────────

    def _on_episode_complete(self, event: dict) -> None:
        if not self._active_agent:
            return
        agent = self._agents.get(self._active_agent)
        if not agent:
            return

        stats = agent['stats']
        stats['episodes'] = event.get('episode', stats['episodes'] + 1)
        reward = event.get('reward', 0.0)
        stats['total_reward'] += reward
        if reward > stats['best_reward']:
            stats['best_reward'] = reward

        # Win tracking (board games set 'winner' in info)
        winner = event.get('winner', 0)
        if winner == 1:
            stats['wins'] += 1
            stats['current_streak'] += 1
            stats['win_streak'] = max(
                stats['win_streak'], stats['current_streak'],
            )
        elif winner == 2:
            stats['losses'] += 1
            stats['current_streak'] = 0

        # Reward ratio
        avg = stats['total_reward'] / max(1, stats['episodes'])
        stats['reward_ratio'] = reward / avg if avg > 0 else 0

        self._check_agent_achievements(self._active_agent)

    def _on_session_complete(self, event: dict) -> None:
        game_id = event.get('game_id', '')
        algorithm = event.get('algorithm', '')
        streamed = event.get('streamed', False)
        self.record_session(game_id, algorithm, streamed)

    # ── Achievement Checking ─────────────────────────────────────────

    def _check_agent_achievements(self, agent_id: str) -> None:
        agent = self._agents[agent_id]
        newly = check_achievements(
            AGENT_ACHIEVEMENTS, agent['stats'], agent['earned'],
        )
        for ach in newly:
            agent['earned'].add(ach.id)
            self._bus.publish({
                'type': 'achievement_earned',
                'level': 'agent',
                'agent_id': agent_id,
                'achievement_id': ach.id,
                'achievement_name': ach.name,
                'description': ach.description,
            })

    def _check_trainer_achievements(self) -> None:
        newly = check_achievements(
            TRAINER_ACHIEVEMENTS, self._trainer_stats, self._trainer_earned,
        )
        for ach in newly:
            self._trainer_earned.add(ach.id)
            self._bus.publish({
                'type': 'achievement_earned',
                'level': 'trainer',
                'achievement_id': ach.id,
                'achievement_name': ach.name,
                'description': ach.description,
            })

    # ── Persistence ──────────────────────────────────────────────────

    def save(self) -> None:
        """Save achievements to disk."""
        os.makedirs(os.path.dirname(self._save_path) or '.', exist_ok=True)
        data = {
            'trainer': {
                'earned': list(self._trainer_earned),
                'stats': {
                    **self._trainer_stats,
                    'algorithms_used': list(self._trainer_stats['algorithms_used']),
                    'games_played': list(self._trainer_stats['games_played']),
                },
            },
            'agents': {
                aid: {
                    'earned': list(a['earned']),
                    'stats': a['stats'],
                }
                for aid, a in self._agents.items()
            },
        }
        with open(self._save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load achievements from disk."""
        if not os.path.exists(self._save_path):
            return
        try:
            with open(self._save_path, 'r') as f:
                data = json.load(f)
            trainer = data.get('trainer', {})
            self._trainer_earned = set(trainer.get('earned', []))
            stats = trainer.get('stats', {})
            self._trainer_stats.update(stats)
            self._trainer_stats['algorithms_used'] = set(
                stats.get('algorithms_used', [])
            )
            self._trainer_stats['games_played'] = set(
                stats.get('games_played', [])
            )
            for aid, adata in data.get('agents', {}).items():
                self._agents[aid] = {
                    'earned': set(adata.get('earned', [])),
                    'stats': adata.get('stats', {}),
                }
        except (json.JSONDecodeError, KeyError):
            pass  # Corrupted file — start fresh
```

**Step 4: Run tests, full suite, commit**

Run: `python -m pytest tests/test_achievement_manager.py tests/test_event_bus.py -v`
Run: `python -m pytest tests/ --tb=short -q`

```bash
git add src/achievements/achievement_manager.py tests/test_achievement_manager.py
git commit -m "feat: add achievement manager with persistence and event integration"
```

---

### Task 4: Base Opponent Interface

**Files:**
- Create: `src/opponents/__init__.py`
- Create: `src/opponents/base_opponent.py`
- Create: `src/opponents/random_opponent.py`
- Test: `tests/test_opponents.py`

**Step 1: Write the failing test**

```python
# tests/test_opponents.py
"""Tests for the opponent system."""
import numpy as np
from src.opponents.base_opponent import BaseOpponent
from src.opponents.random_opponent import RandomOpponent


class TestRandomOpponent:
    def test_picks_from_valid_actions(self):
        opp = RandomOpponent()
        state = {
            'board': np.zeros((3, 3)),
            'valid_actions': [0, 1, 2],
            'game_id': 'tictactoe',
            'turn': 2,
        }
        action = opp.pick_action(state)
        assert action in [0, 1, 2]

    def test_single_valid_action(self):
        opp = RandomOpponent()
        state = {
            'board': np.zeros((3, 3)),
            'valid_actions': [5],
            'game_id': 'tictactoe',
            'turn': 2,
        }
        assert opp.pick_action(state) == 5

    def test_name(self):
        assert RandomOpponent().name == 'Random'

    def test_difficulty_tier(self):
        assert RandomOpponent().difficulty_tier == 0

    def test_reset_does_not_crash(self):
        opp = RandomOpponent()
        opp.reset()  # Should be a no-op
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_opponents.py -v`
Expected: FAIL — ImportError

**Step 3: Write minimal implementation**

```python
# src/opponents/__init__.py
"""Pluggable opponent system for board games."""

# src/opponents/base_opponent.py
"""
Abstract base class for board game opponents.

Every opponent implements pick_action() which receives the current
board state and returns a valid action. This allows swapping between
random, minimax, trained AI, and human opponents.
"""
from abc import ABC, abstractmethod


class BaseOpponent(ABC):
    """Abstract opponent that can play any board game."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for display."""

    @property
    @abstractmethod
    def difficulty_tier(self) -> int:
        """Difficulty level: 0=random, 1-3=minimax, 4-5=trained model."""

    @abstractmethod
    def pick_action(self, board_state: dict) -> int:
        """Choose an action given the current board state.

        Args:
            board_state: Dict with keys:
                'board': np.ndarray or object — current board
                'valid_actions': List[int] — legal moves
                'game_id': str — which game
                'turn': int — whose turn (1=agent, 2=opponent)

        Returns:
            Action from valid_actions.
        """

    def reset(self) -> None:
        """Called at episode start. Override for stateful opponents."""
        pass


# src/opponents/random_opponent.py
"""Random opponent — selects uniformly from valid actions."""
import random as _random
from src.opponents.base_opponent import BaseOpponent


class RandomOpponent(BaseOpponent):
    """Picks a random valid action. The simplest opponent."""

    @property
    def name(self) -> str:
        return 'Random'

    @property
    def difficulty_tier(self) -> int:
        return 0

    def pick_action(self, board_state: dict) -> int:
        return _random.choice(board_state['valid_actions'])
```

**Step 4: Run tests, full suite, commit**

```bash
python -m pytest tests/test_opponents.py -v
python -m pytest tests/ --tb=short -q
git add src/opponents/ tests/test_opponents.py
git commit -m "feat: add opponent base interface and random opponent"
```

---

### Task 5: Minimax Opponent

**Files:**
- Create: `src/opponents/minimax_opponent.py`
- Test: `tests/test_minimax_opponent.py`

**Step 1: Write the failing test**

```python
# tests/test_minimax_opponent.py
"""Tests for minimax opponent."""
import numpy as np
from src.opponents.minimax_opponent import MinimaxOpponent


class TestMinimaxTicTacToe:
    def _make_state(self, board, valid, game_id='tictactoe'):
        return {
            'board': np.array(board),
            'valid_actions': valid,
            'game_id': game_id,
            'turn': 2,
        }

    def test_blocks_winning_move(self):
        """Opponent (player 2) should block player 1 from winning."""
        # Board: player 1 has top-left and top-center.
        # Player 1 wins if top-right (action=2) is open.
        board = [[1, 1, 0],
                 [0, 2, 0],
                 [0, 0, 0]]
        state = self._make_state(board, [2, 3, 5, 6, 7, 8])
        opp = MinimaxOpponent(depth=3, game_id='tictactoe')
        action = opp.pick_action(state)
        assert action == 2  # Must block

    def test_takes_winning_move(self):
        """Opponent (player 2) should take its own winning move."""
        board = [[1, 0, 1],
                 [0, 2, 0],
                 [0, 2, 0]]
        state = self._make_state(board, [1, 3, 5, 6, 8])
        opp = MinimaxOpponent(depth=3, game_id='tictactoe')
        action = opp.pick_action(state)
        assert action == 7  # Win by completing column

    def test_name_includes_depth(self):
        opp = MinimaxOpponent(depth=3)
        assert '3' in opp.name

    def test_difficulty_tier_scales_with_depth(self):
        assert MinimaxOpponent(depth=1).difficulty_tier == 1
        assert MinimaxOpponent(depth=3).difficulty_tier == 2
        assert MinimaxOpponent(depth=5).difficulty_tier == 3


class TestMinimaxConnect4:
    def test_blocks_vertical_three(self):
        """Opponent should block player 1's vertical 3-in-a-row."""
        board = np.zeros((6, 7), dtype=int)
        board[5, 3] = 1
        board[4, 3] = 1
        board[3, 3] = 1  # Player 1 has 3 in column 3
        board[5, 0] = 2
        board[4, 0] = 2
        # Valid: column 3 (row 2 open) should be blocked
        valid = [0, 1, 2, 3, 4, 5, 6]
        state = {
            'board': board, 'valid_actions': valid,
            'game_id': 'connect4', 'turn': 2,
        }
        opp = MinimaxOpponent(depth=3, game_id='connect4')
        assert opp.pick_action(state) == 3
```

**Step 2: Run to verify failure, then implement**

The minimax implementation needs game-specific logic for simulating moves,
checking wins, and evaluating positions. Create evaluators for TicTacToe,
Connect4, Checkers, and Chess.

**Step 3: Implement** (see design doc for evaluation function details per game)

The key pattern:
```python
class MinimaxOpponent(BaseOpponent):
    def __init__(self, depth=3, game_id='tictactoe'):
        self.depth = depth
        self.game_id = game_id

    def pick_action(self, board_state):
        valid = board_state['valid_actions']
        board = board_state['board']
        best_action = valid[0]
        best_score = float('-inf')
        for action in valid:
            new_board = self._simulate_move(board, action, player=2)
            score = self._minimax(new_board, self.depth - 1,
                                  False, float('-inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
```

**Step 4: Run tests, full suite, commit**

```bash
git add src/opponents/minimax_opponent.py tests/test_minimax_opponent.py
git commit -m "feat: add minimax opponent with alpha-beta pruning"
```

---

### Task 6: Model Opponent

**Files:**
- Create: `src/opponents/model_opponent.py`
- Test: `tests/test_model_opponent.py`

Loads any trained checkpoint (DQN, Rainbow, PPO, A2C) and uses it as
an opponent. Uses the same observation preprocessing as the game env.

**Key implementation:**
```python
class ModelOpponent(BaseOpponent):
    def __init__(self, checkpoint_path, algorithm):
        self.algorithm = algorithm
        if algorithm in ('ppo', 'a2c'):
            from stable_baselines3 import PPO, A2C
            cls = PPO if algorithm == 'ppo' else A2C
            self._model = cls.load(checkpoint_path)
        elif algorithm in ('dqn', 'rainbow'):
            # Load torch model
            ...

    def pick_action(self, board_state):
        obs = board_state.get('observation')
        if self._model and obs is not None:
            action, _ = self._model.predict(obs, deterministic=True)
            return int(action)
        return random.choice(board_state['valid_actions'])
```

**Commit:** `feat: add model opponent — load trained checkpoints as opponents`

---

### Task 7: Human Opponent

**Files:**
- Create: `src/opponents/human_opponent.py`
- Test: `tests/test_human_opponent.py`

Uses pygame event loop to wait for human input. Highlights valid moves
on the board. Maps mouse clicks to board positions per game.

**Commit:** `feat: add human opponent with pygame click-to-move input`

---

### Task 8: Refactor Board Games to Use Opponent Interface

**Files:**
- Modify: `games/builtin/tictactoe/game.py:75-81`
- Modify: `games/builtin/connect4/game.py:78-84`
- Modify: `games/builtin/checkers/game.py:129-136`
- Modify: `games/builtin/chess/game.py:115-122`
- Modify: all 4 `adapter.py` files (pass opponent to create_env)

**Pattern for each game:**

Before (tictactoe line 75-81):
```python
empty = list(zip(*np.where(self.board == 0)))
if not empty:
    return self._render_obs(), 0.0, True, self._info()
opp_r, opp_c = random.choice(empty)
```

After:
```python
from src.opponents.random_opponent import RandomOpponent
# In __init__:
self.opponent = kwargs.get('opponent', RandomOpponent())

# In step():
valid = [r * 3 + c for r, c in zip(*np.where(self.board == 0))]
if not valid:
    return self._render_obs(), 0.0, True, self._info()
opp_state = {
    'board': self.board.copy(),
    'valid_actions': valid,
    'game_id': 'tictactoe',
    'turn': 2,
}
opp_action = self.opponent.pick_action(opp_state)
opp_r, opp_c = opp_action // 3, opp_action % 3
```

Do this for all 4 games. Default to RandomOpponent() for backwards compatibility.

**Critical: Run full test suite after each game refactor to ensure nothing breaks.**

```bash
# After each game:
python -m pytest tests/ --tb=short -q
# Must still pass 493+ tests
```

**Commit after all 4:** `refactor: board games use pluggable opponent interface`

---

### Task 9: Session Dataclasses

**Files:**
- Create: `src/scheduler/__init__.py`
- Create: `src/scheduler/session.py`
- Test: `tests/test_session.py`

**Commit:** `feat: add training session and report dataclasses`

---

### Task 10: Calendar Store (Persistence)

**Files:**
- Create: `src/scheduler/calendar_store.py`
- Test: `tests/test_calendar_store.py`

JSON persistence for the training schedule. CRUD operations for sessions.
Tracks completed sessions and scheduler state for crash recovery.

**Commit:** `feat: add calendar store with JSON persistence`

---

### Task 11: Scheduler Engine

**Files:**
- Create: `src/scheduler/scheduler.py`
- Test: `tests/test_scheduler.py`

Core engine that runs as a daemon thread. Pops next session from schedule,
launches training via subprocess, monitors completion, fires events.

**Key methods:**
- `start()` — launch daemon thread
- `stop()` — graceful shutdown after current session
- `run_next_now()` — skip to next pending session
- `_execute_session(session)` — launch main.py, monitor, report
- `_recover_state()` — resume after crash

**Commit:** `feat: add scheduler engine with daemon thread and crash recovery`

---

### Task 12: Wire Event Bus into Base Trainer

**Files:**
- Modify: `src/algorithms/base_trainer.py` — add optional `event_bus` parameter
- Modify: `main.py` — create and pass EventBus + AchievementManager

**Changes to base_trainer.py:**
```python
# In __init__, add:
self._event_bus = kwargs.get('event_bus', None)

# In _fire_episode_complete, add at the end:
if self._event_bus:
    self._event_bus.publish({
        'type': 'episode_complete',
        'episode': self.episode_count,
        'reward': reward,
        'distance': distance,
        'completed': completed,
    })
```

**Changes to main.py:**
```python
# Early in main(), create:
from src.achievements.event_bus import EventBus
from src.achievements.achievement_manager import AchievementManager

event_bus = EventBus()
achievement_mgr = AchievementManager(event_bus)

# Pass to trainer:
trainer = SomeTrainer(..., event_bus=event_bus)
achievement_mgr.set_active_agent(model_path)
```

**Commit:** `feat: wire event bus into training pipeline`

---

## Phase 6: Core Features

### Task 13: Achievement Dashboard Popup

**Files:**
- Modify: `src/visualization/dashboard.py`

Subscribe to `achievement_earned` events, render a popup panel that slides
in from top-right and fades after 3 seconds.

**Commit:** `feat: achievement popup overlay on dashboard`

---

### Task 14: Achievement Stream Overlay Popup

**Files:**
- Modify: `src/streaming/overlay_manager.py`

Add `_draw_achievement_popup()` method. OverlayManager receives achievement
events and renders an amber banner that fades after 3 seconds.

**Commit:** `feat: achievement popup on stream overlay`

---

### Task 15: Tournament Runner

**Files:**
- Create: `src/tournament/__init__.py`
- Create: `src/tournament/tournament_runner.py`
- Test: `tests/test_tournament.py`

Round-robin and single-elimination formats. Uses opponent interface to
pit agents against each other. Publishes results to event bus.

**Commit:** `feat: add tournament runner with round-robin and bracket formats`

---

### Task 16: Tournament Dashboard View

**Files:**
- Create: `src/tournament/tournament_ui.py`
- Modify: `src/visualization/dashboard.py`

Split dashboard view during tournaments: game on left, bracket/standings
on right. Matchup header and live score.

**Commit:** `feat: tournament bracket and standings dashboard view`

---

### Task 17: Calendar UI Widget

**Files:**
- Create: `src/scheduler/calendar_ui.py`
- Modify: `launcher.py`

Tkinter calendar widget with weekly grid, session editor form, quick
presets, and "Run Now" button.

**Commit:** `feat: training calendar UI in launcher`

---

### Task 18: Stream Scheduling Integration

**Files:**
- Modify: `src/scheduler/scheduler.py`
- Modify: `src/scheduler/session.py`

Per-session stream toggle. Auto-start/stop StreamManager around sessions.
Pre-stream countdown overlay.

**Commit:** `feat: stream scheduling — auto-start/stop per session`

---

## Phase 7: Content & Polish

### Task 19: Recording Milestone Markers

**Files:**
- Modify: `src/streaming/recording.py`
- Create: `src/streaming/recording_markers.py`

Subscribe to event bus. Write milestone markers (frame number, timestamp,
type, label) to `<recording>_markers.json` alongside the MP4.

**Commit:** `feat: recording milestone markers from event bus`

---

### Task 20: Milestone Flash on Stream Overlay

**Files:**
- Modify: `src/streaming/overlay_manager.py`

When a milestone fires, show amber banner: "NEW BEST: 142.5" for 3 seconds.
Queue system for multiple rapid milestones.

**Commit:** `feat: milestone flash banner on stream overlay`

---

### Task 21: Best Episode Clip Saving

**Files:**
- Modify: `src/streaming/recording.py`

When `new_best_reward` fires, save the last 5 seconds of frames as a
separate highlight clip in `recordings/highlights/`.

**Commit:** `feat: auto-save best episode clips as highlights`

---

### Task 22: Difficulty Curriculum

**Files:**
- Create: `src/opponents/difficulty_curriculum.py`
- Test: `tests/test_difficulty_curriculum.py`

Wraps opponent selection. Tracks rolling win rate over 50 games. Promotes
to next tier at >70%, demotes at <30%.

**Commit:** `feat: auto-advancing difficulty curriculum for board games`

---

### Task 23: Agent Profile Generation

**Files:**
- Create: `src/achievements/profile.py`
- Test: `tests/test_agent_profile.py`

Generates radar chart scores, play style classification, and stats summary
from training metadata and achievement history.

**Commit:** `feat: agent personality profile generation`

---

### Task 24: Agent Gallery in Launcher

**Files:**
- Modify: `launcher.py`

"Agent Gallery" panel showing profile cards for all trained agents.
Radar chart rendered with tkinter Canvas. Compare mode for two agents.

**Commit:** `feat: agent gallery panel in launcher with profile cards`

---

### Task 25: NEAT Network Visualizer

**Files:**
- Create: `src/visualization/neat_visualizer.py`
- Modify: `src/algorithms/neat/neat_trainer.py`
- Modify: `src/visualization/dashboard.py`

Renders best genome's network topology as a node-link diagram on a pygame
surface. Toggle with 'N' key in dashboard. NEAT trainer passes best genome
via callback each generation.

**Commit:** `feat: NEAT network topology visualizer`

---

## Phase 8: Community & Engagement

### Task 26: Chat Bot (Twitch/YouTube)

**Files:**
- Create: `src/streaming/chat_bot.py`
- Test: `tests/test_chat_bot.py`

Twitch IRC and YouTube Live Chat integration. Background thread.
Commands: !stats, !game, !algorithm, !tournament, !achievements, !schedule, !help.
Autonomous mode: auto-greet, periodic updates, achievement announcements.

**Dependency:** `twitchio` library for Twitch IRC.

**Commit:** `feat: autonomous Twitch/YouTube chat bot`

---

### Task 27: Highlight Reel Generator

**Files:**
- Create: `src/streaming/highlight_reel.py`

After session: scan markers, rank by significance, extract clips, add
title cards, concatenate with transitions. Output to highlights folder.

**Commit:** `feat: auto-generate highlight reel from recording markers`

---

### Task 28: Replay Viewer in Launcher

**Files:**
- Modify: `launcher.py`

Recordings panel with list, timeline scrubber, marker dots, jump-to-marker,
and "Export Clip" button.

**Commit:** `feat: replay viewer with timeline and marker navigation`

---

### Task 29: Model Export/Import (.agent format)

**Files:**
- Create: `src/model_zoo/__init__.py`
- Create: `src/model_zoo/exporter.py`
- Create: `src/model_zoo/importer.py`
- Test: `tests/test_model_zoo.py`

ZIP-based .agent format containing weights, metadata, profile, achievements,
and optional replay clip. Launcher buttons for export and import.

**Commit:** `feat: model zoo — export/import trained agents as .agent packages`

---

### Task 30: Cooperative Game — Dual Snake

**Files:**
- Create: `games/builtin/dual_snake/game.py`
- Create: `games/builtin/dual_snake/adapter.py`

Two snakes on shared board, shared score. Both take actions simultaneously.
Supports AI+AI, Human+AI, and Human+Human.

**Commit:** `feat: add Dual Snake cooperative game`

---

### Task 31: Natural Language Training Commands

**Files:**
- Create: `src/scheduler/nl_parser.py`
- Test: `tests/test_nl_parser.py`
- Modify: `launcher.py`

Keyword matcher that maps natural language to training config. Shows
interpreted config for user confirmation.

**Commit:** `feat: natural language training command parser`

---

### Task 32: Launcher Integration — Tournament + Schedule Buttons

**Files:**
- Modify: `launcher.py`

Add "Tournament" and "Schedule" buttons to main launcher. Wire to
tournament setup flow and calendar UI.

**Commit:** `feat: launcher buttons for tournament and schedule`

---

### Task 33: Final Integration Test

**Files:**
- Create: `tests/test_integration_features.py`

End-to-end test: create event bus + achievement manager, create random
opponent, run a simulated tournament of 2 games, verify achievements
fire, verify markers created, verify profile generated.

```bash
python -m pytest tests/ --tb=short -q
# All tests must pass (500+)
```

**Commit:** `test: integration test for full feature pipeline`

---

## Verification Checklist

After ALL tasks complete:

1. `python -m pytest tests/ --tb=short -q` — all tests pass
2. `python launcher.py` — launcher opens, schedule/tournament/gallery visible
3. Train any game → achievements fire → popup on dashboard
4. Schedule a session → scheduler runs it → session report generated
5. Tournament with 2 agents → bracket displays → winner announced
6. Human vs AI → click to move → moves accepted
7. Stream with overlay → milestones flash → chat bot responds
8. Export agent → import on different machine → loads correctly

---

## Key Principles

- **Test first**: Every task starts with a failing test
- **One commit per task**: Atomic, revertible changes
- **Run full suite after every change**: `python -m pytest tests/ --tb=short -q`
- **Backwards compatible**: Default opponent is Random, event bus is optional
- **No broken windows**: If any test fails, fix before proceeding
