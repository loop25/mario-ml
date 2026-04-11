# Feature Expansion Design: Training Studio v2

**Date:** 2026-03-17
**Status:** Approved
**Author:** Claude + Dakotah

## Overview

With all 4 phases of the multi-game generalist agent platform complete (plugin
registry, 6 algorithms, 6 built-in games, Decision Transformer, streaming,
dashboard), this design covers the next evolution: 13 features that transform
the training studio from a tool into an experience.

The features are ordered by dependency chain — each builds on the ones before
it. Three foundational systems (Scheduler, Opponent Manager, Achievement/Event
Bus) provide shared infrastructure that all features plug into.

## Goals

1. Automated training schedules with calendar UI and stream scheduling
2. Pluggable opponent system: random, minimax, trained AI, human players
3. Tournament mode for agent-vs-agent and agent-vs-human competition
4. Achievement system (agent-level + trainer-level) with stream overlay popups
5. Recording milestones with markers and highlight reel generation
6. Difficulty curriculum that auto-advances board game opponents
7. Agent personality profiles and comparison cards
8. NEAT evolution network topology visualizer
9. Autonomous Twitch/YouTube chat bot for unattended streams
10. Model export/import for community sharing
11. Multi-agent cooperative games
12. Natural language training command parser

## Non-Goals

- Web dashboard (backlog, optional future)
- Mobile app or remote monitoring
- Paid features or monetization
- LLM-dependent features (NL commands use keyword matching, not an API)

---

## Foundational Systems

### System A: Session Scheduler (`src/scheduler/`)

Central engine that owns the training calendar and executes sessions
autonomously.

```
src/scheduler/
    __init__.py
    scheduler.py          # Core engine: runs sessions from queue
    calendar_store.py     # Persists schedule to JSON, supports recurring
    session.py            # Dataclass: game, algorithm, duration, stream?, opponent
    calendar_ui.py        # Tkinter calendar widget for the launcher
```

#### scheduler.py — SchedulerEngine

```python
class SchedulerEngine:
    """Runs training sessions from a persistent schedule.

    Operates as a daemon thread. Checks current time against schedule,
    launches training sessions via subprocess (same as launcher), and
    handles session transitions (save, report, next).
    """
    def __init__(self, schedule_path='config/training_schedule.json'):
        ...

    def start(self) -> None:
        """Start the scheduler daemon thread."""

    def stop(self) -> None:
        """Gracefully stop after current session completes."""

    def run_next_now(self) -> None:
        """Skip to the next pending session immediately."""

    def _execute_session(self, session: TrainingSession) -> SessionReport:
        """Launch a training session and wait for completion.

        1. If session.stream_enabled: start StreamManager
        2. Build command args from session config
        3. Launch main.py via subprocess
        4. Monitor for completion or crash
        5. Save checkpoint, write report, fire events
        6. If streaming: stop StreamManager
        """

    def _recover_state(self) -> None:
        """On startup, check scheduler_state.json for interrupted sessions."""
```

#### session.py — TrainingSession

```python
@dataclass
class TrainingSession:
    session_id: str              # UUID
    game_id: str                 # e.g., 'snake', 'chess'
    algorithm: str               # e.g., 'ppo', 'rainbow'
    duration_episodes: int       # How many episodes to train
    device: str                  # 'auto', 'cuda', 'cpu'
    num_envs: int                # Parallel environments
    stream_enabled: bool         # Whether to stream this session
    stream_title: str            # Auto-generated or custom
    opponent_config: Optional[dict]  # For board games
    scheduled_start: datetime    # When to start (None = ASAP)
    recurring: Optional[str]     # 'daily', 'weekly', None
    model_path: Optional[str]    # Resume from checkpoint
    notes: str                   # User notes

@dataclass
class SessionReport:
    session_id: str
    game_id: str
    algorithm: str
    episodes_completed: int
    best_reward: float
    avg_reward: float
    elapsed_seconds: float
    achievements_earned: List[str]
    checkpoint_path: str
    timestamp: str
```

#### calendar_store.py — CalendarStore

Persists to `config/training_schedule.json`:
```json
{
  "sessions": [
    {
      "session_id": "abc-123",
      "game_id": "snake",
      "algorithm": "ppo",
      "duration_episodes": 5000,
      "stream_enabled": true,
      "scheduled_start": "2026-03-18T02:00:00",
      "recurring": "daily"
    }
  ],
  "completed": [...],
  "scheduler_state": {
    "current_session_id": null,
    "last_completed": "2026-03-17T23:45:00"
  }
}
```

#### calendar_ui.py — CalendarWidget

Tkinter widget for the launcher:
- Weekly grid view (7 columns x 24 rows)
- Session blocks rendered as colored rectangles (color-coded by game)
- Click empty slot -> opens session editor form
- Click existing session -> edit or delete
- Drag edges to resize duration
- "Quick Schedule" presets dropdown:
  - "Overnight All Games" — 6 sessions, 500 ep each, cycles all built-ins
  - "Deep Train Single Game" — 1 session, 5000 ep
  - "DT Generalist Run" — collect phase + train phase
- "Run Now" button starts next pending session immediately

---

### System B: Opponent Manager (`src/opponents/`)

Pluggable opponent slot for all board games. Replaces inline
`random.choice(moves)` with a polymorphic interface.

```
src/opponents/
    __init__.py
    base_opponent.py      # Abstract base class
    random_opponent.py    # Current behavior, extracted
    minimax_opponent.py   # Classic minimax with configurable depth
    model_opponent.py     # Loads any trained checkpoint as opponent
    human_opponent.py     # Keyboard/mouse input, blocks until move
```

#### base_opponent.py

```python
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
        """0=random, 1-3=minimax, 4-5=trained model."""

    @abstractmethod
    def pick_action(self, board_state: dict) -> int:
        """Choose an action given the current board state.

        Args:
            board_state: Dict with keys:
                'board': np.ndarray — current board array
                'valid_actions': List[int] — legal moves
                'game_id': str — which game this is
                'turn': int — whose turn (1=agent, 2=opponent)

        Returns:
            int action index from valid_actions.
        """

    def reset(self) -> None:
        """Called at episode start. Override for stateful opponents."""
        pass
```

#### minimax_opponent.py

```python
class MinimaxOpponent(BaseOpponent):
    """Minimax with alpha-beta pruning.

    Depth controls difficulty:
        depth=1: Easy (looks 1 move ahead)
        depth=3: Medium (decent play)
        depth=5: Hard (strong play, slow for Chess)
    """
    def __init__(self, depth: int = 3, game_id: str = 'tictactoe'):
        self.depth = depth
        self.game_id = game_id
        # Game-specific evaluation functions registered here
        self._evaluators = {
            'tictactoe': self._eval_tictactoe,
            'connect4': self._eval_connect4,
            'checkers': self._eval_checkers,
            'chess': self._eval_chess,
        }
```

Evaluation functions per game:
- **TicTacToe**: Count potential winning lines for each player
- **Connect4**: Score based on 2/3/4-in-a-row counts, center column bonus
- **Checkers**: Material count (pieces + 1.5x kings), mobility
- **Chess**: Material value (standard piece values), mobility, king safety

#### model_opponent.py

```python
class ModelOpponent(BaseOpponent):
    """Loads a trained RL checkpoint as the opponent.

    Supports loading any algorithm's checkpoint:
    - DQN/Rainbow: load network, run forward pass
    - PPO/A2C: load SB3 model, call predict()
    - NEAT: load genome, create network
    """
    def __init__(self, checkpoint_path: str, algorithm: str):
        self.checkpoint_path = checkpoint_path
        self.algorithm = algorithm
        self._model = self._load_model()

    @classmethod
    def from_difficulty(cls, game_id: str, level: str) -> 'ModelOpponent':
        """Load a model by difficulty label.

        Scans models/<game_id>/<algo>/ for checkpoints and picks
        one based on training progress:
            'easy': earliest checkpoint (least trained)
            'medium': middle checkpoint
            'hard': best checkpoint (most trained)
        """
```

#### human_opponent.py

```python
class HumanOpponent(BaseOpponent):
    """Blocks until the human makes a move via pygame input.

    Rendering: the board game's render() output is displayed,
    and pygame mouse events are mapped to valid actions.

    For Chess/Checkers: click source piece, then click destination.
    For Connect4: click a column.
    For TicTacToe: click a cell.
    """
    def pick_action(self, board_state: dict) -> int:
        """Wait for human input. Highlights valid moves on the board."""
```

#### Board Game Refactor

Each board game's `step()` changes from:
```python
# Old: inline random opponent
opp_moves = self._get_valid_moves(2)
move = random.choice(opp_moves)
```
to:
```python
# New: pluggable opponent
opp_state = {
    'board': self.board.copy(),
    'valid_actions': self._get_valid_moves(2),
    'game_id': self.game_id,
    'turn': 2,
}
move = self.opponent.pick_action(opp_state)
```

Default: `RandomOpponent()` if no opponent specified (backwards compatible).

**Human demonstration collection**: When HumanOpponent is active, both the
human's moves and the agent's moves are recorded into the DT experience store
via the existing `_dt_record_step` pipeline.

---

### System C: Achievement & Event Bus (`src/achievements/`)

Lightweight publish/subscribe event system that drives achievements,
recording milestones, stream overlay popups, and chat bot announcements.

```
src/achievements/
    __init__.py
    event_bus.py          # Publish/subscribe hub
    achievement_manager.py # Checks conditions, awards, persists
    definitions.py        # All achievement definitions
    profile.py            # Agent profile card + trainer progression
```

#### event_bus.py

```python
class EventBus:
    """Simple pub/sub for training events.

    Events are dicts with a 'type' key and arbitrary data:
        {'type': 'episode_complete', 'episode': 150, 'reward': 42.5, ...}
        {'type': 'new_best_reward', 'reward': 98.0, 'game_id': 'snake'}
        {'type': 'achievement_earned', 'name': 'First Win', 'level': 'agent'}
    """
    def subscribe(self, event_type: str, callback: Callable) -> None: ...
    def publish(self, event: dict) -> None: ...
```

Subscribers:
- `AchievementManager` — checks conditions, awards achievements
- `RecordingManager` — writes milestone markers to recording metadata
- `OverlayManager` — flashes achievement popup on stream
- `ChatBot` — announces milestones in Twitch/YouTube chat
- `ProfileManager` — updates agent/trainer profile stats

#### definitions.py

```python
# Agent Achievements (per model)
AGENT_ACHIEVEMENTS = [
    Achievement('first_steps', 'First Steps', 'Complete 1 episode',
                condition=lambda s: s['episodes'] >= 1),
    Achievement('century', 'Century', 'Complete 100 episodes',
                condition=lambda s: s['episodes'] >= 100),
    Achievement('marathon', 'Marathon', 'Complete 10,000 episodes',
                condition=lambda s: s['episodes'] >= 10000),
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
    Achievement('perfectionist', 'Perfectionist', 'Win with no invalid moves',
                condition=lambda s: s.get('perfect_game', False)),
    Achievement('generalist', 'Generalist', 'Positive avg reward in 3+ games',
                condition=lambda s: s.get('positive_games', 0) >= 3),
]

# Trainer Achievements (global)
TRAINER_ACHIEVEMENTS = [
    Achievement('getting_started', 'Getting Started', 'Train your first agent',
                condition=lambda s: s['total_sessions'] >= 1),
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
    Achievement('streamer', 'Streamer', 'Complete a streamed training session',
                condition=lambda s: s.get('streamed_sessions', 0) >= 1),
]
```

Persisted in `config/achievements.json`:
```json
{
  "trainer": {
    "earned": [
      {"id": "getting_started", "timestamp": "2026-03-17T14:30:00"}
    ],
    "stats": {
      "total_sessions": 12,
      "algorithms_used": ["ppo", "dqn", "rainbow"],
      "games_played": ["snake", "chess", "tetris"]
    }
  },
  "agents": {
    "models/snake/ppo/best_model": {
      "earned": ["first_steps", "century", "first_win"],
      "stats": {"episodes": 5000, "wins": 3200, "win_streak": 14}
    }
  }
}
```

#### profile.py — Agent Personality Profiles

```python
@dataclass
class AgentProfile:
    """Generated report card for a trained agent."""
    agent_id: str               # Checkpoint path
    game_id: str
    algorithm: str
    play_style: str             # "Aggressive Rusher", "Cautious Explorer", etc.
    radar_scores: Dict[str, float]  # 0-100 per dimension
    stats: Dict[str, Any]       # Total episodes, best reward, etc.
    achievements: List[str]     # Achievement IDs earned
    training_time: float        # Total seconds trained
    timestamp: str

    @classmethod
    def generate(cls, metadata: dict, metrics_history: list,
                 achievements: list) -> 'AgentProfile':
        """Analyze training data to generate the profile."""
```

Radar dimensions by game category:
- **Arcade/Platformer**: Speed, Consistency, Exploration, Survival, Score Efficiency
- **Board**: Aggression, Defense, Opening Strength, Endgame, Adaptability
- **Puzzle (Tetris)**: Speed, Line Efficiency, Recovery, T-Spin Rate, Height Management

Play style classification from action distribution:
- High variance + fast actions = "Aggressive Rusher"
- Low variance + cautious actions = "Methodical Planner"
- Balanced distribution = "Adaptive Generalist"
- etc.

---

## Feature Designs

### P1: Training Schedule / Auto-Pilot with Calendar

Uses: System A (Scheduler)

**Launcher integration:**
- New "Schedule" button in launcher sidebar/toolbar
- Opens calendar overlay with weekly grid
- Session editor form for creating/editing sessions
- Quick presets dropdown for common training patterns
- "Run Now" button for immediate execution

**Stream scheduling:**
- Per-session toggle: "Stream this session"
- Auto-generates stream title from session config
- Pre-stream countdown overlay (30 seconds "Starting Soon" screen)
- Auto-starts/stops ffmpeg RTMP stream per session boundaries

**Crash recovery:**
- `config/scheduler_state.json` tracks current session
- On restart, scheduler checks for interrupted sessions
- Option to resume or skip interrupted session

---

### P2: Achievement System

Uses: System C (Event Bus + Achievement Manager)

**Dashboard popup:**
- Achievement slides in from top-right corner
- Semi-transparent panel with icon + name + description
- Amber accent border matching stream overlay style
- Fades after 3 seconds
- Queue system: multiple achievements display sequentially

**Stream overlay popup:**
- Mirrors dashboard popup on the stream feed
- Achievement banner flashes in top bar area
- Auto-announced in chat if chat bot is running

**Launcher panel:**
- "Achievements" button opens grid view
- Earned achievements shown in full color, locked ones grayed out
- Progress bars for partially-met conditions
- Two tabs: "Agent" (per model) and "Trainer" (global)

---

### P3: Tournament Mode

Uses: System B (Opponent Manager), System C (Events)

**Tournament structure:**
- `src/tournament/tournament_runner.py` — manages bracket/round-robin
- `src/tournament/tournament_ui.py` — launcher setup and bracket display

**Launcher setup:**
- Select game (board games only)
- Add participants: browse checkpoints, add bot tiers, add human slots
- Choose format: Round Robin or Single Elimination
- Set games per matchup (default 10)
- "Start Tournament" button

**Dashboard during tournament:**
- Game view on left, bracket/standings on right
- Current matchup header: "PPO Agent vs Rainbow Agent — Game 3/10"
- Standings table with W/L/D columns
- Match result animations (winner highlight)

**Stream overlay:**
- Matchup banner in top bar
- Score overlay for current matchup
- Bracket graphic between matchups

**Event integration:**
- Tournament results fire on event bus
- Achievements: "Tournament Director", "Tournament Champion"
- Recording markers for match starts/ends and tournament winner

---

### P4: Pluggable Opponents

Uses: System B (Opponent Manager)

See System B section above for full design. Key integration points:

- Board game `create_env()` accepts `opponent=` parameter
- Launcher dropdown for opponent selection in board game panel
- "Auto-difficulty" checkbox enables curriculum (P6)
- Human play mode with pygame click-to-move input
- Human vs Human mode: both players use HumanOpponent
- All human moves collected into DT experience store

---

### P5: Recording Milestones & Replay System

Uses: System C (Event Bus)

**Milestone marker format:**
```json
{
  "recording": "training_20260317_140000.mp4",
  "markers": [
    {"frame": 14523, "time_sec": 484.1, "type": "new_best_reward",
     "value": 142.5, "label": "New Best: 142.5"},
    {"frame": 28901, "time_sec": 963.4, "type": "achievement",
     "name": "Winning Streak 10", "label": "Achievement: On Fire"},
    {"frame": 45002, "time_sec": 1500.1, "type": "stage_cleared",
     "stage": "1-2", "label": "Stage 1-2 Cleared"}
  ]
}
```

**Recording system changes:**
- `RecordingManager` subscribes to event bus
- On milestone event: appends marker to in-memory list
- On recording stop: writes `<recording_name>_markers.json` alongside MP4

**Best episode clips:**
- When `new_best_reward` fires, save the last N seconds as a separate clip
- Stored in `recordings/highlights/`
- Linked from marker metadata

**Stream overlay milestone flash:**
- When a milestone fires, overlay shows amber banner for 3 seconds
- Banner text from marker label: "NEW BEST: 142.5"

**Replay viewer in launcher:**
- List recordings with duration, game, algorithm
- Timeline scrubber with colored marker dots
- Click marker to jump to timestamp
- "Export Clip" saves segment as standalone MP4

---

### P6: Difficulty Curriculum for Board Games

Uses: System B (Opponent Manager)

**Tier progression table:**

| Tier | Opponent | Promote at | Demote at |
|------|----------|-----------|----------|
| 1 | Random | >70% win rate (50 games) | — |
| 2 | Minimax depth=1 | >70% win rate (50 games) | <30% win rate |
| 3 | Minimax depth=3 | >70% win rate (50 games) | <30% win rate |
| 4 | Trained model (early) | >60% win rate (50 games) | <25% win rate |
| 5 | Trained model (best) | — (max) | <25% win rate |

**Implementation:**
- `src/opponents/difficulty_curriculum.py` wraps opponent selection
- Tracks rolling win rate over last 50 games
- Swaps opponent when promotion/demotion triggers
- Dashboard shows current tier + progress bar toward promotion
- Stream overlay shows difficulty badge
- Tier changes fire achievement events

---

### P7: Agent Personality Profiles

Uses: System C (Profile)

**Generation triggers:**
- Automatically after training session completes
- Manual "Generate Profile" button in launcher agent gallery

**Display:**
- Launcher "Agent Gallery" panel
- Profile card: radar chart, play style label, stats, achievement badges
- "Compare" mode: select two agents, overlay radar charts side-by-side
- Profile saved as `models/<game>/<algo>/profile.json`

---

### P8: NEAT Evolution Visualizer

Standalone feature (no system dependency).

**Network renderer:**
- `src/visualization/neat_visualizer.py`
- Input: NEAT genome (nodes + connections)
- Output: pygame Surface with node-link diagram
- Layout: input nodes at bottom, hidden in middle, output at top
- Node size proportional to node count (scales with complexity)
- Edge color: green=positive weight, red=negative, opacity=|weight|
- Animate: redraw each generation, nodes appear/disappear

**Dashboard integration:**
- Toggle with 'N' key: swaps game view panel for network view
- Generation counter, species count, best fitness shown alongside
- NEAT trainer passes best genome to dashboard via callback

---

### P9: Live Chat Integration (Autonomous)

**Architecture:**
- `src/streaming/chat_bot.py`
- Connects to Twitch IRC and/or YouTube Live Chat API
- Background thread, non-blocking

**Commands:**

| Command | Action |
|---------|--------|
| `!game <name>` | Queue game switch for next session |
| `!algorithm <name>` | Queue algorithm switch |
| `!stats` | Post current training stats |
| `!tournament` | Start quick tournament |
| `!challenge` | Request human-vs-agent game |
| `!achievements` | Post recent achievements |
| `!schedule` | Post upcoming sessions |
| `!help` | List available commands |

**Autonomous mode:**
- Auto-greet new chatters with status message
- Periodic progress posts every 15 minutes
- Achievement announcements auto-posted
- Configurable greeting message in launcher
- Enable/disable per setting

---

### P10: Highlight Reel Generator

Uses: P5 (Recording Milestones)

**Auto-compilation after session:**
1. Scan recording markers
2. Rank by significance: achievements > new bests > stage clears
3. Extract 5-second clips around top N markers
4. Add title card overlay on each clip
5. Concatenate with 1-second fade transitions
6. Output to `recordings/highlights/reel_YYYYMMDD.mp4`

**Manual mode in launcher:**
- Select markers to include from replay viewer
- Set clip duration per marker
- "Generate Reel" button

---

### P11: Model Zoo / Community Hub

**Export format (.agent ZIP):**
```
my_snake_agent.agent
    weights.pth           # Model checkpoint
    metadata.json         # Training config, game, algorithm, episodes
    profile.json          # Personality profile + radar scores
    achievements.json     # Earned achievements
    best_replay.mp4       # Best episode recording (if available)
    thumbnail.png         # Last frame of best episode
```

**Launcher integration:**
- "Export Agent" button on any trained model
- "Import Agent" loads .agent file, extracts to models/
- Imported agents available as tournament participants + opponents

---

### P12: Multi-Agent Cooperative Games

**New adapter category:** `"cooperative"`

**Example: Dual Snake**
- Two snakes on shared board, shared score
- Both take actions simultaneously each step
- Environment returns joint observation + joint reward
- Adapter specifies `num_agents: 2`

**Training modes:**
- Two separate models (one per snake)
- Single model with multi-head output (both snakes from one brain)
- Human + AI cooperation (one snake human-controlled)

---

### P13: Natural Language Training Commands

**Launcher text input:**
- Text field: "Describe what you want to train..."
- Keyword parser maps phrases to config:
  - "fast" / "speed" -> high speed_bonus_multiplier
  - "aggressive" -> low death_penalty, attack-focused
  - "overnight" -> create scheduled session, high episode count
  - "all games" -> cycle through built-ins
  - "generalist" -> DT pipeline
- Shows interpreted config for user confirmation
- "Looks good? [Start] [Edit]" buttons

---

## Implementation Priority

| Priority | Feature | Depends On | Estimated Complexity |
|----------|---------|-----------|---------------------|
| P1 | Training Schedule / Calendar | — | High |
| P2 | Achievement System | — | Medium |
| P3 | Tournament Mode | P4 (opponents) | Medium |
| P4 | Pluggable Opponents | — | Medium |
| P5 | Recording Milestones | P2 (events) | Medium |
| P6 | Difficulty Curriculum | P4 (opponents) | Low |
| P7 | Agent Profiles | P2 (achievements) | Medium |
| P8 | NEAT Visualizer | — | Medium |
| P9 | Chat Bot | P1 (scheduler) | Medium |
| P10 | Highlight Reel | P5 (milestones) | Low |
| P11 | Model Zoo | P7 (profiles) | Low |
| P12 | Cooperative Games | P4 (opponents) | High |
| P13 | NL Commands | P1 (scheduler) | Low |

**Optimal build order (respecting dependencies):**

Phase 5: Foundations
  - System C: Event Bus + Achievement Manager
  - System B: Opponent Manager + board game refactor
  - System A: Session Scheduler + Calendar UI

Phase 6: Core Features
  - P2: Achievement System (wires into event bus)
  - P4: Pluggable Opponents (uses opponent manager)
  - P1: Training Schedule (uses scheduler + calendar UI)
  - P3: Tournament Mode (uses opponents + events)

Phase 7: Content & Polish
  - P5: Recording Milestones (uses events)
  - P6: Difficulty Curriculum (uses opponents)
  - P7: Agent Profiles (uses achievements)
  - P8: NEAT Visualizer (standalone)

Phase 8: Community & Engagement
  - P9: Chat Bot (uses scheduler + events)
  - P10: Highlight Reel (uses milestones)
  - P11: Model Zoo (uses profiles)
  - P12: Cooperative Games (uses opponents)
  - P13: NL Commands (uses scheduler)

---

## Testing Strategy

Each feature gets:
1. Unit tests for core logic (opponent pick_action, achievement conditions, etc.)
2. Integration tests verifying event bus wiring
3. Smoke tests for UI components (launcher panels render without crash)
4. Existing 493 tests must continue passing after every change

---

## Key Files (New)

| Path | Purpose |
|------|---------|
| `src/scheduler/scheduler.py` | Session scheduler engine |
| `src/scheduler/calendar_store.py` | Schedule persistence |
| `src/scheduler/session.py` | Session/report dataclasses |
| `src/scheduler/calendar_ui.py` | Tkinter calendar widget |
| `src/opponents/base_opponent.py` | Opponent interface |
| `src/opponents/random_opponent.py` | Random opponent (extracted) |
| `src/opponents/minimax_opponent.py` | Minimax with depth control |
| `src/opponents/model_opponent.py` | Trained model as opponent |
| `src/opponents/human_opponent.py` | Human player input |
| `src/opponents/difficulty_curriculum.py` | Auto-advancing difficulty |
| `src/achievements/event_bus.py` | Pub/sub event system |
| `src/achievements/achievement_manager.py` | Achievement logic |
| `src/achievements/definitions.py` | Achievement definitions |
| `src/achievements/profile.py` | Agent profile generation |
| `src/tournament/tournament_runner.py` | Tournament execution |
| `src/tournament/tournament_ui.py` | Tournament launcher UI |
| `src/streaming/chat_bot.py` | Twitch/YouTube chat bot |
| `src/visualization/neat_visualizer.py` | NEAT network renderer |
| `config/training_schedule.json` | Persisted schedule |
| `config/achievements.json` | Earned achievements |
