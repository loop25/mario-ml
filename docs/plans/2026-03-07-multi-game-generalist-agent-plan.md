# Multi-Game Generalist Agent — Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the plugin registry foundation so the project supports multiple games behind a unified adapter interface, with Snake and Connect Four as the first built-in games, Mario migrated to the new system, and a new A2C algorithm.

**Architecture:** Every game implements `BaseGameAdapter` (identity, env creation, metrics, rewards). A `GameRegistry` auto-discovers adapters from `games/builtin/`, `games/retro/`, and `games/user/` folders. A universal `TimeRewardWrapper` applies per-game reward shaping. The CLI and launcher gain `--game` selection. All 39 existing tests continue to pass.

**Tech Stack:** Python 3.11, gym 0.26, PyTorch, stable-baselines3 (A2C), pygame, tkinter, pytest

**Design doc:** `docs/plans/2026-03-07-multi-game-generalist-agent-design.md`

---

## Phase 1 Overview

| Task | Component | Creates/Modifies |
|------|-----------|-----------------|
| 1 | Directory scaffold | `games/` tree + `__init__.py` files |
| 2 | Reward config dataclasses | `games/reward_config.py` |
| 3 | Base game adapter | `games/base_adapter.py` |
| 4 | Game registry | `games/registry.py` |
| 5 | Time reward wrapper | `src/rewards/time_reward.py` |
| 6 | Universal env factory | `src/environment/universal_env.py` |
| 7 | Mario adapter migration | `games/retro/mario/` |
| 8 | Snake game + adapter | `games/builtin/snake/` |
| 9 | Connect Four game + adapter | `games/builtin/connect4/` |
| 10 | A2C trainer | `src/algorithms/a2c/` |
| 11 | CLI `--game` flag | `main.py` |
| 12 | Launcher game selection | `launcher.py` |
| 13 | Integration tests | `tests/test_integration_multigame.py` |

---

### Task 1: Directory Scaffold

**Files:**
- Create: `games/__init__.py`
- Create: `games/builtin/__init__.py`
- Create: `games/builtin/snake/__init__.py`
- Create: `games/builtin/connect4/__init__.py`
- Create: `games/retro/__init__.py`
- Create: `games/retro/mario/__init__.py`
- Create: `games/user/__init__.py`
- Create: `games/user/_template/__init__.py`
- Create: `src/rewards/__init__.py`

**Step 1: Create the directory tree**

```bash
mkdir -p games/builtin/snake games/builtin/connect4 games/retro/mario games/user/_template src/rewards
```

**Step 2: Create all `__init__.py` files**

Create empty `__init__.py` in every new package directory:

```python
# games/__init__.py
# games/builtin/__init__.py
# games/builtin/snake/__init__.py
# games/builtin/connect4/__init__.py
# games/retro/__init__.py
# games/retro/mario/__init__.py
# games/user/__init__.py
# games/user/_template/__init__.py
# src/rewards/__init__.py
```

All files are empty (just the standard Python package marker).

**Step 3: Verify structure**

Run: `python -c "import games; import games.builtin; import games.retro; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add games/ src/rewards/__init__.py
git commit -m "scaffold: create games/ plugin directory tree and src/rewards/"
```

---

### Task 2: Reward Config Dataclasses

**Files:**
- Create: `games/reward_config.py`
- Create: `tests/test_reward_config.py`

**Step 1: Write the failing test**

```python
# tests/test_reward_config.py
"""Tests for reward config dataclasses."""
import pytest
from games.reward_config import StandardMetrics, RewardConfig, ActionSpaceInfo, TokenConfig


class TestStandardMetrics:
    def test_create_with_defaults(self):
        m = StandardMetrics(progress=0.5, score=100.0, completed=False, time_elapsed=10.0)
        assert m.progress == 0.5
        assert m.score == 100.0
        assert m.completed is False
        assert m.time_elapsed == 10.0


class TestRewardConfig:
    def test_default_values(self):
        rc = RewardConfig()
        assert rc.time_penalty_per_second == 0.01
        assert rc.completion_bonus == 100.0
        assert rc.death_penalty == -15.0
        assert rc.idle_penalty_per_second == 0.005
        assert rc.speed_bonus_multiplier == 1.5
        assert rc.par_time_seconds == 120.0

    def test_custom_values(self):
        rc = RewardConfig(time_penalty_per_second=0.0, completion_bonus=50.0)
        assert rc.time_penalty_per_second == 0.0
        assert rc.completion_bonus == 50.0


class TestActionSpaceInfo:
    def test_create(self):
        info = ActionSpaceInfo(num_actions=4, action_labels=['up', 'down', 'left', 'right'])
        assert info.num_actions == 4
        assert len(info.action_labels) == 4


class TestTokenConfig:
    def test_create(self):
        tc = TokenConfig(
            obs_resolution=(84, 84),
            obs_channels=1,
            action_vocab_size=7,
            game_token_id=42,
        )
        assert tc.obs_resolution == (84, 84)
        assert tc.obs_channels == 1
        assert tc.action_vocab_size == 7
        assert tc.game_token_id == 42
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reward_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'games.reward_config'`

**Step 3: Write the implementation**

```python
# games/reward_config.py
"""
Shared dataclasses for the game adapter system.

These define the universal contracts between game adapters and the framework:
- StandardMetrics: what every game reports after each step
- RewardConfig: how the framework shapes rewards per game
- ActionSpaceInfo: describes the action space for the launcher/algorithms
- TokenConfig: how to tokenize obs/actions for the Decision Transformer
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class StandardMetrics:
    """Universal metrics that every game maps to."""
    progress: float        # 0.0 to 1.0, how far through the level/game
    score: float           # Game-specific score (coins, points, pieces)
    completed: bool        # Did the agent finish/win this episode?
    time_elapsed: float    # Seconds spent this episode


@dataclass
class RewardConfig:
    """Reward shaping parameters. The framework applies these automatically."""
    time_penalty_per_second: float = 0.01
    completion_bonus: float = 100.0
    death_penalty: float = -15.0
    idle_penalty_per_second: float = 0.005
    speed_bonus_multiplier: float = 1.5
    par_time_seconds: float = 120.0


@dataclass
class ActionSpaceInfo:
    """Describes the game's action space for the launcher and algorithms."""
    num_actions: int
    action_labels: List[str]


@dataclass
class TokenConfig:
    """How to tokenize this game for the Decision Transformer."""
    obs_resolution: Tuple[int, int]
    obs_channels: int
    action_vocab_size: int
    game_token_id: int
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reward_config.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add games/reward_config.py tests/test_reward_config.py
git commit -m "feat: add reward config dataclasses (StandardMetrics, RewardConfig, etc.)"
```

---

### Task 3: Base Game Adapter

**Files:**
- Create: `games/base_adapter.py`
- Create: `tests/test_base_adapter.py`

**Step 1: Write the failing test**

```python
# tests/test_base_adapter.py
"""Tests for BaseGameAdapter abstract interface."""
import pytest
import numpy as np
import gym
from gym.spaces import Box, Discrete

from games.base_adapter import BaseGameAdapter
from games.reward_config import (
    StandardMetrics, RewardConfig, ActionSpaceInfo, TokenConfig,
)


class FakeEnv(gym.Env):
    """Minimal gym env for testing."""
    def __init__(self):
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = Discrete(4)

    def reset(self):
        return np.zeros((84, 84, 1), dtype=np.uint8)

    def step(self, action):
        obs = np.zeros((84, 84, 1), dtype=np.uint8)
        return obs, 1.0, False, {'score': 10}

    def render(self, mode='rgb_array'):
        return np.zeros((84, 84, 3), dtype=np.uint8)


class StubAdapter(BaseGameAdapter):
    """Concrete adapter for testing the abstract interface."""

    @property
    def name(self):
        return 'Stub Game'

    @property
    def game_id(self):
        return 'stub'

    @property
    def category(self):
        return 'arcade'

    @property
    def description(self):
        return 'A test stub game.'

    def create_env(self, **kwargs):
        return FakeEnv()

    def get_action_space_info(self):
        return ActionSpaceInfo(num_actions=4, action_labels=['up', 'down', 'left', 'right'])

    def get_observation_shape(self):
        return (84, 84, 1)

    def extract_metrics(self, info, episode_time):
        return StandardMetrics(
            progress=0.5,
            score=info.get('score', 0),
            completed=False,
            time_elapsed=episode_time,
        )


class TestBaseGameAdapter:
    def test_identity_properties(self):
        adapter = StubAdapter()
        assert adapter.name == 'Stub Game'
        assert adapter.game_id == 'stub'
        assert adapter.category == 'arcade'
        assert adapter.description == 'A test stub game.'

    def test_create_env_returns_gym_env(self):
        adapter = StubAdapter()
        env = adapter.create_env()
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        env.close()

    def test_get_action_space_info(self):
        adapter = StubAdapter()
        info = adapter.get_action_space_info()
        assert info.num_actions == 4
        assert info.action_labels[0] == 'up'

    def test_get_observation_shape(self):
        adapter = StubAdapter()
        shape = adapter.get_observation_shape()
        assert shape == (84, 84, 1)

    def test_extract_metrics(self):
        adapter = StubAdapter()
        metrics = adapter.extract_metrics({'score': 42}, episode_time=5.0)
        assert isinstance(metrics, StandardMetrics)
        assert metrics.score == 42
        assert metrics.time_elapsed == 5.0

    def test_default_reward_config(self):
        adapter = StubAdapter()
        rc = adapter.get_reward_config()
        assert isinstance(rc, RewardConfig)
        assert rc.time_penalty_per_second == 0.01

    def test_default_curriculum_is_none(self):
        adapter = StubAdapter()
        assert adapter.get_curriculum() is None

    def test_default_game_specific_options_empty(self):
        adapter = StubAdapter()
        assert adapter.get_game_specific_options() == {}

    def test_default_token_config(self):
        adapter = StubAdapter()
        tc = adapter.get_token_config()
        assert isinstance(tc, TokenConfig)
        assert tc.obs_resolution == (84, 84)
        assert tc.obs_channels == 1
        assert tc.action_vocab_size == 4

    def test_default_needs_sb3_compat_false(self):
        adapter = StubAdapter()
        assert adapter.needs_sb3_compat() is False

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseGameAdapter()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_base_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'games.base_adapter'`

**Step 3: Write the implementation**

```python
# games/base_adapter.py
"""
Abstract base class for all game adapters.

Every game in the platform (built-in, retro, or user-contributed)
must implement this interface. It defines how the framework
discovers, creates, wraps, and measures each game.

The adapter pattern keeps game-specific logic (env creation,
metrics extraction, reward tuning) encapsulated within each
game module, while the framework handles everything else
(training loops, visualization, checkpointing).
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import gym

from games.reward_config import (
    StandardMetrics,
    RewardConfig,
    ActionSpaceInfo,
    TokenConfig,
)


class BaseGameAdapter(ABC):
    """Abstract interface that every game must implement."""

    # ---- Identity (required properties) ----

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name, e.g. 'Super Mario Bros'."""

    @property
    @abstractmethod
    def game_id(self) -> str:
        """Unique identifier, e.g. 'mario'. Used as folder name."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Game category: 'platformer', 'puzzle', 'board', 'arcade', 'rpg'."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description for the launcher GUI."""

    # ---- Environment (required methods) ----

    @abstractmethod
    def create_env(self, **kwargs) -> gym.Env:
        """Create and return a ready-to-use Gym environment.

        Kwargs may include game-specific options (world, stage, difficulty).
        The returned env should NOT have reward shaping applied -- the
        framework handles that via get_reward_config().

        Returns:
            gym.Env with observations suitable for ML.
        """

    @abstractmethod
    def get_action_space_info(self) -> ActionSpaceInfo:
        """Describe the discrete action space."""

    @abstractmethod
    def get_observation_shape(self) -> Tuple[int, ...]:
        """Shape of preprocessed observations, e.g. (84, 84, 4)."""

    # ---- Metrics Translation (required) ----

    @abstractmethod
    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        """Map game-specific info dict to universal StandardMetrics.

        Args:
            info: The info dict from env.step().
            episode_time: Seconds elapsed in this episode.

        Returns:
            StandardMetrics with progress, score, completed, time_elapsed.
        """

    # ---- Reward Shaping (optional override) ----

    def get_reward_config(self) -> RewardConfig:
        """Return reward shaping parameters.

        Override to customize. Defaults tuned for action games.
        Board games should set time_penalty_per_second=0.
        """
        return RewardConfig()

    # ---- Optional Features ----

    def get_curriculum(self) -> Optional[object]:
        """Return a CurriculumManager for multi-level games.

        Single-screen games (Snake, Tetris) return None.
        Multi-level games (Mario, Sonic) return a curriculum.
        """
        return None

    def get_human_render_frame(self, env) -> Optional[np.ndarray]:
        """Return an RGB frame for dashboard display.

        Defaults to env.render(). Override if your game needs
        custom rendering for the dashboard.
        """
        try:
            return env.render(mode='rgb_array')
        except TypeError:
            return env.render()

    def get_game_specific_options(self) -> dict:
        """Return dict of option_name -> (type, default, description)
        for the launcher GUI to display game-specific settings.

        Example: {'world': (int, 1, 'World number 1-8')}
        """
        return {}

    # ---- For Generalist Agent (optional) ----

    def get_token_config(self) -> TokenConfig:
        """How to tokenize this game for the Decision Transformer."""
        obs_shape = self.get_observation_shape()
        return TokenConfig(
            obs_resolution=(obs_shape[0], obs_shape[1]),
            obs_channels=obs_shape[2] if len(obs_shape) > 2 else 1,
            action_vocab_size=self.get_action_space_info().num_actions,
            game_token_id=hash(self.game_id) % 1024,
        )

    # ---- For SB3 Algorithms (optional) ----

    def needs_sb3_compat(self) -> bool:
        """Whether this game needs the SB3CompatWrapper.

        True for old-gym-API environments (like gym-super-mario-bros).
        False for gymnasium-native environments (built-in games).
        """
        return False
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_base_adapter.py -v`
Expected: PASS (all 11 tests)

**Step 5: Commit**

```bash
git add games/base_adapter.py tests/test_base_adapter.py
git commit -m "feat: add BaseGameAdapter abstract interface"
```

---

### Task 4: Game Registry

**Files:**
- Create: `games/registry.py`
- Create: `tests/test_registry.py`

**Step 1: Write the failing test**

```python
# tests/test_registry.py
"""Tests for GameRegistry discovery and lookup."""
import os
import pytest

from games.registry import GameRegistry
from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics, ActionSpaceInfo


class TestGameRegistry:
    def test_register_and_get(self):
        """Manually register an adapter and retrieve it."""
        registry = GameRegistry()

        # Create a stub via the test helper
        from tests.test_base_adapter import StubAdapter
        adapter = StubAdapter()

        registry.register(adapter)
        assert registry.get_game('stub') is adapter

    def test_get_unknown_raises(self):
        registry = GameRegistry()
        with pytest.raises(KeyError):
            registry.get_game('nonexistent')

    def test_list_games(self):
        registry = GameRegistry()
        from tests.test_base_adapter import StubAdapter
        registry.register(StubAdapter())
        games = registry.list_games()
        assert len(games) == 1
        assert games[0].game_id == 'stub'

    def test_list_by_category(self):
        registry = GameRegistry()
        from tests.test_base_adapter import StubAdapter
        registry.register(StubAdapter())
        arcade_games = registry.list_by_category('arcade')
        assert len(arcade_games) == 1
        board_games = registry.list_by_category('board')
        assert len(board_games) == 0

    def test_discover_finds_nothing_in_empty_dirs(self, tmp_path):
        """discover() scans directories without crashing on empty folders."""
        registry = GameRegistry()
        # Create empty game dirs
        (tmp_path / 'builtin').mkdir()
        (tmp_path / 'retro').mkdir()
        (tmp_path / 'user').mkdir()
        registry.discover(base_path=str(tmp_path))
        assert len(registry.list_games()) == 0

    def test_list_game_ids(self):
        registry = GameRegistry()
        from tests.test_base_adapter import StubAdapter
        registry.register(StubAdapter())
        ids = registry.list_game_ids()
        assert 'stub' in ids
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'games.registry'`

**Step 3: Write the implementation**

```python
# games/registry.py
"""
Game registry — discovers and manages game adapters.

The registry scans three folders for game adapters:
  - games/builtin/   (games we build: Snake, Tetris, etc.)
  - games/retro/     (emulated ROM games: Mario, Sonic, etc.)
  - games/user/      (user-contributed game plugins)

Each subfolder that contains an adapter.py with a class inheriting
from BaseGameAdapter is auto-discovered and registered.
"""
import importlib
import importlib.util
import os
import sys
from typing import Dict, List, Optional

from games.base_adapter import BaseGameAdapter


class GameRegistry:
    """Discovers and manages game adapters."""

    def __init__(self):
        self._adapters: Dict[str, BaseGameAdapter] = {}

    def register(self, adapter: BaseGameAdapter) -> None:
        """Manually register a game adapter.

        Args:
            adapter: An instance of a BaseGameAdapter subclass.
        """
        self._adapters[adapter.game_id] = adapter

    def get_game(self, game_id: str) -> BaseGameAdapter:
        """Get a specific game adapter by ID.

        Args:
            game_id: The game's unique identifier (e.g. 'mario', 'snake').

        Returns:
            The registered BaseGameAdapter instance.

        Raises:
            KeyError: If no game with that ID is registered.
        """
        if game_id not in self._adapters:
            raise KeyError(
                f"Unknown game '{game_id}'. "
                f"Available: {', '.join(sorted(self._adapters.keys()))}"
            )
        return self._adapters[game_id]

    def list_games(self) -> List[BaseGameAdapter]:
        """Return all registered game adapters."""
        return list(self._adapters.values())

    def list_game_ids(self) -> List[str]:
        """Return sorted list of all registered game IDs."""
        return sorted(self._adapters.keys())

    def list_by_category(self, category: str) -> List[BaseGameAdapter]:
        """Filter games by category (e.g. 'board', 'arcade')."""
        return [g for g in self._adapters.values() if g.category == category]

    def discover(self, base_path: Optional[str] = None) -> None:
        """Auto-discover game adapters from the standard folders.

        Scans builtin/, retro/, and user/ subdirectories for adapter.py
        files. Each adapter.py must define a class that inherits from
        BaseGameAdapter. The first such class found is instantiated
        and registered.

        Args:
            base_path: Root path containing builtin/, retro/, user/
                       subdirectories. Defaults to the games/ directory.
        """
        if base_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))

        for folder_name in ['builtin', 'retro', 'user']:
            folder_path = os.path.join(base_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for subfolder in sorted(os.listdir(folder_path)):
                # Skip __pycache__, _template, hidden dirs
                if subfolder.startswith(('_', '.')):
                    continue
                adapter_path = os.path.join(folder_path, subfolder, 'adapter.py')
                if not os.path.isfile(adapter_path):
                    continue
                try:
                    adapter = self._load_adapter(adapter_path, folder_name, subfolder)
                    if adapter:
                        self._adapters[adapter.game_id] = adapter
                except Exception as e:
                    print(f"  Warning: failed to load adapter from {adapter_path}: {e}")

    def _load_adapter(
        self, path: str, folder_name: str, subfolder: str
    ) -> Optional[BaseGameAdapter]:
        """Load a single adapter from an adapter.py file.

        Uses importlib to load the module, then searches for the first
        class that is a subclass of BaseGameAdapter (but not
        BaseGameAdapter itself).

        Returns:
            An instantiated adapter, or None if no adapter class found.
        """
        module_name = f"games.{folder_name}.{subfolder}.adapter"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find the adapter class
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseGameAdapter)
                and obj is not BaseGameAdapter
            ):
                return obj()
        return None
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_registry.py -v`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add games/registry.py tests/test_registry.py
git commit -m "feat: add GameRegistry with auto-discovery"
```

---

### Task 5: Time Reward Wrapper

**Files:**
- Create: `src/rewards/time_reward.py`
- Create: `tests/test_time_reward.py`

**Step 1: Write the failing test**

```python
# tests/test_time_reward.py
"""Tests for TimeRewardWrapper."""
import time
import pytest
import numpy as np
import gym
from gym.spaces import Box, Discrete
from unittest.mock import MagicMock

from src.rewards.time_reward import TimeRewardWrapper
from games.reward_config import RewardConfig, StandardMetrics


class FakeGameEnv(gym.Env):
    """Minimal env that returns controllable info dicts."""

    def __init__(self):
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = Discrete(4)
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        return np.zeros((84, 84, 1), dtype=np.uint8)

    def step(self, action):
        self._step_count += 1
        obs = np.zeros((84, 84, 1), dtype=np.uint8)
        reward = 1.0
        done = self._step_count >= 10
        info = {'score': self._step_count * 10, 'progress': self._step_count / 10}
        return obs, reward, done, info


class FakeAdapter:
    """Stub adapter for metric extraction."""

    def __init__(self, completed=False):
        self._completed = completed

    def extract_metrics(self, info, episode_time):
        return StandardMetrics(
            progress=info.get('progress', 0.0),
            score=info.get('score', 0.0),
            completed=self._completed,
            time_elapsed=episode_time,
        )


class TestTimeRewardWrapper:
    def test_reset_returns_obs(self):
        env = FakeGameEnv()
        adapter = FakeAdapter()
        wrapped = TimeRewardWrapper(env, RewardConfig(), adapter)
        obs = wrapped.reset()
        assert obs.shape == (84, 84, 1)

    def test_step_returns_shaped_reward(self):
        env = FakeGameEnv()
        adapter = FakeAdapter()
        config = RewardConfig(time_penalty_per_second=0.0, idle_penalty_per_second=0.0)
        wrapped = TimeRewardWrapper(env, config, adapter)
        wrapped.reset()
        obs, reward, done, info = wrapped.step(0)
        # Base reward is 1.0, no time/idle penalty
        assert reward == pytest.approx(1.0, abs=0.1)

    def test_time_penalty_reduces_reward(self):
        env = FakeGameEnv()
        adapter = FakeAdapter()
        config = RewardConfig(
            time_penalty_per_second=100.0,  # Very large to be detectable
            idle_penalty_per_second=0.0,
        )
        wrapped = TimeRewardWrapper(env, config, adapter)
        wrapped.reset()
        time.sleep(0.05)  # Ensure some time passes
        _, reward, _, _ = wrapped.step(0)
        # Reward should be less than 1.0 due to time penalty
        assert reward < 1.0

    def test_completion_bonus_added(self):
        env = FakeGameEnv()
        adapter = FakeAdapter(completed=True)
        config = RewardConfig(
            time_penalty_per_second=0.0,
            idle_penalty_per_second=0.0,
            completion_bonus=50.0,
        )
        wrapped = TimeRewardWrapper(env, config, adapter)
        wrapped.reset()
        _, reward, _, _ = wrapped.step(0)
        # Base (1.0) + completion (50.0)
        assert reward >= 50.0

    def test_standard_metrics_in_info(self):
        env = FakeGameEnv()
        adapter = FakeAdapter()
        config = RewardConfig()
        wrapped = TimeRewardWrapper(env, config, adapter)
        wrapped.reset()
        _, _, _, info = wrapped.step(0)
        assert '_standard_metrics' in info
        assert isinstance(info['_standard_metrics'], StandardMetrics)

    def test_death_penalty_on_done_without_completion(self):
        """Agent dies (done=True, completed=False) -> death penalty applied."""
        env = FakeGameEnv()
        env._step_count = 9  # Next step will set done=True
        adapter = FakeAdapter(completed=False)
        config = RewardConfig(
            time_penalty_per_second=0.0,
            idle_penalty_per_second=0.0,
            death_penalty=-50.0,
        )
        wrapped = TimeRewardWrapper(env, config, adapter)
        wrapped.reset()
        env._step_count = 9  # Force done on next step
        _, reward, done, _ = wrapped.step(0)
        assert done is True
        # Base (1.0) + death (-50.0) = -49.0
        assert reward < 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_time_reward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rewards.time_reward'`

**Step 3: Write the implementation**

```python
# src/rewards/time_reward.py
"""
Universal time-based reward wrapper.

Applies reward shaping from a game's RewardConfig to any environment.
This encourages agents to play efficiently by penalizing slow play
and rewarding fast completion.

The wrapper is game-agnostic — it relies on the game adapter's
extract_metrics() to translate game-specific info dicts into
StandardMetrics, then applies the universal reward formula.
"""
import time

import gym

from games.reward_config import RewardConfig


class TimeRewardWrapper(gym.Wrapper):
    """Applies time-based rewards from the game's RewardConfig.

    Wraps any game environment and applies:
    - Time penalty: small negative reward per second of wall-clock time
    - Idle penalty: extra penalty when no progress is made
    - Speed bonus: multiplied reward for completing under par time
    - Completion bonus: fixed reward for finishing
    - Death penalty: fixed penalty for dying/losing without completing

    Args:
        env: The gym environment to wrap.
        reward_config: A RewardConfig with tuning parameters.
        adapter: The game adapter (used for extract_metrics).
    """

    def __init__(self, env: gym.Env, reward_config: RewardConfig, adapter):
        super().__init__(env)
        self.config = reward_config
        self.adapter = adapter
        self._episode_start = None
        self._last_progress = 0.0
        self._last_step_time = None

    def reset(self, **kwargs):
        self._episode_start = time.time()
        self._last_progress = 0.0
        self._last_step_time = self._episode_start
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        now = time.time()
        dt = now - self._last_step_time
        self._last_step_time = now
        episode_time = now - self._episode_start

        # Extract standard metrics from game-specific info
        metrics = self.adapter.extract_metrics(info, episode_time)
        info['_standard_metrics'] = metrics

        shaped_reward = reward

        # Time penalty: penalize slow play
        shaped_reward -= self.config.time_penalty_per_second * dt

        # Idle penalty: extra cost when not making progress
        if metrics.progress <= self._last_progress:
            shaped_reward -= self.config.idle_penalty_per_second * dt
        self._last_progress = metrics.progress

        # Completion bonus with speed multiplier
        if metrics.completed:
            shaped_reward += self.config.completion_bonus
            if episode_time < self.config.par_time_seconds:
                speed_ratio = 1.0 - (episode_time / self.config.par_time_seconds)
                shaped_reward += (
                    self.config.completion_bonus
                    * speed_ratio
                    * self.config.speed_bonus_multiplier
                )

        # Death/loss penalty
        if done and not metrics.completed:
            shaped_reward += self.config.death_penalty

        return obs, shaped_reward, done, info
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_time_reward.py -v`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add src/rewards/time_reward.py tests/test_time_reward.py
git commit -m "feat: add TimeRewardWrapper for universal reward shaping"
```

---

### Task 6: Universal Environment Factory

**Files:**
- Create: `src/environment/universal_env.py`
- Create: `tests/test_universal_env.py`

**Step 1: Write the failing test**

```python
# tests/test_universal_env.py
"""Tests for universal environment factory."""
import pytest
import numpy as np

from src.environment.universal_env import create_env_from_adapter
from tests.test_base_adapter import StubAdapter


class TestUniversalEnvFactory:
    def test_creates_env(self):
        adapter = StubAdapter()
        env = create_env_from_adapter(adapter)
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        env.close()

    def test_env_reset_returns_obs(self):
        adapter = StubAdapter()
        env = create_env_from_adapter(adapter)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        env.close()

    def test_env_step_returns_4_tuple(self):
        adapter = StubAdapter()
        env = create_env_from_adapter(adapter)
        env.reset()
        result = env.step(0)
        assert len(result) == 4  # obs, reward, done, info
        env.close()

    def test_time_rewards_enabled_by_default(self):
        adapter = StubAdapter()
        env = create_env_from_adapter(adapter, use_time_rewards=True)
        env.reset()
        _, _, _, info = env.step(0)
        assert '_standard_metrics' in info
        env.close()

    def test_time_rewards_disabled(self):
        adapter = StubAdapter()
        env = create_env_from_adapter(adapter, use_time_rewards=False)
        env.reset()
        _, _, _, info = env.step(0)
        assert '_standard_metrics' not in info
        env.close()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_universal_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.environment.universal_env'`

**Step 3: Write the implementation**

```python
# src/environment/universal_env.py
"""
Universal environment factory.

Creates a ready-to-use Gym environment from any game adapter.
Applies the standard wrapper pipeline (TimeRewardWrapper) and
optional SB3 compatibility wrapping.

This replaces the game-specific factory functions (create_mario_env,
create_cnn_env, etc.) with a single function that works for any game.
"""
import gym

from games.base_adapter import BaseGameAdapter
from src.rewards.time_reward import TimeRewardWrapper


def create_env_from_adapter(
    adapter: BaseGameAdapter,
    use_time_rewards: bool = True,
    sb3_compat: bool = False,
    **game_kwargs,
) -> gym.Env:
    """Create a fully configured environment from a game adapter.

    This is the universal entry point for creating environments.
    It delegates env creation to the adapter, then wraps with
    the framework's standard wrappers.

    Args:
        adapter: The game adapter to create an environment from.
        use_time_rewards: Whether to apply TimeRewardWrapper.
                          Default True.
        sb3_compat: Whether to apply SB3CompatWrapper for
                    stable-baselines3 algorithms. Default False.
        **game_kwargs: Passed to adapter.create_env() for
                       game-specific options (world, stage, etc.).

    Returns:
        gym.Env: A fully wrapped environment ready for training.
    """
    # Create the base environment from the adapter
    env = adapter.create_env(**game_kwargs)

    # Apply time-based reward shaping
    if use_time_rewards:
        reward_config = adapter.get_reward_config()
        env = TimeRewardWrapper(env, reward_config, adapter)

    # Apply SB3 compatibility wrapper if needed
    if sb3_compat or adapter.needs_sb3_compat():
        from src.environment.wrappers import SB3CompatWrapper
        env = SB3CompatWrapper(env)

    return env
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_universal_env.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/environment/universal_env.py tests/test_universal_env.py
git commit -m "feat: add universal env factory (create_env_from_adapter)"
```

---

### Task 7: Migrate Mario Behind the Adapter Interface

**Files:**
- Create: `games/retro/mario/adapter.py`
- Create: `games/retro/mario/reward_shaper.py`
- Create: `tests/test_mario_adapter.py`
- Keep: `src/environment/mario_env.py` (unchanged, backwards compat)

**Step 1: Write the failing test**

```python
# tests/test_mario_adapter.py
"""Tests for the Mario game adapter.

These tests verify the adapter interface without needing
the actual NES emulator — they test the adapter's identity,
config, and metrics extraction using mocked info dicts.
"""
import pytest

from games.retro.mario.adapter import MarioAdapter
from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics, RewardConfig, ActionSpaceInfo


class TestMarioAdapter:
    def test_is_base_game_adapter(self):
        adapter = MarioAdapter()
        assert isinstance(adapter, BaseGameAdapter)

    def test_identity(self):
        adapter = MarioAdapter()
        assert adapter.name == 'Super Mario Bros'
        assert adapter.game_id == 'mario'
        assert adapter.category == 'platformer'
        assert 'Mario' in adapter.description or 'mario' in adapter.description.lower()

    def test_action_space_info(self):
        adapter = MarioAdapter()
        info = adapter.get_action_space_info()
        assert isinstance(info, ActionSpaceInfo)
        assert info.num_actions == 7  # Our simplified action space
        assert len(info.action_labels) == 7

    def test_observation_shape(self):
        adapter = MarioAdapter()
        shape = adapter.get_observation_shape()
        assert shape == (84, 84, 4)  # 84x84 with 4 stacked frames

    def test_extract_metrics_normal_play(self):
        adapter = MarioAdapter()
        info = {
            'x_pos': 500,
            'coins': 3,
            'flag_get': False,
            'life': 2,
        }
        metrics = adapter.extract_metrics(info, episode_time=30.0)
        assert isinstance(metrics, StandardMetrics)
        assert 0.0 <= metrics.progress <= 1.0
        assert metrics.score == 3  # coins
        assert metrics.completed is False
        assert metrics.time_elapsed == 30.0

    def test_extract_metrics_flag_reached(self):
        adapter = MarioAdapter()
        info = {'x_pos': 3000, 'coins': 10, 'flag_get': True, 'life': 2}
        metrics = adapter.extract_metrics(info, episode_time=60.0)
        assert metrics.completed is True

    def test_reward_config(self):
        adapter = MarioAdapter()
        rc = adapter.get_reward_config()
        assert isinstance(rc, RewardConfig)
        # Mario should have nonzero time penalty (action game)
        assert rc.time_penalty_per_second > 0

    def test_needs_sb3_compat(self):
        adapter = MarioAdapter()
        # Mario uses old gym API, needs SB3 compat
        assert adapter.needs_sb3_compat() is True

    def test_game_specific_options(self):
        adapter = MarioAdapter()
        opts = adapter.get_game_specific_options()
        assert 'world' in opts
        assert 'stage' in opts

    def test_has_curriculum(self):
        adapter = MarioAdapter()
        curriculum = adapter.get_curriculum()
        # Mario has multi-level curriculum
        assert curriculum is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mario_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'games.retro.mario.adapter'`

**Step 3: Write the adapter implementation**

```python
# games/retro/mario/adapter.py
"""
Mario game adapter — wraps existing mario_env.py behind BaseGameAdapter.

This migrates our existing Super Mario Bros environment into the
plugin system without changing any of the original code. The
original mario_env.py and CustomRewardWrapper continue to work
unchanged for backwards compatibility.
"""
from typing import Optional, Tuple

import gym

from games.base_adapter import BaseGameAdapter
from games.reward_config import (
    StandardMetrics,
    RewardConfig,
    ActionSpaceInfo,
)
from src.environment.mario_env import create_mario_env, MARIO_ACTIONS
from src.training.curriculum import CurriculumManager


# Approximate max x_pos for progress calculation
_MARIO_MAX_X = 3200


class MarioAdapter(BaseGameAdapter):
    """Adapter for Super Mario Bros via gym-super-mario-bros."""

    @property
    def name(self) -> str:
        return 'Super Mario Bros'

    @property
    def game_id(self) -> str:
        return 'mario'

    @property
    def category(self) -> str:
        return 'platformer'

    @property
    def description(self) -> str:
        return 'Classic NES platformer — run, jump, and stomp through 32 stages.'

    def create_env(self, **kwargs) -> gym.Env:
        """Create a Mario environment.

        Accepted kwargs:
            world (int): World number 1-8. Default 1.
            stage (int): Stage number 1-4. Default 1.
            resize_shape (tuple): Observation size. Default (84, 84).
            frame_stack (int): Frames to stack. Default 4.

        Note: We set apply_reward_shaping=False because the framework
        applies its own TimeRewardWrapper from get_reward_config().
        """
        world = kwargs.get('world', 1)
        stage = kwargs.get('stage', 1)
        resize_shape = kwargs.get('resize_shape', (84, 84))
        frame_stack = kwargs.get('frame_stack', 4)

        return create_mario_env(
            world=world,
            stage=stage,
            resize_shape=resize_shape,
            frame_stack=frame_stack,
            apply_reward_shaping=False,  # Framework handles rewards
            normalize=True,
        )

    def get_action_space_info(self) -> ActionSpaceInfo:
        return ActionSpaceInfo(
            num_actions=len(MARIO_ACTIONS),
            action_labels=[
                'NOOP', 'Right', 'Right+Jump', 'Right+Run',
                'Right+Run+Jump', 'Jump', 'Left',
            ],
        )

    def get_observation_shape(self) -> Tuple[int, ...]:
        return (84, 84, 4)

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        x_pos = info.get('x_pos', 0)
        progress = min(1.0, x_pos / _MARIO_MAX_X)
        return StandardMetrics(
            progress=progress,
            score=float(info.get('coins', 0)),
            completed=bool(info.get('flag_get', False)),
            time_elapsed=episode_time,
        )

    def get_reward_config(self) -> RewardConfig:
        return RewardConfig(
            time_penalty_per_second=0.01,
            completion_bonus=100.0,
            death_penalty=-15.0,
            idle_penalty_per_second=0.005,
            speed_bonus_multiplier=1.5,
            par_time_seconds=120.0,
        )

    def needs_sb3_compat(self) -> bool:
        return True

    def get_game_specific_options(self) -> dict:
        return {
            'world': (int, 1, 'World number 1-8'),
            'stage': (int, 1, 'Stage number 1-4'),
        }

    def get_curriculum(self) -> Optional[CurriculumManager]:
        return CurriculumManager()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mario_adapter.py -v`
Expected: PASS (all 10 tests)

**Step 5: Commit**

```bash
git add games/retro/mario/adapter.py tests/test_mario_adapter.py
git commit -m "feat: migrate Mario behind BaseGameAdapter interface"
```

---

### Task 8: Snake Game + Adapter

**Files:**
- Create: `games/builtin/snake/game.py`
- Create: `games/builtin/snake/adapter.py`
- Create: `tests/test_snake.py`

**Step 1: Write the failing test for the game engine**

```python
# tests/test_snake.py
"""Tests for the Snake game engine and adapter."""
import pytest
import numpy as np

from games.builtin.snake.game import SnakeEnv
from games.builtin.snake.adapter import SnakeAdapter
from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics, ActionSpaceInfo


class TestSnakeEnv:
    def test_reset_returns_observation(self):
        env = SnakeEnv(grid_size=8)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (84, 84, 1)  # Grayscale rendered
        env.close()

    def test_step_returns_4_tuple(self):
        env = SnakeEnv(grid_size=8)
        env.reset()
        obs, reward, done, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        env.close()

    def test_info_contains_score(self):
        env = SnakeEnv(grid_size=8)
        env.reset()
        _, _, _, info = env.step(0)
        assert 'score' in info
        assert 'snake_length' in info
        env.close()

    def test_action_space_is_4(self):
        env = SnakeEnv(grid_size=8)
        assert env.action_space.n == 4
        env.close()

    def test_game_terminates_on_wall_collision(self):
        env = SnakeEnv(grid_size=4)
        env.reset()
        # Move in one direction until wall hit
        done = False
        for _ in range(20):
            _, _, done, _ = env.step(1)  # Right
            if done:
                break
        assert done is True
        env.close()


class TestSnakeAdapter:
    def test_is_base_game_adapter(self):
        adapter = SnakeAdapter()
        assert isinstance(adapter, BaseGameAdapter)

    def test_identity(self):
        adapter = SnakeAdapter()
        assert adapter.game_id == 'snake'
        assert adapter.category == 'arcade'

    def test_action_space_info(self):
        adapter = SnakeAdapter()
        info = adapter.get_action_space_info()
        assert info.num_actions == 4

    def test_observation_shape(self):
        adapter = SnakeAdapter()
        assert adapter.get_observation_shape() == (84, 84, 1)

    def test_extract_metrics(self):
        adapter = SnakeAdapter()
        info = {'score': 5, 'snake_length': 6, 'max_length': 64}
        metrics = adapter.extract_metrics(info, 10.0)
        assert isinstance(metrics, StandardMetrics)
        assert metrics.score == 5
        assert 0 <= metrics.progress <= 1.0

    def test_create_env(self):
        adapter = SnakeAdapter()
        env = adapter.create_env(grid_size=8)
        obs = env.reset()
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_does_not_need_sb3_compat(self):
        adapter = SnakeAdapter()
        assert adapter.needs_sb3_compat() is False
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_snake.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'games.builtin.snake.game'`

**Step 3: Write the Snake game engine**

```python
# games/builtin/snake/game.py
"""
Snake game engine as a Gym environment.

A self-contained implementation of the classic Snake game that
follows the Gym interface. The snake moves on a grid, eats food
to grow, and dies if it hits a wall or itself.

Observations are rendered as 84x84 grayscale images:
- Snake body: white (255)
- Food: gray (180)
- Background: black (0)
"""
import random

import cv2
import numpy as np
import gym
from gym.spaces import Box, Discrete


class SnakeEnv(gym.Env):
    """Snake game as a Gym environment.

    Args:
        grid_size: Size of the grid (grid_size x grid_size). Default 16.
        max_steps_without_food: Steps before timeout. Default grid_size^2.
        render_size: Pixel size of rendered observation. Default 84.
    """

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, grid_size: int = 16, max_steps_without_food: int = 0,
                 render_size: int = 84):
        super().__init__()
        self.grid_size = grid_size
        self.render_size = render_size
        self.max_steps_without_food = max_steps_without_food or grid_size * grid_size

        # 4 actions: 0=up, 1=right, 2=down, 3=left
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=0, high=255,
            shape=(render_size, render_size, 1),
            dtype=np.uint8,
        )

        # Direction vectors: up, right, down, left
        self._directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        self.snake = []
        self.food = None
        self.direction = 1  # Start moving right
        self._steps_since_food = 0
        self._score = 0

    def reset(self):
        # Place snake in center
        center = self.grid_size // 2
        self.snake = [(center, center)]
        self.direction = 1  # Right
        self._steps_since_food = 0
        self._score = 0
        self._place_food()
        return self._render_obs()

    def step(self, action):
        # Prevent 180-degree turns (can't reverse into yourself)
        opposite = {0: 2, 1: 3, 2: 0, 3: 1}
        if action != opposite.get(self.direction, -1):
            self.direction = action

        # Move head
        head_r, head_c = self.snake[0]
        dr, dc = self._directions[self.direction]
        new_head = (head_r + dr, head_c + dc)

        # Check wall collision
        r, c = new_head
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return self._render_obs(), -1.0, True, self._info()

        # Check self collision
        if new_head in self.snake:
            return self._render_obs(), -1.0, True, self._info()

        # Move
        self.snake.insert(0, new_head)

        # Check food
        reward = 0.0
        if new_head == self.food:
            self._score += 1
            self._steps_since_food = 0
            self._place_food()
            reward = 1.0
        else:
            self.snake.pop()  # Remove tail (no growth)
            self._steps_since_food += 1

        # Timeout check
        done = self._steps_since_food >= self.max_steps_without_food

        return self._render_obs(), reward, done, self._info()

    def _place_food(self):
        """Place food on a random empty cell."""
        empty = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in self.snake:
                    empty.append((r, c))
        if empty:
            self.food = random.choice(empty)
        else:
            self.food = None  # Board full (win condition)

    def _render_obs(self) -> np.ndarray:
        """Render grid as an 84x84 grayscale image."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Draw snake
        for segment in self.snake:
            grid[segment[0], segment[1]] = 255

        # Draw food
        if self.food:
            grid[self.food[0], self.food[1]] = 180

        # Resize to render_size x render_size
        obs = cv2.resize(grid, (self.render_size, self.render_size),
                         interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(obs, axis=-1)

    def render(self, mode='rgb_array'):
        """Render as RGB for dashboard display."""
        gray = self._render_obs()[:, :, 0]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def _info(self) -> dict:
        return {
            'score': self._score,
            'snake_length': len(self.snake),
            'max_length': self.grid_size * self.grid_size,
        }

    def close(self):
        pass
```

**Step 4: Write the Snake adapter**

```python
# games/builtin/snake/adapter.py
"""Snake game adapter for the plugin registry."""
from typing import Tuple

import gym

from games.base_adapter import BaseGameAdapter
from games.reward_config import (
    StandardMetrics,
    RewardConfig,
    ActionSpaceInfo,
)
from games.builtin.snake.game import SnakeEnv


class SnakeAdapter(BaseGameAdapter):
    """Adapter for the built-in Snake game."""

    @property
    def name(self) -> str:
        return 'Snake'

    @property
    def game_id(self) -> str:
        return 'snake'

    @property
    def category(self) -> str:
        return 'arcade'

    @property
    def description(self) -> str:
        return 'Classic snake game — eat food, grow longer, avoid walls.'

    def create_env(self, **kwargs) -> gym.Env:
        grid_size = kwargs.get('grid_size', 16)
        return SnakeEnv(grid_size=grid_size)

    def get_action_space_info(self) -> ActionSpaceInfo:
        return ActionSpaceInfo(
            num_actions=4,
            action_labels=['Up', 'Right', 'Down', 'Left'],
        )

    def get_observation_shape(self) -> Tuple[int, ...]:
        return (84, 84, 1)

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        snake_len = info.get('snake_length', 1)
        max_len = info.get('max_length', 256)
        return StandardMetrics(
            progress=snake_len / max_len,
            score=float(info.get('score', 0)),
            completed=(snake_len >= max_len),
            time_elapsed=episode_time,
        )

    def get_reward_config(self) -> RewardConfig:
        return RewardConfig(
            time_penalty_per_second=0.005,
            completion_bonus=100.0,
            death_penalty=-10.0,
            idle_penalty_per_second=0.003,
            speed_bonus_multiplier=1.0,
            par_time_seconds=60.0,
        )

    def get_game_specific_options(self) -> dict:
        return {
            'grid_size': (int, 16, 'Grid size (8, 16, or 32)'),
        }
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_snake.py -v`
Expected: PASS (all 12 tests)

**Step 6: Commit**

```bash
git add games/builtin/snake/game.py games/builtin/snake/adapter.py tests/test_snake.py
git commit -m "feat: add Snake game engine and adapter"
```

---

### Task 9: Connect Four Game + Adapter

**Files:**
- Create: `games/builtin/connect4/game.py`
- Create: `games/builtin/connect4/adapter.py`
- Create: `tests/test_connect4.py`

**Step 1: Write the failing test**

```python
# tests/test_connect4.py
"""Tests for Connect Four game engine and adapter."""
import pytest
import numpy as np

from games.builtin.connect4.game import ConnectFourEnv
from games.builtin.connect4.adapter import ConnectFourAdapter
from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics


class TestConnectFourEnv:
    def test_reset_returns_observation(self):
        env = ConnectFourEnv()
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_step_returns_4_tuple(self):
        env = ConnectFourEnv()
        env.reset()
        obs, reward, done, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(done, bool)
        env.close()

    def test_action_space_is_7(self):
        env = ConnectFourEnv()
        assert env.action_space.n == 7
        env.close()

    def test_info_has_pieces_played(self):
        env = ConnectFourEnv()
        env.reset()
        _, _, _, info = env.step(0)
        assert 'pieces_played' in info
        env.close()

    def test_full_column_is_illegal(self):
        """Dropping in a full column returns done=True (illegal move)."""
        env = ConnectFourEnv()
        env.reset()
        # Fill column 0 (6 rows)
        for _ in range(6):
            env.step(0)
        # 7th drop in column 0 should end game (illegal)
        _, _, done, info = env.step(0)
        assert done is True
        env.close()

    def test_horizontal_win(self):
        """Four in a row horizontally wins."""
        env = ConnectFourEnv()
        env.reset()
        # Player 1 drops in cols 0,1,2,3 (opponent drops in col 6 between)
        env.step(0)  # P1
        env.step(6)  # P2 (opponent)
        env.step(1)  # P1
        env.step(6)  # P2
        env.step(2)  # P1
        env.step(6)  # P2
        _, reward, done, info = env.step(3)  # P1 wins
        assert done is True
        assert info.get('winner') == 1
        env.close()


class TestConnectFourAdapter:
    def test_is_base_game_adapter(self):
        adapter = ConnectFourAdapter()
        assert isinstance(adapter, BaseGameAdapter)

    def test_identity(self):
        adapter = ConnectFourAdapter()
        assert adapter.game_id == 'connect4'
        assert adapter.category == 'board'

    def test_action_space(self):
        adapter = ConnectFourAdapter()
        info = adapter.get_action_space_info()
        assert info.num_actions == 7

    def test_observation_shape(self):
        adapter = ConnectFourAdapter()
        assert adapter.get_observation_shape() == (84, 84, 1)

    def test_extract_metrics(self):
        adapter = ConnectFourAdapter()
        info = {'pieces_played': 10, 'winner': 0}
        metrics = adapter.extract_metrics(info, 5.0)
        assert isinstance(metrics, StandardMetrics)
        assert metrics.progress == pytest.approx(10 / 42)

    def test_board_game_no_time_penalty(self):
        adapter = ConnectFourAdapter()
        rc = adapter.get_reward_config()
        assert rc.time_penalty_per_second == 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_connect4.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the Connect Four game engine**

```python
# games/builtin/connect4/game.py
"""
Connect Four game engine as a Gym environment.

Two-player game on a 6x7 grid. Players alternate dropping pieces
into columns. First to get four in a row (horizontal, vertical,
or diagonal) wins.

The agent plays as Player 1. A simple random opponent plays as
Player 2 (opponent auto-plays after each agent action).

Observations are rendered as 84x84 grayscale images:
- Player 1 pieces: white (255)
- Player 2 pieces: gray (128)
- Empty: black (0)
"""
import random

import cv2
import numpy as np
import gym
from gym.spaces import Box, Discrete


ROWS = 6
COLS = 7


class ConnectFourEnv(gym.Env):
    """Connect Four as a Gym environment with a random opponent."""

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, render_size: int = 84):
        super().__init__()
        self.render_size = render_size
        self.action_space = Discrete(COLS)
        self.observation_space = Box(
            low=0, high=255,
            shape=(render_size, render_size, 1),
            dtype=np.uint8,
        )
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self._pieces_played = 0
        self._winner = 0

    def reset(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self._pieces_played = 0
        self._winner = 0
        return self._render_obs()

    def step(self, action):
        # Player 1 move
        if not self._is_valid_column(action):
            # Illegal move — game over with penalty
            return self._render_obs(), -1.0, True, self._info()

        self._drop_piece(action, player=1)
        self._pieces_played += 1

        # Check if player 1 wins
        if self._check_winner(1):
            self._winner = 1
            return self._render_obs(), 1.0, True, self._info()

        # Check draw
        if self._pieces_played >= ROWS * COLS:
            return self._render_obs(), 0.0, True, self._info()

        # Opponent move (random valid column)
        valid_cols = [c for c in range(COLS) if self._is_valid_column(c)]
        if not valid_cols:
            return self._render_obs(), 0.0, True, self._info()

        opp_col = random.choice(valid_cols)
        self._drop_piece(opp_col, player=2)
        self._pieces_played += 1

        # Check if opponent wins
        if self._check_winner(2):
            self._winner = 2
            return self._render_obs(), -1.0, True, self._info()

        # Check draw after opponent move
        if self._pieces_played >= ROWS * COLS:
            return self._render_obs(), 0.0, True, self._info()

        return self._render_obs(), 0.0, False, self._info()

    def _is_valid_column(self, col: int) -> bool:
        return 0 <= col < COLS and self.board[0, col] == 0

    def _drop_piece(self, col: int, player: int):
        for row in range(ROWS - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = player
                return

    def _check_winner(self, player: int) -> bool:
        """Check all four-in-a-row possibilities for player."""
        b = self.board
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if all(b[r, c + i] == player for i in range(4)):
                    return True
        # Vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                if all(b[r + i, c] == player for i in range(4)):
                    return True
        # Diagonal (down-right)
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(b[r + i, c + i] == player for i in range(4)):
                    return True
        # Diagonal (down-left)
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                if all(b[r + i, c - i] == player for i in range(4)):
                    return True
        return False

    def _render_obs(self) -> np.ndarray:
        """Render board as 84x84 grayscale."""
        grid = np.zeros((ROWS, COLS), dtype=np.uint8)
        grid[self.board == 1] = 255
        grid[self.board == 2] = 128
        obs = cv2.resize(grid, (self.render_size, self.render_size),
                         interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(obs, axis=-1)

    def render(self, mode='rgb_array'):
        gray = self._render_obs()[:, :, 0]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def _info(self) -> dict:
        return {
            'pieces_played': self._pieces_played,
            'winner': self._winner,
        }

    def close(self):
        pass
```

**Step 4: Write the Connect Four adapter**

```python
# games/builtin/connect4/adapter.py
"""Connect Four adapter for the plugin registry."""
from typing import Tuple

import gym

from games.base_adapter import BaseGameAdapter
from games.reward_config import (
    StandardMetrics,
    RewardConfig,
    ActionSpaceInfo,
)
from games.builtin.connect4.game import ConnectFourEnv


class ConnectFourAdapter(BaseGameAdapter):
    """Adapter for the built-in Connect Four game."""

    @property
    def name(self) -> str:
        return 'Connect Four'

    @property
    def game_id(self) -> str:
        return 'connect4'

    @property
    def category(self) -> str:
        return 'board'

    @property
    def description(self) -> str:
        return 'Drop pieces to get four in a row — vertical, horizontal, or diagonal.'

    def create_env(self, **kwargs) -> gym.Env:
        return ConnectFourEnv()

    def get_action_space_info(self) -> ActionSpaceInfo:
        return ActionSpaceInfo(
            num_actions=7,
            action_labels=[f'Col {i+1}' for i in range(7)],
        )

    def get_observation_shape(self) -> Tuple[int, ...]:
        return (84, 84, 1)

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        return StandardMetrics(
            progress=info.get('pieces_played', 0) / 42,
            score=float(1 if info.get('winner') == 1 else 0),
            completed=info.get('winner', 0) == 1,
            time_elapsed=episode_time,
        )

    def get_reward_config(self) -> RewardConfig:
        return RewardConfig(
            time_penalty_per_second=0.0,   # Board game — thinking is fine
            completion_bonus=50.0,
            death_penalty=-10.0,
            idle_penalty_per_second=0.0,
            speed_bonus_multiplier=0.0,
            par_time_seconds=300.0,
        )
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_connect4.py -v`
Expected: PASS (all 12 tests)

**Step 6: Commit**

```bash
git add games/builtin/connect4/game.py games/builtin/connect4/adapter.py tests/test_connect4.py
git commit -m "feat: add Connect Four game engine and adapter"
```

---

### Task 10: A2C Trainer

**Files:**
- Create: `src/algorithms/a2c/__init__.py`
- Create: `src/algorithms/a2c/a2c_trainer.py`
- Create: `config/a2c_config.yaml`
- Create: `tests/test_a2c_trainer.py`

**Step 1: Write the failing test**

```python
# tests/test_a2c_trainer.py
"""Tests for A2C trainer.

These test the trainer's interface without running actual training
(which would require the NES emulator). We verify construction,
config loading, and method signatures.
"""
import pytest
import yaml
import os


class TestA2CConfig:
    def test_config_file_exists(self):
        config_path = os.path.join('config', 'a2c_config.yaml')
        assert os.path.exists(config_path)

    def test_config_has_required_keys(self):
        config_path = os.path.join('config', 'a2c_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert 'n_steps' in config
        assert 'learning_rate' in config
        assert 'gamma' in config
        assert 'ent_coef' in config
        assert 'total_timesteps' in config


class TestA2CTrainerImport:
    def test_can_import_trainer(self):
        from src.algorithms.a2c.a2c_trainer import A2CTrainer
        assert A2CTrainer is not None

    def test_trainer_is_base_trainer_subclass(self):
        from src.algorithms.a2c.a2c_trainer import A2CTrainer
        from src.algorithms.base_trainer import BaseTrainer
        assert issubclass(A2CTrainer, BaseTrainer)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_a2c_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Create A2C config**

```yaml
# config/a2c_config.yaml
# ============================================================================
# A2C (Advantage Actor-Critic) Configuration
# ============================================================================
# A2C is a synchronous version of A3C. It collects short rollouts (n_steps)
# and updates the policy using advantage estimation. Faster than PPO per
# wall-clock second but less stable.
#
# Key difference from PPO: no clip_range (A2C doesn't clip gradients).
# Uses same CnnPolicy backbone as PPO.
# ============================================================================

# --- Training ---
total_timesteps: 1_000_000     # Total env steps before stopping
n_steps: 5                     # Rollout length per update (shorter than PPO)
learning_rate: 0.0007          # Higher than PPO default
gamma: 0.99                    # Discount factor
gae_lambda: 1.0                # GAE lambda (1.0 = no GAE, standard A2C)

# --- Regularization ---
ent_coef: 0.01                 # Entropy bonus (encourages exploration)
vf_coef: 0.5                   # Value function loss coefficient
max_grad_norm: 0.5             # Gradient clipping

# --- Checkpointing ---
save_freq: 50                  # Save checkpoint every N episodes

# --- Multi-env ---
num_envs: 4                    # Parallel environments (sync updates)
```

**Step 4: Create A2C trainer**

```python
# src/algorithms/a2c/__init__.py
```

```python
# src/algorithms/a2c/a2c_trainer.py
"""
A2C (Advantage Actor-Critic) Trainer.

A2C is a synchronous variant of A3C that uses multiple parallel
environments to collect experience and updates the policy using
advantage estimation. It's simpler and faster per wall-clock second
than PPO, but less stable.

This implementation wraps stable-baselines3's A2C with the same
dashboard integration pattern used by the PPO trainer.

Key differences from PPO:
    - Uses n_steps=5 (shorter rollouts)
    - No clip_range parameter
    - Synchronous gradient updates (all envs contribute to one update)
    - Higher learning rate (0.0007 vs PPO's 0.0003)
"""
import os
import yaml
import numpy as np
from typing import Dict, Any, Optional

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from src.algorithms.base_trainer import BaseTrainer
from src.visualization.dashboard import Dashboard


class A2CDashboardCallback(BaseCallback):
    """SB3 callback to feed A2C metrics to the dashboard."""

    def __init__(self, dashboard, trainer, verbose=0):
        super().__init__(verbose)
        self.dashboard = dashboard
        self.trainer = trainer
        self.episode_count = 0

    def _on_step(self):
        # Check for completed episodes
        for i, done in enumerate(self.locals.get('dones', [])):
            if done:
                infos = self.locals.get('infos', [])
                if i < len(infos):
                    ep_info = infos[i].get('episode', {})
                    reward = ep_info.get('r', 0)
                    self.episode_count += 1
                    self.trainer.episode_count = self.episode_count

                    if reward > self.trainer.best_reward:
                        self.trainer.best_reward = reward

                    # Fire episode callback for curriculum
                    distance = infos[i].get('x_pos', 0)
                    completed = infos[i].get('stage_completed', False)
                    self.trainer._fire_episode_complete(
                        reward=reward, distance=distance, completed=completed,
                    )

        # Update dashboard
        if self.dashboard and self.episode_count % 5 == 0:
            metrics = {
                'episode': self.episode_count,
                'reward': self.trainer.best_reward,
                'timestep': self.num_timesteps,
            }
            if not self.trainer.update_visualization(metrics=metrics):
                return False  # Dashboard closed
        return True


class A2CTrainer(BaseTrainer):
    """A2C trainer wrapping stable-baselines3.

    Args:
        env: The Gym environment (will be vectorized internally).
        config: Dict of hyperparameters (or loaded from a2c_config.yaml).
        visualizer: Optional Dashboard for live visualization.
        save_dir: Directory for saving checkpoints.
        log_dir: Directory for saving logs.
    """

    def __init__(
        self,
        env,
        config: Dict[str, Any],
        visualizer: Optional[Dashboard] = None,
        save_dir: str = 'models',
        log_dir: str = 'logs',
    ):
        super().__init__(env, config, visualizer, save_dir, log_dir)
        self.model = None
        self._env_factory = None

    def train(self, num_episodes: int = None) -> None:
        """Train using A2C.

        Args:
            num_episodes: Ignored (A2C uses total_timesteps from config).
        """
        total_timesteps = self.config.get('total_timesteps', 1_000_000)
        self.is_training = True

        # Create vectorized environment
        num_envs = self.config.get('num_envs', 4)
        if self._env_factory:
            vec_env = DummyVecEnv([self._env_factory for _ in range(num_envs)])
        else:
            vec_env = DummyVecEnv([lambda: self.env])
        vec_env = VecTransposeImage(vec_env)

        # Create A2C model
        self.model = A2C(
            'CnnPolicy',
            vec_env,
            n_steps=self.config.get('n_steps', 5),
            learning_rate=self.config.get('learning_rate', 0.0007),
            gamma=self.config.get('gamma', 0.99),
            gae_lambda=self.config.get('gae_lambda', 1.0),
            ent_coef=self.config.get('ent_coef', 0.01),
            vf_coef=self.config.get('vf_coef', 0.5),
            max_grad_norm=self.config.get('max_grad_norm', 0.5),
            verbose=0,
        )

        callback = A2CDashboardCallback(self.visualizer, self)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

        self.is_training = False
        self._save_on_exit()

    def evaluate(self, num_episodes: int = 5) -> float:
        """Evaluate the trained A2C model."""
        if self.model is None:
            raise RuntimeError('No model loaded. Train first or load a checkpoint.')

        rewards = []
        for _ in range(num_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
            total_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                result = self.env.step(action)
                obs, reward, done = result[0], result[1], result[2]
                total_reward += reward
            rewards.append(total_reward)
        return float(np.mean(rewards))

    def save_checkpoint(self, path: str) -> None:
        if self.model:
            self.model.save(path)

    def load_checkpoint(self, path: str) -> None:
        self.model = A2C.load(path)
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_a2c_trainer.py -v`
Expected: PASS (all 4 tests)

**Step 6: Commit**

```bash
git add src/algorithms/a2c/ config/a2c_config.yaml tests/test_a2c_trainer.py
git commit -m "feat: add A2C trainer wrapping stable-baselines3"
```

---

### Task 11: Update CLI with `--game` Flag

**Files:**
- Modify: `main.py`
- Create: `tests/test_cli_game_flag.py`

**Step 1: Write the failing test**

```python
# tests/test_cli_game_flag.py
"""Tests for the --game CLI flag."""
import subprocess
import sys
import pytest


class TestCLIGameFlag:
    def test_help_shows_game_flag(self):
        """The --game flag appears in --help output."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--help'],
            capture_output=True, text=True, timeout=10,
        )
        assert '--game' in result.stdout

    def test_algorithm_choices_include_a2c(self):
        """The --algorithm flag accepts 'a2c'."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--help'],
            capture_output=True, text=True, timeout=10,
        )
        assert 'a2c' in result.stdout

    def test_unknown_game_errors(self):
        """Passing an unknown game ID fails gracefully."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--algorithm', 'dqn',
             '--game', 'nonexistent_game_xyz', '--episodes', '1'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_game_flag.py::TestCLIGameFlag::test_help_shows_game_flag -v`
Expected: FAIL (--game not yet in help output)

**Step 3: Modify main.py**

Add these changes to `main.py`:

1. Add `--game` argument to `parse_args()`:

In the argument parser (after the `--algorithm` argument, around line 88):

```python
    # Game selection (new)
    parser.add_argument(
        '--game', '-g',
        type=str,
        default='mario',
        help='Game to play. Use --list-games to see available games. Default: mario',
    )

    parser.add_argument(
        '--list-games',
        action='store_true',
        help='List all available games and exit.',
    )
```

2. Update algorithm choices to include `a2c`:

Change line 86 from:
```python
        choices=['neat', 'ppo', 'dqn'],
```
to:
```python
        choices=['neat', 'ppo', 'dqn', 'a2c'],
```

Also make `--algorithm` not required (for `--list-games`):
```python
        required=False,
        default='dqn',
```

3. Add game registry initialization and `--list-games` handling at the top of the main function. Add these imports near the top of the file:

```python
from games.registry import GameRegistry
```

4. At the start of the main logic (after `parse_args()`), add:

```python
    # Initialize game registry
    registry = GameRegistry()
    registry.discover()

    if args.list_games:
        print('\nAvailable games:')
        for adapter in registry.list_games():
            print(f'  {adapter.game_id:12s}  [{adapter.category:10s}]  {adapter.name}')
        sys.exit(0)

    # Validate game selection
    try:
        game_adapter = registry.get_game(args.game)
    except KeyError as e:
        print(f'\nError: {e}')
        print('Use --list-games to see available games.')
        sys.exit(1)

    print(f'\nGame: {game_adapter.name} ({game_adapter.game_id})')
```

Full changes: These are surgical edits to main.py. The existing Mario-specific env creation code continues to work when `--game mario` (the default).

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli_game_flag.py -v`
Expected: PASS (all 3 tests)

**Step 5: Run existing tests to verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: All existing + new tests PASS

**Step 6: Commit**

```bash
git add main.py tests/test_cli_game_flag.py
git commit -m "feat: add --game flag and --list-games to CLI"
```

---

### Task 12: Launcher Game Selection Panel

**Files:**
- Modify: `launcher.py`

This task adds a game selection dropdown to the tkinter launcher.

**Step 1: Identify the insertion point**

The launcher currently has an algorithm selector dropdown. We add a game selector above it.

Key changes to `launcher.py`:

1. Import the game registry at the top:
```python
from games.registry import GameRegistry
```

2. In `__init__` of `MarioLauncher`, after creating the registry:
```python
        # Discover games
        self.game_registry = GameRegistry()
        self.game_registry.discover()
        self.available_games = self.game_registry.list_games()
```

3. Add a game selection combobox (before algorithm selector):
```python
        # Game selection
        game_frame = tk.Frame(left_frame, bg=BG_DARK)
        game_frame.pack(fill='x', padx=20, pady=(10, 5))
        tk.Label(game_frame, text='Game', bg=BG_DARK, fg=TEXT_DIM,
                 font=('Consolas', 9)).pack(anchor='w')
        self.game_var = tk.StringVar(value='mario')
        game_names = [f"{g.name} ({g.game_id})" for g in self.available_games]
        if not game_names:
            game_names = ['Super Mario Bros (mario)']
        self.game_combo = ttk.Combobox(
            game_frame, textvariable=self.game_var,
            values=game_names, state='readonly',
        )
        self.game_combo.pack(fill='x')
        self.game_combo.set(game_names[0])
```

4. Update `ALGO_INFO` to include A2C:
```python
        'a2c': {
            'color': ACCENT_GREEN,
            'desc': 'Advantage Actor-Critic (SB3)',
            'duration_label': '~30 min for 500K steps',
            'file_ext': '.zip',
            'model_subdir': 'a2c',
        },
```

5. In the command builder method, extract the game_id from the selected game and add `--game {game_id}` to the CLI command.

**Step 2: Manually verify the launcher opens**

Run: `python launcher.py`
Expected: Launcher window opens with game selection dropdown showing discovered games.

**Step 3: Commit**

```bash
git add launcher.py
git commit -m "feat: add game selection dropdown and A2C to launcher"
```

---

### Task 13: Integration Tests

**Files:**
- Create: `tests/test_integration_multigame.py`

**Step 1: Write integration tests**

```python
# tests/test_integration_multigame.py
"""Integration tests for the multi-game plugin system.

These verify that the full pipeline works end-to-end:
registry discovery -> adapter -> env creation -> step/reset.
"""
import pytest

from games.registry import GameRegistry
from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics
from src.environment.universal_env import create_env_from_adapter


class TestRegistryDiscovery:
    def test_discovers_snake(self):
        registry = GameRegistry()
        registry.discover()
        game = registry.get_game('snake')
        assert game.name == 'Snake'

    def test_discovers_connect4(self):
        registry = GameRegistry()
        registry.discover()
        game = registry.get_game('connect4')
        assert game.name == 'Connect Four'

    def test_discovers_mario(self):
        registry = GameRegistry()
        registry.discover()
        game = registry.get_game('mario')
        assert game.name == 'Super Mario Bros'

    def test_list_games_returns_all(self):
        registry = GameRegistry()
        registry.discover()
        ids = registry.list_game_ids()
        assert 'snake' in ids
        assert 'connect4' in ids
        assert 'mario' in ids

    def test_list_by_category(self):
        registry = GameRegistry()
        registry.discover()
        board_games = registry.list_by_category('board')
        assert any(g.game_id == 'connect4' for g in board_games)
        arcade_games = registry.list_by_category('arcade')
        assert any(g.game_id == 'snake' for g in arcade_games)


class TestSnakeEndToEnd:
    def test_snake_env_with_time_rewards(self):
        registry = GameRegistry()
        registry.discover()
        adapter = registry.get_game('snake')
        env = create_env_from_adapter(adapter, use_time_rewards=True)
        obs = env.reset()
        assert obs.shape == (84, 84, 1)
        obs, reward, done, info = env.step(0)
        assert '_standard_metrics' in info
        metrics = info['_standard_metrics']
        assert isinstance(metrics, StandardMetrics)
        env.close()


class TestConnectFourEndToEnd:
    def test_connect4_env_without_time_rewards(self):
        registry = GameRegistry()
        registry.discover()
        adapter = registry.get_game('connect4')
        env = create_env_from_adapter(adapter, use_time_rewards=False)
        obs = env.reset()
        assert obs.shape == (84, 84, 1)
        obs, reward, done, info = env.step(3)  # Drop in middle column
        assert isinstance(reward, (int, float))
        env.close()


class TestAllAdaptersConformToInterface:
    """Verify every discovered adapter implements the full interface."""

    def test_all_adapters_have_required_properties(self):
        registry = GameRegistry()
        registry.discover()
        for adapter in registry.list_games():
            assert isinstance(adapter.name, str) and len(adapter.name) > 0
            assert isinstance(adapter.game_id, str) and len(adapter.game_id) > 0
            assert adapter.category in ('platformer', 'puzzle', 'board', 'arcade', 'rpg')
            assert isinstance(adapter.description, str)

    def test_all_adapters_have_action_space(self):
        registry = GameRegistry()
        registry.discover()
        for adapter in registry.list_games():
            info = adapter.get_action_space_info()
            assert info.num_actions > 0
            assert len(info.action_labels) == info.num_actions

    def test_all_adapters_have_observation_shape(self):
        registry = GameRegistry()
        registry.discover()
        for adapter in registry.list_games():
            shape = adapter.get_observation_shape()
            assert len(shape) >= 2
```

**Step 2: Run all integration tests**

Run: `python -m pytest tests/test_integration_multigame.py -v`
Expected: PASS (all tests)

**Step 3: Run the FULL test suite to verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: ALL tests PASS (existing 39 + new tests)

**Step 4: Commit**

```bash
git add tests/test_integration_multigame.py
git commit -m "test: add integration tests for multi-game plugin system"
```

---

## Phase 1 Complete Checklist

After all 13 tasks:
- [ ] `games/` directory tree exists with `__init__.py` files
- [ ] `games/reward_config.py` — dataclasses for metrics/rewards/tokens
- [ ] `games/base_adapter.py` — abstract game adapter interface
- [ ] `games/registry.py` — auto-discovery registry
- [ ] `src/rewards/time_reward.py` — universal time-based reward wrapper
- [ ] `src/environment/universal_env.py` — creates env from any adapter
- [ ] `games/retro/mario/adapter.py` — Mario migrated to adapter pattern
- [ ] `games/builtin/snake/` — Snake game engine + adapter
- [ ] `games/builtin/connect4/` — Connect Four game engine + adapter
- [ ] `src/algorithms/a2c/` — A2C trainer wrapping SB3
- [ ] `main.py` — `--game` flag, `--list-games`, A2C in choices
- [ ] `launcher.py` — game selection dropdown, A2C in algorithm list
- [ ] All existing 39 tests still pass
- [ ] New tests for every component
- [ ] All changes committed

---

## Phases 2–4 (Future Plans)

These phases will be planned in detail in their own planning sessions after Phase 1 is complete and merged.

### Phase 2: Algorithm Expansion + ROM Support
- Rainbow DQN (prioritized replay, dueling networks, noisy nets, C51, multi-step)
- stable-retro integration for emulated ROM games
- ROM import wizard in launcher
- Chess, Checkers, Tic-Tac-Toe built-in games
- Cross-game comparison dashboard panel

### Phase 3: Generalist Agent
- Shared experience store (collects tokenized trajectories from all games)
- Decision Transformer network architecture (GPT-2 style)
- Decision Transformer trainer
- Multi-game training mode in launcher
- Cross-game metrics visualization

### Phase 4: Community + Polish
- User plugin folder with template generator
- "Add Game" wizard for custom Python environments
- Training presets (Quick Demo, Deep Training, Overnight)
- Documentation and tutorials
