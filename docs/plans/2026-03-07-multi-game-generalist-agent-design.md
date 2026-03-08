# Multi-Game Generalist Agent Platform Design

**Date:** 2026-03-07
**Status:** Approved
**Author:** Claude + Dakotah

## Overview

Evolve the mario-ml project from a single-game Mario trainer into a
multi-game reinforcement learning platform. The platform supports built-in
games (Snake, Chess, Tetris, etc.), emulated ROM games (Mario, Sonic,
Pokemon via stable-retro), and user-contributed game plugins. A shared
experience pool enables training a single generalist agent across all games
using a Decision Transformer architecture.

## Goals

1. Plugin registry architecture where games are self-contained modules
2. Built-in traditional games (Snake, Tetris, Chess, Checkers, Tic-Tac-Toe, Connect Four)
3. Emulated ROM support via stable-retro (NES, SNES, Genesis, Game Boy, GBA)
4. Three new algorithms: A2C, Rainbow DQN, Decision Transformer
5. Universal time-based reward system to encourage fast play
6. Shared experience pool for training a generalist agent across all games
7. Ease of use for non-technical users via enhanced launcher GUI
8. Preserve all existing features (dashboard, streaming, music, pause/resume)

## Non-Goals

- 3D game support (pixel-based 2D only)
- Superhuman performance on complex board games (educational, not competitive)
- ROM distribution (users must supply their own legally-owned ROMs)
- Replacing specialized agents (generalist trades peak performance for breadth)

---

## Architecture

### Project Structure

```
rl-arcade/
|
+-- launcher.py                    # GUI launcher (enhanced for game selection)
+-- main.py                        # CLI entry point
|
+-- games/                         # PLUGIN REGISTRY
|   +-- registry.py                # Discovers and loads game adapters
|   +-- base_adapter.py            # Abstract interface all games implement
|   +-- reward_config.py           # RewardConfig, StandardMetrics, TokenConfig
|   +-- builtin/                   # Games we build ourselves
|   |   +-- snake/
|   |   |   +-- adapter.py         # SnakeAdapter(BaseGameAdapter)
|   |   |   +-- game.py            # Snake game engine + Gym env
|   |   +-- tetris/
|   |   |   +-- adapter.py
|   |   |   +-- game.py
|   |   +-- chess/
|   |   |   +-- adapter.py
|   |   |   +-- game.py
|   |   +-- checkers/
|   |   |   +-- adapter.py
|   |   |   +-- game.py
|   |   +-- tictactoe/
|   |   |   +-- adapter.py
|   |   |   +-- game.py
|   |   +-- connect4/
|   |       +-- adapter.py
|   |       +-- game.py
|   +-- retro/                     # Emulated ROM games (stable-retro)
|   |   +-- mario/
|   |   |   +-- adapter.py         # Current Mario, migrated here
|   |   |   +-- reward_shaper.py   # Mario-specific rewards
|   |   |   +-- curriculum.py      # 32-stage curriculum (moved from src/)
|   |   +-- sonic/
|   |   |   +-- adapter.py
|   |   +-- megaman/
|   |   |   +-- adapter.py
|   |   +-- pokemon/
|   |       +-- adapter.py
|   +-- user/                      # Drop-in folder for user-contributed games
|       +-- _template/             # Example adapter for users to copy
|           +-- adapter.py
|
+-- src/
|   +-- algorithms/                # EXTENDED with new algorithms
|   |   +-- base_trainer.py        # Existing (preserved, no changes)
|   |   +-- neat/                  # Existing (preserved)
|   |   +-- ppo/                   # Existing (preserved)
|   |   +-- dqn/                   # Existing (preserved)
|   |   +-- rainbow/               # NEW: Rainbow DQN
|   |   |   +-- rainbow_trainer.py
|   |   |   +-- rainbow_network.py # Dueling + Noisy nets
|   |   |   +-- prioritized_replay.py
|   |   +-- a2c/                   # NEW: A2C via stable-baselines3
|   |   |   +-- a2c_trainer.py
|   |   +-- decision_transformer/  # NEW: Multi-game generalist
|   |       +-- dt_trainer.py
|   |       +-- dt_network.py      # Transformer architecture
|   |       +-- trajectory_dataset.py
|   |
|   +-- rewards/                   # NEW: Universal reward system
|   |   +-- base_reward.py         # Abstract reward shaper
|   |   +-- time_reward.py         # Time-based penalties/bonuses
|   |   +-- composite_reward.py    # Chains multiple reward shapers
|   |
|   +-- experience/                # NEW: Shared experience for generalist
|   |   +-- experience_store.py    # Unified buffer across games
|   |   +-- tokenizer.py           # Obs/action -> token sequences
|   |
|   +-- environment/               # Generalized wrappers
|   |   +-- wrappers.py            # SkipFrame, GrayScale, Resize, FrameStack (kept)
|   |   +-- universal_env.py       # NEW: Creates env from any adapter
|   |
|   +-- visualization/             # Existing (preserved, game-agnostic)
|   +-- streaming/                 # Existing (preserved)
|   +-- audio/                     # Existing (preserved)
|   +-- training/                  # Existing curriculum base (generalized)
|
+-- config/                        # Algorithm configs (existing + new)
|   +-- neat_config.txt
|   +-- ppo_config.yaml
|   +-- dqn_config.yaml
|   +-- rainbow_config.yaml        # NEW
|   +-- a2c_config.yaml            # NEW
|   +-- dt_config.yaml             # NEW
|
+-- models/                        # Organized by game + algorithm
|   +-- mario/
|   |   +-- dqn/
|   |   +-- ppo/
|   +-- snake/
|   |   +-- rainbow/
|   +-- generalist/                # Decision Transformer checkpoints
|       +-- experience_pool/       # Shared experience data
|
+-- tests/                         # Existing + new tests
+-- docs/plans/                    # Design documents
```

### What Stays the Same

These components are already game-agnostic and require zero changes:

- `BaseTrainer` abstract class and all its utilities (pause/resume,
  callbacks, checkpointing, signal handling, metadata)
- `Dashboard`, `GameRenderer`, `GridRenderer`, `GraphPanel`, `MetricsTracker`
- `StreamManager`, `OverlayManager`, `MusicManager`
- All existing wrappers: `SkipFrame`, `GrayScaleObservation`,
  `ResizeObservation`, `FrameStackObservation`
- `SB3CompatWrapper` (bridges old gym -> gymnasium for SB3 algorithms)
- All 39 existing tests

### What Moves

- `mario_env.py` -> `games/retro/mario/adapter.py` (wrapped in adapter)
- `CustomRewardWrapper` -> `games/retro/mario/reward_shaper.py`
- `CurriculumManager` -> `games/retro/mario/curriculum.py`
- `MARIO_ACTIONS` -> `games/retro/mario/adapter.py`

The existing CLI flags (`--world`, `--stage`) still work when Mario
is the selected game.

---

## Game Adapter Interface

Every game implements this abstract interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import gym


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
    time_penalty_per_second: float = 0.01   # Penalize slow play
    completion_bonus: float = 100.0         # Reward for finishing
    death_penalty: float = -15.0            # Penalty for dying/losing
    idle_penalty_per_second: float = 0.005  # Penalize no progress
    speed_bonus_multiplier: float = 1.5     # Bonus for fast completion
    par_time_seconds: float = 120.0         # "Par" time for speed bonus


@dataclass
class ActionSpaceInfo:
    """Describes the game's action space for the launcher and algorithms."""
    num_actions: int              # Total discrete actions
    action_labels: List[str]      # Human-readable names per action


@dataclass
class TokenConfig:
    """How to tokenize this game for the Decision Transformer."""
    obs_resolution: Tuple[int, int]  # Resize observations to this
    obs_channels: int                 # Number of channels (1=gray, 3=RGB)
    action_vocab_size: int            # Max discrete actions
    game_token_id: int                # Unique ID for this game in token stream


class BaseGameAdapter(ABC):
    """Abstract interface that every game must implement."""

    # ---- Identity (class-level attributes) ----

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

    # ---- Environment ----

    @abstractmethod
    def create_env(self, **kwargs) -> gym.Env:
        """Create and return a ready-to-use Gym environment.

        Kwargs may include game-specific options (world, stage, difficulty).
        The returned env should NOT have reward shaping applied -- the
        framework handles that via get_reward_config().

        Returns:
            gym.Env with observations suitable for ML (grayscale, resized).
        """

    @abstractmethod
    def get_action_space_info(self) -> ActionSpaceInfo:
        """Describe the discrete action space."""

    @abstractmethod
    def get_observation_shape(self) -> Tuple[int, ...]:
        """Shape of preprocessed observations, e.g. (84, 84, 4)."""

    # ---- Metrics Translation ----

    @abstractmethod
    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        """Map game-specific info dict to universal StandardMetrics.

        Args:
            info: The info dict from env.step().
            episode_time: Seconds elapsed in this episode.

        Returns:
            StandardMetrics with progress, score, completed, time_elapsed.
        """

    # ---- Reward Shaping ----

    def get_reward_config(self) -> RewardConfig:
        """Return reward shaping parameters.

        Override to customize. Defaults are tuned for action games.
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

        Example: {'world': (int, 1, 'World number 1-8'),
                  'stage': (int, 1, 'Stage number 1-4')}
        """
        return {}

    # ---- For Generalist Agent ----

    def get_token_config(self) -> TokenConfig:
        """How to tokenize this game for the Decision Transformer.

        Override if the defaults don't suit your game.
        """
        obs_shape = self.get_observation_shape()
        return TokenConfig(
            obs_resolution=(obs_shape[0], obs_shape[1]),
            obs_channels=obs_shape[2] if len(obs_shape) > 2 else 1,
            action_vocab_size=self.get_action_space_info().num_actions,
            game_token_id=hash(self.game_id) % 1024,
        )

    # ---- For SB3 Algorithms ----

    def needs_sb3_compat(self) -> bool:
        """Whether this game needs the SB3CompatWrapper.

        True for old-gym-API environments (like gym-super-mario-bros).
        False for gymnasium-native environments (built-in games).
        """
        return False
```

---

## Game Registry

The registry discovers and loads game adapters from three locations:

```python
class GameRegistry:
    """Discovers and manages game adapters."""

    def __init__(self):
        self._adapters: Dict[str, BaseGameAdapter] = {}

    def discover(self):
        """Auto-discover games from builtin/, retro/, and user/ folders."""
        for folder in ['games/builtin', 'games/retro', 'games/user']:
            for subfolder in os.listdir(folder):
                adapter_path = os.path.join(folder, subfolder, 'adapter.py')
                if os.path.exists(adapter_path):
                    adapter = self._load_adapter(adapter_path)
                    self._adapters[adapter.game_id] = adapter

    def list_games(self) -> List[BaseGameAdapter]:
        """Return all discovered game adapters."""
        return list(self._adapters.values())

    def get_game(self, game_id: str) -> BaseGameAdapter:
        """Get a specific game adapter by ID."""
        return self._adapters[game_id]

    def list_by_category(self, category: str) -> List[BaseGameAdapter]:
        """Filter games by category."""
        return [g for g in self._adapters.values() if g.category == category]
```

---

## Time-Based Reward System

Universal reward wrapper that applies to every game:

```python
class TimeRewardWrapper(gym.Wrapper):
    """Applies time-based rewards from the game's RewardConfig.

    Wraps any game environment and applies:
    - Time penalty: small negative reward per second
    - Idle penalty: extra penalty when no progress is made
    - Speed bonus: multiplied reward for fast level completion
    - Completion bonus: fixed reward for finishing

    The reward config comes from the game adapter, so each game
    can tune these values independently.
    """

    def __init__(self, env, reward_config, adapter):
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
                shaped_reward += self.config.completion_bonus * speed_ratio * self.config.speed_bonus_multiplier

        # Death/loss penalty
        if done and not metrics.completed:
            shaped_reward += self.config.death_penalty

        return obs, shaped_reward, done, info
```

---

## New Algorithms

### A2C (Advantage Actor-Critic)

Wraps stable-baselines3's A2C implementation. Nearly identical to
the existing PPO trainer. Uses synchronous gradient updates instead
of PPO's clipped objective. Faster wall-clock training but less stable.

Key differences from PPO trainer:
- Uses `stable_baselines3.A2C` instead of `stable_baselines3.PPO`
- Config: `n_steps=5` (shorter rollouts), `ent_coef=0.01`
- No clip_range parameter (A2C doesn't clip)
- Same CNN policy, same device selection, same checkpoint logic

### Rainbow DQN

Extends the existing vanilla DQN with six improvements:

1. **Double DQN**: Use online network to select actions, target
   network to evaluate them. Fixes Q-value overestimation.
2. **Prioritized Experience Replay**: Sample important transitions
   more often based on TD-error magnitude.
3. **Dueling Networks**: Separate network heads for state value V(s)
   and action advantage A(s,a). Better value estimation.
4. **Multi-step Returns**: Use n-step (n=3) bootstrap targets instead
   of single-step. Faster propagation of reward signal.
5. **Distributional RL (C51)**: Learn full return distribution using
   51 atoms instead of just the mean. More stable learning.
6. **Noisy Networks**: Replace epsilon-greedy exploration with learnable
   noise in network weights. State-dependent exploration.

Implementation approach:
- New `rainbow_network.py` with dueling + noisy + distributional heads
- New `prioritized_replay.py` with sum-tree data structure
- `rainbow_trainer.py` extends the training loop with n-step returns
- Keep existing vanilla DQN as a separate, simpler option

### Decision Transformer

Treats RL as sequence prediction. Architecture:

```
Input sequence per timestep:
    [game_token] [return_to_go] [obs_embedding] [action]

The transformer attends over the last K timesteps (context window)
and predicts the next action given the desired return-to-go.
```

Training:
1. Collect experience from individual game agents (DQN, PPO, etc.)
2. Tokenize and store in shared experience pool
3. Train transformer on batches sampled across all games
4. Each batch contains trajectories from multiple games

Inference:
1. User selects a game and desired performance level
2. Set return-to-go to the desired reward
3. Transformer autoregressively generates actions

Architecture details:
- GPT-2 style decoder-only transformer
- 4-8 layers, 256-512 embedding dim (fits RTX 5070 12GB VRAM)
- Separate embedding heads per modality (obs, action, reward, game)
- Context window of 20-30 timesteps
- Trained with standard cross-entropy loss on action prediction

---

## Built-In Games

### Snake
- **Category:** arcade
- **Observation:** 84x84 grayscale grid
- **Actions:** 4 (up, down, left, right)
- **Progress:** snake_length / max_possible_length
- **Completion:** fill entire grid (practically never)
- **Time reward:** moderate penalty (encourage quick food collection)
- **Curriculum:** grid size progression (8x8 -> 16x16 -> 32x32)

### Tetris
- **Category:** puzzle
- **Observation:** 84x84 grayscale board rendering
- **Actions:** 5 (left, right, rotate_cw, rotate_ccw, drop)
- **Progress:** lines_cleared / target_lines
- **Completion:** survive N lines or reach score threshold
- **Time reward:** low penalty (speed matters less than survival)

### Chess
- **Category:** board
- **Observation:** 84x84 rendered board OR 8x8x12 piece planes
- **Actions:** variable (legal moves, up to ~218)
- **Progress:** material_advantage normalized
- **Completion:** checkmate
- **Time reward:** zero (thinking is good in chess)
- **Note:** use python-chess library for game logic and move validation

### Checkers
- **Category:** board
- **Observation:** 84x84 rendered board OR 8x8x4 piece planes
- **Actions:** variable (legal moves)
- **Progress:** piece_advantage normalized
- **Completion:** opponent has no moves
- **Time reward:** zero

### Tic-Tac-Toe
- **Category:** board
- **Observation:** 84x84 rendered board OR 3x3x2 planes
- **Actions:** 9 (grid positions)
- **Progress:** moves_made / 9
- **Completion:** three in a row
- **Time reward:** zero
- **Note:** primarily for testing, agent should learn optimal play quickly

### Connect Four
- **Category:** board
- **Observation:** 84x84 rendered board OR 6x7x2 planes
- **Actions:** 7 (column drops)
- **Progress:** pieces_played / 42
- **Completion:** four in a row
- **Time reward:** zero

---

## ROM / Emulator Support

Integration with stable-retro for emulated games:

### Supported Platforms
- NES (Super Mario Bros, Mega Man, Castlevania)
- SNES (Super Mario World, Donkey Kong Country)
- Sega Genesis (Sonic the Hedgehog)
- Game Boy / GBA (Pokemon)
- Atari 2600 (Pong, Breakout, Space Invaders)

### ROM Import Flow
1. User clicks "Add Game" -> "ROM Game" in launcher
2. File picker for ROM file
3. Auto-detect console type from file extension / header
4. Generate default adapter with score-based rewards
5. User can customize reward config via launcher panel
6. Adapter saved to `games/user/` folder

### Legal Notice
ROMs are not included. Users must provide their own legally-owned
ROM files. The launcher displays this notice when adding ROM games.

---

## Launcher GUI Enhancements

The existing tkinter launcher gains these new features:

### Game Selection Panel
- Scrollable list of discovered games with icons and categories
- [Built-in] and [ROM] badges to distinguish game types
- "Add Game" button with wizard for ROM imports

### Training Mode Toggle
- "Single Game" mode: train one agent on one game (current behavior)
- "Generalist" mode: train Decision Transformer on selected games
  with checkboxes for which games to include

### Algorithm Selector
- Dropdown with all 6 algorithms + descriptions
- "Auto-select" checkbox: framework picks best algorithm for game category
  - Platformers: Rainbow DQN or PPO
  - Board games: DQN or NEAT
  - Arcade: PPO or A2C
  - Generalist mode: Decision Transformer (forced)

### Speed Rewards Toggle
- "Speed Rewards" checkbox (on by default for action games)
- Enables/disables time-based reward shaping

### Game-Specific Options
- Dynamic panel that shows options from adapter.get_game_specific_options()
- Mario shows: World, Stage dropdowns
- Snake shows: Grid size, Speed
- Chess shows: Difficulty (opponent strength)

---

## Honest Limitations

### Will Work Well
- Training individual agents per game with all algorithms
- Comparing algorithm performance across game types
- Transfer learning between similar games (Mario -> Mega Man)
- Time-based rewards accelerating action game training
- Plugin system for community game contributions

### Hard But Possible
- Decision Transformer generalist playing 5-10 games with one model
- Pokemon RL training (very sparse rewards, needs heavy shaping)
- Measuring cross-game transfer effects

### Not Realistic
- Generalist agent outperforming specialized agents at individual games
- Superhuman Chess via RL alone (would need MCTS integration)
- 3D game support (architecture is pixel-based 2D)
- Negative transfer: some game combinations may hurt rather than help

---

## Implementation Phases

### Phase 1: Foundation (Multi-Game Framework)
- [ ] Game adapter interface (BaseGameAdapter, StandardMetrics, RewardConfig)
- [ ] Game registry with auto-discovery
- [ ] Migrate Mario behind the adapter interface
- [ ] Universal TimeRewardWrapper
- [ ] Build Snake game + adapter
- [ ] Build Tetris game + adapter
- [ ] Build Connect Four game + adapter
- [ ] A2C trainer (wraps SB3.A2C)
- [ ] Universal environment factory (creates env from any adapter)
- [ ] Updated launcher with game selection dropdown
- [ ] Updated main.py with --game flag
- [ ] Tests for adapter interface, registry, reward system
- [ ] All existing features preserved and tested

### Phase 2: Algorithm Expansion + ROM Support
- [ ] Rainbow DQN: prioritized replay buffer
- [ ] Rainbow DQN: dueling network architecture
- [ ] Rainbow DQN: noisy nets + distributional RL
- [ ] Rainbow DQN: multi-step returns
- [ ] Rainbow DQN: trainer integration
- [ ] stable-retro integration for emulated games
- [ ] ROM import wizard in launcher
- [ ] Retro adapter template (generic score-based rewards)
- [ ] Sonic adapter
- [ ] Pokemon adapter (with exploration-based rewards)
- [ ] Build Chess game + adapter (using python-chess)
- [ ] Build Checkers game + adapter
- [ ] Build Tic-Tac-Toe game + adapter
- [ ] Cross-game comparison in dashboard
- [ ] Per-game config panel in launcher

### Phase 3: Generalist Agent
- [ ] Shared experience store (collects from all game agents)
- [ ] Observation/action tokenizer
- [ ] Decision Transformer network architecture
- [ ] Decision Transformer trainer
- [ ] Multi-game training mode in launcher
- [ ] Experience collection pipeline (agents -> store -> DT)
- [ ] Cross-game metrics visualization
- [ ] Transfer learning experiment tooling

### Phase 4: Community + Polish
- [ ] User plugin folder with template generator
- [ ] "Add Game" wizard for custom Python environments
- [ ] Training presets ("Quick Demo", "Deep Training", "Overnight")
- [ ] Documentation and tutorials
- [ ] Example adapters for popular ROMs
