"""
Pokemon Red gymnasium environment using PyBoy emulator.

Adapted from PokemonRedExperiments v2 (MIT License):
  https://github.com/PWhiddy/PokemonRedExperiments

Core changes from the original:
  - Removed streaming (WebSocket), video recording (mediapy), and
    matplotlib dependencies
  - Added 'simple_obs' mode that returns a plain image observation
    (compatible with standard CNN policies) instead of the Dict space
  - Integrated with this project's game adapter system
  - ROM and state file paths are configurable via kwargs

The environment reads Game Boy RAM directly to extract game state
(player position, HP, badges, event flags) and computes a rich
exploration-based reward signal.

Reward components:
  - event:   Event flags triggered (story progress)
  - heal:    Cumulative healing (incentivizes Pokemon Centers)
  - badge:   Gym badges earned
  - explore: Unique map tiles visited (primary exploration signal)
  - stuck:   Penalty for lingering on heavily-visited tiles
"""

import json
import os
import uuid

import numpy as np
from gymnasium import Env, spaces

try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
    HAS_PYBOY = True
except ImportError:
    HAS_PYBOY = False

# Memory address ranges for event flags
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E
MUSEUM_TICKET = (0xD754, 0)

# Pokemon party level addresses
LEVEL_ADDRS = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]

# Current HP addresses (2 bytes each, big-endian)
HP_ADDRS = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]

# Max HP addresses
MAX_HP_ADDRS = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]

# Game Boy button mappings
VALID_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START,
] if HAS_PYBOY else []

RELEASE_ACTIONS = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_START,
] if HAS_PYBOY else []

ACTION_LABELS = ['Down', 'Left', 'Right', 'Up', 'A', 'B', 'Start']

# Data file directory
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class RedGymEnv(Env):
    """Pokemon Red environment using PyBoy emulator.

    Args (passed via config dict or kwargs):
        gb_path: Path to PokemonRed.gb ROM file.
        init_state: Path to .state file (PyBoy save state to skip intro).
        headless: If True, run without display window. Default True.
        action_freq: Emulator ticks per action. Default 24.
        max_steps: Maximum steps per episode. Default 2048*80.
        explore_weight: Weight for exploration reward. Default 1.0.
        reward_scale: Global reward scale. Default 0.5.
        simple_obs: If True, return plain image obs instead of Dict.
                    Compatible with standard CnnPolicy. Default True.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config=None, **kwargs):
        super().__init__()

        if not HAS_PYBOY:
            raise ImportError(
                "PyBoy is required for Pokemon Red. "
                "Install with: pip install pyboy"
            )

        # Merge config dict and kwargs
        if config is None:
            config = {}
        config.update(kwargs)

        self.gb_path = config.get("gb_path", "PokemonRed.gb")
        self.init_state = config.get("init_state", None)
        self.headless = config.get("headless", True)
        self.act_freq = config.get("action_freq", 24)
        self.max_steps = config.get("max_steps", 2048 * 80)
        self.explore_weight = config.get("explore_weight", 1.0)
        self.reward_scale = config.get("reward_scale", 0.5)
        self.simple_obs = config.get("simple_obs", True)

        # Auto-find init state if not specified
        if self.init_state is None:
            candidates = [
                os.path.join(_DATA_DIR, "init.state"),
                os.path.join(os.path.dirname(self.gb_path), "init.state"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    self.init_state = c
                    break

        self.frame_stacks = 3
        self.coords_pad = 12
        self.enc_freqs = 8
        self.instance_id = str(uuid.uuid4())[:8]

        # Load event names for tracking
        events_path = os.path.join(_DATA_DIR, "events.json")
        if os.path.isfile(events_path):
            with open(events_path) as f:
                self.event_names = json.load(f)
        else:
            self.event_names = {}

        # Import global map utilities
        from games.retro.pokemon.global_map import (
            local_to_global, GLOBAL_MAP_SHAPE,
        )
        self._local_to_global = local_to_global
        self._global_map_shape = GLOBAL_MAP_SHAPE

        # Define spaces
        self.action_space = spaces.Discrete(len(VALID_ACTIONS))

        if self.simple_obs:
            # Simple image observation: Game Boy screen (144x160 -> 72x80)
            # with 3 stacked grayscale frames
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(72, 80, self.frame_stacks),
                dtype=np.uint8,
            )
        else:
            # Full Dict observation (matches PokemonRedExperiments v2)
            self.observation_space = spaces.Dict({
                "screens": spaces.Box(
                    low=0, high=255,
                    shape=(72, 80, self.frame_stacks),
                    dtype=np.uint8,
                ),
                "health": spaces.Box(low=0, high=1, shape=(1,)),
                "level": spaces.Box(
                    low=-1, high=1, shape=(self.enc_freqs,)
                ),
                "badges": spaces.MultiBinary(8),
                "events": spaces.MultiBinary(
                    (EVENT_FLAGS_END - EVENT_FLAGS_START) * 8
                ),
                "map": spaces.Box(
                    low=0, high=255,
                    shape=(self.coords_pad * 4, self.coords_pad * 4, 1),
                    dtype=np.uint8,
                ),
                "recent_actions": spaces.MultiDiscrete(
                    [len(VALID_ACTIONS)] * self.frame_stacks
                ),
            })

        # Create PyBoy instance
        head = "null" if self.headless else "SDL2"
        self.pyboy = PyBoy(self.gb_path, window=head)

        if not self.headless:
            self.pyboy.set_emulation_speed(6)

        # Internal state (initialized in reset)
        self.step_count = 0
        self.reset_count = 0
        self.seen_coords = {}
        self.explore_map = None
        self.recent_screens = None
        self.recent_actions = None
        self.total_reward = 0.0
        self.progress_reward = {}
        self.last_health = 1.0
        self.total_healing_rew = 0.0
        self.died_count = 0
        self.party_size = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.max_opponent_level = 0
        self.max_map_progress = 0
        self.base_event_flags = 0
        self.current_event_flags_set = {}

        # Essential map locations for progress tracking
        self.essential_map_locations = {
            v: i for i, v in enumerate([
                40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65
            ])
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Load save state to skip game intro
        if self.init_state and os.path.isfile(self.init_state):
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)

        # Reset tracking state
        self.seen_coords = {}
        self.explore_map = np.zeros(self._global_map_shape, dtype=np.uint8)
        self.recent_screens = np.zeros(
            (72, 80, self.frame_stacks), dtype=np.uint8
        )
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1.0
        self.total_healing_rew = 0.0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.max_opponent_level = 0
        self.max_map_progress = 0
        self.current_event_flags_set = {}

        self.base_event_flags = sum(
            bin(self._read_mem(i)).count("1")
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
        )

        self.progress_reward = self._get_game_state_reward()
        self.total_reward = sum(self.progress_reward.values())
        self.reset_count += 1

        return self._get_obs(), {}

    def step(self, action):
        self._run_action(action)
        self._update_recent_actions(action)
        self._update_seen_coords()
        self._update_explore_map()
        self._update_heal_reward()

        self.party_size = self._read_mem(0xD163)

        # Compute reward delta
        self.progress_reward = self._get_game_state_reward()
        new_total = sum(self.progress_reward.values())
        reward = new_total - self.total_reward
        self.total_reward = new_total

        self.last_health = self._read_hp_fraction()
        self._update_map_progress()

        self.step_count += 1
        truncated = self.step_count >= self.max_steps - 1

        obs = self._get_obs()

        # Info dict for metrics extraction
        info = {
            "badges": self._get_badges(),
            "explore_count": len(self.seen_coords),
            "event_reward": self.progress_reward.get("event", 0),
            "hp_fraction": self._read_hp_fraction(),
            "map_progress": self.max_map_progress,
            "deaths": self.died_count,
            "step": self.step_count,
            "total_reward": self.total_reward,
        }

        return obs, reward, False, truncated, info

    def render(self, mode="rgb_array"):
        """Return the current Game Boy screen as an RGB array."""
        # PyBoy screen is (144, 160, 4) RGBA
        screen = self.pyboy.screen.ndarray
        if screen.shape[-1] == 4:
            return screen[:, :, :3].copy()
        return screen.copy()

    def close(self):
        if hasattr(self, 'pyboy') and self.pyboy is not None:
            self.pyboy.stop()

    # ---- Internal: Observation ----

    def _get_obs(self):
        # Render downscaled grayscale frame (144x160 -> 72x80)
        screen = self.pyboy.screen.ndarray[:, :, 0:1]  # grayscale channel
        from skimage.transform import downscale_local_mean
        screen = downscale_local_mean(screen, (2, 2, 1)).astype(np.uint8)

        # Update frame stack
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = screen[:, :, 0]

        if self.simple_obs:
            return self.recent_screens

        # Full Dict observation
        level_sum = 0.02 * sum(
            self._read_mem(a) for a in LEVEL_ADDRS
        )
        return {
            "screens": self.recent_screens,
            "health": np.array([self._read_hp_fraction()], dtype=np.float32),
            "level": self._fourier_encode(level_sum),
            "badges": np.array(
                [int(b) for b in f"{self._read_mem(0xD356):08b}"],
                dtype=np.int8,
            ),
            "events": np.array(self._read_event_bits(), dtype=np.int8),
            "map": self._get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions,
        }

    # ---- Internal: Emulator interaction ----

    def _run_action(self, action):
        """Press a button for 8 ticks, release, then idle."""
        self.pyboy.send_input(VALID_ACTIONS[action])
        render = not self.headless
        self.pyboy.tick(8, render)
        self.pyboy.send_input(RELEASE_ACTIONS[action])
        self.pyboy.tick(self.act_freq - 8 - 1, render)
        self.pyboy.tick(1, True)  # Always render final tick

    def _read_mem(self, addr):
        return self.pyboy.memory[addr]

    def _read_bit(self, addr, bit):
        return bin(256 + self._read_mem(addr))[-bit - 1] == "1"

    def _read_hp(self, start):
        return 256 * self._read_mem(start) + self._read_mem(start + 1)

    def _read_hp_fraction(self):
        hp_sum = sum(self._read_hp(a) for a in HP_ADDRS)
        max_hp_sum = sum(self._read_hp(a) for a in MAX_HP_ADDRS)
        return hp_sum / max(max_hp_sum, 1)

    def _read_event_bits(self):
        return [
            int(bit)
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
            for bit in f"{self._read_mem(i):08b}"
        ]

    # ---- Internal: Game state ----

    def _get_game_coords(self):
        return (
            self._read_mem(0xD362),  # x
            self._read_mem(0xD361),  # y
            self._read_mem(0xD35E),  # map
        )

    def _get_badges(self):
        return bin(self._read_mem(0xD356)).count("1")

    def _get_all_events_reward(self):
        return max(
            sum(
                bin(self._read_mem(i)).count("1")
                for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
            )
            - self.base_event_flags
            - int(self._read_bit(MUSEUM_TICKET[0], MUSEUM_TICKET[1])),
            0,
        )

    def _update_max_event_rew(self):
        cur = self._get_all_events_reward()
        self.max_event_rew = max(cur, self.max_event_rew)
        return self.max_event_rew

    # ---- Internal: Reward ----

    def _get_game_state_reward(self):
        return {
            "event": self.reward_scale * self._update_max_event_rew() * 4,
            "heal": self.reward_scale * self.total_healing_rew * 10,
            "badge": self.reward_scale * self._get_badges() * 10,
            "explore": (
                self.reward_scale * self.explore_weight
                * len(self.seen_coords) * 0.1
            ),
            "stuck": (
                self.reward_scale
                * self._get_stuck_penalty() * -0.05
            ),
        }

    def _get_stuck_penalty(self):
        x, y, m = self._get_game_coords()
        key = f"x:{x} y:{y} m:{m}"
        count = self.seen_coords.get(key, 0)
        return 1 if count >= 600 else 0

    def _update_seen_coords(self):
        if self._read_mem(0xD057) == 0:  # Not in battle
            x, y, m = self._get_game_coords()
            key = f"x:{x} y:{y} m:{m}"
            self.seen_coords[key] = self.seen_coords.get(key, 0) + 1

    def _update_heal_reward(self):
        cur_health = self._read_hp_fraction()
        if cur_health > self.last_health and self._read_mem(0xD163) == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount * heal_amount
            else:
                self.died_count += 1

    # ---- Internal: Exploration map ----

    def _update_explore_map(self):
        c = self._get_global_coords()
        if (0 <= c[0] < self.explore_map.shape[0]
                and 0 <= c[1] < self.explore_map.shape[1]):
            self.explore_map[c[0], c[1]] = 255

    def _get_explore_map(self):
        c = self._get_global_coords()
        pad = self.coords_pad
        if (0 <= c[0] < self.explore_map.shape[0]
                and 0 <= c[1] < self.explore_map.shape[1]):
            out = self.explore_map[
                c[0] - pad:c[0] + pad,
                c[1] - pad:c[1] + pad,
            ]
        else:
            out = np.zeros((pad * 2, pad * 2), dtype=np.uint8)

        # Upscale 2x using simple repeat (avoids einops dependency)
        return np.repeat(np.repeat(out, 2, axis=0), 2, axis=1)

    def _get_global_coords(self):
        x, y, m = self._get_game_coords()
        return self._local_to_global(y, x, m)

    def _update_map_progress(self):
        map_idx = self._read_mem(0xD35E)
        if map_idx in self.essential_map_locations:
            self.max_map_progress = max(
                self.max_map_progress,
                self.essential_map_locations[map_idx],
            )

    # ---- Internal: Encoding ----

    def _fourier_encode(self, val):
        return np.sin(
            val * 2 ** np.arange(self.enc_freqs)
        ).astype(np.float32)

    def _update_recent_actions(self, action):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action
