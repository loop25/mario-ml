"""Tests for retro game adapters (BaseRetroAdapter, Sonic, Pokemon).

These tests verify adapter identity, metrics extraction, reward configs,
and dashboard configs WITHOUT requiring stable-retro or ROM files.
Environment creation tests are skipped when stable-retro is not installed.
"""
import pytest
import numpy as np

from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics, RewardConfig, ActionSpaceInfo


# ─── BaseRetroAdapter ───────────────────────────────────────────


class TestBaseRetroAdapter:
    """Test the abstract base class for retro adapters."""

    def test_import(self):
        from games.retro.base_retro_adapter import BaseRetroAdapter
        assert BaseRetroAdapter is not None

    def test_is_abstract(self):
        from games.retro.base_retro_adapter import BaseRetroAdapter
        with pytest.raises(TypeError):
            # Can't instantiate — rom_name and other abstracts not defined
            BaseRetroAdapter()

    def test_default_frame_skip(self):
        from games.retro.base_retro_adapter import BaseRetroAdapter
        # Create a minimal concrete subclass
        class DummyRetro(BaseRetroAdapter):
            @property
            def name(self): return 'Dummy'
            @property
            def game_id(self): return 'dummy'
            @property
            def category(self): return 'test'
            @property
            def description(self): return 'Test adapter'
            @property
            def rom_name(self): return 'DummyGame-Console'
            def get_action_space_info(self):
                return ActionSpaceInfo(num_actions=2, action_labels=['A', 'B'])
            def extract_metrics(self, info, episode_time):
                return StandardMetrics(progress=0.0, score=0.0,
                                       completed=False, time_elapsed=0.0)

        adapter = DummyRetro()
        assert adapter.frame_skip == 4
        assert adapter.frame_stack == 4
        assert adapter.render_size == (84, 84)

    def test_default_reward_config(self):
        from games.retro.base_retro_adapter import BaseRetroAdapter

        class DummyRetro(BaseRetroAdapter):
            @property
            def name(self): return 'Dummy'
            @property
            def game_id(self): return 'dummy'
            @property
            def category(self): return 'test'
            @property
            def description(self): return 'Test adapter'
            @property
            def rom_name(self): return 'DummyGame-Console'
            def get_action_space_info(self):
                return ActionSpaceInfo(num_actions=2, action_labels=['A', 'B'])
            def extract_metrics(self, info, episode_time):
                return StandardMetrics(progress=0.0, score=0.0,
                                       completed=False, time_elapsed=0.0)

        adapter = DummyRetro()
        rc = adapter.get_reward_config()
        assert isinstance(rc, RewardConfig)
        assert rc.time_penalty_per_second > 0
        assert rc.completion_bonus > 0
        assert rc.death_penalty < 0

    def test_needs_sb3_compat(self):
        from games.retro.base_retro_adapter import BaseRetroAdapter

        class DummyRetro(BaseRetroAdapter):
            @property
            def name(self): return 'Dummy'
            @property
            def game_id(self): return 'dummy'
            @property
            def category(self): return 'test'
            @property
            def description(self): return 'Test adapter'
            @property
            def rom_name(self): return 'DummyGame-Console'
            def get_action_space_info(self):
                return ActionSpaceInfo(num_actions=2, action_labels=['A', 'B'])
            def extract_metrics(self, info, episode_time):
                return StandardMetrics(progress=0.0, score=0.0,
                                       completed=False, time_elapsed=0.0)

        adapter = DummyRetro()
        assert adapter.needs_sb3_compat() is True

    def test_supported_algorithms_no_neat(self):
        from games.retro.base_retro_adapter import BaseRetroAdapter

        class DummyRetro(BaseRetroAdapter):
            @property
            def name(self): return 'Dummy'
            @property
            def game_id(self): return 'dummy'
            @property
            def category(self): return 'test'
            @property
            def description(self): return 'Test adapter'
            @property
            def rom_name(self): return 'DummyGame-Console'
            def get_action_space_info(self):
                return ActionSpaceInfo(num_actions=2, action_labels=['A', 'B'])
            def extract_metrics(self, info, episode_time):
                return StandardMetrics(progress=0.0, score=0.0,
                                       completed=False, time_elapsed=0.0)

        adapter = DummyRetro()
        algos = adapter.supported_algorithms()
        assert 'neat' not in algos
        assert 'ppo' in algos
        assert 'rainbow' in algos

    def test_observation_shape(self):
        from games.retro.base_retro_adapter import BaseRetroAdapter

        class DummyRetro(BaseRetroAdapter):
            @property
            def name(self): return 'Dummy'
            @property
            def game_id(self): return 'dummy'
            @property
            def category(self): return 'test'
            @property
            def description(self): return 'Test adapter'
            @property
            def rom_name(self): return 'DummyGame-Console'
            def get_action_space_info(self):
                return ActionSpaceInfo(num_actions=2, action_labels=['A', 'B'])
            def extract_metrics(self, info, episode_time):
                return StandardMetrics(progress=0.0, score=0.0,
                                       completed=False, time_elapsed=0.0)

        adapter = DummyRetro()
        assert adapter.get_observation_shape() == (84, 84, 4)

    def test_is_available_without_retro(self):
        """Without stable-retro installed, is_available should return False."""
        from games.retro.base_retro_adapter import BaseRetroAdapter, HAS_RETRO

        class DummyRetro(BaseRetroAdapter):
            @property
            def name(self): return 'Dummy'
            @property
            def game_id(self): return 'dummy'
            @property
            def category(self): return 'test'
            @property
            def description(self): return 'Test adapter'
            @property
            def rom_name(self): return 'DummyGame-Console'
            def get_action_space_info(self):
                return ActionSpaceInfo(num_actions=2, action_labels=['A', 'B'])
            def extract_metrics(self, info, episode_time):
                return StandardMetrics(progress=0.0, score=0.0,
                                       completed=False, time_elapsed=0.0)

        adapter = DummyRetro()
        if not HAS_RETRO:
            assert adapter.is_available() is False

    def test_create_env_raises_without_retro(self):
        """create_env should raise ImportError if stable-retro not installed."""
        from games.retro.base_retro_adapter import BaseRetroAdapter, HAS_RETRO

        class DummyRetro(BaseRetroAdapter):
            @property
            def name(self): return 'Dummy'
            @property
            def game_id(self): return 'dummy'
            @property
            def category(self): return 'test'
            @property
            def description(self): return 'Test adapter'
            @property
            def rom_name(self): return 'DummyGame-Console'
            def get_action_space_info(self):
                return ActionSpaceInfo(num_actions=2, action_labels=['A', 'B'])
            def extract_metrics(self, info, episode_time):
                return StandardMetrics(progress=0.0, score=0.0,
                                       completed=False, time_elapsed=0.0)

        if not HAS_RETRO:
            adapter = DummyRetro()
            with pytest.raises(ImportError, match='stable-retro'):
                adapter.create_env()


# ─── Sonic Adapter ──────────────────────────────────────────────


class TestSonicAdapter:
    def _make(self):
        from games.retro.sonic.adapter import SonicAdapter
        return SonicAdapter()

    def test_is_base_game_adapter(self):
        adapter = self._make()
        assert isinstance(adapter, BaseGameAdapter)

    def test_identity(self):
        adapter = self._make()
        assert adapter.game_id == 'sonic'
        assert adapter.name == 'Sonic the Hedgehog'
        assert adapter.category == 'platformer'

    def test_rom_name(self):
        adapter = self._make()
        assert adapter.rom_name == 'SonicTheHedgehog-Genesis'

    def test_initial_state(self):
        adapter = self._make()
        assert adapter.initial_state == 'GreenHillZone.Act1'

    def test_action_space(self):
        adapter = self._make()
        info = adapter.get_action_space_info()
        assert info.num_actions == 12
        assert len(info.action_labels) == 12
        assert 'Right' in info.action_labels
        assert 'Jump' in info.action_labels

    def test_observation_shape(self):
        adapter = self._make()
        assert adapter.get_observation_shape() == (84, 84, 4)

    def test_extract_metrics_progress(self):
        adapter = self._make()
        info = {'x': 4800, 'rings': 50, 'level_end_bonus': 0}
        metrics = adapter.extract_metrics(info, 30.0)
        assert isinstance(metrics, StandardMetrics)
        assert metrics.progress == pytest.approx(0.5, abs=0.01)
        assert metrics.score == 50.0
        assert metrics.completed is False
        assert metrics.time_elapsed == 30.0

    def test_extract_metrics_completed(self):
        adapter = self._make()
        info = {'x': 9600, 'rings': 100, 'level_end_bonus': 1000}
        metrics = adapter.extract_metrics(info, 60.0)
        assert metrics.progress == pytest.approx(1.0)
        assert metrics.completed is True

    def test_extract_metrics_empty_info(self):
        adapter = self._make()
        metrics = adapter.extract_metrics({}, 0.0)
        assert metrics.progress == 0.0
        assert metrics.score == 0.0
        assert metrics.completed is False

    def test_reward_config_speed_bonus(self):
        adapter = self._make()
        rc = adapter.get_reward_config()
        assert isinstance(rc, RewardConfig)
        assert rc.speed_bonus_multiplier == 2.0  # Sonic emphasizes speed
        assert rc.par_time_seconds == 90.0

    def test_reward_config_penalties(self):
        adapter = self._make()
        rc = adapter.get_reward_config()
        assert rc.time_penalty_per_second > 0
        assert rc.death_penalty < 0
        assert rc.completion_bonus > 0

    def test_dashboard_config(self):
        adapter = self._make()
        config = adapter.get_dashboard_config()
        assert config['graph_2_metric'] == 'distance'
        assert config['graph_2_info_key'] == 'x'
        assert 'Distance' in config['graph_2_title']

    def test_completion_criteria(self):
        adapter = self._make()
        criteria = adapter.get_completion_criteria()
        assert criteria['metric'] == 'reward'
        assert criteria['threshold'] == 500.0

    def test_game_specific_options(self):
        adapter = self._make()
        opts = adapter.get_game_specific_options()
        assert 'state' in opts

    def test_needs_sb3_compat(self):
        adapter = self._make()
        assert adapter.needs_sb3_compat() is True

    def test_no_neat_support(self):
        adapter = self._make()
        algos = adapter.supported_algorithms()
        assert 'neat' not in algos

    def test_is_available_without_retro(self):
        from games.retro.base_retro_adapter import HAS_RETRO
        adapter = self._make()
        if not HAS_RETRO:
            assert adapter.is_available() is False

    def test_create_env_raises_without_retro(self):
        from games.retro.base_retro_adapter import HAS_RETRO
        if not HAS_RETRO:
            adapter = self._make()
            with pytest.raises(ImportError, match='stable-retro'):
                adapter.create_env()


# ─── Pokemon Adapter ────────────────────────────────────────────


class TestPokemonAdapter:
    """Tests for the PyBoy-based Pokemon Red adapter."""

    def _make(self):
        from games.retro.pokemon.adapter import PokemonAdapter
        return PokemonAdapter()

    def test_is_base_game_adapter(self):
        adapter = self._make()
        assert isinstance(adapter, BaseGameAdapter)

    def test_identity(self):
        adapter = self._make()
        assert adapter.game_id == 'pokemon'
        assert adapter.name == 'Pokemon Red'
        assert adapter.category == 'rpg'

    def test_action_space(self):
        adapter = self._make()
        info = adapter.get_action_space_info()
        # 7 actions: D-pad(4) + A, B, Start (no Select in training)
        assert info.num_actions == 7
        assert len(info.action_labels) == 7
        assert 'A' in info.action_labels
        assert 'B' in info.action_labels
        assert 'Up' in info.action_labels

    def test_observation_shape(self):
        adapter = self._make()
        # PyBoy Game Boy: 144x160 downscaled 2x = 72x80, 3 stacked frames
        assert adapter.get_observation_shape() == (72, 80, 3)

    def test_extract_metrics_no_badges(self):
        adapter = self._make()
        info = {'badges': 0}
        metrics = adapter.extract_metrics(info, 120.0)
        assert isinstance(metrics, StandardMetrics)
        assert metrics.progress == 0.0
        assert metrics.score == 0.0
        assert metrics.completed is False

    def test_extract_metrics_one_badge(self):
        adapter = self._make()
        # New env returns badge count directly (not bitmask)
        info = {'badges': 1}
        metrics = adapter.extract_metrics(info, 300.0)
        assert metrics.progress == pytest.approx(1.0 / 8.0)
        assert metrics.score == 1.0
        assert metrics.completed is False

    def test_extract_metrics_multiple_badges(self):
        adapter = self._make()
        info = {'badges': 3}
        metrics = adapter.extract_metrics(info, 600.0)
        assert metrics.progress == pytest.approx(3.0 / 8.0)
        assert metrics.score == 3.0

    def test_extract_metrics_all_badges(self):
        adapter = self._make()
        info = {'badges': 8}
        metrics = adapter.extract_metrics(info, 3600.0)
        assert metrics.progress == pytest.approx(1.0)
        assert metrics.score == 8.0
        assert metrics.completed is True

    def test_extract_metrics_empty_info(self):
        adapter = self._make()
        metrics = adapter.extract_metrics({}, 0.0)
        assert metrics.progress == 0.0
        assert metrics.score == 0.0

    def test_reward_config_rpg_style(self):
        """RPGs should have no time pressure or speed bonus."""
        adapter = self._make()
        rc = adapter.get_reward_config()
        assert isinstance(rc, RewardConfig)
        assert rc.time_penalty_per_second == 0.0
        assert rc.speed_bonus_multiplier == 0.0

    def test_reward_config_env_internal(self):
        """Reward shaping is handled internally by the env."""
        adapter = self._make()
        rc = adapter.get_reward_config()
        # All zeros because the env handles reward shaping itself
        assert rc.completion_bonus == 0.0
        assert rc.death_penalty == 0.0

    def test_dashboard_config(self):
        adapter = self._make()
        config = adapter.get_dashboard_config()
        assert config['graph_2_metric'] == 'badges'
        assert 'Badges' in config['graph_2_title']

    def test_completion_criteria(self):
        adapter = self._make()
        criteria = adapter.get_completion_criteria()
        assert criteria['metric'] == 'badges'
        assert criteria['threshold'] == 1
        assert 'badge' in criteria['description'].lower()

    def test_game_specific_options(self):
        adapter = self._make()
        opts = adapter.get_game_specific_options()
        assert 'gb_path' in opts
        assert 'simple_obs' in opts

    def test_needs_sb3_compat(self):
        """PyBoy env is gymnasium-native, no compat wrapper needed."""
        adapter = self._make()
        assert adapter.needs_sb3_compat() is False

    def test_no_neat_support(self):
        adapter = self._make()
        algos = adapter.supported_algorithms()
        assert 'neat' not in algos

    def test_is_available_with_pyboy(self):
        """PyBoy adapter reports available when pyboy is installed."""
        adapter = self._make()
        try:
            import pyboy
            assert adapter.is_available() is True
        except ImportError:
            assert adapter.is_available() is False

    def test_create_env_raises_without_rom(self):
        """create_env should raise FileNotFoundError without ROM."""
        adapter = self._make()
        with pytest.raises(FileNotFoundError):
            adapter.create_env(gb_path='nonexistent_rom.gb', headless=True)
