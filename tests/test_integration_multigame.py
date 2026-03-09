"""Integration tests for the multi-game plugin system.

These verify that the full pipeline works end-to-end:
registry discovery -> adapter -> env creation -> step/reset.
"""
import pytest

from games.registry import GameRegistry
from games.base_adapter import BaseGameAdapter
from games.reward_config import StandardMetrics
from src.environment.universal_env import create_env_from_adapter

try:
    from src.environment.mario_env import create_mario_env
    _HAS_MARIO = True
except ImportError:
    _HAS_MARIO = False


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

    @pytest.mark.skipif(not _HAS_MARIO, reason='gym-super-mario-bros not installed')
    def test_discovers_mario(self):
        registry = GameRegistry()
        registry.discover()
        game = registry.get_game('mario')
        assert game.name == 'Super Mario Bros'

    def test_list_games_returns_builtin(self):
        registry = GameRegistry()
        registry.discover()
        ids = registry.list_game_ids()
        assert 'snake' in ids
        assert 'connect4' in ids

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
