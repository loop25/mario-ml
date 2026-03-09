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
