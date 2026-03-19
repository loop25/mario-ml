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


class TestAlgorithmCompatibility:
    """Verify supported_algorithms() is correct for each game."""

    def test_snake_excludes_neat(self):
        registry = GameRegistry()
        registry.discover()
        adapter = registry.get_game('snake')
        supported = adapter.supported_algorithms()
        assert 'neat' not in supported
        assert 'ppo' in supported
        assert 'dqn' in supported
        assert 'a2c' in supported

    def test_connect4_excludes_neat(self):
        registry = GameRegistry()
        registry.discover()
        adapter = registry.get_game('connect4')
        supported = adapter.supported_algorithms()
        assert 'neat' not in supported
        assert 'ppo' in supported

    @pytest.mark.skipif(not _HAS_MARIO, reason='gym-super-mario-bros not installed')
    def test_mario_supports_all(self):
        registry = GameRegistry()
        registry.discover()
        adapter = registry.get_game('mario')
        supported = adapter.supported_algorithms()
        assert 'neat' in supported
        assert 'ppo' in supported
        assert 'dqn' in supported
        assert 'a2c' in supported

    def test_all_adapters_return_list(self):
        registry = GameRegistry()
        registry.discover()
        for adapter in registry.list_games():
            supported = adapter.supported_algorithms()
            assert isinstance(supported, list)
            assert len(supported) > 0


class TestGameAwareAutoResume:
    """Verify find_resume_checkpoint filters by game_id."""

    def test_skips_checkpoint_from_different_game(self, tmp_path):
        import json
        from main import find_resume_checkpoint
        # Create a fake checkpoint from mario
        algo_dir = tmp_path / 'ppo'
        algo_dir.mkdir()
        (algo_dir / 'metadata.json').write_text(json.dumps({
            'algorithm': 'PPOTrainer',
            'game_id': 'mario',
            'episode': 100,
        }))
        (algo_dir / 'final.zip').write_text('fake')
        # Should NOT match when looking for snake
        path, meta = find_resume_checkpoint('ppo', 'snake', save_dir=str(tmp_path))
        assert path is None

    def test_matches_checkpoint_from_same_game(self, tmp_path):
        import json
        from main import find_resume_checkpoint
        algo_dir = tmp_path / 'ppo'
        algo_dir.mkdir()
        (algo_dir / 'metadata.json').write_text(json.dumps({
            'algorithm': 'PPOTrainer',
            'game_id': 'snake',
            'episode': 100,
        }))
        (algo_dir / 'final.zip').write_text('fake')
        path, meta = find_resume_checkpoint('ppo', 'snake', save_dir=str(tmp_path))
        assert path is not None
        assert 'final.zip' in path

    def test_old_checkpoint_without_game_id_matches_mario(self, tmp_path):
        import json
        from main import find_resume_checkpoint
        algo_dir = tmp_path / 'neat'
        algo_dir.mkdir()
        (algo_dir / 'metadata.json').write_text(json.dumps({
            'algorithm': 'NEATTrainer',
            'episode': 50,
        }))
        (algo_dir / 'final_best_genome.pkl').write_text('fake')
        # Old checkpoints (no game_id) default to 'mario'
        path, meta = find_resume_checkpoint('neat', 'mario', save_dir=str(tmp_path))
        assert path is not None
        # But should NOT match for other games
        path2, _ = find_resume_checkpoint('neat', 'connect4', save_dir=str(tmp_path))
        assert path2 is None


class TestAllAdaptersConformToInterface:
    """Verify every discovered adapter implements the full interface."""

    def test_all_adapters_have_required_properties(self):
        registry = GameRegistry()
        registry.discover()
        for adapter in registry.list_games():
            assert isinstance(adapter.name, str) and len(adapter.name) > 0
            assert isinstance(adapter.game_id, str) and len(adapter.game_id) > 0
            assert adapter.category in ('platformer', 'puzzle', 'board', 'arcade', 'rpg', 'cooperative')
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


class TestDashboardConfig:
    """Test that each adapter provides valid dashboard configuration."""

    def test_all_adapters_have_dashboard_config(self):
        registry = GameRegistry()
        registry.discover()
        required_keys = {
            'graph_2_title', 'graph_2_metric', 'graph_2_info_key',
            'status_metric_label', 'status_metric_key',
        }
        for adapter in registry.list_games():
            config = adapter.get_dashboard_config()
            assert isinstance(config, dict)
            assert required_keys.issubset(config.keys()), (
                f'{adapter.name} dashboard_config missing keys: '
                f'{required_keys - config.keys()}'
            )

    def test_snake_dashboard_config_uses_score(self):
        registry = GameRegistry()
        registry.discover()
        snake = registry.get_game('snake')
        config = snake.get_dashboard_config()
        assert config['graph_2_metric'] == 'score'
        assert config['graph_2_info_key'] == 'score'

    def test_connect4_dashboard_config_uses_win_rate(self):
        registry = GameRegistry()
        registry.discover()
        c4 = registry.get_game('connect4')
        config = c4.get_dashboard_config()
        assert config['graph_2_metric'] == 'win_rate'
        assert config['graph_2_info_key'] == 'winner'

    def test_all_adapters_have_completion_criteria(self):
        registry = GameRegistry()
        registry.discover()
        required_keys = {'metric', 'threshold', 'window', 'description'}
        for adapter in registry.list_games():
            criteria = adapter.get_completion_criteria()
            assert isinstance(criteria, dict)
            assert required_keys.issubset(criteria.keys()), (
                f'{adapter.name} completion_criteria missing keys: '
                f'{required_keys - criteria.keys()}'
            )
            assert criteria['threshold'] > 0
            assert criteria['window'] > 0

    def test_snake_completion_criteria(self):
        registry = GameRegistry()
        registry.discover()
        snake = registry.get_game('snake')
        criteria = snake.get_completion_criteria()
        assert criteria['metric'] == 'score'
        assert criteria['threshold'] == 50.0

    def test_connect4_completion_criteria(self):
        registry = GameRegistry()
        registry.discover()
        c4 = registry.get_game('connect4')
        criteria = c4.get_completion_criteria()
        assert criteria['metric'] == 'win_rate'
        assert criteria['threshold'] == 0.9
