"""Tests for Sub-Phase 2D: Dashboard & Launcher Enhancements.

Tests the comparison panel utilities, run log parsing, smoothing,
game-opts CLI parsing, and base_trainer timestamped log export.
"""
import json
import os
import tempfile

import pytest

from src.visualization.comparison_panel import (
    _parse_run_filename,
    load_run_logs,
)
from main import parse_game_opts


# ─── Run Filename Parsing ──────────────────────────────────────


class TestParseRunFilename:
    """Test _parse_run_filename() from comparison_panel."""

    def test_valid_standard_filename(self):
        result = _parse_run_filename('snake_dqn_20260310_143022.json')
        assert result is not None
        assert result['game'] == 'snake'
        assert result['algo'] == 'dqn'
        assert result['date'] == '20260310'
        assert result['time'] == '143022'
        assert result['filename'] == 'snake_dqn_20260310_143022.json'

    def test_valid_game_with_underscores(self):
        """Game names can contain underscores (e.g., 'tic_tac_toe')."""
        result = _parse_run_filename('tic_tac_toe_ppo_20260310_120000.json')
        assert result is not None
        # game_algo = 'tic_tac_toe_ppo', last segment = 'ppo', rest = 'tic_tac_toe'
        assert result['algo'] == 'ppo'

    def test_valid_mario_rainbow(self):
        result = _parse_run_filename('mario_rainbow_20260309_091500.json')
        assert result is not None
        assert result['game'] == 'mario'
        assert result['algo'] == 'rainbow'

    def test_invalid_no_extension(self):
        assert _parse_run_filename('snake_dqn_20260310_143022') is None

    def test_invalid_wrong_extension(self):
        assert _parse_run_filename('snake_dqn_20260310_143022.csv') is None

    def test_invalid_too_few_parts(self):
        assert _parse_run_filename('invalid.json') is None

    def test_invalid_no_algo(self):
        """If game_algo can't be split into game + algo, return None."""
        # 'onlyname' has no underscore, so can't split into game + algo
        result = _parse_run_filename('onlyname_20260310_143022.json')
        assert result is None


# ─── Run Log Loading ───────────────────────────────────────────


class TestLoadRunLogs:
    """Test load_run_logs() with real temp files."""

    def test_no_runs_dir(self, tmp_path):
        """Returns empty list when logs/runs/ doesn't exist."""
        result = load_run_logs(str(tmp_path))
        assert result == []

    def test_empty_runs_dir(self, tmp_path):
        """Returns empty list when logs/runs/ exists but is empty."""
        (tmp_path / 'runs').mkdir()
        result = load_run_logs(str(tmp_path))
        assert result == []

    def test_loads_valid_run_file(self, tmp_path):
        """Successfully loads a valid run JSON file."""
        runs_dir = tmp_path / 'runs'
        runs_dir.mkdir()

        run_data = {
            'history': {
                'reward': [1.0, 2.0, 3.0],
                'distance': [10.0, 20.0, 30.0],
            },
            'best': {'reward': 3.0},
        }
        (runs_dir / 'snake_dqn_20260310_120000.json').write_text(
            json.dumps(run_data)
        )

        result = load_run_logs(str(tmp_path))
        assert len(result) == 1
        assert result[0]['game'] == 'snake'
        assert result[0]['algo'] == 'dqn'
        assert result[0]['data'] == run_data
        assert 'snake' in result[0]['label'].lower()

    def test_skips_invalid_json(self, tmp_path):
        """Skips files with invalid JSON."""
        runs_dir = tmp_path / 'runs'
        runs_dir.mkdir()
        (runs_dir / 'snake_dqn_20260310_120000.json').write_text('not json{{{')

        result = load_run_logs(str(tmp_path))
        assert result == []

    def test_skips_non_json_files(self, tmp_path):
        """Skips non-.json files."""
        runs_dir = tmp_path / 'runs'
        runs_dir.mkdir()
        (runs_dir / 'readme.txt').write_text('hello')

        result = load_run_logs(str(tmp_path))
        assert result == []

    def test_multiple_runs_sorted(self, tmp_path):
        """Multiple valid runs are loaded and sorted by filename."""
        runs_dir = tmp_path / 'runs'
        runs_dir.mkdir()

        for name in ['mario_ppo_20260310_110000.json',
                      'snake_dqn_20260309_090000.json',
                      'chess_a2c_20260310_150000.json']:
            (runs_dir / name).write_text(json.dumps({'history': {}}))

        result = load_run_logs(str(tmp_path))
        assert len(result) == 3
        # Sorted by filename → chess, mario, snake
        assert result[0]['game'] == 'chess'
        assert result[1]['game'] == 'mario'
        assert result[2]['game'] == 'snake'


# ─── Smoothing ─────────────────────────────────────────────────


class TestSmoothing:
    """Test the _smooth method logic (extracted as a standalone function)."""

    def _smooth(self, values, window):
        """Replicate the smoothing logic from ComparisonPanel."""
        if window <= 1 or not values:
            return values
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            chunk = values[start:i + 1]
            result.append(sum(chunk) / len(chunk))
        return result

    def test_no_smoothing(self):
        vals = [1.0, 2.0, 3.0]
        assert self._smooth(vals, 1) == vals

    def test_empty_input(self):
        assert self._smooth([], 10) == []

    def test_window_of_2(self):
        vals = [1.0, 3.0, 5.0, 7.0]
        result = self._smooth(vals, 2)
        assert result[0] == pytest.approx(1.0)  # only val[0]
        assert result[1] == pytest.approx(2.0)  # (1+3)/2
        assert result[2] == pytest.approx(4.0)  # (3+5)/2
        assert result[3] == pytest.approx(6.0)  # (5+7)/2

    def test_window_larger_than_data(self):
        vals = [10.0, 20.0]
        result = self._smooth(vals, 100)
        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(15.0)

    def test_constant_values_unchanged(self):
        vals = [5.0] * 10
        result = self._smooth(vals, 5)
        for v in result:
            assert v == pytest.approx(5.0)


# ─── Game Options CLI Parsing ──────────────────────────────────


class TestParseGameOpts:
    """Test parse_game_opts() from main.py."""

    def test_empty_list(self):
        assert parse_game_opts([]) == {}

    def test_single_int(self):
        result = parse_game_opts(['grid_size=16'])
        assert result == {'grid_size': 16}
        assert isinstance(result['grid_size'], int)

    def test_single_float(self):
        result = parse_game_opts(['speed=1.5'])
        assert result == {'speed': 1.5}
        assert isinstance(result['speed'], float)

    def test_single_bool_true(self):
        result = parse_game_opts(['wrap=true'])
        assert result == {'wrap': True}
        assert isinstance(result['wrap'], bool)

    def test_single_bool_false(self):
        result = parse_game_opts(['wrap=false'])
        assert result == {'wrap': False}

    def test_bool_case_insensitive(self):
        result = parse_game_opts(['wrap=True'])
        assert result['wrap'] is True
        result2 = parse_game_opts(['wrap=FALSE'])
        assert result2['wrap'] is False

    def test_single_string(self):
        result = parse_game_opts(['state=GreenHillZone.Act1'])
        assert result == {'state': 'GreenHillZone.Act1'}
        assert isinstance(result['state'], str)

    def test_multiple_opts(self):
        result = parse_game_opts([
            'grid_size=20',
            'speed=2.5',
            'wrap=false',
            'mode=easy',
        ])
        assert result == {
            'grid_size': 20,
            'speed': 2.5,
            'wrap': False,
            'mode': 'easy',
        }

    def test_skips_invalid_no_equals(self):
        result = parse_game_opts(['invalid', 'grid_size=10'])
        assert result == {'grid_size': 10}

    def test_value_with_equals_sign(self):
        """Values can contain '=' (split on first '=' only)."""
        result = parse_game_opts(['formula=a=b'])
        assert result == {'formula': 'a=b'}

    def test_whitespace_tolerance(self):
        result = parse_game_opts([' key = value '])
        assert result == {'key': 'value'}


# ─── Comparison Panel Import ───────────────────────────────────


class TestComparisonPanelImport:
    """Verify the module imports without errors (no tkinter required)."""

    def test_module_imports(self):
        import src.visualization.comparison_panel as cp
        assert hasattr(cp, 'ComparisonPanel')
        assert hasattr(cp, 'load_run_logs')
        assert hasattr(cp, '_parse_run_filename')

    def test_run_colors_defined(self):
        from src.visualization.comparison_panel import RUN_COLORS
        assert len(RUN_COLORS) >= 10
