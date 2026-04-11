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
