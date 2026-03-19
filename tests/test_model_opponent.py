"""Tests for ModelOpponent — trained checkpoint opponent."""

import os
import tempfile

import pytest

from src.opponents.model_opponent import ModelOpponent


# ------------------------------------------------------------------
# Constructor / file handling
# ------------------------------------------------------------------

class TestModelOpponentInit:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match='Checkpoint not found'):
            ModelOpponent('/nonexistent/path/model.zip')

    def test_accepts_existing_zip(self, tmp_path):
        p = tmp_path / 'dummy_model.zip'
        p.write_bytes(b'\x00')
        opp = ModelOpponent(str(p))
        assert opp._model is None  # corrupt file, but no crash

    def test_accepts_existing_pt(self, tmp_path):
        p = tmp_path / 'dummy_model.pt'
        p.write_bytes(b'\x00')
        opp = ModelOpponent(str(p))
        assert opp._model is None


# ------------------------------------------------------------------
# Property / metadata tests
# ------------------------------------------------------------------

class TestModelOpponentProperties:
    @pytest.fixture()
    def dummy_zip(self, tmp_path):
        p = tmp_path / 'checkpoint_1000.zip'
        p.write_bytes(b'\x00')
        return str(p)

    @pytest.fixture()
    def best_zip(self, tmp_path):
        p = tmp_path / 'best_model.zip'
        p.write_bytes(b'\x00')
        return str(p)

    def test_name_includes_filename(self, dummy_zip):
        opp = ModelOpponent(dummy_zip)
        assert 'checkpoint_1000.zip' in opp.name
        assert opp.name == 'Model (checkpoint_1000.zip)'

    def test_name_includes_best(self, best_zip):
        opp = ModelOpponent(best_zip)
        assert 'best_model.zip' in opp.name

    def test_difficulty_tier_best(self, best_zip):
        opp = ModelOpponent(best_zip)
        assert opp.difficulty_tier == 5

    def test_difficulty_tier_regular(self, dummy_zip):
        opp = ModelOpponent(dummy_zip)
        assert opp.difficulty_tier == 4

    def test_difficulty_tier_best_case_insensitive(self, tmp_path):
        p = tmp_path / 'BEST_checkpoint.zip'
        p.write_bytes(b'\x00')
        opp = ModelOpponent(str(p))
        assert opp.difficulty_tier == 5


# ------------------------------------------------------------------
# Action selection / graceful degradation
# ------------------------------------------------------------------

class TestModelOpponentActions:
    @pytest.fixture()
    def broken_opponent(self, tmp_path):
        """Opponent with a corrupt checkpoint — model will be None."""
        p = tmp_path / 'broken.zip'
        p.write_bytes(b'\x00')
        return ModelOpponent(str(p))

    def test_fallback_to_random_when_no_model(self, broken_opponent):
        state = {
            'board': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            'valid_actions': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'game_id': 'tictactoe',
            'turn': 0,
        }
        action = broken_opponent.pick_action(state)
        assert action in state['valid_actions']

    def test_pick_action_returns_valid(self, broken_opponent):
        valid = [3, 5, 7]
        state = {
            'board': [[1, 2, 1], [0, 0, 0], [0, 0, 0]],
            'valid_actions': valid,
            'game_id': 'tictactoe',
            'turn': 2,
        }
        for _ in range(20):
            action = broken_opponent.pick_action(state)
            assert action in valid

    def test_pick_action_empty_valid_actions(self, broken_opponent):
        state = {
            'board': [[1, 2, 1], [2, 1, 2], [2, 1, 2]],
            'valid_actions': [],
            'game_id': 'tictactoe',
            'turn': 9,
        }
        assert broken_opponent.pick_action(state) == 0

    def test_reset_is_noop(self, broken_opponent):
        # Should not raise
        broken_opponent.reset()


# ------------------------------------------------------------------
# Torch model path (.pt / .pth)
# ------------------------------------------------------------------

class TestModelOpponentTorch:
    def test_pth_extension_accepted(self, tmp_path):
        p = tmp_path / 'model.pth'
        p.write_bytes(b'\x00')
        opp = ModelOpponent(str(p))
        assert opp._model_type == 'torch'
        assert opp._model is None  # corrupt, graceful degradation

    def test_pth_fallback_action(self, tmp_path):
        p = tmp_path / 'model.pth'
        p.write_bytes(b'\x00')
        opp = ModelOpponent(str(p))
        state = {
            'board': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            'valid_actions': [0, 1, 2],
            'game_id': 'tictactoe',
            'turn': 0,
        }
        assert opp.pick_action(state) in [0, 1, 2]
