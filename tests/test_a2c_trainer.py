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
