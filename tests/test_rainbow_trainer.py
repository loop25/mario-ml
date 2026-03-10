"""Tests for the Rainbow DQN trainer.

Tests import, config loading, base class inheritance,
and the RainbowTrainer's method signatures.
"""
import pytest
import yaml
import os

from src.algorithms.rainbow.rainbow_trainer import RainbowTrainer
from src.algorithms.base_trainer import BaseTrainer


class TestRainbowConfig:
    def test_config_file_exists(self):
        config_path = os.path.join('config', 'rainbow_config.yaml')
        assert os.path.exists(config_path)

    def test_config_has_core_keys(self):
        config_path = os.path.join('config', 'rainbow_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert 'learning_rate' in config
        assert 'gamma' in config
        assert 'batch_size' in config
        assert 'buffer_size' in config
        assert 'num_episodes' in config

    def test_config_has_feature_toggles(self):
        config_path = os.path.join('config', 'rainbow_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert 'use_double_dqn' in config
        assert 'use_prioritized_replay' in config
        assert 'use_dueling' in config
        assert 'use_noisy' in config
        assert 'use_distributional' in config
        assert 'use_multistep' in config

    def test_config_has_per_params(self):
        config_path = os.path.join('config', 'rainbow_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert 'per_alpha' in config
        assert 'per_beta_start' in config
        assert 'per_beta_end' in config
        assert 'per_beta_frames' in config

    def test_config_has_c51_params(self):
        config_path = os.path.join('config', 'rainbow_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert 'num_atoms' in config
        assert 'v_min' in config
        assert 'v_max' in config
        assert config['num_atoms'] == 51

    def test_config_has_multistep_params(self):
        config_path = os.path.join('config', 'rainbow_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert 'n_step' in config
        assert config['n_step'] == 3

    def test_config_has_noisy_params(self):
        config_path = os.path.join('config', 'rainbow_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert 'sigma_init' in config

    def test_config_has_environment_settings(self):
        config_path = os.path.join('config', 'rainbow_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert 'frame_skip' in config
        assert 'frame_stack' in config
        assert 'downscale_size' in config


class TestRainbowTrainerImport:
    def test_can_import_trainer(self):
        assert RainbowTrainer is not None

    def test_trainer_is_base_trainer_subclass(self):
        assert issubclass(RainbowTrainer, BaseTrainer)


class TestRainbowTrainerMethods:
    """Verify the trainer has the expected method signatures."""

    def test_has_train_method(self):
        assert hasattr(RainbowTrainer, 'train')

    def test_has_evaluate_method(self):
        assert hasattr(RainbowTrainer, 'evaluate')

    def test_has_save_checkpoint_method(self):
        assert hasattr(RainbowTrainer, 'save_checkpoint')

    def test_has_load_checkpoint_method(self):
        assert hasattr(RainbowTrainer, 'load_checkpoint')

    def test_has_select_action_method(self):
        assert hasattr(RainbowTrainer, 'select_action')


class TestRainbowModuleImports:
    """Verify all Rainbow submodules import cleanly."""

    def test_import_noisy_linear(self):
        from src.algorithms.rainbow.noisy_linear import NoisyLinear
        assert NoisyLinear is not None

    def test_import_prioritized_replay(self):
        from src.algorithms.rainbow.prioritized_replay import (
            PrioritizedReplayBuffer,
            SumTree,
            PrioritizedExperience,
        )
        assert PrioritizedReplayBuffer is not None
        assert SumTree is not None

    def test_import_rainbow_network(self):
        from src.algorithms.rainbow.rainbow_network import RainbowNetwork
        assert RainbowNetwork is not None

    def test_import_rainbow_package(self):
        import src.algorithms.rainbow
        assert src.algorithms.rainbow is not None
