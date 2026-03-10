"""Tests for the Decision Transformer network (src/algorithms/decision_transformer/).

Validates:
    - DecisionTransformer forward pass shapes and types
    - CausalSelfAttention masking
    - ObservationEncoder output shape
    - get_action() returns valid action integer
    - count_parameters() returns positive int
    - TrajectoryDataset builds correctly from ExperienceStore
    - DTTrainer construction and checkpoint save/load
"""

import os
import numpy as np
import pytest
import torch

from games.reward_config import TokenConfig
from src.experience.tokenizer import Tokenizer, TokenizedTrajectory
from src.experience.experience_store import ExperienceStore
from src.algorithms.decision_transformer.dt_network import (
    DecisionTransformer,
    CausalSelfAttention,
    ObservationEncoder,
)
from src.algorithms.decision_transformer.trajectory_dataset import TrajectoryDataset


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def dt_config():
    """Minimal DT config for testing."""
    return {
        'obs_channels': 1,
        'max_action_vocab': 16,
        'max_game_tokens': 64,
        'embed_dim': 32,
        'num_heads': 2,
        'num_layers': 2,
        'context_length': 5,
        'max_timestep': 100,
        'dropout': 0.0,
    }


@pytest.fixture
def model(dt_config):
    return DecisionTransformer(
        obs_channels=dt_config['obs_channels'],
        max_action_vocab=dt_config['max_action_vocab'],
        max_game_tokens=dt_config['max_game_tokens'],
        embed_dim=dt_config['embed_dim'],
        num_heads=dt_config['num_heads'],
        num_layers=dt_config['num_layers'],
        context_length=dt_config['context_length'],
        max_timestep=dt_config['max_timestep'],
        dropout=dt_config['dropout'],
    )


def make_trajectory(game_token_id=1, length=10, total_return=50.0):
    """Create a minimal TokenizedTrajectory for testing."""
    return TokenizedTrajectory(
        game_token_id=game_token_id,
        observations=np.random.randint(0, 255, (length, 1, 84, 84), dtype=np.uint8),
        actions=np.random.randint(0, 7, length, dtype=np.int32),
        rewards=np.ones(length, dtype=np.float32) * (total_return / length),
        returns_to_go=np.linspace(total_return, total_return / length, length).astype(np.float32),
        timesteps=np.arange(length, dtype=np.int32),
        dones=np.zeros(length, dtype=bool),
        total_return=total_return,
        length=length,
    )


@pytest.fixture
def filled_store(tmp_path):
    """ExperienceStore with some trajectories for dataset testing."""
    store_dir = str(tmp_path / 'dt_test_store')
    store = ExperienceStore(store_dir=store_dir, max_trajectories=100)
    for _ in range(5):
        store.add_trajectory(make_trajectory(game_token_id=1, length=20, total_return=50.0))
    for _ in range(3):
        store.add_trajectory(make_trajectory(game_token_id=2, length=15, total_return=30.0))
    return store


# ── ObservationEncoder ────────────────────────────────────────────────


class TestObservationEncoder:

    def test_output_shape(self):
        encoder = ObservationEncoder(in_channels=1, embed_dim=32)
        obs = torch.randn(4, 1, 84, 84)  # (B, C, H, W)
        output = encoder(obs)
        assert output.shape == (4, 32)

    def test_rgb_input(self):
        encoder = ObservationEncoder(in_channels=3, embed_dim=64)
        obs = torch.randn(2, 3, 84, 84)
        output = encoder(obs)
        assert output.shape == (2, 64)


# ── CausalSelfAttention ──────────────────────────────────────────────


class TestCausalSelfAttention:

    def test_output_shape(self):
        attn = CausalSelfAttention(embed_dim=32, num_heads=2, dropout=0.0)
        x = torch.randn(2, 15, 32)
        out = attn(x)
        assert out.shape == (2, 15, 32)

    def test_causal_masking(self):
        """Future positions should be masked (output at pos 0 should not depend on pos 1+)."""
        attn = CausalSelfAttention(embed_dim=32, num_heads=2, dropout=0.0)
        attn.eval()

        x = torch.randn(1, 10, 32)
        out1 = attn(x).detach()

        # Modify future positions
        x_modified = x.clone()
        x_modified[0, 5:, :] = torch.randn(5, 32)
        out2 = attn(x_modified).detach()

        # Positions 0-4 should be unchanged (they can't see positions 5+)
        torch.testing.assert_close(out1[0, :5], out2[0, :5], atol=1e-5, rtol=1e-5)


# ── DecisionTransformer ──────────────────────────────────────────────


class TestDecisionTransformer:

    def test_forward_shape(self, model):
        B, K = 2, 5
        obs = torch.randn(B, K, 1, 84, 84)
        actions = torch.randint(0, 16, (B, K))
        rtg = torch.randn(B, K)
        timesteps = torch.arange(K).unsqueeze(0).expand(B, -1)
        game_tokens = torch.ones(B, dtype=torch.long)
        mask = torch.ones(B, K, dtype=torch.bool)

        logits = model(obs, actions, rtg, timesteps, game_tokens, mask=mask)
        assert logits.shape == (B, K, 16)  # (B, K, action_vocab)

    def test_forward_no_mask(self, model):
        B, K = 1, 3
        obs = torch.randn(B, K, 1, 84, 84)
        actions = torch.randint(0, 16, (B, K))
        rtg = torch.randn(B, K)
        timesteps = torch.arange(K).unsqueeze(0)
        game_tokens = torch.zeros(B, dtype=torch.long)

        logits = model(obs, actions, rtg, timesteps, game_tokens)
        assert logits.shape == (B, K, 16)

    def test_get_action(self, model):
        K = 5
        obs = torch.randn(K, 1, 84, 84)
        actions = torch.randint(0, 16, (K,))
        rtg = torch.randn(K)
        timesteps = torch.arange(K)

        model.eval()
        action = model.get_action(obs, actions, rtg, timesteps, game_token=1)
        assert isinstance(action, int)
        assert 0 <= action < 16

    def test_count_parameters(self, model):
        count = model.count_parameters()
        assert isinstance(count, int)
        assert count > 0

    def test_different_context_lengths(self, model):
        """Model should handle shorter-than-max context."""
        B = 1
        for K in [1, 3, 5]:
            obs = torch.randn(B, K, 1, 84, 84)
            actions = torch.randint(0, 16, (B, K))
            rtg = torch.randn(B, K)
            timesteps = torch.arange(K).unsqueeze(0)
            game_tokens = torch.zeros(B, dtype=torch.long)

            logits = model(obs, actions, rtg, timesteps, game_tokens)
            assert logits.shape == (B, K, 16)

    def test_gradient_flows(self, model):
        """Verify backward pass works (gradients flow through)."""
        B, K = 2, 5
        obs = torch.randn(B, K, 1, 84, 84)
        actions = torch.randint(0, 16, (B, K))
        rtg = torch.randn(B, K)
        timesteps = torch.arange(K).unsqueeze(0).expand(B, -1)
        game_tokens = torch.ones(B, dtype=torch.long)

        logits = model(obs, actions, rtg, timesteps, game_tokens)
        loss = logits.sum()
        loss.backward()

        # At least some parameters should have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad


# ── TrajectoryDataset ─────────────────────────────────────────────────


class TestTrajectoryDataset:

    def test_dataset_length(self, filled_store):
        dataset = TrajectoryDataset(filled_store, context_length=5)
        assert len(dataset) > 0

    def test_dataset_item_keys(self, filled_store):
        dataset = TrajectoryDataset(filled_store, context_length=5)
        item = dataset[0]
        assert 'observations' in item
        assert 'actions' in item
        assert 'returns_to_go' in item
        assert 'timesteps' in item
        assert 'game_tokens' in item
        assert 'mask' in item

    def test_dataset_item_shapes(self, filled_store):
        K = 5
        dataset = TrajectoryDataset(filled_store, context_length=K)
        item = dataset[0]
        assert item['observations'].shape == (K, 1, 84, 84)
        assert item['actions'].shape == (K,)
        assert item['returns_to_go'].shape == (K,)
        assert item['timesteps'].shape == (K,)
        assert item['mask'].shape == (K,)

    def test_dataset_item_dtypes(self, filled_store):
        dataset = TrajectoryDataset(filled_store, context_length=5)
        item = dataset[0]
        assert item['observations'].dtype == torch.float32
        assert item['actions'].dtype == torch.long
        assert item['returns_to_go'].dtype == torch.float32
        assert item['timesteps'].dtype == torch.long
        assert item['mask'].dtype == torch.bool

    def test_obs_normalized(self, filled_store):
        dataset = TrajectoryDataset(filled_store, context_length=5)
        item = dataset[0]
        assert item['observations'].max() <= 1.0
        assert item['observations'].min() >= 0.0

    def test_game_filter(self, filled_store):
        """Filtering by game should reduce dataset size."""
        full = TrajectoryDataset(filled_store, context_length=5)
        game1_only = TrajectoryDataset(filled_store, context_length=5, game_filter=1)
        game2_only = TrajectoryDataset(filled_store, context_length=5, game_filter=2)
        assert len(game1_only) < len(full)
        assert len(game2_only) < len(full)
        assert len(game1_only) + len(game2_only) == len(full)


# ── DTTrainer ─────────────────────────────────────────────────────────


class TestDTTrainer:

    def test_trainer_construction(self, dt_config, filled_store):
        from src.algorithms.decision_transformer.dt_trainer import DTTrainer
        trainer = DTTrainer(
            config=dt_config,
            store=filled_store,
            device_preference='cpu',
        )
        assert trainer.model is not None
        assert trainer.global_step == 0

    def test_trainer_short_train(self, dt_config, filled_store, tmp_path):
        """Train for a few steps — should not crash."""
        from src.algorithms.decision_transformer.dt_trainer import DTTrainer
        dt_config['total_train_steps'] = 5
        dt_config['batch_size'] = 2
        dt_config['log_interval'] = 2
        dt_config['save_interval'] = 100  # Don't save during short test
        dt_config['warmup_steps'] = 2

        save_dir = str(tmp_path / 'dt_ckpt')
        trainer = DTTrainer(
            config=dt_config,
            store=filled_store,
            save_dir=save_dir,
            device_preference='cpu',
        )
        trainer.train()
        assert trainer.global_step == 5

    def test_checkpoint_save_load(self, dt_config, filled_store, tmp_path):
        """Save and load checkpoint roundtrip."""
        from src.algorithms.decision_transformer.dt_trainer import DTTrainer
        dt_config['total_train_steps'] = 3
        dt_config['batch_size'] = 2
        dt_config['save_interval'] = 100
        dt_config['warmup_steps'] = 1

        save_dir = str(tmp_path / 'dt_ckpt2')
        trainer = DTTrainer(
            config=dt_config,
            store=filled_store,
            save_dir=save_dir,
            device_preference='cpu',
        )
        trainer.train()

        # Check final checkpoint exists
        final_path = os.path.join(save_dir, 'dt_final.pt')
        assert os.path.exists(final_path)

        # Load into new trainer
        trainer2 = DTTrainer(
            config=dt_config,
            store=filled_store,
            save_dir=save_dir,
            device_preference='cpu',
        )
        trainer2.load_checkpoint(final_path)
        assert trainer2.global_step == 3

    def test_empty_store_raises(self, dt_config, tmp_path):
        """Training with empty store should raise ValueError."""
        from src.algorithms.decision_transformer.dt_trainer import DTTrainer
        empty_store = ExperienceStore(
            store_dir=str(tmp_path / 'empty_store'),
            max_trajectories=10,
        )
        trainer = DTTrainer(
            config=dt_config,
            store=empty_store,
            device_preference='cpu',
        )
        with pytest.raises(ValueError, match="Experience store is empty"):
            trainer.train()

    def test_metadata_json_saved(self, dt_config, filled_store, tmp_path):
        """Metadata JSON should be written after training."""
        from src.algorithms.decision_transformer.dt_trainer import DTTrainer
        dt_config['total_train_steps'] = 3
        dt_config['batch_size'] = 2
        dt_config['save_interval'] = 100
        dt_config['warmup_steps'] = 1

        save_dir = str(tmp_path / 'dt_meta')
        trainer = DTTrainer(
            config=dt_config,
            store=filled_store,
            save_dir=save_dir,
            device_preference='cpu',
        )
        trainer.train()

        meta_path = os.path.join(save_dir, 'metadata.json')
        assert os.path.exists(meta_path)
        import json
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        assert meta['algorithm'] == 'decision_transformer'
        assert meta['global_step'] == 3
