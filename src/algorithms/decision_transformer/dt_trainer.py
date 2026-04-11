"""
Decision Transformer Trainer.

Trains the Decision Transformer on collected experience from multiple
games using standard supervised learning (cross-entropy on action
prediction). Unlike traditional RL trainers, the DT trainer:

    1. Does NOT interact with environments during training
    2. Learns from a fixed dataset of pre-collected trajectories
    3. Uses return-conditioning to achieve desired performance levels

Training pipeline:
    ExperienceStore → TrajectoryDataset → DataLoader → DT → Loss → Backprop

The trainer also supports:
    - Multi-game experience collection (run game agents to fill the store)
    - Evaluation by rolling out the DT in actual game environments
    - Checkpoint saving/loading of the transformer model
    - Dashboard integration for training visualization

Usage:
    trainer = DTTrainer(
        config=config,
        store=experience_store,
        visualizer=dashboard,
    )
    trainer.train()
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List

from src.algorithms.device import select_device
from src.algorithms.decision_transformer.dt_network import DecisionTransformer
from src.algorithms.decision_transformer.trajectory_dataset import TrajectoryDataset
from src.experience.experience_store import ExperienceStore
from src.visualization.dashboard import Dashboard


class DTTrainer:
    """Decision Transformer trainer.

    This trainer is different from BaseTrainer because:
    - It doesn't wrap a single environment
    - It trains on offline data from multiple games
    - It uses supervised learning, not RL
    - Evaluation requires creating environments on-the-fly

    Args:
        config: Hyperparameters dict (loaded from dt_config.yaml).
        store: ExperienceStore with pre-collected trajectories.
        visualizer: Optional Dashboard for live training visualization.
        save_dir: Directory for saving model checkpoints.
        device_preference: Force a specific compute device.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        store: ExperienceStore,
        visualizer: Optional[Dashboard] = None,
        save_dir: str = 'models/generalist',
        device_preference: Optional[str] = None,
    ):
        self.config = config
        self.store = store
        self.visualizer = visualizer
        self.save_dir = save_dir

        # Device selection
        self.device = select_device(
            preference=device_preference, algo_name='DT', verbose=True,
        )

        # Build the Decision Transformer
        self.model = DecisionTransformer(
            obs_channels=config.get('obs_channels', 1),
            max_action_vocab=config.get('max_action_vocab', 64),
            max_game_tokens=config.get('max_game_tokens', 1024),
            embed_dim=config.get('embed_dim', 256),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 4),
            context_length=config.get('context_length', 20),
            max_timestep=config.get('max_timestep', 5000),
            dropout=config.get('dropout', 0.1),
        ).to(self.device)

        print(f'Decision Transformer: {self.model.count_parameters():,} parameters')

        # Optimizer (AdamW as in the original DT paper)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler (cosine annealing)
        total_steps = config.get('total_train_steps', 100000)
        warmup_steps = config.get('warmup_steps', 1000)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps,
        )
        self.warmup_steps = warmup_steps

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.is_training = False
        self.train_losses: List[float] = []

        os.makedirs(save_dir, exist_ok=True)

    def _warmup_lr(self):
        """Linear learning rate warmup."""
        if self.global_step < self.warmup_steps:
            lr_scale = self.global_step / max(1, self.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.config.get('learning_rate', 1e-4) * lr_scale

    def train(self) -> None:
        """Train the Decision Transformer on collected experience.

        Runs for `total_train_steps` gradient updates, sampling batches
        from the TrajectoryDataset.
        """
        total_steps = self.config.get('total_train_steps', 100000)
        batch_size = self.config.get('batch_size', 32)
        context_length = self.config.get('context_length', 20)
        log_interval = self.config.get('log_interval', 100)
        save_interval = self.config.get('save_interval', 5000)

        if len(self.store) == 0:
            raise ValueError(
                "Experience store is empty. Collect trajectories first "
                "by training individual game agents."
            )

        # Create dataset and dataloader
        dataset = TrajectoryDataset(
            self.store,
            context_length=context_length,
            min_return_pct=self.config.get('min_return_pct', 0.0),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # In-process to avoid multiprocessing issues
            drop_last=True,
        )

        self.model.train()
        self.is_training = True
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        print(f'\nTraining Decision Transformer for {total_steps} steps')
        print(f'  Dataset: {len(dataset)} samples from {len(self.store)} trajectories')
        print(f'  Batch size: {batch_size}, Context: {context_length}')
        print(f'  Games in store: {self.store.get_game_ids()}')
        print()

        start_time = time.time()
        data_iter = iter(loader)

        while self.global_step < total_steps:
            # Get next batch (re-create iterator when exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            # Move to device
            obs = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)
            rtg = batch['returns_to_go'].to(self.device)
            timesteps = batch['timesteps'].to(self.device)
            game_tokens = batch['game_tokens'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Forward pass
            action_logits = self.model(
                obs, actions, rtg, timesteps, game_tokens, mask=mask,
            )

            # Compute loss (only on valid positions)
            # action_logits: (B, K, vocab_size), actions: (B, K)
            B, K, V = action_logits.shape
            logits_flat = action_logits.reshape(B * K, V)
            targets_flat = actions.reshape(B * K)
            mask_flat = mask.reshape(B * K)

            per_token_loss = loss_fn(logits_flat, targets_flat)
            # Mask out padding positions
            loss = (per_token_loss * mask_flat.float()).sum() / mask_flat.float().sum()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            max_grad_norm = self.config.get('max_grad_norm', 1.0)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.optimizer.step()

            # Learning rate schedule
            self._warmup_lr()
            if self.global_step >= self.warmup_steps:
                self.scheduler.step()

            self.global_step += 1
            loss_val = loss.item()
            self.train_losses.append(loss_val)

            # Logging
            if self.global_step % log_interval == 0:
                elapsed = time.time() - start_time
                avg_loss = np.mean(self.train_losses[-log_interval:])
                lr = self.optimizer.param_groups[0]['lr']

                # Compute action accuracy
                with torch.no_grad():
                    preds = action_logits.argmax(dim=-1)
                    correct = ((preds == actions) & mask).float()
                    accuracy = correct.sum() / mask.float().sum()

                print(
                    f'Step {self.global_step}/{total_steps} | '
                    f'Loss: {avg_loss:.4f} | '
                    f'Acc: {accuracy:.3f} | '
                    f'LR: {lr:.2e} | '
                    f'Time: {elapsed:.0f}s'
                )

                # Update dashboard if available
                if self.visualizer:
                    metrics = {
                        'episode': self.global_step,
                        'reward': -avg_loss,  # Use negative loss as "reward"
                        'accuracy': accuracy.item(),
                    }
                    self.visualizer.update(frame=None, metrics=metrics)

            # Save checkpoint periodically
            if self.global_step % save_interval == 0:
                self._save_checkpoint()

            # Track best loss
            if loss_val < self.best_loss:
                self.best_loss = loss_val

        # Final save
        self._save_checkpoint(final=True)
        self.is_training = False

        total_time = time.time() - start_time
        print(f'\nTraining complete in {total_time:.0f}s')
        print(f'Best loss: {self.best_loss:.4f}')

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint and metadata."""
        suffix = 'final' if final else f'step_{self.global_step}'
        model_path = os.path.join(self.save_dir, f'dt_{suffix}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': self.config,
        }, model_path)

        # Save metadata JSON
        meta_path = os.path.join(self.save_dir, 'metadata.json')
        meta = {
            'algorithm': 'decision_transformer',
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'parameters': self.model.count_parameters(),
            'games': self.store.get_game_ids(),
            'total_trajectories': len(self.store),
            'config': self.config,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f'  Checkpoint saved: {model_path}')

    def load_checkpoint(self, path: str) -> None:
        """Load a model checkpoint.

        Args:
            path: Path to the .pt checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f'Loaded DT checkpoint from step {self.global_step}')

    def evaluate_on_game(
        self,
        env,
        game_token_id: int,
        target_return: float,
        num_episodes: int = 5,
        max_steps: int = 1000,
        visualizer=None,
    ) -> float:
        """Evaluate the DT on a specific game environment.

        Args:
            env: Gym environment for the game.
            game_token_id: Game identity token for conditioning.
            target_return: Desired return-to-go (conditions the DT).
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            visualizer: Optional Dashboard for live gameplay display.

        Returns:
            Mean reward across evaluation episodes.
        """
        from src.experience.tokenizer import Tokenizer
        from games.reward_config import TokenConfig

        self.model.eval()
        context_length = self.config.get('context_length', 20)
        obs_channels = self.config.get('obs_channels', 1)

        # Create a simple tokenizer for this game
        token_config = TokenConfig(
            obs_resolution=(84, 84),
            obs_channels=obs_channels,
            action_vocab_size=self.config.get('max_action_vocab', 64),
            game_token_id=game_token_id,
        )
        tokenizer = Tokenizer(token_config)

        rewards = []
        for ep in range(num_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            # Initialize context buffers
            obs_history = []
            action_history = []
            rtg_history = []
            ts_history = []

            total_reward = 0.0
            remaining_return = target_return

            for step in range(max_steps):
                # Tokenize current observation
                proc_obs = tokenizer._resize_observation(obs)
                obs_history.append(proc_obs)
                rtg_history.append(remaining_return)
                ts_history.append(step)

                # Build context tensors (last K timesteps)
                K = min(len(obs_history), context_length)
                obs_tensor = torch.zeros(K, *proc_obs.shape, dtype=torch.float32)
                act_tensor = torch.zeros(K, dtype=torch.long)
                rtg_tensor = torch.zeros(K, dtype=torch.float32)
                ts_tensor = torch.zeros(K, dtype=torch.long)

                for t in range(K):
                    idx = len(obs_history) - K + t
                    obs_tensor[t] = torch.from_numpy(
                        obs_history[idx].astype(np.float32) / 255.0
                    )
                    rtg_tensor[t] = rtg_history[idx]
                    ts_tensor[t] = ts_history[idx]
                    if idx < len(action_history):
                        act_tensor[t] = action_history[idx]

                # Get action from DT
                action = self.model.get_action(
                    obs_tensor, act_tensor, rtg_tensor, ts_tensor,
                    game_token=game_token_id,
                )
                action_history.append(action)

                # Step environment
                result = env.step(action)
                obs, reward, done = result[0], result[1], result[2]
                if isinstance(obs, tuple):
                    obs = obs[0]

                total_reward += reward
                remaining_return -= reward

                # Live visualization — show the DT playing the game
                if visualizer and step % 4 == 0:
                    try:
                        from src.visualization.frame_capture import capture_display_frame
                        frame = capture_display_frame(env, fallback_obs=obs)
                        if frame is not None:
                            metrics = {
                                'episode': ep + 1,
                                'reward': total_reward,
                                'score': total_reward,
                            }
                            if not visualizer.update(
                                frame=frame, metrics=metrics,
                            ):
                                break  # Dashboard closed
                    except Exception:
                        pass

                if done:
                    break

            rewards.append(total_reward)
            print(f'  Episode {ep+1}/{num_episodes}: reward={total_reward:.1f}')

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        print(
            f'\nDT evaluation on game {game_token_id}: '
            f'{mean_reward:.1f} +/- {std_reward:.1f} over {num_episodes} episodes'
        )
        return mean_reward
