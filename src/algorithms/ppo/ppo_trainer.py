"""
PPO (Proximal Policy Optimization) Trainer.

PPO is a modern policy gradient reinforcement learning algorithm.
Unlike NEAT which evolves network topology, PPO learns through
gradient descent on collected experience:

1. Collect experience: Run the policy in the environment for N steps
2. Compute advantages: How much better each action was vs. expected
3. Update policy: Increase probability of good actions (with clipping)
4. Update value function: Better predict future rewards

Key advantages of PPO:
    - Stable training (clipped objective prevents large policy changes)
    - Sample efficient (reuses data for multiple optimization epochs)
    - Works well with CNNs for visual input

This implementation uses stable-baselines3 which provides:
    - CnnPolicy: Convolutional neural network for processing game frames
    - Vectorized environments: Multiple parallel environments for speed
    - Built-in logging and callback system

Usage:
    trainer = PPOTrainer(env, config, visualizer=dashboard)
    trainer.train(total_timesteps=1000000)
    trainer.save_checkpoint('models/ppo/ppo_model')
"""

import os
import yaml
import numpy as np
from typing import Dict, Any, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from src.algorithms.base_trainer import BaseTrainer
from src.visualization.dashboard import Dashboard
from src.environment.mario_env import create_mario_env, create_sb3_env


class DashboardCallback(BaseCallback):
    """
    Custom callback for stable-baselines3 that feeds metrics to the dashboard.

    stable-baselines3 uses a callback system to hook into training events.
    This callback captures episode rewards, losses, and other metrics
    and sends them to the Dashboard for live visualization.

    Args:
        dashboard: Dashboard instance for visualization.
        trainer: PPOTrainer instance to update metrics on.
        render_interval: Render game frame every N episodes.
        verbose: Verbosity level (0=silent, 1=info, 2=debug).
    """

    def __init__(
        self,
        dashboard: Optional[Dashboard],
        trainer: 'PPOTrainer',
        render_interval: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.dashboard = dashboard
        self.trainer = trainer
        self.render_interval = render_interval
        self.episode_count = 0
        self.episode_rewards = []
        self._total_stage_completions = 0
        self._latest_frame = None
        self._latest_frames: list = []  # For multi-env grid display

        # Per-environment episode tracking.
        # SB3's DummyVecEnv returns arrays of rewards/dones/infos with
        # one element per env.  We need to accumulate per-env episode
        # stats so we detect episode completion in ALL environments,
        # not just env 0.
        num_envs = max(1, trainer.num_envs)
        self._env_episode_rewards = [0.0] * num_envs
        self._env_episode_distances = [0] * num_envs
        self._env_stage_completed = [False] * num_envs
        self._env_action_counts = [[0] * 7 for _ in range(num_envs)]

    def _on_step(self) -> bool:
        """
        Called after each environment step during training.

        SB3's DummyVecEnv steps ALL N environments each call, so
        ``self.locals`` contains arrays with one element per env.
        We track per-env episode stats and detect episode completion
        across ALL environments (not just env 0).

        Returns:
            bool: True to continue training, False to stop.
        """
        # Check for pause (blocks this thread until resumed)
        if not self.trainer.check_pause():
            return False  # Abort training

        # If dashboard was already closed, stop training immediately
        if self.trainer._dashboard_closed:
            return False

        num_envs = self.trainer.num_envs
        rewards = self.locals.get('rewards', [0])
        actions = self.locals.get('actions', [0])
        infos = self.locals.get('infos', [{}])
        dones = self.locals.get('dones', [False])

        # Accumulate per-env episode stats
        for i in range(min(num_envs, len(rewards))):
            self._env_episode_rewards[i] += rewards[i]

            # Track action
            if i < len(actions):
                action = int(actions[i])
                if 0 <= action < 7:
                    self._env_action_counts[i][action] += 1

            # Track distance & stage completion
            if i < len(infos):
                info = infos[i]
                x_pos = info.get('x_pos', 0)
                self._env_episode_distances[i] = max(
                    self._env_episode_distances[i], x_pos,
                )
                if info.get('stage_completed', False):
                    self._env_stage_completed[i] = True

        # --- Live gameplay display ---
        # Frame capture is expensive, so we throttle it.
        display_interval = 8 if num_envs <= 1 else 16

        if self.dashboard and self.num_timesteps % display_interval == 0:
            if num_envs > 1:
                # Multi-env: grab full-res NES frames from each env.
                self._latest_frames = []
                try:
                    vec_env = self.trainer.vec_env
                    dummy_env = vec_env.venv if hasattr(vec_env, 'venv') else vec_env
                    for i in range(min(num_envs, len(dummy_env.envs))):
                        try:
                            frame = dummy_env.envs[i].unwrapped.screen.copy()
                            self._latest_frames.append(frame)
                        except (AttributeError, Exception):
                            pass
                except (AttributeError, Exception):
                    pass

                if self._latest_frames:
                    self._latest_frame = self._latest_frames[0]
                    if not self.trainer.update_visualization_grid(
                        frames=self._latest_frames, metrics=None,
                    ):
                        return False
            else:
                # Single env: grab the full-res NES screen
                try:
                    self._latest_frame = self.trainer.env.unwrapped.screen
                except AttributeError:
                    if self.locals.get('new_obs') is not None:
                        obs = self.locals['new_obs']
                        if len(obs) > 0:
                            frame = obs[0]
                            if frame.ndim == 3:
                                self._latest_frame = frame[-1] if frame.shape[0] <= 4 else frame
                            else:
                                self._latest_frame = frame

                if self._latest_frame is not None:
                    if not self.trainer.update_visualization(
                        frame=self._latest_frame, metrics=None,
                    ):
                        return False

        # Check for episode completion in ALL environments.
        # SB3's DummyVecEnv auto-resets done envs, so each True in
        # dones[] means that env just finished an episode.
        for i in range(min(num_envs, len(dones))):
            if dones[i]:
                if not self._on_episode_end(env_index=i):
                    return False  # Dashboard closed — stop training

        return True  # Continue training

    def _on_episode_end(self, env_index: int = 0) -> bool:
        """Called when an episode finishes in any environment.

        Records metrics from the completed episode and updates the
        dashboard.  For multi-env, uses update_visualization_grid()
        so the GridRenderer is used correctly.

        Args:
            env_index: Which environment just finished its episode.

        Returns:
            bool: True to continue training, False if dashboard was closed.
        """
        self.episode_count += 1
        reward = self._env_episode_rewards[env_index]
        distance = self._env_episode_distances[env_index]

        # Log stage completion
        if self._env_stage_completed[env_index]:
            self._total_stage_completions += 1
            print(f'  *** STAGE COMPLETED! (Episode {self.episode_count}, '
                  f'Env {env_index}, '
                  f'Total completions: {self._total_stage_completions}) ***')

        # Update trainer tracking
        self.trainer.episode_count = self.episode_count
        if reward > self.trainer.best_reward:
            self.trainer.best_reward = reward
        if distance > self.trainer.best_distance:
            self.trainer.best_distance = distance

        # Notify episode callbacks (curriculum learning, etc.)
        self.trainer._fire_episode_complete(
            reward=reward,
            distance=distance,
            completed=self._env_stage_completed[env_index],
        )

        # Update dashboard
        if self.dashboard:
            loss = 0.0  # Updated in _on_rollout_end instead

            # Aggregate action counts across all envs for the graph
            combined_actions = [0] * 7
            for env_counts in self._env_action_counts:
                for j in range(7):
                    combined_actions[j] += env_counts[j]

            metrics = {
                'episode': self.episode_count,
                'reward': reward,
                'distance': distance,
                'loss': loss,
                'action_distribution': combined_actions,
            }

            num_envs = self.trainer.num_envs

            if num_envs > 1:
                # Multi-env: use grid display with latest frames
                frames = self._latest_frames if self._latest_frames else None
                if not self.trainer.update_visualization_grid(
                    frames=frames, metrics=metrics,
                ):
                    return False
            else:
                # Single env: use standard single-frame display
                frame = self._latest_frame if (
                    self.episode_count % self.render_interval == 0
                ) else None
                if not self.trainer.update_visualization(
                    frame=frame, metrics=metrics,
                ):
                    return False

        # Reset tracking for this env only (other envs may still be mid-episode)
        self._env_episode_rewards[env_index] = 0.0
        self._env_episode_distances[env_index] = 0
        self._env_stage_completed[env_index] = False
        self._env_action_counts[env_index] = [0] * 7
        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout (data collection phase).

        This is where we can access loss values from the latest update.
        """
        # Try to extract loss from the model's logger
        if self.dashboard and hasattr(self, 'model') and self.model is not None:
            try:
                # Access the most recent loss values
                if hasattr(self.model, 'logger') and self.model.logger is not None:
                    # The logger stores name_to_value after each update
                    name_to_value = getattr(self.model.logger, 'name_to_value', {})
                    policy_loss = name_to_value.get('train/policy_gradient_loss', 0)
                    value_loss = name_to_value.get('train/value_loss', 0)
                    # Combined loss for the graph
                    total_loss = abs(policy_loss) + abs(value_loss) * 0.5
                    if total_loss > 0:
                        self.dashboard.metrics.record(loss=total_loss)
            except Exception:
                pass  # Loss reporting is optional, don't break training


class PPOTrainer(BaseTrainer):
    """
    PPO algorithm trainer using stable-baselines3.

    Wraps the stable-baselines3 PPO implementation with:
    - Custom dashboard visualization via callbacks
    - Automatic environment wrapping for CNN input
    - Checkpoint management with metadata
    - Graceful shutdown support

    Args:
        env: Base Mario environment (will be re-wrapped for PPO).
        config: Dictionary of hyperparameters (loaded from YAML).
        visualizer: Optional Dashboard for live visualization.
        save_dir: Directory for checkpoints. Default 'models'.
        log_dir: Directory for logs. Default 'logs'.

    Example:
        env = create_mario_env(world=1, stage=1)
        config = yaml.safe_load(open('config/ppo_config.yaml'))
        dashboard = Dashboard(algorithm='ppo')
        trainer = PPOTrainer(env, config, dashboard)
        trainer.train(total_timesteps=1000000)
    """

    def __init__(
        self,
        env,
        config: Dict[str, Any],
        visualizer: Optional[Dashboard] = None,
        save_dir: str = 'models',
        log_dir: str = 'logs',
        num_envs: int = 1,
        world: int = 1,
        stage: int = 1,
    ):
        super().__init__(env, config, visualizer, save_dir, log_dir)

        self.num_envs = num_envs
        self.world = world
        self.stage = stage

        # Wrap the environment for stable-baselines3 compatibility
        # PPO needs: vectorized env, transposed images (channels first)
        self.vec_env = self._wrap_env_for_sb3(env, num_envs=num_envs)

        # Determine safe device (CUDA may report available but fail
        # on newer GPUs like RTX 5070 Blackwell/sm_120 with cu124)
        device = self._select_device()

        # Create PPO model with CNN policy
        self.model = PPO(
            policy='CnnPolicy',
            env=self.vec_env,
            learning_rate=config.get('learning_rate', 2.5e-4),
            n_steps=config.get('n_steps', 512),
            batch_size=config.get('batch_size', 64),
            n_epochs=config.get('n_epochs', 10),
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_range=config.get('clip_range', 0.2),
            ent_coef=config.get('ent_coef', 0.01),
            vf_coef=config.get('vf_coef', 0.5),
            max_grad_norm=config.get('max_grad_norm', 0.5),
            verbose=1,
            tensorboard_log=None,  # Disabled: use our dashboard instead
            device=device,
        )

    @staticmethod
    def _select_device() -> str:
        """Return ``'cuda'`` if CUDA works, otherwise ``'cpu'``.

        Some GPUs report CUDA available but fail on actual kernel
        execution (e.g. RTX 5070 Blackwell/sm_120 with older CUDA).

        Fix: Install PyTorch nightly with CUDA 12.8+:
            pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
        """
        try:
            import torch
            if torch.cuda.is_available():
                a = torch.randn(4, 4, device='cuda')
                _ = a @ a.T
                del a
                torch.cuda.empty_cache()
                gpu = torch.cuda.get_device_name(0)
                print(f'PPO using device: cuda ({gpu})')
                return 'cuda'
        except (RuntimeError, Exception):
            pass
        print('PPO using device: cpu')
        print('  Tip: RTX 50-series needs PyTorch nightly with cu128+')
        print('  Run: pip install --pre torch torchvision torchaudio '
              '--index-url https://download.pytorch.org/whl/nightly/cu128')
        return 'cpu'

    def _wrap_env_for_sb3(self, env, num_envs: int = 1):
        """
        Wrap environment for stable-baselines3 PPO compatibility.

        stable-baselines3 requires:
        1. New gymnasium API: reset() → (obs, info), step() → 5-tuple
        2. Vectorized environment (DummyVecEnv)
        3. Channel-first image format (VecTransposeImage)

        When num_envs > 1, creates additional environments and wraps them
        all in DummyVecEnv. We use DummyVecEnv (same process) instead of
        SubprocVecEnv because nes_py's C-based NES emulator crashes when
        used in subprocesses (access violation in _LIB.Restore).

        The multi-env setup still provides benefit: SB3 collects experience
        from all envs in parallel batches, providing more diverse training
        data per rollout.

        Args:
            env: The base Mario environment (old gym API).
            num_envs: Number of parallel environments. Default 1.

        Returns:
            Wrapped environment compatible with stable-baselines3.
        """
        from src.environment.wrappers import SB3CompatWrapper
        from src.environment.mario_env import create_cnn_env

        if num_envs <= 1:
            # Single env — simple DummyVecEnv wrapper
            compat_env = SB3CompatWrapper(env)
            vec_env = DummyVecEnv([lambda: compat_env])
        else:
            # Multi-env — create N environments in the SAME process.
            # nes_py's NES emulator uses a C library that crashes when
            # used in subprocesses (SubprocVecEnv), so we use DummyVecEnv
            # which steps envs sequentially in the main process.
            world, stage = self.world, self.stage

            # Store extra envs so we can access .unwrapped for frames.
            # Pump pygame events between env creations to prevent
            # Windows "Not Responding" during long setup.
            try:
                import pygame
                _pump = pygame.event.pump
            except (ImportError, Exception):
                _pump = lambda: None

            self._extra_envs = []
            env_wrappers = [SB3CompatWrapper(env)]  # First env is the one passed in

            for i in range(num_envs - 1):
                new_env = create_cnn_env(world=world, stage=stage)
                self._extra_envs.append(new_env)
                env_wrappers.append(SB3CompatWrapper(new_env))
                _pump()  # Keep window responsive

            # Use DummyVecEnv — all envs run in the main process.
            # Each lambda captures its own wrapper instance.
            vec_env = DummyVecEnv([
                (lambda w: lambda: w)(wrapper)
                for wrapper in env_wrappers
            ])
            print(f'  PPO: Using DummyVecEnv with {num_envs} environments (same process)')

        # Transpose observations to channel-first (H,W,C) -> (C,H,W)
        vec_env = VecTransposeImage(vec_env)

        return vec_env

    def train(self, total_timesteps: int = None) -> None:
        """
        Train PPO for specified total timesteps.

        Uses stable-baselines3's built-in training loop with
        a custom callback for dashboard visualization.

        Args:
            total_timesteps: Total environment steps to train for.
                             If None, uses value from config.
        """
        if total_timesteps is None:
            total_timesteps = self.config.get('total_timesteps', 1000000)

        self.is_training = True
        device = self.model.device
        print(f'\n{"="*60}')
        print(f'Starting PPO Training - {total_timesteps:,} timesteps')
        print(f'Device: {device}')
        print(f'Parallel envs: {self.num_envs}')
        print(f'Learning rate: {self.config.get("learning_rate", 2.5e-4)}')
        print(f'{"="*60}\n')

        # Create callback for dashboard updates
        callback = DashboardCallback(
            dashboard=self.visualizer,
            trainer=self,
            render_interval=self.config.get('render_every_n_episodes', 5),
        )

        # Run training
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
            )
        except (SystemExit, KeyboardInterrupt):
            print('\nTraining interrupted.')

        self.is_training = False
        print(f'\n{"="*60}')
        print(f'PPO Training Complete!')
        print(f'Episodes: {self.episode_count}')
        print(f'Best reward: {self.best_reward:.0f}')
        print(f'{"="*60}\n')

        # Save final model
        self._save_on_exit()

    def evaluate(self, num_episodes: int = 5) -> float:
        """
        Evaluate the trained PPO model.

        Runs the policy in the environment without learning,
        rendering every frame to the dashboard.

        Args:
            num_episodes: Number of episodes to evaluate.

        Returns:
            float: Average reward across evaluation episodes.
        """
        total_rewards = []

        for ep in range(num_episodes):
            obs = self.vec_env.reset()
            episode_reward = 0.0
            done = False
            steps = 0

            while not done and steps < 5000:
                # Get action from the trained policy
                action, _states = self.model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, done, info = self.vec_env.step(action)
                episode_reward += reward[0]
                steps += 1

                # Render to dashboard
                if self.visualizer:
                    # Get raw NES frame instead of processed observation
                    try:
                        frame = self.env.unwrapped.screen
                    except AttributeError:
                        frame = obs[0][-1] if obs[0].ndim == 3 else obs[0]
                    self.update_visualization(
                        frame=frame,
                        metrics={
                            'episode': ep + 1,
                            'reward': episode_reward,
                            'distance': info[0].get('x_pos', 0),
                        },
                    )

                if done[0]:
                    break

            total_rewards.append(episode_reward)
            distance = info[0].get('x_pos', 0) if info else 0
            print(f'  Eval Episode {ep+1}: '
                  f'Reward={episode_reward:.0f}, Distance={distance}')

        avg_reward = np.mean(total_rewards)
        print(f'\n  Average Eval Reward: {avg_reward:.0f}')
        return avg_reward

    def save_checkpoint(self, path: str) -> None:
        """
        Save PPO model checkpoint.

        Uses stable-baselines3's native save format (.zip file containing
        policy network, value network, and optimizer state).

        Args:
            path: Path for the checkpoint (without extension).
                  The .zip extension is added automatically.
        """
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else '.',
            exist_ok=True,
        )
        save_path = f'{path}.zip'
        self.model.save(save_path)
        print(f'  Saved PPO model to: {save_path}')

    def load_checkpoint(self, path: str) -> None:
        """
        Load PPO model from checkpoint.

        Args:
            path: Path to the checkpoint file (.zip).

        Raises:
            FileNotFoundError: If the checkpoint doesn't exist.
        """
        # Handle path with or without .zip extension
        if not path.endswith('.zip'):
            path_with_ext = f'{path}.zip'
        else:
            path_with_ext = path

        if not os.path.exists(path_with_ext) and not os.path.exists(path):
            raise FileNotFoundError(f'Checkpoint not found: {path}')

        actual_path = path_with_ext if os.path.exists(path_with_ext) else path
        self.model = PPO.load(actual_path, env=self.vec_env)
        print(f'Loaded PPO model from: {actual_path}')
