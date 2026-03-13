"""
A2C (Advantage Actor-Critic) Trainer.

A2C is a synchronous variant of A3C that uses multiple parallel
environments to collect experience and updates the policy using
advantage estimation. It's simpler and faster per wall-clock second
than PPO, but less stable.

This implementation wraps stable-baselines3's A2C with the same
dashboard integration pattern used by the PPO trainer.

Key differences from PPO:
    - Uses n_steps=5 (shorter rollouts)
    - No clip_range parameter
    - Synchronous gradient updates (all envs contribute to one update)
    - Higher learning rate (0.0007 vs PPO's 0.0003)
"""
import os
import yaml
import numpy as np
from typing import Dict, Any, Optional

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from src.algorithms.base_trainer import BaseTrainer
from src.algorithms.device import select_device
from src.algorithms.frame_utils import capture_display_frame_from_vec_env
from src.visualization.dashboard import Dashboard


class A2CDashboardCallback(BaseCallback):
    """SB3 callback to feed A2C metrics and gameplay frames to the dashboard."""

    def __init__(self, dashboard, trainer, verbose=0):
        super().__init__(verbose)
        self.dashboard = dashboard
        self.trainer = trainer
        self.episode_count = 0
        self._latest_frame = None

    def _on_step(self):
        # Read dashboard config for dynamic metric extraction
        dash_cfg = self.trainer.dashboard_config or {}
        info_key = dash_cfg.get('graph_2_info_key', 'x_pos')
        metric_name = dash_cfg.get('graph_2_metric', 'distance')
        num_actions = self.trainer.num_actions

        # --- Live gameplay frame capture ---
        display_interval = 16
        if self.dashboard and self.num_timesteps % display_interval == 0:
            frame = capture_display_frame_from_vec_env(self.trainer.vec_env)
            if frame is not None:
                self._latest_frame = frame

            if self._latest_frame is not None:
                if not self.trainer.update_visualization(
                    frame=self._latest_frame, metrics=None,
                ):
                    return False

        # Check for completed episodes
        for i, done in enumerate(self.locals.get('dones', [])):
            if done:
                infos = self.locals.get('infos', [])
                if i < len(infos):
                    ep_info = infos[i].get('episode', {})
                    reward = ep_info.get('r', 0)
                    self.episode_count += 1
                    self.trainer.episode_count = self.episode_count

                    if reward > self.trainer.best_reward:
                        self.trainer.best_reward = reward

                    # Extract game-specific metric for curriculum callback
                    game_metric = infos[i].get(info_key, 0)
                    completed = infos[i].get('stage_completed', False)

                    # For Connect4: track win_rate from winner field
                    if metric_name == 'win_rate':
                        winner = infos[i].get('winner', 0)
                        if not hasattr(self, '_win_tracker'):
                            self._win_tracker = {
                                'wins': 0, 'losses': 0, 'total': 0,
                            }
                        self._win_tracker['total'] += 1
                        if winner == 1:
                            self._win_tracker['wins'] += 1
                        elif winner == 2:
                            self._win_tracker['losses'] += 1
                        game_metric = (
                            self._win_tracker['wins'] / self._win_tracker['total']
                        )

                    self.trainer._fire_episode_complete(
                        reward=reward, distance=game_metric, completed=completed,
                    )

                    # Track action distribution from this step
                    actions = self.locals.get('actions', [])
                    action_counts = [0] * num_actions
                    for a in actions:
                        a_int = int(a)
                        if 0 <= a_int < num_actions:
                            action_counts[a_int] += 1

                    # Report game-specific metric to dashboard
                    if self.dashboard and self.episode_count % 5 == 0:
                        metrics = {
                            'episode': self.episode_count,
                            'reward': reward,
                            metric_name: game_metric,
                            'action_distribution': action_counts,
                        }
                        # Also report opponent_win_rate for dual-line chart
                        if metric_name == 'win_rate' and hasattr(self, '_win_tracker'):
                            total = self._win_tracker['total']
                            metrics['opponent_win_rate'] = (
                                self._win_tracker['losses'] / total if total > 0 else 0
                            )
                        if not self.trainer.update_visualization(
                            frame=self._latest_frame, metrics=metrics,
                        ):
                            return False  # Dashboard closed

        # Periodic dashboard heartbeat (keeps frame rendering alive)
        if self.dashboard and self.episode_count % 5 != 0:
            if self.num_timesteps % 100 == 0:
                if not self.trainer.update_visualization(
                    frame=self._latest_frame, metrics=None,
                ):
                    return False
        return True


class A2CTrainer(BaseTrainer):
    """A2C trainer wrapping stable-baselines3.

    Args:
        env: The Gym environment (will be vectorized internally).
        config: Dict of hyperparameters (or loaded from a2c_config.yaml).
        visualizer: Optional Dashboard for live visualization.
        save_dir: Directory for saving checkpoints.
        log_dir: Directory for saving logs.
    """

    def __init__(
        self,
        env,
        config: Dict[str, Any],
        visualizer: Optional[Dashboard] = None,
        save_dir: str = 'models',
        log_dir: str = 'logs',
        num_envs: int = 1,
        device_preference: Optional[str] = None,
    ):
        super().__init__(env, config, visualizer, save_dir, log_dir)
        self.model = None
        self._env_factory = None
        self.num_envs = num_envs
        self.device_preference = device_preference
        self.vec_env = None  # Stored so callback can access envs for frame capture

    def _wrap_env_for_sb3(self, env, num_envs: int = 1):
        """Wrap environment with SB3 compatibility for DummyVecEnv.

        SB3's DummyVecEnv expects gymnasium-style reset() returning
        (obs, info). Our built-in games use old-gym reset() returning
        just obs. SB3CompatWrapper bridges this gap.
        """
        from src.environment.wrappers import SB3CompatWrapper
        compat_env = SB3CompatWrapper(env)
        vec_env = DummyVecEnv([lambda: compat_env])
        vec_env = VecTransposeImage(vec_env)
        return vec_env

    def train(self, num_episodes: int = None) -> None:
        """Train using A2C.

        Args:
            num_episodes: Ignored (A2C uses total_timesteps from config).
        """
        total_timesteps = self.config.get('total_timesteps', 1_000_000)
        self.is_training = True

        # Create vectorized environment with SB3 compatibility wrapping.
        num_envs = self.config.get('num_envs', 4)
        if self._env_factory:
            vec_env = DummyVecEnv([self._env_factory for _ in range(num_envs)])
            vec_env = VecTransposeImage(vec_env)
        else:
            vec_env = self._wrap_env_for_sb3(self.env, num_envs=1)

        # Store vec_env so callback can access envs for frame capture
        self.vec_env = vec_env

        # Centralized device selection with auto-detection
        device = select_device(
            preference=self.device_preference, algo_name='A2C'
        )

        # Create or reuse A2C model (reuse when resuming from checkpoint)
        if self.model is None:
            self.model = A2C(
                'CnnPolicy',
                vec_env,
                n_steps=self.config.get('n_steps', 5),
                learning_rate=self.config.get('learning_rate', 0.0007),
                gamma=self.config.get('gamma', 0.99),
                gae_lambda=self.config.get('gae_lambda', 1.0),
                ent_coef=self.config.get('ent_coef', 0.01),
                vf_coef=self.config.get('vf_coef', 0.5),
                max_grad_norm=self.config.get('max_grad_norm', 0.5),
                verbose=0,
                device=device,
            )
        else:
            self.model.set_env(vec_env)

        callback = A2CDashboardCallback(self.visualizer, self)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

        self.is_training = False
        self._save_on_exit()

    def evaluate(self, num_episodes: int = 5) -> float:
        """Evaluate the trained A2C model."""
        if self.model is None:
            raise RuntimeError('No model loaded. Train first or load a checkpoint.')

        rewards = []
        for _ in range(num_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
            total_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                result = self.env.step(action)
                obs, reward, done = result[0], result[1], result[2]
                total_reward += reward
            rewards.append(total_reward)
        return float(np.mean(rewards))

    def save_checkpoint(self, path: str) -> None:
        if self.model:
            self.model.save(path)

    def load_checkpoint(self, path: str) -> None:
        self.model = A2C.load(path)
