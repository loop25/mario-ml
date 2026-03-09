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
from src.visualization.dashboard import Dashboard


class A2CDashboardCallback(BaseCallback):
    """SB3 callback to feed A2C metrics to the dashboard."""

    def __init__(self, dashboard, trainer, verbose=0):
        super().__init__(verbose)
        self.dashboard = dashboard
        self.trainer = trainer
        self.episode_count = 0

    def _on_step(self):
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

                    # Fire episode callback for curriculum
                    distance = infos[i].get('x_pos', 0)
                    completed = infos[i].get('stage_completed', False)
                    self.trainer._fire_episode_complete(
                        reward=reward, distance=distance, completed=completed,
                    )

        # Update dashboard
        if self.dashboard and self.episode_count % 5 == 0:
            metrics = {
                'episode': self.episode_count,
                'reward': self.trainer.best_reward,
                'timestep': self.num_timesteps,
            }
            if not self.trainer.update_visualization(metrics=metrics):
                return False  # Dashboard closed
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
    ):
        super().__init__(env, config, visualizer, save_dir, log_dir)
        self.model = None
        self._env_factory = None

    def train(self, num_episodes: int = None) -> None:
        """Train using A2C.

        Args:
            num_episodes: Ignored (A2C uses total_timesteps from config).
        """
        total_timesteps = self.config.get('total_timesteps', 1_000_000)
        self.is_training = True

        # Create vectorized environment
        num_envs = self.config.get('num_envs', 4)
        if self._env_factory:
            vec_env = DummyVecEnv([self._env_factory for _ in range(num_envs)])
        else:
            vec_env = DummyVecEnv([lambda: self.env])
        vec_env = VecTransposeImage(vec_env)

        # Create A2C model
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
        )

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
