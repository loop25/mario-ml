"""
DQN (Deep Q-Network) Trainer.

DQN learns to estimate the Q-value of each action in each state.
The Q-value Q(s, a) represents the expected total future reward
for taking action `a` in state `s` and then acting optimally.

Training algorithm:
    1. Observe state s, choose action a (epsilon-greedy)
    2. Execute action, observe reward r and next state s'
    3. Store experience (s, a, r, s', done) in replay buffer
    4. Sample random batch from replay buffer
    5. Compute target: y = r + gamma * max(Q_target(s', a'))
    6. Compute loss: MSE(Q_policy(s, a) - y)
    7. Backpropagate and update policy network
    8. Periodically copy policy weights to target network

Two networks are used (Double DQN pattern):
    - Policy network: Updated every step, used to select actions
    - Target network: Updated periodically, provides stable Q targets
    Without a target network, the training targets keep shifting
    as the network updates, causing instability.

Usage:
    trainer = DQNTrainer(env, config, visualizer=dashboard)
    trainer.train(num_episodes=5000)
    trainer.save_checkpoint('models/dqn/dqn_model')
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional

from src.algorithms.base_trainer import BaseTrainer
from src.algorithms.dqn.dqn_network import DQNNetwork
from src.algorithms.dqn.replay_buffer import ReplayBuffer
from src.visualization.dashboard import Dashboard


class DQNTrainer(BaseTrainer):
    """
    DQN trainer with custom PyTorch implementation.

    Features:
        - Epsilon-greedy exploration with decay
        - Experience replay for stable training
        - Target network for stable Q-value targets
        - Automatic GPU detection and usage
        - Periodic checkpoint saving
        - Live dashboard visualization

    Args:
        env: Mario environment with stacked frames.
        config: Dictionary of hyperparameters (from YAML).
        visualizer: Optional Dashboard for live visualization.
        save_dir: Checkpoint directory. Default 'models'.
        log_dir: Log directory. Default 'logs'.

    Example:
        env = create_cnn_env(world=1, stage=1)
        config = yaml.safe_load(open('config/dqn_config.yaml'))
        dashboard = Dashboard(algorithm='dqn')
        trainer = DQNTrainer(env, config, dashboard)
        trainer.train(num_episodes=5000)
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

        # Device selection: Use GPU if available AND functional.
        # Some GPUs (e.g. RTX 5070 Blackwell/sm_120) report CUDA as
        # available but fail on actual kernel execution with cu124.
        self.device = self._select_device()

        # Get environment dimensions
        obs_shape = env.observation_space.shape  # (84, 84, 4)
        n_actions = env.action_space.n            # 7

        # Input channels = number of stacked frames (last dim of obs)
        input_channels = obs_shape[-1] if len(obs_shape) == 3 else obs_shape[0]

        # ================================================================
        # Create two networks: policy (updated each step) and target (stable)
        # ================================================================
        self.policy_net = DQNNetwork(
            input_channels=input_channels,
            num_actions=n_actions,
        ).to(self.device)

        self.target_net = DQNNetwork(
            input_channels=input_channels,
            num_actions=n_actions,
        ).to(self.device)

        # Copy policy weights to target (start synchronized)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Target network is never trained directly
        self.target_net.eval()

        # Optimizer: Adam is the standard choice for DQN
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.get('learning_rate', 1e-4),
        )

        # Loss function: Huber loss is more robust than MSE to outliers
        self.loss_fn = nn.SmoothL1Loss()

        # ================================================================
        # Experience Replay Buffer
        # ================================================================
        self.replay_buffer = ReplayBuffer(
            capacity=config.get('buffer_size', 100000),
            observation_shape=(input_channels, 84, 84),
        )

        # ================================================================
        # Exploration Parameters (epsilon-greedy)
        # ================================================================
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)

        # Training parameters
        self.batch_size = config.get('batch_size', 32)
        self.gamma = config.get('gamma', 0.99)
        self.target_update_freq = config.get('target_update_freq', 1000)
        self.learning_starts = config.get('learning_starts', 10000)
        self.max_steps = config.get('max_steps_per_episode', 5000)

        # Counters
        self.total_steps = 0
        self.n_actions = n_actions

        # ================================================================
        # Multi-Environment Support (round-robin)
        # ================================================================
        self.extra_envs = []
        if num_envs > 1:
            from src.environment.mario_env import create_cnn_env
            # Pump pygame events between env creations so Windows
            # doesn't flag the window as "Not Responding" during setup.
            try:
                import pygame
                _pump = pygame.event.pump
            except (ImportError, Exception):
                _pump = lambda: None
            for i in range(num_envs - 1):
                extra_env = create_cnn_env(world=world, stage=stage)
                self.extra_envs.append(extra_env)
                _pump()  # Keep window responsive
            print(f'  DQN: Created {num_envs} environments (round-robin, shared replay buffer)')

    @staticmethod
    def _select_device() -> torch.device:
        """Select CUDA if available and functional, otherwise CPU.

        Some GPUs (e.g. RTX 5070 Blackwell/sm_120) report
        ``torch.cuda.is_available() == True`` but crash on actual
        kernel execution with older CUDA toolkit versions.  We run a
        quick smoke-test to catch that and fall back to CPU.

        Fix: Install PyTorch nightly with CUDA 12.8+:
            pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
        """
        if not torch.cuda.is_available():
            print('DQN using device: cpu')
            print('  Tip: RTX 50-series needs PyTorch nightly with cu128+')
            print('  Run: pip install --pre torch torchvision torchaudio '
                  '--index-url https://download.pytorch.org/whl/nightly/cu128')
            return torch.device('cpu')
        try:
            a = torch.randn(4, 4, device='cuda')
            _ = a @ a.T
            del a
            torch.cuda.empty_cache()
            gpu = torch.cuda.get_device_name(0)
            print(f'DQN using device: cuda ({gpu})')
            return torch.device('cuda')
        except RuntimeError:
            print('DQN using device: cpu (CUDA kernels not supported on this GPU)')
            print('  Tip: RTX 50-series needs PyTorch nightly with cu128+')
            print('  Run: pip install --pre torch torchvision torchaudio '
                  '--index-url https://download.pytorch.org/whl/nightly/cu128')
            return torch.device('cpu')

    def _preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert observation to channels-first format for PyTorch.

        The environment outputs (H, W, C) but PyTorch expects (C, H, W).

        Args:
            obs: Observation from environment, shape (84, 84, 4).

        Returns:
            Transposed observation, shape (4, 84, 84), float32 [0, 1].
        """
        if obs.ndim == 3 and obs.shape[-1] <= 4:
            # (H, W, C) → (C, H, W)
            obs = np.transpose(obs, (2, 0, 1))
        # Normalize to [0, 1]. Use dtype check instead of max() which
        # scans every pixel — much faster for 84x84x4 arrays.
        if obs.dtype == np.uint8:
            obs = obs.astype(np.float32) * (1.0 / 255.0)
        elif obs.dtype != np.float32:
            obs = obs.astype(np.float32)
        return obs

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy strategy.

        With probability epsilon: choose random action (exploration)
        With probability 1-epsilon: choose best action (exploitation)

        Epsilon starts high (lots of exploration) and decays over time
        as the network learns better Q-values.

        Args:
            state: Preprocessed observation, shape (C, H, W).

        Returns:
            int: Selected action index (0-6).
        """
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return self.env.action_space.sample()
        else:
            # Exploitation: use the policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def _train_step(self) -> float:
        """
        Perform one training step on a batch from the replay buffer.

        The DQN loss is:
            L = E[(Q(s,a) - y)^2]
            where y = r + gamma * max_a' Q_target(s', a') * (1 - done)

        Returns:
            float: Training loss value (for logging).
        """
        # Don't train until we have enough experience
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0

        # Sample a random batch of experiences
        batch = self.replay_buffer.sample(self.batch_size, self.device)

        # ================================================================
        # Compute Q(s, a) for the actions that were actually taken
        # ================================================================
        # policy_net(states) gives Q-values for ALL actions: (batch, 7)
        # We gather the Q-values for the specific actions taken
        current_q = self.policy_net(batch.states)
        current_q = current_q.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        # ================================================================
        # Compute target Q-values: r + gamma * max Q_target(s', a')
        # ================================================================
        with torch.no_grad():
            # Get max Q-value from target network for next states
            next_q = self.target_net(batch.next_states).max(dim=1)[0]
            # Zero out Q-values for terminal states (done=True)
            next_q[batch.dones] = 0.0
            # Bellman equation: Q-target = reward + discounted future Q
            target_q = batch.rewards + self.gamma * next_q

        # ================================================================
        # Compute loss and update policy network
        # ================================================================
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def _update_target_network(self) -> None:
        """
        Copy policy network weights to the target network.

        This is done periodically (every target_update_freq steps)
        to provide stable training targets. If we updated every step,
        the targets would be a "moving goalpost" causing instability.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, num_episodes: int = None) -> None:
        """
        Main DQN training loop.

        For each episode:
        1. Reset environment
        2. For each step: select action, step env, store experience, train
        3. Decay epsilon
        4. Update dashboard with metrics

        When num_envs > 1, uses round-robin stepping across all environments:
        each step picks the next env, steps it, and pushes the experience
        into the shared replay buffer. This provides more diverse training
        data and shows multiple Mario games on the grid display.

        Args:
            num_episodes: Number of episodes to train. If None, uses config.
        """
        if num_episodes is None:
            num_episodes = self.config.get('num_episodes', 5000)

        if self.num_envs > 1:
            self._train_multi_env(num_episodes)
        else:
            self._train_single_env(num_episodes)

    def _train_single_env(self, num_episodes: int) -> None:
        """Original single-environment training loop."""
        self.is_training = True
        print(f'\n{"="*60}')
        print(f'Starting DQN Training - {num_episodes} episodes')
        print(f'Device: {self.device}')
        print(f'Parallel envs: 1')
        print(f'Buffer size: {self.config.get("buffer_size", 100000):,}')
        print(f'Learning starts after: {self.learning_starts:,} steps')
        print(f'{"="*60}\n')

        for episode in range(1, num_episodes + 1):
            # Stop if dashboard was closed
            if self._dashboard_closed:
                break

            self.episode_count = episode

            # Reset environment for new episode
            obs = self.env.reset()
            obs = self._preprocess_observation(obs)

            episode_reward = 0.0
            episode_loss = 0.0
            loss_count = 0
            max_distance = 0
            action_counts = [0] * self.n_actions
            last_frame = None

            for step in range(self.max_steps):
                # Check for pause
                if not self.check_pause():
                    return

                self.total_steps += 1

                # Select action (epsilon-greedy)
                action = self.select_action(obs)
                action_counts[action] += 1

                # Execute action in environment
                next_obs, reward, done, info = self.env.step(action)
                next_obs_processed = self._preprocess_observation(next_obs)

                # Store experience in replay buffer
                self.replay_buffer.push(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs_processed,
                    done=done,
                )

                # Perform training step (if enough experience collected)
                if self.total_steps > self.learning_starts:
                    loss = self._train_step()
                    if loss > 0:
                        episode_loss += loss
                        loss_count += 1

                # Update target network periodically
                if self.total_steps % self.target_update_freq == 0:
                    self._update_target_network()

                # Track metrics
                episode_reward += reward
                x_pos = info.get('x_pos', 0)
                max_distance = max(max_distance, x_pos)
                obs = next_obs_processed

                # Capture the raw NES frame (240x256 RGB) for visualization
                try:
                    last_frame = self.env.unwrapped.screen
                except AttributeError:
                    last_frame = next_obs

                # Show live gameplay every 4 steps
                if self.visualizer and step % 4 == 0:
                    if not self.update_visualization(frame=last_frame, metrics=None):
                        break

                if done:
                    break

            # Decay epsilon after each episode
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon * self.epsilon_decay,
            )

            # Update best tracking
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
            if max_distance > self.best_distance:
                self.best_distance = max_distance

            # Average loss for this episode
            avg_loss = episode_loss / max(loss_count, 1)

            # Check for stage completion
            stage_completed = info.get('stage_completed', False)
            if stage_completed:
                completions = info.get('stage_completions', 0)
                print(f'  *** STAGE COMPLETED! (Episode {episode}, '
                      f'Total completions: {completions}) ***')

            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(
                    f'  Ep {episode}: '
                    f'Reward={episode_reward:.0f}, '
                    f'Dist={max_distance}, '
                    f'Loss={avg_loss:.4f}, '
                    f'Eps={self.epsilon:.3f}, '
                    f'Buffer={len(self.replay_buffer):,}'
                )

            # Update visualization dashboard
            if self.visualizer:
                render_interval = self.config.get('render_every_n_episodes', 5)
                frame = last_frame if episode % render_interval == 0 else None

                metrics = {
                    'episode': episode,
                    'reward': episode_reward,
                    'distance': max_distance,
                    'loss': avg_loss,
                    'epsilon': self.epsilon,
                    'action_distribution': action_counts,
                }

                if not self.update_visualization(frame=frame, metrics=metrics):
                    break  # Dashboard closed

            # Auto-checkpoint
            self._auto_checkpoint(episode)

        # Training complete
        self.is_training = False
        print(f'\n{"="*60}')
        print(f'DQN Training Complete!')
        print(f'Episodes: {self.episode_count}')
        print(f'Best reward: {self.best_reward:.0f}')
        print(f'Final epsilon: {self.epsilon:.4f}')
        print(f'{"="*60}\n')

        # Save final model
        self._save_on_exit()

    def _train_multi_env(self, num_episodes: int) -> None:
        """
        Multi-environment round-robin training loop.

        Runs N environments in the same process. Each training step
        picks the next environment in round-robin order, steps it,
        and pushes the experience into the shared replay buffer.

        This provides more diverse training data (from N independent
        game runs) and enables the multi-Mario grid display.
        """
        all_envs = [self.env] + self.extra_envs
        n = len(all_envs)

        self.is_training = True
        print(f'\n{"="*60}')
        print(f'Starting DQN Training - {num_episodes} episodes')
        print(f'Device: {self.device}')
        print(f'Parallel envs: {n} (round-robin)')
        print(f'Buffer size: {self.config.get("buffer_size", 100000):,}')
        print(f'Learning starts after: {self.learning_starts:,} steps')
        print(f'{"="*60}\n')

        # Per-env state tracking
        env_obs = [None] * n
        env_rewards = [0.0] * n
        env_distances = [0] * n
        env_steps = [0] * n
        env_done = [True] * n  # Start all as done so they reset
        env_action_counts = [[0] * self.n_actions for _ in range(n)]
        env_frames = [None] * n

        episodes_completed = 0
        total_loss = 0.0
        loss_count = 0

        while episodes_completed < num_episodes:
            if self._dashboard_closed:
                break

            # Step each environment in round-robin
            for env_idx in range(n):
                env = all_envs[env_idx]

                # Reset if episode ended
                if env_done[env_idx]:
                    obs = env.reset()
                    env_obs[env_idx] = self._preprocess_observation(obs)
                    env_rewards[env_idx] = 0.0
                    env_distances[env_idx] = 0
                    env_steps[env_idx] = 0
                    env_done[env_idx] = False
                    env_action_counts[env_idx] = [0] * self.n_actions

                obs = env_obs[env_idx]
                self.total_steps += 1

                # Select action
                action = self.select_action(obs)
                env_action_counts[env_idx][action] += 1

                # Step environment
                next_obs, reward, done, info = env.step(action)
                next_obs_processed = self._preprocess_observation(next_obs)

                # Store in shared replay buffer
                self.replay_buffer.push(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs_processed,
                    done=done,
                )

                # Train
                if self.total_steps > self.learning_starts:
                    loss = self._train_step()
                    if loss > 0:
                        total_loss += loss
                        loss_count += 1

                # Update target network
                if self.total_steps % self.target_update_freq == 0:
                    self._update_target_network()

                # Track metrics
                env_rewards[env_idx] += reward
                x_pos = info.get('x_pos', 0)
                env_distances[env_idx] = max(env_distances[env_idx], x_pos)
                env_obs[env_idx] = next_obs_processed
                env_steps[env_idx] += 1

                # Capture frame
                try:
                    env_frames[env_idx] = env.unwrapped.screen
                except AttributeError:
                    env_frames[env_idx] = next_obs

                # Check max steps
                if env_steps[env_idx] >= self.max_steps:
                    done = True

                if done:
                    env_done[env_idx] = True
                    episodes_completed += 1
                    self.episode_count = episodes_completed
                    ep_reward = env_rewards[env_idx]
                    ep_dist = env_distances[env_idx]

                    # Decay epsilon
                    self.epsilon = max(
                        self.epsilon_end,
                        self.epsilon * self.epsilon_decay,
                    )

                    # Track best
                    if ep_reward > self.best_reward:
                        self.best_reward = ep_reward
                    if ep_dist > self.best_distance:
                        self.best_distance = ep_dist

                    # Stage completion
                    if info.get('stage_completed', False):
                        print(f'  *** STAGE COMPLETED! (Episode {episodes_completed}, '
                              f'Env {env_idx}) ***')

                    # Print progress
                    if episodes_completed % 10 == 0:
                        avg_loss = total_loss / max(loss_count, 1)
                        print(
                            f'  Ep {episodes_completed}: '
                            f'Reward={ep_reward:.0f}, '
                            f'Dist={ep_dist}, '
                            f'Loss={avg_loss:.4f}, '
                            f'Eps={self.epsilon:.3f}, '
                            f'Buffer={len(self.replay_buffer):,}'
                        )
                        total_loss = 0.0
                        loss_count = 0

                    # Update dashboard with metrics
                    if self.visualizer:
                        avg_loss = total_loss / max(loss_count, 1) if loss_count > 0 else 0
                        metrics = {
                            'episode': episodes_completed,
                            'reward': ep_reward,
                            'distance': ep_dist,
                            'loss': avg_loss,
                            'epsilon': self.epsilon,
                            'action_distribution': env_action_counts[env_idx].copy(),
                        }
                        if not self.update_visualization_grid(
                            frames=env_frames, metrics=metrics,
                        ):
                            break

                    self._auto_checkpoint(episodes_completed)

                    if episodes_completed >= num_episodes:
                        break

            # Show live grid display every 4 total_steps
            if self.visualizer and self.total_steps % (4 * n) == 0:
                if not self.update_visualization_grid(
                    frames=env_frames, metrics=None,
                ):
                    break

        # Close extra environments
        for extra_env in self.extra_envs:
            try:
                extra_env.close()
            except Exception:
                pass

        self.is_training = False
        print(f'\n{"="*60}')
        print(f'DQN Training Complete!')
        print(f'Episodes: {self.episode_count}')
        print(f'Best reward: {self.best_reward:.0f}')
        print(f'Final epsilon: {self.epsilon:.4f}')
        print(f'{"="*60}\n')

        self._save_on_exit()

    def evaluate(self, num_episodes: int = 5) -> float:
        """
        Evaluate the trained DQN model.

        Runs with epsilon=0 (always greedy, no exploration).

        Args:
            num_episodes: Number of evaluation episodes.

        Returns:
            float: Average reward across evaluation episodes.
        """
        # Save current epsilon and set to 0 for pure exploitation
        saved_epsilon = self.epsilon
        self.epsilon = 0.0

        total_rewards = []

        for ep in range(num_episodes):
            obs = self.env.reset()
            obs = self._preprocess_observation(obs)
            episode_reward = 0.0
            done = False
            steps = 0

            while not done and steps < self.max_steps:
                action = self.select_action(obs)

                next_obs, reward, done, info = self.env.step(action)
                next_obs = self._preprocess_observation(next_obs)

                episode_reward += reward
                obs = next_obs
                steps += 1

                # Render every frame during evaluation
                if self.visualizer:
                    try:
                        raw_frame = self.env.unwrapped.screen
                    except AttributeError:
                        raw_frame = None
                    self.update_visualization(
                        frame=raw_frame,
                        metrics={
                            'episode': ep + 1,
                            'reward': episode_reward,
                            'distance': info.get('x_pos', 0),
                        },
                    )

            total_rewards.append(episode_reward)
            distance = info.get('x_pos', 0)
            print(f'  Eval Episode {ep+1}: '
                  f'Reward={episode_reward:.0f}, Distance={distance}')

        # Restore epsilon
        self.epsilon = saved_epsilon

        avg_reward = np.mean(total_rewards)
        print(f'\n  Average Eval Reward: {avg_reward:.0f}')
        return avg_reward

    def save_checkpoint(self, path: str) -> None:
        """
        Save DQN checkpoint.

        Saves everything needed to resume training:
        - Policy network weights
        - Target network weights
        - Optimizer state
        - Epsilon value
        - Step counters

        Args:
            path: Base path for the checkpoint (without extension).
        """
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else '.',
            exist_ok=True,
        )
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'best_reward': self.best_reward,
            'best_distance': self.best_distance,
        }
        save_path = f'{path}.pt'
        torch.save(checkpoint, save_path)
        print(f'  Saved DQN checkpoint to: {save_path}')

    def load_checkpoint(self, path: str) -> None:
        """
        Load DQN checkpoint and restore all training state.

        Args:
            path: Path to the checkpoint file (.pt).

        Raises:
            FileNotFoundError: If the checkpoint doesn't exist.
        """
        if not path.endswith('.pt'):
            path = f'{path}.pt'

        if not os.path.exists(path):
            raise FileNotFoundError(f'Checkpoint not found: {path}')

        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        self.best_distance = checkpoint.get('best_distance', 0)

        print(f'Loaded DQN checkpoint from: {path}')
        print(f'  Epsilon: {self.epsilon:.4f}')
        print(f'  Total steps: {self.total_steps:,}')
        print(f'  Best reward: {self.best_reward:.0f}')
