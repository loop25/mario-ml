"""
Rainbow DQN Trainer.

Combines six improvements to DQN into one agent:
    1. Double DQN — uses online net for action selection in targets,
       reducing Q-value overestimation.
    2. Prioritized Experience Replay — samples surprising transitions
       more often via SumTree.
    3. Dueling Networks — separates V(s) and A(s,a) estimation.
    4. Noisy Networks — learned exploration replaces epsilon-greedy.
    5. Distributional RL (C51) — learns return distributions.
    6. Multi-step Returns — uses n-step bootstrapping.

Each improvement can be toggled independently via config for
ablation studies. With all disabled, behaves like vanilla DQN.

Reference: Hessel et al., 2018 — "Rainbow: Combining Improvements
in Deep Reinforcement Learning"

Usage:
    trainer = RainbowTrainer(env, config, visualizer=dashboard)
    trainer.train(num_episodes=5000)
    trainer.save_checkpoint('models/rainbow/model')
"""

import os
from collections import deque

import numpy as np
import torch
import torch.optim as optim
from typing import Dict, Any, Optional

from src.algorithms.base_trainer import BaseTrainer
from src.algorithms.device import select_device
from src.algorithms.frame_utils import capture_display_frame
from src.algorithms.rainbow.rainbow_network import RainbowNetwork
from src.algorithms.rainbow.prioritized_replay import (
    PrioritizedReplayBuffer,
    PrioritizedExperience,
)
from src.algorithms.dqn.replay_buffer import ReplayBuffer
from src.visualization.dashboard import Dashboard


class RainbowTrainer(BaseTrainer):
    """Rainbow DQN trainer with all six improvements.

    Extends BaseTrainer directly (not DQNTrainer) because nearly
    every method differs: distributional loss, PER priority updates,
    multi-step returns, noise reset.

    Args:
        env: Gym environment with observations suitable for CNN.
        config: Hyperparameters (from rainbow_config.yaml).
        visualizer: Optional Dashboard for live visualization.
        save_dir: Checkpoint directory. Default 'models'.
        log_dir: Log directory. Default 'logs'.
        num_envs: Number of environments (currently single-env only).
        world: World number (for Mario curriculum).
        stage: Stage number (for Mario curriculum).
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
        device_preference: Optional[str] = None,
    ):
        super().__init__(env, config, visualizer, save_dir, log_dir)

        self.world = world
        self.stage = stage

        # Centralized device selection with auto-detection
        self.device = select_device(
            preference=device_preference, algo_name='Rainbow'
        )

        # Environment dimensions
        obs_shape = env.observation_space.shape  # (84, 84, 4) or (4, 84, 84)
        n_actions = env.action_space.n

        # Input channels = stacked frames
        input_channels = (
            obs_shape[-1] if len(obs_shape) == 3 and obs_shape[-1] <= 4
            else obs_shape[0]
        )

        # Feature toggles from config
        self.use_double = config.get('use_double_dqn', True)
        self.use_per = config.get('use_prioritized_replay', True)
        self.use_dueling = config.get('use_dueling', True)
        self.use_noisy = config.get('use_noisy', True)
        self.use_distributional = config.get('use_distributional', True)
        self.use_multistep = config.get('use_multistep', True)

        # C51 parameters
        self.num_atoms = config.get('num_atoms', 51)
        self.v_min = config.get('v_min', -10.0)
        self.v_max = config.get('v_max', 10.0)

        # Multi-step parameters
        self.n_step = config.get('n_step', 3) if self.use_multistep else 1
        self.gamma = config.get('gamma', 0.99)

        # ================================================================
        # Networks: online + target (both RainbowNetwork)
        # ================================================================
        net_kwargs = dict(
            input_channels=input_channels,
            num_actions=n_actions,
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            use_dueling=self.use_dueling,
            use_noisy=self.use_noisy,
            use_distributional=self.use_distributional,
            sigma_init=config.get('sigma_init', 0.5),
        )

        self.online_net = RainbowNetwork(**net_kwargs).to(self.device)
        self.target_net = RainbowNetwork(**net_kwargs).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer (Adam with higher epsilon per Rainbow paper)
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=config.get('learning_rate', 6.25e-5),
            eps=config.get('adam_epsilon', 1.5e-4),
        )

        # ================================================================
        # Replay Buffer — PER or vanilla
        # ================================================================
        obs_storage_shape = (input_channels, 84, 84)
        if self.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=config.get('buffer_size', 100000),
                observation_shape=obs_storage_shape,
                alpha=config.get('per_alpha', 0.6),
            )
            self.beta_start = config.get('per_beta_start', 0.4)
            self.beta_end = config.get('per_beta_end', 1.0)
            self.beta_frames = config.get('per_beta_frames', 100000)
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=config.get('buffer_size', 100000),
                observation_shape=obs_storage_shape,
            )

        # ================================================================
        # N-step transition buffer
        # ================================================================
        self.n_step_buffer = deque(maxlen=self.n_step)

        # ================================================================
        # Epsilon-greedy (only when noisy nets are disabled)
        # ================================================================
        if not self.use_noisy:
            self.epsilon = config.get('epsilon_start', 1.0)
            self.epsilon_end = config.get('epsilon_end', 0.01)
            self.epsilon_decay = config.get('epsilon_decay', 0.995)
        else:
            self.epsilon = 0.0  # Noisy nets handle exploration

        # Training parameters
        self.batch_size = config.get('batch_size', 32)
        self.target_update_freq = config.get('target_update_freq', 1000)
        self.learning_starts = config.get('learning_starts', 10000)
        self.max_steps = config.get('max_steps_per_episode', 5000)

        # Counters
        self.total_steps = 0
        self.n_actions = n_actions

    def _preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert HWC observation to CHW float32 [0, 1]."""
        if obs.ndim == 3 and obs.shape[-1] <= 4:
            obs = np.transpose(obs, (2, 0, 1))
        if obs.dtype == np.uint8:
            obs = obs.astype(np.float32) * (1.0 / 255.0)
        elif obs.dtype != np.float32:
            obs = obs.astype(np.float32)
        return obs

    def _compute_n_step_return(self) -> tuple:
        """Compute n-step discounted return from the transition buffer.

        R_n = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^(n-1) * r_{n-1}

        Returns:
            (state_0, action_0, n_step_return, last_next_state, last_done)
            where state_0/action_0 are from the oldest transition.
        """
        n_step_return = 0.0
        for i, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma ** i) * r
            if d:
                # Episode ended before n steps — truncate
                return s if i == 0 else self.n_step_buffer[0][0], \
                    self.n_step_buffer[0][1], n_step_return, ns, True

        # All n steps completed without episode ending
        first = self.n_step_buffer[0]
        last = self.n_step_buffer[-1]
        return first[0], first[1], n_step_return, last[3], last[4]

    def select_action(self, state: np.ndarray) -> int:
        """Select action using noisy nets or epsilon-greedy.

        With noisy nets: always use the online network (noise provides
        exploration automatically).
        Without noisy nets: epsilon-greedy like vanilla DQN.
        """
        if not self.use_noisy and np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net.q_values(state_t)
            return q_values.argmax(dim=1).item()

    def _train_step(self) -> float:
        """Perform one training step with all Rainbow improvements.

        Returns:
            Training loss value (for logging).
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0

        # Sample batch (with or without priorities)
        if self.use_per:
            beta = self._current_beta()
            batch, is_weights, tree_indices = self.replay_buffer.sample(
                self.batch_size, beta=beta, device=str(self.device),
            )
        else:
            batch = self.replay_buffer.sample(
                self.batch_size, device=str(self.device),
            )
            is_weights = torch.ones(self.batch_size, device=self.device)
            tree_indices = None

        if self.use_distributional:
            loss, td_errors = self._distributional_loss(batch, is_weights)
        else:
            loss, td_errors = self._scalar_loss(batch, is_weights)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities in PER buffer
        if self.use_per and tree_indices is not None:
            self.replay_buffer.update_priorities(
                tree_indices, td_errors.cpu().numpy(),
            )

        # Reset noise for next forward pass
        if self.use_noisy:
            self.online_net.reset_noise()
            self.target_net.reset_noise()

        return loss.item()

    def _distributional_loss(
        self,
        batch: PrioritizedExperience,
        is_weights: torch.Tensor,
    ) -> tuple:
        """Compute C51 distributional cross-entropy loss.

        Projects the target distribution onto the fixed support using
        the categorical projection algorithm from Bellemare et al., 2017.

        Returns:
            (loss, td_errors) tuple.
        """
        # Determine which states/dones to use (n-step or 1-step)
        if self.use_multistep and hasattr(batch, 'n_step_returns'):
            next_states = batch.n_step_next_states
            dones = batch.n_step_dones
            rewards = batch.n_step_returns
            discount = self.gamma ** self.n_step
        else:
            next_states = batch.next_states
            dones = batch.dones
            rewards = batch.rewards
            discount = self.gamma

        batch_size = batch.states.size(0)
        support = self.online_net.support  # (num_atoms,)
        delta_z = self.online_net.delta_z

        # Current distribution: log p(s, a)
        log_probs = self.online_net(batch.states)  # (B, A, atoms)
        log_probs_a = log_probs[
            range(batch_size), batch.actions
        ]  # (B, atoms)

        with torch.no_grad():
            # Double DQN: online net selects action, target evaluates
            if self.use_double:
                next_q = self.online_net.q_values(next_states)
                next_actions = next_q.argmax(dim=1)  # (B,)
            else:
                next_q = self.target_net.q_values(next_states)
                next_actions = next_q.argmax(dim=1)

            # Target distribution for selected action
            target_log_probs = self.target_net(next_states)  # (B, A, atoms)
            target_probs = target_log_probs[
                range(batch_size), next_actions
            ].exp()  # (B, atoms)

            # ============================================================
            # Categorical projection algorithm
            # ============================================================
            # Shift support by rewards: Tz = r + gamma^n * z
            Tz = rewards.unsqueeze(1) + discount * support.unsqueeze(0)
            # Clip to [v_min, v_max]
            Tz = Tz.clamp(self.v_min, self.v_max)

            # Compute projection indices
            b = (Tz - self.v_min) / delta_z  # (B, atoms)
            lower = b.floor().long()
            upper = b.ceil().long()

            # Handle edge case where lower == upper (exact atom hit)
            lower = lower.clamp(0, self.num_atoms - 1)
            upper = upper.clamp(0, self.num_atoms - 1)

            # Distribute probability mass to adjacent atoms
            projected = torch.zeros_like(target_probs)
            offset = (
                torch.arange(batch_size, device=self.device)
                .unsqueeze(1)
                .expand(batch_size, self.num_atoms)
                * self.num_atoms
            )

            projected.view(-1).index_add_(
                0, (lower + offset).view(-1),
                (target_probs * (upper.float() - b)).view(-1),
            )
            projected.view(-1).index_add_(
                0, (upper + offset).view(-1),
                (target_probs * (b - lower.float())).view(-1),
            )

            # Zero out distributions for terminal states
            dones_mask = dones.float().unsqueeze(1)
            projected = projected * (1 - dones_mask)
            # For terminal states: all mass on the reward atom
            if dones.any():
                reward_atoms = rewards[dones].unsqueeze(1)
                reward_b = (reward_atoms.clamp(self.v_min, self.v_max) - self.v_min) / delta_z
                r_lower = reward_b.floor().long().clamp(0, self.num_atoms - 1)
                r_upper = reward_b.ceil().long().clamp(0, self.num_atoms - 1)

                terminal_dist = torch.zeros(
                    dones.sum().item(), self.num_atoms, device=self.device,
                )
                terminal_dist.scatter_add_(
                    1, r_lower, (r_upper.float() - reward_b),
                )
                terminal_dist.scatter_add_(
                    1, r_upper, (reward_b - r_lower.float()),
                )
                projected[dones] = terminal_dist

        # Cross-entropy loss: -sum(target * log(predicted))
        element_loss = -(projected * log_probs_a).sum(dim=1)  # (B,)
        loss = (element_loss * is_weights).mean()

        # TD errors for PER (use element-wise loss as proxy)
        td_errors = element_loss.detach().abs()

        return loss, td_errors

    def _scalar_loss(
        self,
        batch,
        is_weights: torch.Tensor,
    ) -> tuple:
        """Standard DQN loss (Huber) for non-distributional mode.

        With Double DQN: online net selects action, target evaluates.
        Without: target net selects and evaluates.

        Returns:
            (loss, td_errors) tuple.
        """
        if self.use_multistep and hasattr(batch, 'n_step_returns'):
            next_states = batch.n_step_next_states
            dones = batch.n_step_dones
            rewards = batch.n_step_returns
            discount = self.gamma ** self.n_step
        else:
            next_states = batch.next_states
            dones = batch.dones
            rewards = batch.rewards
            discount = self.gamma

        # Current Q-values for chosen actions
        q_values = self.online_net(batch.states)  # (B, A)
        current_q = q_values.gather(
            1, batch.actions.unsqueeze(1),
        ).squeeze(1)  # (B,)

        with torch.no_grad():
            if self.use_double:
                # Online net selects action, target evaluates
                next_q_online = self.online_net(next_states)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.target_net(next_states)
                next_q = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1)[0]

            next_q[dones] = 0.0
            target_q = rewards + discount * next_q

        # Element-wise Huber loss
        td_errors = (current_q - target_q).abs()
        huber = torch.where(
            td_errors < 1.0,
            0.5 * td_errors ** 2,
            td_errors - 0.5,
        )
        loss = (huber * is_weights).mean()

        return loss, td_errors.detach()

    def _current_beta(self) -> float:
        """Compute current PER beta (annealed linearly)."""
        fraction = min(1.0, self.total_steps / self.beta_frames)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def train(self, num_episodes: int = None) -> None:
        """Main Rainbow DQN training loop.

        Args:
            num_episodes: Episodes to train. Uses config if None.
        """
        if num_episodes is None:
            num_episodes = self.config.get('num_episodes', 5000)

        self.is_training = True

        # Read dashboard config for dynamic metric extraction
        dash_cfg = self.dashboard_config or {}
        info_key = dash_cfg.get('graph_2_info_key', 'x_pos')
        metric_name = dash_cfg.get('graph_2_metric', 'distance')

        # Print training banner
        features = []
        if self.use_double:
            features.append('Double')
        if self.use_per:
            features.append('PER')
        if self.use_dueling:
            features.append('Dueling')
        if self.use_noisy:
            features.append('Noisy')
        if self.use_distributional:
            features.append(f'C51({self.num_atoms})')
        if self.use_multistep:
            features.append(f'{self.n_step}-step')

        print(f'\n{"="*60}')
        print(f'Starting Rainbow DQN Training - {num_episodes} episodes')
        print(f'Device: {self.device}')
        print(f'Features: {", ".join(features) if features else "Vanilla DQN"}')
        print(f'Buffer: {"PER" if self.use_per else "Uniform"} '
              f'({self.config.get("buffer_size", 100000):,})')
        print(f'Learning starts: {self.learning_starts:,} steps')
        print(f'{"="*60}\n')

        # Publish training start event (achievements, milestones, etc.)
        self.publish_training_start()

        for episode in range(1, num_episodes + 1):
            if self._dashboard_closed:
                break

            self.episode_count = episode

            # Reset environment
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            obs = self._preprocess_observation(obs)

            # Reset n-step buffer and noise at episode start
            self.n_step_buffer.clear()
            if self.use_noisy:
                self.online_net.reset_noise()

            episode_reward = 0.0
            episode_loss = 0.0
            loss_count = 0
            game_metric = 0
            action_counts = [0] * self.n_actions
            last_frame = None

            for step in range(self.max_steps):
                if not self.check_pause():
                    return

                self.total_steps += 1

                # Select action
                action = self.select_action(obs)
                action_counts[action] += 1

                # Step environment
                result = self.env.step(action)
                next_obs, reward, done = result[0], result[1], result[2]
                info = result[3] if len(result) > 3 else {}

                next_obs_proc = self._preprocess_observation(next_obs)

                # Accumulate n-step transitions
                self.n_step_buffer.append(
                    (obs, action, reward, next_obs_proc, done),
                )

                # Store in replay buffer once we have n transitions
                if len(self.n_step_buffer) == self.n_step:
                    s0, a0, n_ret, ns_n, d_n = self._compute_n_step_return()
                    first_transition = self.n_step_buffer[0]

                    if self.use_per:
                        self.replay_buffer.push(
                            state=s0,
                            action=a0,
                            reward=first_transition[2],  # immediate reward
                            next_state=first_transition[3],
                            done=first_transition[4],
                            n_step_return=n_ret,
                            n_step_next_state=ns_n,
                            n_step_done=d_n,
                        )
                    else:
                        self.replay_buffer.push(
                            state=s0,
                            action=a0,
                            reward=n_ret if self.use_multistep else first_transition[2],
                            next_state=ns_n if self.use_multistep else first_transition[3],
                            done=d_n if self.use_multistep else first_transition[4],
                        )

                # Train step
                if self.total_steps > self.learning_starts:
                    loss = self._train_step()
                    if loss > 0:
                        episode_loss += loss
                        loss_count += 1

                # Update target network
                if self.total_steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(
                        self.online_net.state_dict(),
                    )

                # DT trajectory collection
                self._dt_record_step(next_obs, action, reward)

                # Track metrics
                episode_reward += reward
                game_metric_val = info.get(info_key, 0)
                game_metric = max(game_metric, game_metric_val)
                obs = next_obs_proc

                # Capture display-quality frame for visualization
                last_frame = capture_display_frame(self.env, fallback_obs=next_obs)

                # Live gameplay display
                if self.visualizer and step % 4 == 0:
                    if not self.update_visualization(
                        frame=last_frame, metrics=None,
                    ):
                        break

                if done:
                    # Flush remaining n-step transitions
                    while len(self.n_step_buffer) > 0:
                        s0, a0, n_ret, ns_n, d_n = self._compute_n_step_return()
                        first = self.n_step_buffer[0]
                        if self.use_per:
                            self.replay_buffer.push(
                                state=s0, action=a0,
                                reward=first[2],
                                next_state=first[3], done=first[4],
                                n_step_return=n_ret,
                                n_step_next_state=ns_n,
                                n_step_done=d_n,
                            )
                        else:
                            self.replay_buffer.push(
                                state=s0, action=a0,
                                reward=n_ret if self.use_multistep else first[2],
                                next_state=ns_n if self.use_multistep else first[3],
                                done=d_n if self.use_multistep else first[4],
                            )
                        self.n_step_buffer.popleft()
                    break

            # Finalize DT trajectory for this episode
            self._dt_finalize_episode()

            # Decay epsilon (only if not using noisy nets)
            if not self.use_noisy:
                self.epsilon = max(
                    self.epsilon_end,
                    self.epsilon * self.epsilon_decay,
                )

            # Update best tracking
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.publish_new_best(episode_reward)
            if game_metric > self.best_distance:
                self.best_distance = game_metric

            # Win tracking for board games
            completed = info.get('stage_completed', False)
            if metric_name == 'win_rate':
                winner = info.get('winner', 0)
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

            # Publish event bus episode_complete (achievements, milestones, etc.)
            self.publish_episode_complete(episode_reward, {
                'distance': game_metric,
                'stage_completed': completed,
            })

            # Fire episode callbacks
            self._fire_episode_complete(
                reward=episode_reward,
                distance=game_metric,
                completed=completed,
            )

            # Average loss
            avg_loss = episode_loss / max(loss_count, 1)

            # Print progress
            if episode % 10 == 0:
                extra = f'Eps={self.epsilon:.3f}, ' if not self.use_noisy else ''
                beta_str = f'Beta={self._current_beta():.3f}, ' if self.use_per else ''
                print(
                    f'  Ep {episode}: '
                    f'Reward={episode_reward:.0f}, '
                    f'{metric_name}={game_metric}, '
                    f'Loss={avg_loss:.4f}, '
                    f'{extra}{beta_str}'
                    f'Buffer={len(self.replay_buffer):,}'
                )

            # Update dashboard
            if self.visualizer and episode % 5 == 0:
                metrics = {
                    'episode': episode,
                    'reward': episode_reward,
                    metric_name: game_metric,
                    'loss': avg_loss,
                    'action_distribution': action_counts,
                }
                if metric_name == 'win_rate' and hasattr(self, '_win_tracker'):
                    total = self._win_tracker['total']
                    metrics['opponent_win_rate'] = (
                        self._win_tracker['losses'] / total if total > 0 else 0
                    )
                if not self.use_noisy:
                    metrics['epsilon'] = self.epsilon

                if not self.update_visualization(
                    frame=last_frame, metrics=metrics,
                ):
                    break

            # Auto-checkpoint
            self._auto_checkpoint(episode)

        # Training complete
        self.is_training = False

        # Publish training end event (achievements, milestones, etc.)
        self.publish_training_end(self.episode_count)

        print(f'\n{"="*60}')
        print(f'Rainbow DQN Training Complete!')
        print(f'Episodes: {self.episode_count}')
        print(f'Best reward: {self.best_reward:.0f}')
        if not self.use_noisy:
            print(f'Final epsilon: {self.epsilon:.4f}')
        print(f'{"="*60}\n')

        self._save_on_exit()

    def evaluate(self, num_episodes: int = 5) -> float:
        """Evaluate the trained Rainbow model.

        Sets the network to eval mode (disables noise in NoisyLinear).

        Args:
            num_episodes: Number of episodes to evaluate.

        Returns:
            Average reward across evaluation episodes.
        """
        self.online_net.eval()

        rewards = []
        for _ in range(num_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            obs = self._preprocess_observation(obs)
            done = False
            total_reward = 0.0

            while not done:
                with torch.no_grad():
                    state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    q_values = self.online_net.q_values(state_t)
                    action = q_values.argmax(dim=1).item()

                result = self.env.step(action)
                obs = self._preprocess_observation(result[0])
                total_reward += result[1]
                done = result[2]

            rewards.append(total_reward)

        self.online_net.train()
        return float(np.mean(rewards))

    def save_checkpoint(self, path: str) -> None:
        """Save model state, optimizer, and training state.

        Args:
            path: File path (without extension).
        """
        checkpoint = {
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode_count,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'best_distance': self.best_distance,
            'epsilon': self.epsilon,
            'config': self.config,
        }
        save_path = f'{path}.pt'
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, path: str) -> None:
        """Load model state from disk.

        Args:
            path: Path to .pt checkpoint file.
        """
        if not path.endswith('.pt'):
            path = f'{path}.pt'

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode_count = checkpoint.get('episode', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        self.best_distance = checkpoint.get('best_distance', 0)
        self.epsilon = checkpoint.get('epsilon', 0.0)
        print(f'  Loaded Rainbow checkpoint: ep {self.episode_count}, '
              f'best={self.best_reward:.0f}')
