"""
Universal time-based reward wrapper.

Applies reward shaping from a game's RewardConfig to any environment.
This encourages agents to play efficiently by penalizing slow play
and rewarding fast completion.

The wrapper is game-agnostic — it relies on the game adapter's
extract_metrics() to translate game-specific info dicts into
StandardMetrics, then applies the universal reward formula.
"""
import time

try:
    import gymnasium as gym
except ImportError:
    import gym  # Legacy fallback

from games.reward_config import RewardConfig


class TimeRewardWrapper(gym.Wrapper):
    """Applies time-based rewards from the game's RewardConfig.

    Wraps any game environment and applies:
    - Time penalty: small negative reward per second of wall-clock time
    - Idle penalty: extra penalty when no progress is made
    - Speed bonus: multiplied reward for completing under par time
    - Completion bonus: fixed reward for finishing
    - Death penalty: fixed penalty for dying/losing without completing

    Args:
        env: The gym environment to wrap.
        reward_config: A RewardConfig with tuning parameters.
        adapter: The game adapter (used for extract_metrics).
    """

    def __init__(self, env: gym.Env, reward_config: RewardConfig, adapter):
        super().__init__(env)
        self.config = reward_config
        self.adapter = adapter
        self._episode_start = None
        self._last_progress = 0.0
        self._last_step_time = None

    def reset(self, **kwargs):
        self._episode_start = time.time()
        self._last_progress = 0.0
        self._last_step_time = self._episode_start
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        now = time.time()
        dt = now - self._last_step_time
        self._last_step_time = now
        episode_time = now - self._episode_start

        # Extract standard metrics from game-specific info
        metrics = self.adapter.extract_metrics(info, episode_time)
        info['_standard_metrics'] = metrics

        shaped_reward = reward

        # Time penalty: penalize slow play
        shaped_reward -= self.config.time_penalty_per_second * dt

        # Idle penalty: extra cost when not making progress
        if metrics.progress <= self._last_progress:
            shaped_reward -= self.config.idle_penalty_per_second * dt
        self._last_progress = metrics.progress

        # Completion bonus with speed multiplier
        if metrics.completed:
            shaped_reward += self.config.completion_bonus
            if episode_time < self.config.par_time_seconds:
                speed_ratio = 1.0 - (episode_time / self.config.par_time_seconds)
                shaped_reward += (
                    self.config.completion_bonus
                    * speed_ratio
                    * self.config.speed_bonus_multiplier
                )

        # Death/loss penalty
        if done and not metrics.completed:
            shaped_reward += self.config.death_penalty

        return obs, shaped_reward, done, info
