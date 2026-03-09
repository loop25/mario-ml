"""
Universal environment factory.

Creates a ready-to-use Gym environment from any game adapter.
Applies the standard wrapper pipeline (TimeRewardWrapper) and
optional SB3 compatibility wrapping.

This replaces the game-specific factory functions (create_mario_env,
create_cnn_env, etc.) with a single function that works for any game.
"""
import gym

from games.base_adapter import BaseGameAdapter
from src.rewards.time_reward import TimeRewardWrapper


def create_env_from_adapter(
    adapter: BaseGameAdapter,
    use_time_rewards: bool = True,
    sb3_compat: bool = False,
    **game_kwargs,
) -> gym.Env:
    """Create a fully configured environment from a game adapter.

    This is the universal entry point for creating environments.
    It delegates env creation to the adapter, then wraps with
    the framework's standard wrappers.

    Args:
        adapter: The game adapter to create an environment from.
        use_time_rewards: Whether to apply TimeRewardWrapper.
                          Default True.
        sb3_compat: Whether to apply SB3CompatWrapper for
                    stable-baselines3 algorithms. Default False.
        **game_kwargs: Passed to adapter.create_env() for
                       game-specific options (world, stage, etc.).

    Returns:
        gym.Env: A fully wrapped environment ready for training.
    """
    # Create the base environment from the adapter
    env = adapter.create_env(**game_kwargs)

    # Apply time-based reward shaping
    if use_time_rewards:
        reward_config = adapter.get_reward_config()
        env = TimeRewardWrapper(env, reward_config, adapter)

    # Apply SB3 compatibility wrapper if needed
    if sb3_compat or adapter.needs_sb3_compat():
        from src.environment.wrappers import SB3CompatWrapper
        env = SB3CompatWrapper(env)

    return env
