"""
Mario game adapter — wraps existing mario_env.py behind BaseGameAdapter.

This migrates our existing Super Mario Bros environment into the
plugin system without changing any of the original code. The
original mario_env.py and CustomRewardWrapper continue to work
unchanged for backwards compatibility.
"""
from typing import Optional, Tuple

import gym

from games.base_adapter import BaseGameAdapter
from games.reward_config import (
    StandardMetrics,
    RewardConfig,
    ActionSpaceInfo,
)
from src.environment.mario_env import create_mario_env, MARIO_ACTIONS
from src.training.curriculum import CurriculumManager


# Approximate max x_pos for progress calculation
_MARIO_MAX_X = 3200


class MarioAdapter(BaseGameAdapter):
    """Adapter for Super Mario Bros via gym-super-mario-bros."""

    @property
    def name(self) -> str:
        return 'Super Mario Bros'

    @property
    def game_id(self) -> str:
        return 'mario'

    @property
    def category(self) -> str:
        return 'platformer'

    @property
    def description(self) -> str:
        return 'Classic NES platformer — run, jump, and stomp through 32 stages.'

    def create_env(self, **kwargs) -> gym.Env:
        """Create a Mario environment.

        Accepted kwargs:
            world (int): World number 1-8. Default 1.
            stage (int): Stage number 1-4. Default 1.
            resize_shape (tuple): Observation size. Default (84, 84).
            frame_stack (int): Frames to stack. Default 4.

        Note: We set apply_reward_shaping=False because the framework
        applies its own TimeRewardWrapper from get_reward_config().
        """
        world = kwargs.get('world', 1)
        stage = kwargs.get('stage', 1)
        resize_shape = kwargs.get('resize_shape', (84, 84))
        frame_stack = kwargs.get('frame_stack', 4)

        return create_mario_env(
            world=world,
            stage=stage,
            resize_shape=resize_shape,
            frame_stack=frame_stack,
            apply_reward_shaping=False,  # Framework handles rewards
            normalize=True,
        )

    def get_action_space_info(self) -> ActionSpaceInfo:
        return ActionSpaceInfo(
            num_actions=len(MARIO_ACTIONS),
            action_labels=[
                'NOOP', 'Right', 'Right+Jump', 'Right+Run',
                'Right+Run+Jump', 'Jump', 'Left',
            ],
        )

    def get_observation_shape(self) -> Tuple[int, ...]:
        return (84, 84, 4)

    def extract_metrics(self, info: dict, episode_time: float) -> StandardMetrics:
        x_pos = info.get('x_pos', 0)
        progress = min(1.0, x_pos / _MARIO_MAX_X)
        return StandardMetrics(
            progress=progress,
            score=float(info.get('coins', 0)),
            completed=bool(info.get('flag_get', False)),
            time_elapsed=episode_time,
        )

    def get_reward_config(self) -> RewardConfig:
        return RewardConfig(
            time_penalty_per_second=0.01,
            completion_bonus=100.0,
            death_penalty=-15.0,
            idle_penalty_per_second=0.005,
            speed_bonus_multiplier=1.5,
            par_time_seconds=120.0,
        )

    def needs_sb3_compat(self) -> bool:
        return True

    def get_game_specific_options(self) -> dict:
        return {
            'world': (int, 1, 'World number 1-8'),
            'stage': (int, 1, 'Stage number 1-4'),
        }

    def get_curriculum(self) -> Optional[CurriculumManager]:
        return CurriculumManager()
