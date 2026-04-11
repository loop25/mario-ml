"""
Observation/Action Tokenizer for the Decision Transformer.

Converts heterogeneous game data into a standardized format that the
Decision Transformer can consume. Each game has different observation
shapes, action spaces, and reward ranges — the tokenizer normalizes
all of these.

Tokenization pipeline per timestep:
    1. Observation → resize to game's obs_resolution → uint8 tensor
    2. Action → integer in [0, action_vocab_size)
    3. Return-to-go → float (sum of future rewards in episode)
    4. Game token → integer game_token_id from TokenConfig

The tokenizer works at the trajectory level: it takes a complete
episode from a game agent and produces a TokenizedTrajectory that
the Experience Store can persist.

Usage:
    from src.experience.tokenizer import Tokenizer
    from games.reward_config import TokenConfig

    config = TokenConfig(
        obs_resolution=(84, 84),
        obs_channels=1,
        action_vocab_size=7,
        game_token_id=42,
    )
    tokenizer = Tokenizer(config)
    trajectory = tokenizer.tokenize_episode(
        observations, actions, rewards, dones
    )
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from games.reward_config import TokenConfig


@dataclass
class TokenizedTrajectory:
    """A tokenized episode ready for the Decision Transformer.

    All arrays have shape (T, ...) where T is the episode length.

    Attributes:
        game_token_id: Integer identifying which game this came from.
        observations: uint8 array of shape (T, C, H, W) — CHW format.
        actions: int32 array of shape (T,) — action indices.
        rewards: float32 array of shape (T,) — per-step rewards.
        returns_to_go: float32 array of shape (T,) — cumulative future rewards.
        timesteps: int32 array of shape (T,) — position within episode.
        dones: bool array of shape (T,) — episode termination flags.
        total_return: float — sum of all rewards in this episode.
        length: int — number of timesteps (T).
    """
    game_token_id: int
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    returns_to_go: np.ndarray
    timesteps: np.ndarray
    dones: np.ndarray
    total_return: float
    length: int


class Tokenizer:
    """Converts raw game data into standardized token sequences.

    Args:
        token_config: TokenConfig from the game adapter specifying
            observation resolution, channels, action vocab, and game ID.
    """

    def __init__(self, token_config: TokenConfig):
        self.config = token_config

    def _resize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Resize and normalize an observation to the target resolution.

        Args:
            obs: Raw observation from the game environment.
                Can be (H, W), (H, W, C), or already the target size.

        Returns:
            uint8 array of shape (C, H, W) — channels-first format.
        """
        target_h, target_w = self.config.obs_resolution
        target_c = self.config.obs_channels

        # Ensure we have a numpy array
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)

        # Handle different input shapes
        if obs.ndim == 2:
            # (H, W) grayscale
            obs = obs[:, :, np.newaxis]
        elif obs.ndim == 3 and obs.shape[0] in (1, 3, 4):
            # Might be CHW already — transpose to HWC for resizing
            if obs.shape[0] <= 4 and obs.shape[1] > 4 and obs.shape[2] > 4:
                obs = np.transpose(obs, (1, 2, 0))

        h, w = obs.shape[0], obs.shape[1]
        c = obs.shape[2] if obs.ndim == 3 else 1

        # Convert to grayscale if needed
        if target_c == 1 and c == 3:
            if HAS_CV2:
                obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
                obs = obs[:, :, np.newaxis]
            else:
                # Simple luminance conversion
                obs = np.mean(obs, axis=2, keepdims=True).astype(np.uint8)
        elif target_c == 3 and c == 1:
            obs = np.repeat(obs, 3, axis=2)

        # Resize if needed
        if h != target_h or w != target_w:
            if HAS_CV2:
                # cv2.resize expects HWC or HW
                if obs.ndim == 3 and obs.shape[2] == 1:
                    obs = cv2.resize(obs[:, :, 0], (target_w, target_h))
                    obs = obs[:, :, np.newaxis]
                else:
                    obs = cv2.resize(obs, (target_w, target_h))
                    if obs.ndim == 2:
                        obs = obs[:, :, np.newaxis]
            else:
                # Basic nearest-neighbor resize without cv2
                obs = self._naive_resize(obs, target_h, target_w)

        # Ensure uint8
        if obs.dtype != np.uint8:
            if obs.max() <= 1.0:
                obs = (obs * 255).astype(np.uint8)
            else:
                obs = obs.astype(np.uint8)

        # Convert to CHW (channels first) for PyTorch
        if obs.ndim == 3:
            obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
        else:
            obs = obs[np.newaxis, :, :]  # HW -> 1HW

        return obs

    @staticmethod
    def _naive_resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Simple nearest-neighbor resize fallback when cv2 is unavailable."""
        h, w = img.shape[0], img.shape[1]
        row_indices = (np.arange(target_h) * h / target_h).astype(int)
        col_indices = (np.arange(target_w) * w / target_w).astype(int)
        return img[np.ix_(row_indices, col_indices)]

    def _compute_returns_to_go(self, rewards: np.ndarray) -> np.ndarray:
        """Compute discounted returns-to-go (no discounting, sum of future rewards).

        For the Decision Transformer, return-to-go at timestep t is:
            R_t = sum(rewards[t:])

        Using undiscounted returns as in the original DT paper.

        Args:
            rewards: float32 array of shape (T,).

        Returns:
            float32 array of shape (T,) with returns-to-go.
        """
        rtg = np.zeros_like(rewards, dtype=np.float32)
        running_sum = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running_sum += rewards[t]
            rtg[t] = running_sum
        return rtg

    def tokenize_episode(
        self,
        observations: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        dones: Optional[List[bool]] = None,
    ) -> TokenizedTrajectory:
        """Tokenize a complete episode into a standardized trajectory.

        Args:
            observations: List of T observations from env.reset()/step().
            actions: List of T action integers taken by the agent.
            rewards: List of T rewards received.
            dones: Optional list of T done flags. If None, last step is True.

        Returns:
            TokenizedTrajectory ready for the Experience Store.
        """
        T = len(actions)
        assert len(observations) >= T, (
            f"Need at least {T} observations for {T} actions, got {len(observations)}"
        )

        # Resize and stack observations
        processed_obs = np.stack(
            [self._resize_observation(obs) for obs in observations[:T]],
            axis=0,
        )

        # Actions as int32
        action_arr = np.array(actions, dtype=np.int32)

        # Rewards as float32
        reward_arr = np.array(rewards, dtype=np.float32)

        # Returns-to-go
        rtg = self._compute_returns_to_go(reward_arr)

        # Timesteps (0-indexed positions)
        timestep_arr = np.arange(T, dtype=np.int32)

        # Done flags
        if dones is not None:
            done_arr = np.array(dones, dtype=bool)
        else:
            done_arr = np.zeros(T, dtype=bool)
            done_arr[-1] = True

        return TokenizedTrajectory(
            game_token_id=self.config.game_token_id,
            observations=processed_obs,
            actions=action_arr,
            rewards=reward_arr,
            returns_to_go=rtg,
            timesteps=timestep_arr,
            dones=done_arr,
            total_return=float(reward_arr.sum()),
            length=T,
        )

    def tokenize_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ) -> dict:
        """Tokenize a single transition (for online collection).

        Returns a dict that can be accumulated into a trajectory.
        Less efficient than tokenize_episode() but useful for streaming.
        """
        return {
            'observation': self._resize_observation(obs),
            'action': np.int32(action),
            'reward': np.float32(reward),
            'done': done,
            'game_token_id': self.config.game_token_id,
        }
