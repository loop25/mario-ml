"""Tests for the Observation/Action Tokenizer (src/experience/tokenizer.py).

Validates:
    - TokenizedTrajectory dataclass structure
    - Observation resizing (HW, HWC, CHW inputs; grayscale conversion)
    - Returns-to-go computation (backward cumulative sum)
    - Episode tokenization (correct shapes, dtypes, values)
    - Single transition tokenization
    - Edge cases (1-step episode, already-correct-size obs)
"""

import numpy as np
import pytest

from games.reward_config import TokenConfig
from src.experience.tokenizer import Tokenizer, TokenizedTrajectory


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def token_config():
    """Standard token config for 84x84 grayscale with 7 actions."""
    return TokenConfig(
        obs_resolution=(84, 84),
        obs_channels=1,
        action_vocab_size=7,
        game_token_id=42,
    )


@pytest.fixture
def tokenizer(token_config):
    return Tokenizer(token_config)


@pytest.fixture
def simple_episode():
    """5-step episode with small observations."""
    T = 5
    observations = [np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8) for _ in range(T)]
    actions = [0, 1, 2, 3, 4]
    rewards = [1.0, 2.0, 3.0, 0.0, 5.0]
    dones = [False, False, False, False, True]
    return observations, actions, rewards, dones


# ── TokenizedTrajectory ──────────────────────────────────────────────


class TestTokenizedTrajectory:

    def test_dataclass_fields(self, token_config, tokenizer, simple_episode):
        obs, acts, rews, dones = simple_episode
        traj = tokenizer.tokenize_episode(obs, acts, rews, dones)
        assert isinstance(traj, TokenizedTrajectory)
        assert traj.game_token_id == 42
        assert traj.length == 5
        assert traj.total_return == pytest.approx(11.0)

    def test_array_shapes(self, tokenizer, simple_episode):
        obs, acts, rews, dones = simple_episode
        traj = tokenizer.tokenize_episode(obs, acts, rews, dones)
        T = 5
        assert traj.observations.shape == (T, 1, 84, 84)
        assert traj.actions.shape == (T,)
        assert traj.rewards.shape == (T,)
        assert traj.returns_to_go.shape == (T,)
        assert traj.timesteps.shape == (T,)
        assert traj.dones.shape == (T,)

    def test_array_dtypes(self, tokenizer, simple_episode):
        obs, acts, rews, dones = simple_episode
        traj = tokenizer.tokenize_episode(obs, acts, rews, dones)
        assert traj.observations.dtype == np.uint8
        assert traj.actions.dtype == np.int32
        assert traj.rewards.dtype == np.float32
        assert traj.returns_to_go.dtype == np.float32
        assert traj.timesteps.dtype == np.int32
        assert traj.dones.dtype == bool


# ── Observation Resizing ─────────────────────────────────────────────


class TestObservationResize:

    def test_hw_grayscale_input(self, tokenizer):
        """2D grayscale input (H, W) should become (1, 84, 84)."""
        obs = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
        result = tokenizer._resize_observation(obs)
        assert result.shape == (1, 84, 84)
        assert result.dtype == np.uint8

    def test_hwc_rgb_input(self, tokenizer):
        """3D RGB input (H, W, 3) should be converted to (1, 84, 84) grayscale."""
        obs = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        result = tokenizer._resize_observation(obs)
        assert result.shape == (1, 84, 84)
        assert result.dtype == np.uint8

    def test_chw_input_detected(self, tokenizer):
        """CHW input (1, 100, 120) should be transposed then resized."""
        obs = np.random.randint(0, 255, (1, 100, 120), dtype=np.uint8)
        result = tokenizer._resize_observation(obs)
        assert result.shape == (1, 84, 84)

    def test_already_correct_size(self, tokenizer):
        """Input already at (84, 84) should pass through without resize."""
        obs = np.random.randint(0, 255, (84, 84), dtype=np.uint8)
        result = tokenizer._resize_observation(obs)
        assert result.shape == (1, 84, 84)

    def test_float_input_normalized(self, tokenizer):
        """Float [0, 1] input should be scaled to uint8."""
        obs = np.random.rand(100, 120).astype(np.float32)
        result = tokenizer._resize_observation(obs)
        assert result.dtype == np.uint8
        assert result.max() > 0  # Should have been scaled

    def test_rgb_token_config(self):
        """3-channel output when obs_channels=3."""
        config = TokenConfig(
            obs_resolution=(84, 84),
            obs_channels=3,
            action_vocab_size=4,
            game_token_id=1,
        )
        tokenizer = Tokenizer(config)
        obs = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        result = tokenizer._resize_observation(obs)
        assert result.shape == (3, 84, 84)


# ── Returns-to-Go ────────────────────────────────────────────────────


class TestReturnsToGo:

    def test_simple_rtg(self, tokenizer):
        """RTG at each step should be the sum of future rewards."""
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rtg = tokenizer._compute_returns_to_go(rewards)
        np.testing.assert_array_almost_equal(rtg, [6.0, 5.0, 3.0])

    def test_single_step_rtg(self, tokenizer):
        rewards = np.array([5.0], dtype=np.float32)
        rtg = tokenizer._compute_returns_to_go(rewards)
        np.testing.assert_array_almost_equal(rtg, [5.0])

    def test_zero_rewards(self, tokenizer):
        rewards = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        rtg = tokenizer._compute_returns_to_go(rewards)
        np.testing.assert_array_almost_equal(rtg, [0.0, 0.0, 0.0])

    def test_negative_rewards(self, tokenizer):
        rewards = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
        rtg = tokenizer._compute_returns_to_go(rewards)
        np.testing.assert_array_almost_equal(rtg, [-2.0, -1.0, -3.0])


# ── Episode Tokenization ─────────────────────────────────────────────


class TestTokenizeEpisode:

    def test_returns_to_go_values(self, tokenizer, simple_episode):
        obs, acts, rews, dones = simple_episode
        traj = tokenizer.tokenize_episode(obs, acts, rews, dones)
        # rewards = [1, 2, 3, 0, 5]  → RTG = [11, 10, 8, 5, 5]
        expected_rtg = np.array([11.0, 10.0, 8.0, 5.0, 5.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(traj.returns_to_go, expected_rtg)

    def test_timesteps_are_sequential(self, tokenizer, simple_episode):
        obs, acts, rews, dones = simple_episode
        traj = tokenizer.tokenize_episode(obs, acts, rews, dones)
        np.testing.assert_array_equal(traj.timesteps, [0, 1, 2, 3, 4])

    def test_actions_preserved(self, tokenizer, simple_episode):
        obs, acts, rews, dones = simple_episode
        traj = tokenizer.tokenize_episode(obs, acts, rews, dones)
        np.testing.assert_array_equal(traj.actions, [0, 1, 2, 3, 4])

    def test_dones_preserved(self, tokenizer, simple_episode):
        obs, acts, rews, dones = simple_episode
        traj = tokenizer.tokenize_episode(obs, acts, rews, dones)
        assert traj.dones[-1] is True or traj.dones[-1] == True
        assert traj.dones[0] is False or traj.dones[0] == False

    def test_default_dones(self, tokenizer, simple_episode):
        """When dones=None, last step should be True."""
        obs, acts, rews, _ = simple_episode
        traj = tokenizer.tokenize_episode(obs, acts, rews, dones=None)
        assert traj.dones[-1] == True
        assert all(not d for d in traj.dones[:-1])

    def test_single_step_episode(self, tokenizer):
        """1-step episodes should work."""
        obs = [np.zeros((84, 84), dtype=np.uint8)]
        acts = [0]
        rews = [10.0]
        traj = tokenizer.tokenize_episode(obs, acts, rews)
        assert traj.length == 1
        assert traj.total_return == 10.0
        assert traj.observations.shape == (1, 1, 84, 84)


# ── Transition Tokenization ──────────────────────────────────────────


class TestTokenizeTransition:

    def test_transition_keys(self, tokenizer):
        obs = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        result = tokenizer.tokenize_transition(obs, action=3, reward=1.5, done=False)
        assert 'observation' in result
        assert 'action' in result
        assert 'reward' in result
        assert 'done' in result
        assert 'game_token_id' in result

    def test_transition_values(self, tokenizer):
        obs = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        result = tokenizer.tokenize_transition(obs, action=3, reward=1.5, done=False)
        assert result['observation'].shape == (1, 84, 84)
        assert result['action'] == 3
        assert result['reward'] == pytest.approx(1.5)
        assert result['done'] == False
        assert result['game_token_id'] == 42


# ── Naive Resize Fallback ────────────────────────────────────────────


class TestNaiveResize:

    def test_downsample(self):
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = Tokenizer._naive_resize(img, 84, 84)
        assert result.shape == (84, 84, 3)

    def test_upsample(self):
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        result = Tokenizer._naive_resize(img, 84, 84)
        assert result.shape == (84, 84)
