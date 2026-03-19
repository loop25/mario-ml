"""Tests for the Agent Personality Profile system."""

import json
import os
import tempfile

import pytest

from src.achievements.profile import AgentProfile, generate_profile, scan_all_profiles

SAMPLE_METADATA = {
    "game_id": "snake",
    "algorithm": "ppo",
    "episode": 5000,
    "best_reward": 42.5,
    "avg_reward": 22.3,
    "reward_std": 8.1,
    "elapsed_time": "1:30:00",
    "best_distance": 0,
}

VALID_PLAY_STYLES = {
    "Consistent",
    "Explorer",
    "Speedrunner",
    "Resilient",
    "Peak Performer",
    "Balanced",
}


def _write_metadata(directory: str, data: dict | None = None) -> str:
    """Write a metadata.json into *directory* and return its path."""
    path = os.path.join(directory, "metadata.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data or SAMPLE_METADATA, fh)
    return path


class TestGenerateProfile:
    """generate_profile produces correct fields from metadata."""

    def test_generate_profile_from_metadata(self, tmp_path):
        meta_path = _write_metadata(str(tmp_path))
        profile = generate_profile(meta_path)

        assert isinstance(profile, AgentProfile)
        assert profile.game_id == "snake"
        assert profile.algorithm == "ppo"
        assert profile.total_episodes == 5000
        assert profile.best_reward == pytest.approx(42.5)
        assert profile.avg_reward == pytest.approx(22.3)
        assert profile.training_time_hours == pytest.approx(1.5, abs=0.01)
        assert profile.agent_id == "snake_ppo"
        assert profile.display_name == "Snake PPO Agent"

    def test_profile_dimensions_in_range(self, tmp_path):
        meta_path = _write_metadata(str(tmp_path))
        profile = generate_profile(meta_path)

        for dim_name in (
            "consistency",
            "exploration",
            "speed",
            "resilience",
            "peak_performance",
        ):
            value = getattr(profile, dim_name)
            assert 0 <= value <= 100, f"{dim_name}={value} out of range"

    def test_play_style_assigned(self, tmp_path):
        meta_path = _write_metadata(str(tmp_path))
        profile = generate_profile(meta_path)

        assert profile.play_style in VALID_PLAY_STYLES

    def test_badges_is_list(self, tmp_path):
        meta_path = _write_metadata(str(tmp_path))
        profile = generate_profile(meta_path)

        assert isinstance(profile.badges, list)

    def test_handles_minimal_metadata(self, tmp_path):
        """Profile generation works even with very sparse metadata."""
        meta_path = _write_metadata(str(tmp_path), {"game_id": "tetris"})
        profile = generate_profile(meta_path)

        assert profile.game_id == "tetris"
        assert profile.total_episodes == 0
        assert profile.play_style in VALID_PLAY_STYLES

    def test_handles_short_elapsed_format(self, tmp_path):
        """Parses '1m 17s' style elapsed_time correctly."""
        data = {**SAMPLE_METADATA, "elapsed_time": "1m 17s"}
        meta_path = _write_metadata(str(tmp_path), data)
        profile = generate_profile(meta_path)

        # 1m17s ~= 0.0214 hours
        assert 0.01 < profile.training_time_hours < 0.05


class TestScanAllProfiles:
    """scan_all_profiles discovers metadata files in a directory tree."""

    def test_scan_empty_dir(self, tmp_path):
        result = scan_all_profiles(str(tmp_path))
        assert result == []

    def test_scan_with_metadata(self, tmp_path):
        model_dir = tmp_path / "ppo"
        model_dir.mkdir()
        _write_metadata(str(model_dir))

        profiles = scan_all_profiles(str(tmp_path))
        assert len(profiles) == 1
        assert profiles[0].game_id == "snake"

    def test_scan_multiple_sorted_by_reward(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        _write_metadata(str(dir_a), {**SAMPLE_METADATA, "best_reward": 10.0})

        dir_b = tmp_path / "b"
        dir_b.mkdir()
        _write_metadata(str(dir_b), {**SAMPLE_METADATA, "best_reward": 50.0})

        profiles = scan_all_profiles(str(tmp_path))
        assert len(profiles) == 2
        assert profiles[0].best_reward > profiles[1].best_reward

    def test_scan_skips_bad_json(self, tmp_path):
        model_dir = tmp_path / "bad"
        model_dir.mkdir()
        bad_path = model_dir / "metadata.json"
        bad_path.write_text("NOT JSON", encoding="utf-8")

        profiles = scan_all_profiles(str(tmp_path))
        assert profiles == []

    def test_scan_nonexistent_dir(self):
        profiles = scan_all_profiles("/nonexistent/path/that/does/not/exist")
        assert profiles == []
