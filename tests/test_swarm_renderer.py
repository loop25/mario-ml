"""Tests for SwarmRenderer and Snake swarm rendering."""
import pytest
import numpy as np

from src.visualization.swarm_renderer import (
    SwarmRenderer,
    generate_agent_colors,
    dim_color,
)
from games.builtin.snake.game import SnakeEnv


# ---------------------------------------------------------------------------
# Color generation
# ---------------------------------------------------------------------------

class TestColorGeneration:
    def test_zero_colors(self):
        assert generate_agent_colors(0) == []

    def test_single_color_is_lime(self):
        colors = generate_agent_colors(1)
        assert len(colors) == 1
        assert colors[0] == (60, 255, 60)

    def test_n_distinct_colors(self):
        """All generated colors should be unique tuples."""
        for n in (2, 4, 8, 16):
            colors = generate_agent_colors(n)
            assert len(colors) == n
            assert len(set(colors)) == n, f"Duplicate colors for n={n}"

    def test_colors_are_valid_rgb(self):
        for r, g, b in generate_agent_colors(10):
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255

    def test_dim_color(self):
        c = dim_color((200, 100, 50), factor=0.5)
        assert c == (100, 50, 25)


# ---------------------------------------------------------------------------
# SwarmRenderer initialisation
# ---------------------------------------------------------------------------

class TestSwarmRendererInit:
    def test_default_init(self):
        sr = SwarmRenderer(640, 700, num_envs=4)
        assert sr.width == 640
        assert sr.height == 700
        assert sr.num_envs == 4
        assert sr.game_type == 'generic'
        assert len(sr.agent_colors) == 4

    def test_board_game_uses_grid_fallback(self):
        sr = SwarmRenderer(640, 700, num_envs=4, game_type='board')
        assert sr._grid_fallback is not None

    def test_grid_type_no_fallback(self):
        sr = SwarmRenderer(640, 700, num_envs=4, game_type='grid')
        assert sr._grid_fallback is None

    def test_last_frame_initially_none(self):
        sr = SwarmRenderer(640, 700, num_envs=2)
        assert sr.last_frame is None

    def test_resize(self):
        sr = SwarmRenderer(640, 700, num_envs=2)
        sr.resize(800, 600)
        assert sr.width == 800
        assert sr.height == 600


# ---------------------------------------------------------------------------
# SwarmRenderer compositing (headless — no pygame screen needed)
# ---------------------------------------------------------------------------

class TestSwarmRendererComposite:
    def _make_frame(self, h=64, w=64, color=(255, 255, 255)):
        """Create a simple solid-color test frame."""
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = color
        return frame

    def test_composite_single_frame(self):
        sr = SwarmRenderer(640, 700, num_envs=1)
        frame = self._make_frame(color=(100, 200, 50))
        result = sr._composite_frames([frame])
        assert result is not None
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_composite_multiple_frames(self):
        sr = SwarmRenderer(640, 700, num_envs=3)
        frames = [
            self._make_frame(color=(100, 0, 0)),
            self._make_frame(color=(0, 100, 0)),
            self._make_frame(color=(0, 0, 100)),
        ]
        result = sr._composite_frames(frames)
        assert result is not None
        assert result.shape == (64, 64, 3)

    def test_composite_with_none_frames(self):
        sr = SwarmRenderer(640, 700, num_envs=3)
        frames = [self._make_frame(color=(100, 100, 100)), None, None]
        result = sr._composite_frames(frames)
        assert result is not None
        assert result.shape == (64, 64, 3)

    def test_composite_all_none_returns_last(self):
        sr = SwarmRenderer(640, 700, num_envs=2)
        result = sr._composite_frames([None, None])
        assert result is None  # no last frame cached yet

    def test_composite_grayscale_input(self):
        sr = SwarmRenderer(640, 700, num_envs=1)
        gray = np.full((64, 64), 128, dtype=np.uint8)
        result = sr._composite_frames([gray])
        assert result.shape == (64, 64, 3)

    def test_composite_preserves_base_when_ghosts_are_black(self):
        """If ghost frames are all black, the base frame is unchanged."""
        sr = SwarmRenderer(640, 700, num_envs=3)
        base = self._make_frame(color=(200, 100, 50))
        black = self._make_frame(color=(0, 0, 0))
        result = sr._composite_frames([base, black, black])
        np.testing.assert_array_equal(result, base)

    def test_sidescroller_composite_runs(self):
        sr = SwarmRenderer(640, 700, num_envs=2, game_type='sidescroller')
        frames = [
            self._make_frame(color=(100, 50, 50)),
            self._make_frame(color=(50, 100, 50)),
        ]
        infos = [{'x_pos': 100}, {'x_pos': 120}]
        result = sr._composite_frames(frames, infos)
        assert result is not None
        assert result.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# Snake render_swarm
# ---------------------------------------------------------------------------

class TestSnakeRenderSwarm:
    def test_returns_correct_shape(self):
        env = SnakeEnv(grid_size=16)
        env.reset()
        snake1 = [(4, 4), (4, 3), (4, 2)]
        snake2 = [(8, 8), (8, 7)]
        foods = [(2, 2), (10, 10)]
        img = env.render_swarm([snake1, snake2], foods)
        expected = env.DISPLAY_SIZE // env.grid_size * env.grid_size
        assert img.shape == (expected, expected, 3)
        assert img.dtype == np.uint8
        env.close()

    def test_empty_snake_list_no_crash(self):
        env = SnakeEnv(grid_size=8)
        env.reset()
        img = env.render_swarm([], [])
        expected = env.DISPLAY_SIZE // env.grid_size * env.grid_size
        assert img.shape == (expected, expected, 3)
        env.close()

    def test_dead_snakes_skipped(self):
        env = SnakeEnv(grid_size=8)
        env.reset()
        # One alive, two dead (empty body lists)
        img = env.render_swarm(
            [[(3, 3), (3, 2)], [], []],
            [(1, 1), None, None],
        )
        assert img.shape[0] > 0
        env.close()

    def test_scores_overlay(self):
        env = SnakeEnv(grid_size=8)
        env.reset()
        img = env.render_swarm(
            [[(3, 3)], [(5, 5)]],
            [(1, 1), (7, 7)],
            scores=[3, 7],
        )
        # Just check it doesn't crash and returns an image
        assert img.dtype == np.uint8
        env.close()

    def test_many_agents(self):
        """Swarm with more agents than the built-in palette."""
        env = SnakeEnv(grid_size=16)
        env.reset()
        snakes = [[(r, 1), (r, 0)] for r in range(12)]
        foods = [(r, 14) for r in range(12)]
        img = env.render_swarm(snakes, foods)
        assert img.shape[0] > 0
        env.close()

    def test_swarm_color_palette_has_no_duplicates(self):
        """The first 8 built-in colors should be distinct."""
        colors = [SnakeEnv._swarm_color(i) for i in range(8)]
        assert len(set(colors)) == 8

    def test_swarm_color_beyond_palette(self):
        """Agent indices beyond the palette fall back to HSV generation."""
        c = SnakeEnv._swarm_color(100)
        assert isinstance(c, tuple)
        assert len(c) == 3
