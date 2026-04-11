"""Tests for the NEAT evolution network visualizer."""

import numpy as np
import pytest

from src.visualization.neat_visualizer import NeatVisualizer


# ---------------------------------------------------------------------------
# Mock NEAT objects
# ---------------------------------------------------------------------------

class MockConnectionGene:
    def __init__(self, weight=1.0, enabled=True):
        self.weight = weight
        self.enabled = enabled


class MockNodeGene:
    def __init__(self, bias=0.0):
        self.bias = bias


class MockGenome:
    """Mimics a neat.DefaultGenome with 2 inputs, 2 hidden, 1 output."""

    def __init__(self):
        self.nodes = {
            0: MockNodeGene(),   # hidden
            1: MockNodeGene(),   # hidden
        }
        self.connections = {
            (-1, 0): MockConnectionGene(0.5, True),    # input -1 -> hidden 0
            (-2, 0): MockConnectionGene(-0.3, True),   # input -2 -> hidden 0
            (0, 1): MockConnectionGene(0.8, True),     # hidden 0 -> hidden 1
            (1, 2): MockConnectionGene(1.2, True),     # hidden 1 -> output 2
            (-1, 2): MockConnectionGene(-0.1, False),  # disabled
        }


class EmptyGenome:
    """Genome with nodes but no connections."""

    def __init__(self):
        self.nodes = {
            0: MockNodeGene(),
        }
        self.connections = {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNeatVisualizer:

    def test_render_returns_numpy_array(self):
        viz = NeatVisualizer()
        genome = MockGenome()
        img = viz.render_genome(genome)
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3  # RGB

    def test_render_correct_dimensions(self):
        viz = NeatVisualizer(width=400, height=300)
        genome = MockGenome()
        img = viz.render_genome(genome)
        assert img.shape == (300, 400, 3)

    def test_render_topology_basic(self):
        """2 inputs, 1 output, 1 connection -> valid image."""
        viz = NeatVisualizer(320, 240)
        nodes = {0, 1, 2}
        connections = [(0, 2, 0.5, True)]
        img = viz.render_topology(
            nodes=nodes,
            connections=connections,
            input_ids={0, 1},
            output_ids={2},
        )
        assert img.shape == (240, 320, 3)
        assert img.dtype == np.uint8
        # Image should not be completely black (nodes are drawn)
        assert img.sum() > 0

    def test_render_genome_mock(self):
        """MockGenome with explicit input/output ids produces a valid image."""
        viz = NeatVisualizer()

        class ConfigStub:
            class genome_config:
                input_keys = [-1, -2]
                output_keys = [2]

        genome = MockGenome()
        img = viz.render_genome(genome, config=ConfigStub())
        assert img.shape == (300, 400, 3)
        assert img.sum() > 0

    def test_disabled_connections_skipped(self):
        """Disabling a connection should change the rendered image."""
        viz = NeatVisualizer(400, 300)

        # All connections enabled
        genome_all = MockGenome()
        # Force the disabled connection to be enabled for the "all" version
        genome_all.connections[(-1, 2)] = MockConnectionGene(-0.1, True)
        img_all = viz.render_genome(genome_all)

        # Original genome has (-1, 2) disabled
        genome_dis = MockGenome()
        img_dis = viz.render_genome(genome_dis)

        # The two images should differ because one has an extra connection drawn
        assert not np.array_equal(img_all, img_dis)

    def test_empty_genome(self):
        """Genome with no connections still produces a valid image."""
        viz = NeatVisualizer()
        genome = EmptyGenome()
        img = viz.render_genome(genome)
        assert img.shape == (300, 400, 3)
        assert img.dtype == np.uint8
        # Should still have at least the node drawn
        assert img.sum() > 0

    def test_custom_size(self):
        viz = NeatVisualizer(200, 150)
        genome = MockGenome()
        img = viz.render_genome(genome)
        assert img.shape == (150, 200, 3)
