"""
NEAT Evolution Network Visualizer.

Renders a NEAT genome's neural network topology as a node-link diagram.
Produces a numpy RGB array suitable for embedding in the training dashboard.

Node layout:
    - Input nodes: bottom row (blue)
    - Output nodes: top row (orange)
    - Hidden nodes: middle rows (white)

Connection colours:
    - Green: positive weight
    - Red: negative weight
    - Thickness proportional to |weight|
    - Disabled connections are skipped
"""

import math
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np


class NeatVisualizer:
    """Renders a NEAT genome's neural network as a node-link diagram."""

    # Node colours (BGR for cv2)
    COLOR_INPUT = (255, 150, 50)    # blue-ish
    COLOR_OUTPUT = (50, 150, 255)   # orange-ish
    COLOR_HIDDEN = (220, 220, 220)  # white-ish
    COLOR_BG = (0, 0, 0)           # black background

    # Connection colours (BGR)
    COLOR_POS = (50, 200, 50)      # green
    COLOR_NEG = (50, 50, 220)      # red

    NODE_RADIUS = 10
    FONT_SCALE = 0.35
    FONT_THICKNESS = 1
    PADDING = 30  # margin from edges

    def __init__(self, width: int = 400, height: int = 300):
        self.width = width
        self.height = height

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_genome(self, genome, config=None) -> np.ndarray:
        """Render a NEAT genome as an RGB numpy array.

        Args:
            genome: A neat.DefaultGenome (or mock with .nodes and .connections).
            config: Optional neat config for input/output node counts.

        Returns:
            (height, width, 3) uint8 numpy array (RGB).
        """
        input_ids, output_ids = self._detect_io_ids(genome, config)
        return self.render_topology(
            nodes=genome.nodes,
            connections=genome.connections,
            input_ids=input_ids,
            output_ids=output_ids,
        )

    def render_topology(
        self,
        nodes: Union[Dict, Set],
        connections: Union[Dict, List],
        input_ids: Optional[Set[int]] = None,
        output_ids: Optional[Set[int]] = None,
    ) -> np.ndarray:
        """Render from raw nodes/connections data.

        Args:
            nodes: dict {node_id: node_gene} or set of node IDs.
            connections: dict {(in, out): connection_gene}
                         or list of (in, out, weight, enabled) tuples.
            input_ids: node IDs to place on the bottom row.
            output_ids: node IDs to place on the top row.

        Returns:
            (height, width, 3) uint8 numpy array (RGB).
        """
        input_ids = set(input_ids) if input_ids else set()
        output_ids = set(output_ids) if output_ids else set()

        # Normalise nodes to a set of IDs
        if isinstance(nodes, dict):
            node_ids = set(nodes.keys())
        else:
            node_ids = set(nodes)

        # Collect any node IDs referenced in connections but missing from nodes
        conn_list = self._normalise_connections(connections)
        for src, dst, _w, _e in conn_list:
            node_ids.add(src)
            node_ids.add(dst)

        # Also ensure io ids are in the full set
        node_ids |= input_ids | output_ids

        # Categorise
        hidden_ids = node_ids - input_ids - output_ids

        # Compute positions
        positions = self._layout(
            sorted(input_ids), sorted(output_ids), sorted(hidden_ids)
        )

        # Draw
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw connections first (behind nodes)
        for src, dst, weight, enabled in conn_list:
            if not enabled:
                continue
            pt1 = positions.get(src)
            pt2 = positions.get(dst)
            if pt1 is None or pt2 is None:
                continue
            colour = self.COLOR_POS if weight >= 0 else self.COLOR_NEG
            thickness = max(1, min(4, int(abs(weight) * 2)))
            cv2.line(canvas, pt1, pt2, colour, thickness, cv2.LINE_AA)

        # Draw nodes
        for nid, pos in positions.items():
            if nid in input_ids:
                colour = self.COLOR_INPUT
            elif nid in output_ids:
                colour = self.COLOR_OUTPUT
            else:
                colour = self.COLOR_HIDDEN
            cv2.circle(canvas, pos, self.NODE_RADIUS, colour, -1, cv2.LINE_AA)
            cv2.circle(canvas, pos, self.NODE_RADIUS, (100, 100, 100), 1, cv2.LINE_AA)

            # Label
            label = str(nid)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_THICKNESS
            )
            tx = pos[0] - tw // 2
            ty = pos[1] + th // 2
            cv2.putText(
                canvas,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.FONT_SCALE,
                (0, 0, 0),
                self.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        # cv2 uses BGR; convert to RGB for the returned array
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_io_ids(self, genome, config) -> Tuple[Set[int], Set[int]]:
        """Determine input and output node IDs.

        If a NEAT config is available, use its genome config to get the
        canonical input/output keys.  Otherwise fall back to a heuristic:
        negative IDs are inputs, IDs present in genome.nodes that are not
        connection sources from inputs are outputs (best-effort).
        """
        if config is not None:
            try:
                gc = config.genome_config
                return set(gc.input_keys), set(gc.output_keys)
            except AttributeError:
                pass

        # Heuristic: negative IDs are inputs
        all_ids: Set[int] = set()
        for nid in genome.nodes:
            all_ids.add(nid)
        for (src, dst) in genome.connections:
            all_ids.add(src)
            all_ids.add(dst)

        input_ids = {nid for nid in all_ids if nid < 0}

        # Heuristic: nodes in genome.nodes that are never a source in a
        # connection *from* a negative node AND are >= 0 are treated as
        # outputs if they appear as the final destination.  Simple approach:
        # output = nodes explicitly in genome.nodes that are >= 0 and are
        # destinations of at least one connection but never sources.
        sources = {src for src, _ in genome.connections}
        non_input = all_ids - input_ids
        output_ids = {nid for nid in non_input if nid not in sources}

        # If heuristic yields nothing, just leave output_ids empty.
        return input_ids, output_ids

    def _normalise_connections(
        self, connections
    ) -> List[Tuple[int, int, float, bool]]:
        """Return a list of (src, dst, weight, enabled) tuples."""
        result = []
        if isinstance(connections, dict):
            for (src, dst), gene in connections.items():
                weight = getattr(gene, "weight", 0.0)
                enabled = getattr(gene, "enabled", True)
                result.append((src, dst, weight, enabled))
        else:
            for item in connections:
                if len(item) == 4:
                    result.append(tuple(item))
                elif len(item) == 3:
                    result.append((*item, True))
                else:
                    result.append((*item, 0.0, True))
        return result

    def _layout(
        self,
        input_ids: List[int],
        output_ids: List[int],
        hidden_ids: List[int],
    ) -> Dict[int, Tuple[int, int]]:
        """Compute (x, y) pixel positions for each node."""
        positions: Dict[int, Tuple[int, int]] = {}
        pad = self.PADDING

        usable_w = self.width - 2 * pad
        usable_h = self.height - 2 * pad

        # Y bands: outputs at top, inputs at bottom, hidden in middle
        y_output = pad
        y_input = self.height - pad

        # Hidden layers — split vertically in the middle region
        num_hidden_layers = max(1, math.ceil(len(hidden_ids) / max(1, len(input_ids))))
        # Simple: put all hidden in one row at the vertical centre
        # For more complex topologies, split into multiple rows.
        if num_hidden_layers == 1 or len(hidden_ids) <= 6:
            hidden_layers = [hidden_ids]
        else:
            per_layer = max(1, len(hidden_ids) // num_hidden_layers)
            hidden_layers = [
                hidden_ids[i : i + per_layer]
                for i in range(0, len(hidden_ids), per_layer)
            ]

        def spread_x(ids: List[int], y: int):
            n = len(ids)
            if n == 0:
                return
            if n == 1:
                positions[ids[0]] = (self.width // 2, y)
                return
            for i, nid in enumerate(ids):
                x = pad + int(i * usable_w / (n - 1))
                positions[nid] = (x, y)

        spread_x(input_ids, y_input)
        spread_x(output_ids, y_output)

        # Hidden layers evenly spaced between output and input rows
        n_layers = len(hidden_layers)
        for li, layer in enumerate(hidden_layers):
            if n_layers == 1:
                y = (y_output + y_input) // 2
            else:
                y = y_output + int((li + 1) * (y_input - y_output) / (n_layers + 1))
            spread_x(layer, y)

        return positions
