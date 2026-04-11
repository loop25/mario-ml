"""
Global coordinate mapping for Pokemon Red.

Converts local in-game coordinates (x, y, map_id) to global pixel
coordinates on a stitched Kanto world map. Used for the exploration
reward system — tracking which tiles the agent has visited.

Adapted from PokemonRedExperiments v2 (MIT License):
  https://github.com/PWhiddy/PokemonRedExperiments
  Originally from pokemonred_puffer by thatguy11325.
"""

import os
import json

MAP_PATH = os.path.join(os.path.dirname(__file__), "data", "map_data.json")
PAD = 20
GLOBAL_MAP_SHAPE = (444 + PAD * 2, 436 + PAD * 2)
MAP_ROW_OFFSET = PAD
MAP_COL_OFFSET = PAD

# Load map coordinate data once at import time
with open(MAP_PATH) as _f:
    _raw = json.load(_f)["regions"]
MAP_DATA = {int(e["id"]): e for e in _raw}


def local_to_global(r: int, c: int, map_n: int):
    """Convert local game coordinates to global map coordinates.

    Args:
        r: Row (y position) in local map.
        c: Column (x position) in local map.
        map_n: Map ID from game memory.

    Returns:
        Tuple (global_row, global_col) on the stitched world map.
        Falls back to center of map if coordinates are invalid.
    """
    try:
        map_x, map_y = MAP_DATA[map_n]["coordinates"]
        gy = r + map_y + MAP_ROW_OFFSET
        gx = c + map_x + MAP_COL_OFFSET
        if 0 <= gy < GLOBAL_MAP_SHAPE[0] and 0 <= gx < GLOBAL_MAP_SHAPE[1]:
            return gy, gx
        # Out of bounds — return center as fallback
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
    except KeyError:
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
