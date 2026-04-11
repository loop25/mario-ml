# Phase 4: Population View, Polish & UX Improvements

> Design doc for the final polish phase. Covers swarm/ghost visualization,
> tiled grid wiring, training presets, Add Game wizard, visual polish,
> dashboard hotkeys, and stream audio shuffle.

**Date:** 2026-03-12
**Branch:** feature/multi-game-generalist-agent

---

## 1. Population / Swarm View ("Ghost Mode")

### Concept
During training with N parallel environments, composite all agents into a
single game frame instead of separate tiles. Each agent appears as a
semi-transparent overlay on the shared game background — like watching
50 Marios all running World 1-1 simultaneously.

### Architecture

New class: `src/visualization/swarm_renderer.py` → `SwarmRenderer`

**How it composites:**
- Frame 0 provides the base background
- For each additional frame, detect the "character pixels" (pixels that
  differ from the background) and alpha-blend them onto the base
- Each agent gets a unique tint color for identification
- The "best" agent (highest reward so far) is rendered at full opacity

**Per-game behavior:**
- **Side-scrollers (Mario, Sonic):** Agents at different scroll positions
  produce different backgrounds. Use frame 0 as the camera. Other agents'
  characters are composited at their relative position offsets using
  info-dict `x_pos` values.
- **Grid games (Snake):** Render all N snakes into the same grid with
  different colors. Each snake has its own hue from an HSV color wheel.
- **Board games (Chess, Connect4, TicTacToe, Checkers):** Swarm mode
  doesn't apply — falls back to tiled grid automatically.

**Dashboard integration:**
- `Dashboard.__init__` gets `display_mode` parameter: `"single"` | `"grid"` | `"swarm"`
- `"single"` = current behavior (one GameRenderer)
- `"grid"` = current GridRenderer (already built)
- `"swarm"` = new SwarmRenderer
- Hotkey `V` cycles through modes during training
- For swarm mode on grid games, auto-fallback to grid

### Snake swarm implementation (simplest starting point)
Snake's `_render_rgb()` already draws onto a black background. For swarm:
1. Run N snake environments
2. Each env tracks its own snake body positions
3. A new `render_swarm(snakes: list)` method draws ALL snakes on one grid
4. Each snake gets a different color from an HSV wheel
5. Food is shared (same random seed) or each env has its own food

### Mario/Sonic swarm implementation
More complex because the level scrolls:
1. All N environments run the same level
2. The camera follows the leading agent (highest x_pos)
3. Other agents are drawn at their x_pos relative to the camera
4. Agents behind the camera left edge or ahead of right edge are
   shown as small arrow indicators at the screen edges
5. Dead agents fade out over 30 frames

---

## 2. Tiled Grid Mode (wire existing GridRenderer)

GridRenderer already exists and works. Needs:
- PPO/DQN/A2C: collect frames from all vec envs in the SB3 callback
- NEAT: pass genome frames from ParallelEvaluator to dashboard
- Launcher: ensure num_envs dropdown feeds through to dashboard creation

---

## 3. Training Presets

Dropdown in launcher replacing raw episode count entry:

| Preset | Episodes | Description |
|--------|----------|-------------|
| Quick Demo | 100 | See results in minutes |
| Standard | 5,000 | Good training run |
| Deep Training | 25,000 | Multi-hour deep learning |
| Overnight | 50,000 | Leave running overnight |
| Custom | (manual) | User enters their own value |

Implementation: `ttk.Combobox` that sets the episode `StringVar`. "Custom"
reveals the manual entry field.

---

## 4. "Add Game" Wizard

Button in launcher extras section:
1. Prompt for game name (simple dialog)
2. Copy `games/user/_template/` to `games/user/<sanitized_name>/`
3. Open the new folder in file explorer
4. Show brief instructions in a message box

---

## 5. Visual Polish: Chess & Checkers

### Chess
- Wooden board texture (alternating tan/brown squares)
- Piece rendering with shadows and anti-aliased outlines
- Last-move highlight (yellow tint on from/to squares)
- Check indicator (red tint on king's square)
- Captured pieces display along edges

### Checkers
- Classic red/black board with beveled squares
- Pieces with 3D-effect shading (gradient circles)
- King pieces with crown symbol
- Valid move highlights during agent turns
- Jump chain visualization

---

## 6. Dashboard Hotkey Overlay

Press `?` or `H` during training:
- Semi-transparent dark overlay covers 60% of screen center
- Lists all keyboard shortcuts in two columns
- Auto-dismisses on any keypress
- Shortcuts: SPACE=pause, ESC=quit, M=mute, +/-=volume,
  N=next track, V=display mode, R=record, ?=help

---

## 7. Stream Audio Shuffle

Before passing playlist to StreamManager, shuffle the list:
```python
import random
audio_files = list(music_manager.playlist)
random.shuffle(audio_files)
stream_manager = StreamManager(audio_files=audio_files)
```

---

## Implementation Order

1. SwarmRenderer for Snake (simplest proof of concept)
2. Wire tiled grid for PPO/A2C/NEAT
3. SwarmRenderer for side-scrollers (Mario/Sonic)
4. Display mode toggle in dashboard + launcher
5. Training presets (small launcher change)
6. Add Game wizard (small launcher change)
7. Chess visual polish
8. Checkers visual polish
9. Dashboard hotkey overlay
10. Stream audio shuffle
11. Tests for all new components
