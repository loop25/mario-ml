# Mario-ML: Streaming, Music, and Optimization Design

**Date:** 2026-03-04
**Status:** Approved

## Overview

Comprehensive enhancement of the mario-ml project to add simultaneous Twitch+YouTube streaming, background music, true pause/resume, curriculum-based whole-game training, and performance optimizations.

## Architecture

```
launcher.py (GUI)
  [Algorithm] [World/Stage] [Stream Keys] [Music] [START]
  [Twitch Key: ****] [YouTube Key: ****] [Stream: ON/OFF]
       |
       | subprocess
       v
main.py (CLI)
  --stream-twitch KEY --stream-youtube KEY --music PATH
       |
  +----+----+----+
  v    v    v    v
Trainer  Dashboard  Streamer  MusicManager
(Train)  (Pygame)   (ffmpeg)  (pygame.mixer)
         + Overlay   -> Twitch
                     -> YouTube
                     + Audio
```

## Component Designs

### 1. True Pause/Resume (base_trainer.py)

- `training_paused` property on `BaseTrainer`
- `check_pause()` method blocks training thread when paused
- Each trainer calls `self.check_pause()` at top of episode/step loop
- Dashboard SPACE key toggles the flag; status text shows PAUSED/TRAINING
- On pause: timer stops, episode state preserved, dashboard keeps rendering
- On resume: timer resumes from where it stopped

### 2. Background Music (src/audio/music_manager.py -- new)

- `MusicManager` class wrapping `pygame.mixer.music`
- Loads from `assets/music/` directory (user drops in .ogg or .mp3 files)
- Playlist mode: shuffles available tracks, auto-advances
- Volume control via dashboard keyboard (Up/Down arrows) or config
- Mute toggle (M key)
- Dashboard displays current track name and volume level
- Music starts when training begins, pauses when training pauses

### 3. Streaming (src/streaming/stream_manager.py -- new)

**Approach: ffmpeg RTMP Pipeline**

- `StreamManager` class managing an ffmpeg subprocess
- Input: raw video frames piped via stdin + audio file path
- Output: simultaneous RTMP to Twitch and YouTube via `-f tee` muxer
- Video: 1280x720, 30fps, H.264, 4500kbps
- Audio: AAC 128kbps from same music files as MusicManager
- `start_stream()` / `stop_stream()` methods
- Status indicator on dashboard: red dot = LIVE, gray = OFF
- Stream health monitoring via ffmpeg stderr parsing
- Config via `config/streaming_config.yaml`

**ffmpeg command structure:**
```
ffmpeg -f rawvideo -pix_fmt rgb24 -s 1280x720 -r 30 -i pipe:0
       -stream_loop -1 -i <music_file>
       -c:v libx264 -preset veryfast -b:v 4500k -maxrate 4500k -bufsize 9000k
       -c:a aac -b:a 128k
       -f tee "[f=flv]rtmp://live.twitch.tv/app/{key}|[f=flv]rtmp://a.rtmp.youtube.com/live2/{key}"
```

### 4. Overlay System (src/streaming/overlay_manager.py -- rewrite)

- Composites stream frame: dashboard surface + overlays
- LIVE indicator badge (top-left, pulsing red dot)
- Current algorithm + world/stage text overlay
- Training stats ticker (episode, reward, time elapsed)
- Optional branding watermark

### 5. Curriculum Learning (src/training/curriculum.py -- new)

- `CurriculumManager` class tracking per-stage performance
- Advancement threshold: configurable (avg reward > X over last 10 episodes, or flag reached 3 times)
- Stage schedule: ordered list of all 32 stages (1-1 through 8-4)
- Revisitation: every N episodes, randomly sample a previously-completed stage
- Regression detection: if earlier stage performance drops, re-queue it
- Unified checkpoint: saves curriculum state alongside model weights
- Works with all three algorithms (NEAT, PPO, DQN)

### 6. Launcher UI Additions (launcher.py)

- Streaming section: Twitch key, YouTube key, Start Stream toggle, status indicator
- Audio section: Music folder path selector, volume slider, mute checkbox
- Whole Game mode checkbox: auto-selects curriculum learning, shows progress across 32 stages

### 7. Performance Optimizations

- MetricsStore: cap at 10,000 data points with rolling window
- DQN: tighter gradient clipping (1.0), faster epsilon decay option
- Dashboard: cache matplotlib figures, only redraw on new data
- Config validation: schema check on load

## Data Flow for Streaming

```
Training Step
  -> Dashboard.update(frame, metrics)
     -> Render pygame surface (1400x800)
     -> MusicManager.tick() [advance playlist if needed]
     -> OverlayManager.compose(surface) [add overlays]
     -> StreamManager.send_frame(composed_surface)
          -> ffmpeg stdin (raw RGB) -> tee muxer
               -> rtmp://twitch
               -> rtmp://youtube
     -> Recorder.capture(surface) [if recording locally]
```

## New/Modified Files

```
src/
  audio/
    __init__.py                 # NEW
    music_manager.py            # NEW: pygame.mixer music playback
  streaming/
    stream_manager.py           # NEW: ffmpeg RTMP streaming
    overlay_manager.py          # REWRITE: actual overlay compositing
    recording.py                # MODIFY: use ffmpeg for audio+video
  training/
    __init__.py                 # NEW
    curriculum.py               # NEW: curriculum learning manager
  algorithms/
    base_trainer.py             # MODIFY: true pause/resume
    neat/neat_trainer.py        # MODIFY: pause check in loop
    ppo/ppo_trainer.py          # MODIFY: pause check in callback
    dqn/dqn_trainer.py          # MODIFY: pause check in loop
  visualization/
    dashboard.py                # MODIFY: music controls, stream status, perf
assets/
  music/                        # NEW: directory for user music files
    README.txt                  # Instructions for adding music
config/
  streaming_config.yaml         # MODIFY: add stream keys, audio settings
launcher.py                     # MODIFY: streaming + audio UI sections
main.py                         # MODIFY: --stream, --music CLI args
```

## Error Handling

- ffmpeg not installed: clear error message with install instructions, streaming disabled but training continues
- No music files: silent mode, no crash
- Stream key invalid: ffmpeg stderr detection, notify via dashboard overlay
- Network drop during stream: auto-reconnect via ffmpeg `-reconnect` flags
- Stage advancement failure: log warning, stay on current stage

## Decision Log

| Decision | Chosen | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| Streaming | ffmpeg RTMP | OBS WebSocket, Pure Python RTMP | One-click, multi-platform, audio muxing, production-proven |
| Music | pygame.mixer | python-vlc, playsound | Zero new deps, already in project, simple API |
| Whole-game | Curriculum learning | Random sampling, Sequential | Prevents forgetting, measurable progress, structured |
| Pause | Training-thread block | Display-only (current) | True pause required for meaningful control |
