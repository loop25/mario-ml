# Streaming Setup Guide

Stream your AI training sessions live to Twitch and/or YouTube.
The dashboard is captured in real time, composited with overlays
(LIVE badge, algorithm info, stats ticker), and streamed via ffmpeg.

---

## Prerequisites

### 1. Install ffmpeg

ffmpeg is the engine that encodes and sends your video stream.

**Windows (recommended — winget):**
```
winget install ffmpeg
```

**Windows (manual):**
1. Download from https://ffmpeg.org/download.html (select a Windows build).
2. Extract the archive (e.g. to `C:\ffmpeg`).
3. Add `C:\ffmpeg\bin` to your system PATH:
   - Open **Start** > search **"Environment Variables"**.
   - Under **System variables**, find `Path`, click **Edit**.
   - Click **New** and paste `C:\ffmpeg\bin`.
   - Click **OK** on all dialogs.
4. Open a **new** terminal and verify:
   ```
   ffmpeg -version
   ```
   You should see version info (e.g. `ffmpeg version 7.x ...`).

### 2. Get your stream key(s)

**Twitch:**
1. Go to https://dashboard.twitch.tv/settings/stream
2. Click **Copy** next to "Primary Stream Key".
3. Save it somewhere safe. Never share this key publicly.

**YouTube:**
1. Go to https://studio.youtube.com
2. Click **Go Live** (or **Create** > **Go Live**).
3. Select **Stream** tab.
4. Under **Stream settings**, copy the **Stream key**.

---

## Quick Start

### Stream to Twitch only
```bash
python main.py --algorithm dqn --visualize --stream-twitch YOUR_TWITCH_KEY
```

### Stream to YouTube only
```bash
python main.py --algorithm ppo --visualize --stream-youtube YOUR_YOUTUBE_KEY
```

### Stream to both simultaneously
```bash
python main.py --algorithm neat --visualize ^
  --stream-twitch YOUR_TWITCH_KEY ^
  --stream-youtube YOUR_YOUTUBE_KEY
```
(Use `^` for line continuation on Windows CMD, or `\` in bash/PowerShell.)

### Stream with background music
```bash
python main.py --algorithm dqn --visualize ^
  --stream-twitch YOUR_TWITCH_KEY ^
  --music assets/music
```
The first music file is piped into the stream as audio.
See the [Music Placement Guide](music-placement-guide.md) for details.

### Stream + record locally
```bash
python main.py --algorithm neat --visualize ^
  --stream-twitch YOUR_TWITCH_KEY ^
  --record
```
The recording is saved to `recordings/training_YYYYMMDD_HHMMSS.mp4`.

---

## How It Works (Step by Step)

1. **You run the command** with `--stream-twitch` and/or `--stream-youtube`.
2. `main.py` detects the streaming flags and auto-enables `--visualize`.
3. A `StreamManager` is created with your key(s) and stream settings.
4. An `OverlayManager` is created for compositing overlays on the stream.
5. `StreamManager.start()` launches an **ffmpeg subprocess** in the background.
6. Training begins. Each frame of the dashboard is:
   - Rendered by pygame (the dashboard window you see).
   - Captured as a raw RGB numpy array.
   - Passed through `OverlayManager.compose()` which adds:
     - A pulsing red **LIVE** badge (top-left corner).
     - Algorithm name and stage info (top-right corner).
     - Stats ticker bar at the bottom (episode, reward, elapsed time).
   - Piped to ffmpeg via stdin as raw bytes.
7. ffmpeg encodes the frames to H.264 video and sends via RTMP to your platform(s).
8. When training ends (or you press Ctrl+C), `StreamManager.stop()` gracefully
   terminates ffmpeg and prints stream statistics.

---

## Stream Settings

The default settings are optimized for Twitch's recommended configuration:

| Setting        | Default    | Notes                              |
|----------------|------------|------------------------------------|
| Resolution     | 1280x720   | 720p — good balance of quality/CPU |
| Frame rate     | 30 fps     | Smooth for training visualization  |
| Video bitrate  | 4500 kbps  | Twitch recommended for 720p30     |
| Video codec    | H.264      | libx264, veryfast preset           |
| Audio codec    | AAC        | 128 kbps, 44100 Hz                |
| Audio source   | Music file | Falls back to silence if no music  |

These are configured in `config/streaming_config.yaml`:
```yaml
streaming:
  resolution: [1280, 720]
  fps: 30
  video_bitrate: "4500k"
```

---

## Troubleshooting

### "ffmpeg: command not found" / "'ffmpeg' is not recognized"
- ffmpeg is not installed or not on your PATH.
- Follow the installation steps above.
- Make sure you opened a **new** terminal after editing PATH.

### "[Stream] Failed to start"
- Check that your stream key is correct (no extra spaces).
- Verify ffmpeg works: `ffmpeg -version`.
- Check your internet connection.
- Look at the error message printed after the failure.

### Stream is laggy or dropping frames
- Close other CPU-intensive programs.
- Lower the video bitrate in `config/streaming_config.yaml`.
- Try reducing `--num-envs` to 1.
- Make sure you are on a wired connection (not WiFi).

### No audio on stream
- Place music files in `assets/music/` (see Music Placement Guide).
- Pass `--music assets/music` on the command line.
- If no music files are found, the stream uses silent audio.

### Stream starts but platforms show "offline"
- It can take 10-30 seconds for Twitch/YouTube to pick up the stream.
- Verify your stream key is correct.
- Check that your platform account is in good standing.

---

## Recording Without Streaming

If you just want to save a local recording (no live stream):

```bash
python main.py --algorithm dqn --visualize --record
```

Recordings are saved as MP4 files in the `recordings/` directory.
They capture the full dashboard at 1400x800 resolution, 30 fps.

---

## Dashboard Keyboard Controls During Streaming

| Key         | Action                                      |
|-------------|---------------------------------------------|
| **M**       | Toggle music mute/unmute                    |
| **Up**      | Increase music volume                       |
| **Down**    | Decrease music volume                       |
| **Space**   | Pause/resume training                       |
| **Esc** or close window | Stop training (saves model first) |

---

## Advanced: Using OBS Instead

If you prefer more control over your stream layout, you can use OBS Studio
to capture the dashboard window instead of the built-in streaming:

1. Run training with `--visualize` (no streaming flags).
2. In OBS, add a **Window Capture** source pointing to the dashboard window.
3. Add your own overlays, scenes, and transitions in OBS.
4. Stream from OBS to your platform of choice.

The built-in streaming is designed for zero-configuration simplicity.
OBS gives you full control over layout and multi-scene setups.
