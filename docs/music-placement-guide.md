# Music Placement Guide

Add background music to your AI training sessions and live streams.
Music plays through the dashboard and is piped into live streams
as the audio track.

---

## Quick Start

1. **Create the music folder** (if it doesn't exist):
   ```
   assets/music/
   ```
   This folder is already included in the project. It is the default
   location where the music manager looks for tracks.

2. **Drop your music files** into the folder:
   ```
   assets/music/
   ├── lofi-beats-01.ogg
   ├── chiptune-adventure.mp3
   ├── chill-synth.wav
   └── retro-vibes.flac
   ```

3. **Run training** — music plays automatically:
   ```bash
   python main.py --algorithm dqn --visualize
   ```
   You will see:
   ```
   Loaded 4 music track(s) from C:\Projects\mario-ml\assets\music
   ```

That's it! The music manager scans the folder, shuffles the tracks,
and loops through them during training.

---

## Supported Audio Formats

| Format | Extension | Recommended? | Notes                          |
|--------|-----------|--------------|--------------------------------|
| OGG    | `.ogg`    | Best         | Best pygame compatibility      |
| MP3    | `.mp3`    | Good         | Widely available               |
| WAV    | `.wav`    | OK           | Large file sizes               |
| FLAC   | `.flac`   | OK           | Lossless, larger than OGG/MP3  |

**Recommendation:** Use `.ogg` files for the best compatibility with
pygame's mixer. Most audio converters and tools (Audacity, ffmpeg)
can convert to OGG Vorbis.

**Converting with ffmpeg:**
```bash
ffmpeg -i my-track.mp3 -c:a libvorbis -q:a 5 my-track.ogg
```

---

## Folder Structure

### Default location
```
C:\Projects\mario-ml\assets\music\
```

The music manager scans this directory for any files with a supported
extension. Subdirectories are **not** scanned — place files directly
in the folder.

### Custom location

Use the `--music` flag to point to a different folder:
```bash
python main.py --algorithm dqn --visualize --music "D:\My Music\Training Beats"
```

---

## How It Works

1. On startup, `MusicManager` scans the music directory for files
   matching `.ogg`, `.mp3`, `.wav`, or `.flac`.
2. Found tracks are **shuffled** into a random playlist order.
3. When training starts, the first track begins playing via
   `pygame.mixer.music`.
4. When a track finishes, the next track in the shuffled playlist
   starts automatically (via a pygame `MUSIC_END_EVENT`).
5. After the last track finishes, the playlist loops from the
   beginning (still in the same shuffled order).

---

## Dashboard Keyboard Controls

While the dashboard is open during training:

| Key         | Action                          |
|-------------|---------------------------------|
| **M**       | Mute / unmute music             |
| **Up**      | Increase volume (+5%)           |
| **Down**    | Decrease volume (-5%)           |

The default volume is **50%**. Volume changes are applied immediately.
Muting remembers your previous volume level so unmuting restores it.

---

## Music + Streaming

When streaming to Twitch or YouTube, the **first track** in the
shuffled playlist is used as the audio track for the stream:

```bash
python main.py --algorithm dqn --visualize ^
  --stream-twitch YOUR_KEY ^
  --music assets/music
```

**Important notes:**
- The stream audio comes from the **music file** piped to ffmpeg,
  not from your system's audio output.
- If no music files are found, the stream has **silent audio**.
- The stream loops the first audio file continuously. In-dashboard
  track advancement does not affect the stream audio.

---

## Music + Recording

Local recordings (`--record`) capture **video only** — they do not
include the music audio track. If you need a recording with audio,
use the streaming feature to stream + record via OBS, or combine
the video and audio files afterward:

```bash
ffmpeg -i recordings/training_20260309_143052.mp4 ^
       -i assets/music/lofi-beats-01.ogg ^
       -c:v copy -c:a aac -shortest ^
       output-with-audio.mp4
```

---

## Tips for Choosing Music

- **Instrumental / lo-fi** works best — vocals can be distracting.
- **Keep files under 10 MB each** for faster loading.
- **Royalty-free music** is important if you are streaming publicly.
  Good sources:
  - https://freemusicarchive.org
  - https://incompetech.com
  - https://pixabay.com/music/
  - YouTube Audio Library (in YouTube Studio)
- **Chiptune / retro game music** fits the Mario training aesthetic.
- **3-5 tracks** is plenty for a training session. The playlist loops.

---

## Troubleshooting

### No "Loaded N music track(s)" message at startup
- Check that your music files are in `assets/music/` (or the custom
  path you specified with `--music`).
- Verify the file extensions are `.ogg`, `.mp3`, `.wav`, or `.flac`.
- Make sure the files are actual audio files (not renamed text files).

### Music doesn't play
- Make sure `--visualize` is enabled (music requires the dashboard).
- Music is disabled in `--eval` mode by default.
- Check that pygame's mixer initialized correctly (no error messages
  at startup).

### Music is too loud / too quiet
- Use **Up** and **Down** arrow keys during training to adjust volume.
- Default volume is 50%. Range is 0% to 100%.

### "pygame.error: Unrecognized audio format"
- The file may be corrupted or in an unsupported encoding.
- Convert it to `.ogg` using ffmpeg:
  ```bash
  ffmpeg -i problem-file.mp3 -c:a libvorbis -q:a 5 fixed-file.ogg
  ```

### Music stops and doesn't advance to next track
- This can happen if the dashboard event loop is blocked.
- Ensure training is running (not paused or stuck).
