# Super Mario Bros ML Training

Train AI agents to play Super Mario Bros using three different machine learning algorithms, each with a live visualization dashboard.

## Algorithms

| Algorithm | Type | Library | How It Learns |
|-----------|------|---------|---------------|
| **NEAT** | Neuroevolution | neat-python | Evolves neural network topology through genetic algorithms. Population of networks compete, best reproduce. |
| **PPO** | Policy Gradient RL | stable-baselines3 | Learns a policy (action probabilities) via gradient descent on collected experience. Clips updates for stability. |
| **DQN** | Value-Based RL | PyTorch (custom) | Learns Q-values (expected reward per action) with experience replay and target networks. |

## Live Dashboard

The training dashboard shows game rendering + 4 real-time graphs in a single window:

```
+----------------------------+------------------------------+
|                            |  Reward        Distance      |
|     GAME DISPLAY           |  [graph]       [graph]       |
|     (live Mario gameplay)  |                              |
|                            |  Loss/Cmplx    Actions       |
|                            |  [graph]       [bar chart]   |
+----------------------------+------------------------------+
| Episode: 152 | Reward: 2450 | Best: 3120    Time: 00:15:30|
+---------------------------------------------------------- +
```

## Installation

### Prerequisites
- Windows 10/11
- Python 3.11 (not 3.12+ due to library compatibility)

### Step 1: Install Python 3.11
If you don't have Python 3.11, download the installer:
https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

Install to `C:\Python311` without modifying PATH (to keep your existing Python).

### Step 2: Create Virtual Environment
```bash
C:\Python311\python.exe -m venv C:\Projects\mario-ml\venv
```

### Step 3: Activate and Install Dependencies
```bash
C:\Projects\mario-ml\venv\Scripts\activate
cd C:\Projects\mario-ml
pip install -r requirements.txt
```

## Usage

Always activate the virtual environment first:
```bash
C:\Projects\mario-ml\venv\Scripts\activate
cd C:\Projects\mario-ml
```

### GUI Launcher (Recommended)
```bash
python launcher.py
```

The launcher provides a graphical interface for all training options — algorithm selection, world/stage, streaming keys, music toggle, curriculum learning, and more. No CLI flags to remember.

### Train with Live Dashboard (CLI)
```bash
# NEAT (neuroevolution) - 100 generations
python main.py --algorithm neat --visualize

# PPO (policy optimization) - 1M timesteps
python main.py --algorithm ppo --visualize

# DQN (deep Q-learning) - 5000 episodes
python main.py --algorithm dqn --visualize
```

### Custom Training Duration
```bash
python main.py --algorithm neat --visualize --episodes 200
python main.py --algorithm dqn --visualize --episodes 10000
```

### Different World/Stage
```bash
python main.py --algorithm neat --visualize --world 1 --stage 2
```

### Evaluate a Saved Model
```bash
python main.py --algorithm neat --eval --load models/neat/final_best_genome.pkl --visualize
python main.py --algorithm ppo --eval --load models/ppo/final.zip --visualize
python main.py --algorithm dqn --eval --load models/dqn/final.pt --visualize
```

### Record Training as Video
```bash
python main.py --algorithm neat --visualize --record
```

### Live Stream to Twitch / YouTube
```bash
# Stream to Twitch
python main.py --algorithm dqn --visualize --stream-twitch YOUR_TWITCH_KEY

# Stream to YouTube
python main.py --algorithm ppo --visualize --stream-youtube YOUR_YOUTUBE_KEY

# Simultaneous stream to both platforms
python main.py --algorithm neat --visualize \
    --stream-twitch YOUR_TWITCH_KEY \
    --stream-youtube YOUR_YOUTUBE_KEY
```

Requires **ffmpeg** installed and on PATH (see [Streaming Requirements](#streaming-requirements) below).

### Background Music
```bash
# Play music from the default assets/music directory
python main.py --algorithm neat --visualize --music assets/music
```

Drop `.mp3`, `.ogg`, or `.wav` files into `assets/music/`. Tracks play in shuffled order and auto-advance. Use **M** to mute and **↑/↓** for volume during training.

### Curriculum Learning (Whole Game)
```bash
# Train a single model across all 32 stages (World 1-1 through 8-4)
python main.py --algorithm dqn --visualize --curriculum
```

The curriculum manager tracks performance per stage, advances when the agent consistently scores well, and periodically revisits earlier stages to prevent catastrophic forgetting.

### Train Without Dashboard (Faster)
```bash
python main.py --algorithm dqn
```

## Dashboard Controls

| Key | Action |
|-----|--------|
| **SPACE** | Pause/Resume training (true pause — halts training thread) |
| **S** | Save screenshot |
| **M** | Mute/Unmute background music |
| **↑ / ↓** | Volume up / down |
| **ESC** | Stop training (model auto-saves) |

Closing the window also triggers an auto-save.

## Saved Models

Models are automatically saved during training and on exit:

| Algorithm | Format | Location |
|-----------|--------|----------|
| NEAT | `.pkl` (pickle) | `models/neat/` |
| PPO | `.zip` (stable-baselines3) | `models/ppo/` |
| DQN | `.pt` (PyTorch) | `models/dqn/` |

Each algorithm also saves `metadata.json` with training stats (episodes, best reward, elapsed time).

Checkpoints are auto-saved periodically (configurable in config files).

**Ctrl+C** during training triggers a graceful save before exit.

## Configuration

Hyperparameters are stored in YAML/TXT config files:

| File | Algorithm | Key Parameters |
|------|-----------|----------------|
| `config/neat_config.txt` | NEAT | Population size, mutation rates, species threshold |
| `config/ppo_config.yaml` | PPO | Learning rate, clip range, n_steps, gamma |
| `config/dqn_config.yaml` | DQN | Learning rate, epsilon decay, buffer size, batch size |

## Project Architecture

```
mario-ml/
  main.py                    # CLI entry point
  launcher.py                # GUI launcher (tkinter)
  config/                    # Algorithm hyperparameters
  src/
    environment/             # Game environment + preprocessing
      mario_env.py           # Environment factory
      wrappers.py            # Frame skip, grayscale, resize, stack
    algorithms/              # ML implementations
      base_trainer.py        # Shared interface + true pause/resume
      neat/neat_trainer.py   # NEAT evolution
      ppo/ppo_trainer.py     # PPO via stable-baselines3
      dqn/
        dqn_trainer.py       # Training loop
        dqn_network.py       # CNN Q-network
        replay_buffer.py     # Experience replay
    visualization/           # Live dashboard
      dashboard.py           # Main pygame window
      game_renderer.py       # Game frame display
      graph_panel.py         # Real-time matplotlib graphs
      metrics_tracker.py     # Metric collection + JSON export
    streaming/               # Recording + live streaming
      recording.py           # MP4 video recording
      stream_manager.py      # ffmpeg RTMP streaming (Twitch/YouTube)
      overlay_manager.py     # Stream overlay compositing (LIVE badge, stats)
    audio/                   # Background music
      music_manager.py       # Playlist, volume, mute via pygame.mixer
    training/                # Training utilities
      curriculum.py          # Curriculum learning across all 32 stages
  assets/music/              # Drop .mp3/.ogg/.wav files here
  models/                    # Saved model checkpoints
  logs/                      # Training logs
  recordings/                # Recorded videos
  tests/                     # pytest test suite
```

## Streaming Requirements

For built-in live streaming to Twitch/YouTube, install **ffmpeg**:

- **Windows (winget):** `winget install ffmpeg`
- **Windows (manual):** Download from https://ffmpeg.org/download.html and add to PATH
- **Verify:** `ffmpeg -version`

The stream outputs at 720p 30fps with the dashboard plus LIVE badge, algorithm info overlay, and stats ticker.

### Alternative: OBS Studio

You can also capture the dashboard window with OBS Studio:

1. Open OBS Studio
2. Add Source > Window Capture
3. Select "Mario ML Dashboard" from the window list
4. The 1400x800 window captures cleanly at any stream resolution

## Troubleshooting

**"ModuleNotFoundError: No module named 'gym_super_mario_bros'"**
Make sure the virtual environment is activated:
```bash
C:\Projects\mario-ml\venv\Scripts\activate
```

**"Python version not supported"**
This project requires Python 3.11. Check with:
```bash
python --version  # Should show 3.11.x
```

**Dashboard window is black**
The game needs a few seconds to initialize. If it stays black, try restarting.

**Training is slow**
- Run without `--visualize` for 2-5x faster training
- DQN benefits greatly from GPU. Install CUDA-enabled PyTorch for GPU support
- Reduce population size in `neat_config.txt` for faster NEAT iterations

**Model won't load**
Ensure the path matches the algorithm type (`.pkl` for NEAT, `.zip` for PPO, `.pt` for DQN).

## Adding New Algorithms

To add a new algorithm:

1. Create `src/algorithms/myalgo/myalgo_trainer.py`
2. Subclass `BaseTrainer` and implement `train()`, `evaluate()`, `save_checkpoint()`, `load_checkpoint()`
3. Add a config file in `config/`
4. Add the algorithm choice to `main.py`'s argument parser and trainer creation

The `BaseTrainer` class provides auto-checkpoint, graceful shutdown, and dashboard integration for free.
