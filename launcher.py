"""
Super Mario Bros ML — Launcher GUI.

A simple, clean graphical launcher for the Mario ML training system.
Instead of typing terminal commands with --flags, just click buttons.

This launcher builds the appropriate command and runs main.py as a
subprocess, so the training/pygame dashboard runs independently.

Features:
    - 2-column layout: all settings visible at once (no tabs)
    - Persistent settings: saves your preferences across sessions
    - Resizable/maximizable window
    - Graduated shutdown: CTRL_C_EVENT → terminate → kill

Usage:
    1. Activate the virtual environment:
       C:\\Projects\\mario-ml\\venv\\Scripts\\activate
    2. Run the launcher:
       python launcher.py
    3. Pick your game and algorithm, then click START.

Requirements:
    - Python 3.11 with tkinter (included by default)
    - All project dependencies installed in the venv
"""

import os
import re
import shutil
import sys
import json
import signal
import subprocess
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from games.registry import GameRegistry

# ---------------------------------------------------------------------------
# Path setup — figure out where the project lives
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "main.py")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "recordings")
ROMS_DIR = os.path.join(PROJECT_ROOT, "roms")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
SETTINGS_PATH = os.path.join(CONFIG_DIR, "launcher_settings.json")

def _find_venv_python() -> str:
    """Return the path to the venv's Python, falling back to sys.executable.

    This lets the launcher work correctly even when launched with the
    system Python (e.g. by double-clicking).  We look for the venv
    directory that lives alongside this script.
    """
    if sys.platform == "win32":
        venv_python = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
    if os.path.isfile(venv_python):
        return venv_python
    return sys.executable

VENV_PYTHON = _find_venv_python()

# ---------------------------------------------------------------------------
# Color palette — dark theme matching the pygame dashboard aesthetic
# ---------------------------------------------------------------------------
BG_DARK = "#1a1a2e"       # Main background
BG_MEDIUM = "#16213e"     # Section backgrounds
BG_LIGHT = "#0f3460"      # Input fields / hover
TEXT_PRIMARY = "#e0e0e0"   # Main text
TEXT_DIM = "#8888aa"       # Subtle labels
ACCENT_GREEN = "#00d474"   # NEAT color / success
ACCENT_BLUE = "#4da6ff"    # PPO color
ACCENT_ORANGE = "#ff8c42"  # DQN color
ACCENT_RED = "#ff4757"     # Stop button
BORDER_COLOR = "#2a2a4a"   # Subtle borders

# Training duration presets (preset_name -> episode count)
TRAINING_PRESETS = {
    "Quick Demo": "100",
    "Standard": "5000",
    "Deep Training": "25000",
    "Overnight": "50000",
    "Custom": "",
}
PRESET_NAMES = list(TRAINING_PRESETS.keys())

# Algorithm-specific accent colors, file extensions, and user-facing info
ALGO_INFO = {
    "neat": {
        "color": ACCENT_GREEN,
        "duration_label": "Generations",
        "duration_default": "100",
        "file_ext": [("NEAT Genome", "*.pkl"), ("All Files", "*.*")],
        "model_subdir": "neat",
        "label": "NEAT",
        "short_desc": "Evolves neural networks",
        "tooltip": (
            "NEAT (NeuroEvolution of Augmenting Topologies)\n"
            "Evolves neural network structure through genetic algorithms.\n"
            "Great for: visual learning, watching brains evolve live.\n"
            "Speed: Fast per generation, needs many generations.\n"
            "Best for: action games (Mario, Sonic) with small observation."
        ),
        "badge": None,
    },
    "ppo": {
        "color": ACCENT_BLUE,
        "duration_label": "Timesteps (×1000)",
        "duration_default": "1000",
        "file_ext": [("SB3 Model", "*.zip"), ("All Files", "*.*")],
        "model_subdir": "ppo",
        "label": "PPO",
        "short_desc": "Best all-around agent",
        "tooltip": (
            "PPO (Proximal Policy Optimization)\n"
            "State-of-the-art policy gradient method.\n"
            "Great for: reliable training, GPU acceleration.\n"
            "Speed: Moderate — steady improvement.\n"
            "Best for: any game, most versatile single-game agent."
        ),
        "badge": "RECOMMENDED",
    },
    "dqn": {
        "color": ACCENT_ORANGE,
        "duration_label": "Episodes",
        "duration_default": "5000",
        "file_ext": [("PyTorch Model", "*.pt"), ("All Files", "*.*")],
        "model_subdir": "dqn",
        "label": "DQN",
        "short_desc": "Classic deep Q-learning",
        "tooltip": (
            "DQN (Deep Q-Network)\n"
            "Learns action values from experience replay.\n"
            "Great for: discrete action games, stable learning.\n"
            "Speed: Slow start, then rapid improvement.\n"
            "Best for: games with clear goals (Mario, Snake)."
        ),
        "badge": None,
    },
    "a2c": {
        "color": ACCENT_GREEN,
        "duration_label": "Timesteps (×1000)",
        "duration_default": "1000",
        "file_ext": [("SB3 Model", "*.zip"), ("All Files", "*.*")],
        "model_subdir": "a2c",
        "label": "A2C",
        "short_desc": "Fast parallel training",
        "tooltip": (
            "A2C (Advantage Actor-Critic)\n"
            "Combines value estimation with policy learning.\n"
            "Great for: fast training with multiple envs.\n"
            "Speed: Fast — synchronous parallel updates.\n"
            "Best for: action games with GPU available."
        ),
        "badge": None,
    },
    "rainbow": {
        "color": "#9B59B6",
        "duration_label": "Episodes",
        "duration_default": "5000",
        "file_ext": [("PyTorch Model", "*.pt"), ("All Files", "*.*")],
        "model_subdir": "rainbow",
        "label": "Rainbow",
        "short_desc": "6 DQN improvements combined",
        "tooltip": (
            "Rainbow DQN (6 improvements in one)\n"
            "Combines: Double DQN, PER, Dueling, Noisy Nets, C51, Multi-step.\n"
            "Great for: maximum single-game performance.\n"
            "Speed: Slow but achieves highest scores.\n"
            "Best for: competitive play, beating high scores."
        ),
        "badge": "BEST SCORE",
    },
    "dt": {
        "color": "#E67E22",
        "duration_label": "Steps (×1000)",
        "duration_default": "100",
        "file_ext": [("PyTorch Model", "*.pt"), ("All Files", "*.*")],
        "model_subdir": "generalist",
        "label": "DT",
        "short_desc": "Multi-game generalist AI",
        "tooltip": (
            "Decision Transformer (Generalist Agent)\n"
            "A GPT-2 style transformer that learns from all games at once.\n"
            "Great for: building one AI that plays every game.\n"
            "Speed: Trains offline on collected experience.\n"
            "Requires: Pre-collected data from other agents.\n"
            "This is the ULTIMATE AGENT — trains on all games combined!"
        ),
        "badge": "ULTIMATE",
    },
}

# Default settings for persistence
DEFAULT_SETTINGS = {
    "algorithm": "neat",
    "game_index": 0,
    "device": "auto",
    "duration": "5000",
    "duration_preset": "Standard",
    "visualize": True,
    "record": False,
    "eval_mode": False,
    "next_stage": False,
    "curriculum": False,
    "num_envs": "1",
    "world": "1",
    "stage": "1",
    "model_path": "",
    "stream_enabled": False,
    "twitch_key": "",
    "youtube_key": "",
    "music": True,
    "window_geometry": "",
    "extras_expanded": False,
    "opponent_type": "auto",
    "opponent_depth": "3",
    "opponent_model": "",
}


class MarioLauncher:
    """
    Main launcher window.

    Creates a tkinter GUI with a 2-column layout for algorithm selection,
    game options, training settings, and a START/STOP button.
    Spawns main.py as a subprocess when START is clicked.
    """

    def __init__(self):
        # ---------------------------------------------------------------
        # Window setup — resizable with minimum size
        # ---------------------------------------------------------------
        self.root = tk.Tk()
        self.root.title("Game AI Training Studio")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)
        self.root.minsize(800, 550)

        # Configure ttk styles for dark theme
        style = ttk.Style()
        style.theme_use('clam')  # 'clam' is the most customizable ttk theme
        style.configure('TCombobox',
            fieldbackground=BG_LIGHT,
            background=BG_MEDIUM,
            foreground=TEXT_PRIMARY,
            arrowcolor=TEXT_PRIMARY,
            selectbackground=BG_LIGHT,
            selectforeground=TEXT_PRIMARY,
        )
        style.map('TCombobox',
            fieldbackground=[('readonly', BG_LIGHT)],
            selectbackground=[('readonly', BG_LIGHT)],
            selectforeground=[('readonly', TEXT_PRIMARY)],
            foreground=[('readonly', TEXT_PRIMARY)],
        )
        # Style the dropdown list
        self.root.option_add('*TCombobox*Listbox.background', BG_MEDIUM)
        self.root.option_add('*TCombobox*Listbox.foreground', TEXT_PRIMARY)
        self.root.option_add('*TCombobox*Listbox.selectBackground', BG_LIGHT)
        self.root.option_add('*TCombobox*Listbox.selectForeground', TEXT_PRIMARY)

        # Track the training subprocess (None when idle)
        self.process = None

        # Debounce timer for settings save
        self._save_timer = None

        # State variables
        self.selected_algo = tk.StringVar(value="neat")
        self.world_var = tk.StringVar(value="1")
        self.stage_var = tk.StringVar(value="1")
        self.duration_var = tk.StringVar(value="5000")
        self.preset_var = tk.StringVar(value="Standard")
        self.visualize_var = tk.BooleanVar(value=True)
        self.record_var = tk.BooleanVar(value=False)
        self.eval_var = tk.BooleanVar(value=False)
        self.next_stage_var = tk.BooleanVar(value=False)
        self.curriculum_var = tk.BooleanVar(value=False)
        self.num_envs_var = tk.StringVar(value="1")
        self.model_path_var = tk.StringVar(value="")

        self.device_var = tk.StringVar(value="auto")

        # Opponent selection (for board games)
        self.opponent_var = tk.StringVar(value="auto")
        self.opponent_depth_var = tk.StringVar(value="3")
        self.opponent_model_var = tk.StringVar(value="")

        # Streaming
        self.twitch_key_var = tk.StringVar()
        self.youtube_key_var = tk.StringVar()
        self.stream_var = tk.BooleanVar(value=False)

        # Music
        self.music_var = tk.BooleanVar(value=True)
        self.music_dir_var = tk.StringVar(
            value=os.path.join(PROJECT_ROOT, 'assets', 'music')
        )

        # Extras panel state
        self._extras_expanded = False

        # Discover games
        self.game_registry = GameRegistry()
        self.game_registry.discover()
        self.available_games = self.game_registry.list_games()
        self.game_var = tk.StringVar(value='mario')

        # Build UI — 2-column grid layout
        self._build_header()
        self._build_main_layout()
        self._build_start_area()

        # Load saved settings (overrides defaults above)
        self._load_settings()

        # Apply game-dependent UI visibility
        self._update_world_stage_visibility()
        self._update_opponent_visibility()

        # Set initial window position/size
        self.root.update_idletasks()
        saved_geo = getattr(self, '_saved_geometry', '')
        if saved_geo:
            try:
                self.root.geometry(saved_geo)
            except tk.TclError:
                self._center_window()
        else:
            self._center_window()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Wire up auto-save on settings change
        self._wire_save_triggers()

    def _center_window(self):
        """Center the window on screen with a default size."""
        self.root.geometry("960x680")
        self.root.update_idletasks()
        w = 960
        h = 680
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    # ===================================================================
    # Settings Persistence
    # ===================================================================

    def _load_settings(self):
        """Load saved settings from JSON, merging with defaults."""
        if not os.path.isfile(SETTINGS_PATH):
            return

        try:
            with open(SETTINGS_PATH, 'r') as f:
                saved = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        settings = {**DEFAULT_SETTINGS, **saved}

        # Apply saved values to tk variables
        self.selected_algo.set(settings['algorithm'])
        self.device_var.set(settings['device'])
        self.duration_var.set(settings['duration'])
        self.preset_var.set(settings.get('duration_preset', 'Standard'))
        self._on_preset_changed()
        self.visualize_var.set(settings['visualize'])
        self.record_var.set(settings['record'])
        self.eval_var.set(settings['eval_mode'])
        self.next_stage_var.set(settings['next_stage'])
        self.curriculum_var.set(settings['curriculum'])
        self.num_envs_var.set(settings['num_envs'])
        self.world_var.set(settings['world'])
        self.stage_var.set(settings['stage'])
        self.model_path_var.set(settings['model_path'])
        self.stream_var.set(settings['stream_enabled'])
        self.twitch_key_var.set(settings['twitch_key'])
        self.youtube_key_var.set(settings['youtube_key'])
        self.music_var.set(settings['music'])
        self.opponent_var.set(settings.get('opponent_type', 'auto'))
        self.opponent_depth_var.set(settings.get('opponent_depth', '3'))
        self.opponent_model_var.set(settings.get('opponent_model', ''))

        # Restore game selection
        game_idx = settings.get('game_index', 0)
        game_names = [f"{g.name} ({g.game_id})" for g in self.available_games]
        if game_names and 0 <= game_idx < len(game_names):
            self.game_combo.set(game_names[game_idx])

        # Restore extras panel state
        if settings.get('extras_expanded', False):
            self._extras_expanded = True
            self._toggle_extras(save=False)

        # Restore window geometry
        self._saved_geometry = settings.get('window_geometry', '')

        # Update UI to reflect loaded settings
        self._update_algo_buttons()
        self._update_algo_desc()
        self._update_duration_label()
        self._update_model_display()
        self._update_start_button_text()
        self._rebuild_game_options()

    def _save_settings(self):
        """Save current settings to JSON."""
        # Figure out game index
        game_text = self.game_combo.get()
        game_names = [f"{g.name} ({g.game_id})" for g in self.available_games]
        game_idx = game_names.index(game_text) if game_text in game_names else 0

        settings = {
            'algorithm': self.selected_algo.get(),
            'game_index': game_idx,
            'device': self.device_var.get(),
            'duration': self.duration_var.get(),
            'duration_preset': self.preset_var.get(),
            'visualize': self.visualize_var.get(),
            'record': self.record_var.get(),
            'eval_mode': self.eval_var.get(),
            'next_stage': self.next_stage_var.get(),
            'curriculum': self.curriculum_var.get(),
            'num_envs': self.num_envs_var.get(),
            'world': self.world_var.get(),
            'stage': self.stage_var.get(),
            'model_path': self.model_path_var.get(),
            'stream_enabled': self.stream_var.get(),
            'twitch_key': self.twitch_key_var.get(),
            'youtube_key': self.youtube_key_var.get(),
            'music': self.music_var.get(),
            'opponent_type': self.opponent_var.get(),
            'opponent_depth': self.opponent_depth_var.get(),
            'opponent_model': self.opponent_model_var.get(),
            'window_geometry': self.root.geometry(),
            'extras_expanded': self._extras_expanded,
        }

        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            # Atomic write via temp file
            tmp_path = SETTINGS_PATH + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(settings, f, indent=2)
            # Replace atomically (Windows: os.replace is atomic)
            os.replace(tmp_path, SETTINGS_PATH)
        except OSError as e:
            print(f'[Settings] Failed to save: {e}')

    def _schedule_save(self, *args):
        """Debounced save — waits 300ms after last change before writing."""
        if self._save_timer is not None:
            self.root.after_cancel(self._save_timer)
        self._save_timer = self.root.after(300, self._save_settings)

    def _wire_save_triggers(self):
        """Register trace callbacks on all tk variables for auto-save."""
        for var in [
            self.selected_algo, self.world_var, self.stage_var,
            self.duration_var, self.preset_var, self.visualize_var, self.record_var,
            self.eval_var, self.next_stage_var, self.curriculum_var,
            self.num_envs_var, self.model_path_var, self.device_var,
            self.stream_var, self.twitch_key_var, self.youtube_key_var,
            self.music_var, self.game_var,
        ]:
            var.trace_add('write', self._schedule_save)

    # ===================================================================
    # GUI Building Methods — 2-column grid layout
    # ===================================================================

    def _build_header(self):
        """Compact header: title left, GPU status badge right."""
        header = tk.Frame(self.root, bg=BG_MEDIUM, padx=20, pady=10)
        header.pack(fill="x")

        # Left: title + subtitle
        left = tk.Frame(header, bg=BG_MEDIUM)
        left.pack(side="left", fill="x", expand=True)

        tk.Label(
            left,
            text="Game AI Training Studio",
            font=("Segoe UI", 15, "bold"),
            fg=ACCENT_GREEN,
            bg=BG_MEDIUM,
        ).pack(anchor="w")

        tk.Label(
            left,
            text="Train AI agents to play games — no coding required",
            font=("Segoe UI", 9),
            fg=TEXT_DIM,
            bg=BG_MEDIUM,
        ).pack(anchor="w")

        # Right: GPU status badge
        device_text, device_color = self._detect_device_label()
        self.device_label = tk.Label(
            header,
            text=device_text,
            font=("Segoe UI", 9, "bold"),
            fg=device_color,
            bg=BG_MEDIUM,
            padx=8,
            pady=4,
        )
        self.device_label.pack(side="right", anchor="e")

        # Session count badge (updated dynamically)
        self.session_badge = tk.Label(
            header,
            text="",
            font=("Segoe UI", 8, "bold"),
            fg=BG_DARK, bg=ACCENT_BLUE,
            padx=6, pady=2,
        )
        # Hidden by default, shown when sessions are pending
        self._update_session_badge()

    def _update_session_badge(self):
        """Show/hide session count badge in header."""
        try:
            if hasattr(self, '_calendar_store') and self._calendar_store:
                pending = len(self._calendar_store.get_pending_sessions())
                if pending > 0:
                    self.session_badge.config(text=f"{pending} scheduled")
                    self.session_badge.pack(side="right", padx=(0, 8))
                    return
        except Exception:
            pass
        self.session_badge.pack_forget()

    def _detect_device_label(self):
        """Detect available compute device and return (label_text, color)."""
        try:
            from src.algorithms.device import get_device_info
            info = get_device_info()
            if info['cuda_available']:
                gpu = info.get('gpu_name', 'GPU')
                vram = info.get('vram_gb', '?')
                return f"GPU: {gpu} ({vram}GB VRAM)", ACCENT_GREEN
            if info['mps_available']:
                return "GPU: Apple Silicon (MPS)", ACCENT_GREEN
            return "Device: CPU (no GPU detected)", "#ffaa00"
        except Exception:
            return "Device: CPU", TEXT_DIM

    def _build_main_layout(self):
        """Build the 2-column grid: left (game/algo/duration) + right (options/compute/model)."""
        # Main content frame that stretches with the window
        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=0, pady=0)
        main.columnconfigure(0, weight=1, uniform="col")
        main.columnconfigure(1, weight=1, uniform="col")
        main.rowconfigure(0, weight=1)

        # Left column
        left = tk.Frame(main, bg=BG_DARK, padx=20, pady=10)
        left.grid(row=0, column=0, sticky="nsew")
        self._build_left_column(left)

        # Vertical separator
        tk.Frame(main, bg=BORDER_COLOR, width=1).grid(
            row=0, column=0, sticky="nse", padx=0,
        )

        # Right column
        right = tk.Frame(main, bg=BG_DARK, padx=20, pady=10)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_right_column(right)

        # Extras section (collapsible, spans both columns)
        self._build_extras_section(self.root)

        # Schedule section (collapsible, spans both columns)
        self._build_schedule_section(self.root)

    def _build_left_column(self, parent):
        """Left column: GAME selector + ALGORITHM picker + DURATION."""

        # ── Game selector ──────────────────────────────────────────────
        self._section_label(parent, "GAME")

        game_names = [f"{g.name} ({g.game_id})" for g in self.available_games]
        if not game_names:
            game_names = ["Super Mario Bros (mario)"]

        self.game_combo = ttk.Combobox(
            parent, textvariable=self.game_var,
            values=game_names, state="readonly",
            font=("Segoe UI", 11),
        )
        self.game_combo.pack(fill="x", pady=(4, 0))
        self.game_combo.set(game_names[0])
        self.game_combo.bind("<<ComboboxSelected>>", self._on_game_changed)

        # Dynamic game-specific options
        self.game_opts_frame = tk.Frame(parent, bg=BG_DARK)
        self.game_opts_frame.pack(fill="x", pady=(5, 0))
        self.game_opt_widgets = {}
        self._rebuild_game_options()

        # World / Stage row (for Mario / multi-level games — hidden for others)
        self.ws_frame = tk.Frame(parent, bg=BG_DARK)
        self.ws_frame.pack(fill="x", pady=(8, 0))

        tk.Label(self.ws_frame, text="World", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(side="left")
        ttk.Combobox(
            self.ws_frame, textvariable=self.world_var,
            values=[str(i) for i in range(1, 9)],
            state="readonly", width=4,
        ).pack(side="left", padx=(4, 12))

        tk.Label(self.ws_frame, text="Stage", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(side="left")
        ttk.Combobox(
            self.ws_frame, textvariable=self.stage_var,
            values=[str(i) for i in range(1, 5)],
            state="readonly", width=4,
        ).pack(side="left", padx=(4, 0))

        # ── Separator ──────────────────────────────────────────────────
        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(
            fill="x", pady=(12, 8),
        )

        # ── Algorithm selector ─────────────────────────────────────────
        hdr = self._section_label(parent, "ALGORITHM")
        tk.Label(
            hdr, text="(hover for details)",
            font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(side="left", padx=(8, 0))

        # Top row: 5 single-game algorithms
        btn_frame = tk.Frame(parent, bg=BG_DARK)
        btn_frame.pack(fill="x", pady=(5, 0))

        self.algo_buttons = {}
        for algo in ["neat", "ppo", "dqn", "a2c", "rainbow"]:
            info = ALGO_INFO[algo]
            col = tk.Frame(btn_frame, bg=BG_DARK)
            col.pack(side="left", expand=True, fill="x", padx=2)

            btn = tk.Button(
                col,
                text=info["label"],
                font=("Segoe UI", 11, "bold"),
                cursor="hand2", relief="flat", bd=0,
                command=lambda a=algo: self._select_algorithm(a),
            )
            btn.pack(fill="x")
            self.algo_buttons[algo] = btn

            if info.get("badge"):
                tk.Label(
                    col, text=info["badge"],
                    font=("Segoe UI", 7, "bold"),
                    fg=info["color"], bg=BG_DARK,
                ).pack()

            self._bind_tooltip(btn, info["tooltip"])

        # Bottom row: DT generalist (special prominence)
        dt_frame = tk.Frame(parent, bg=BG_DARK)
        dt_frame.pack(fill="x", pady=(6, 0))

        dt_info = ALGO_INFO["dt"]
        dt_btn = tk.Button(
            dt_frame,
            text="DT — Decision Transformer (Generalist Agent)",
            font=("Segoe UI", 10, "bold"),
            cursor="hand2", relief="flat", bd=0,
            command=lambda: self._select_algorithm("dt"),
        )
        dt_btn.pack(fill="x")
        self.algo_buttons["dt"] = dt_btn
        tk.Label(
            dt_frame,
            text="ULTIMATE AGENT — One AI that plays ALL games",
            font=("Segoe UI", 8, "bold"),
            fg=dt_info["color"], bg=BG_DARK,
        ).pack()
        self._bind_tooltip(dt_btn, dt_info["tooltip"])

        self.algo_desc_label = tk.Label(
            parent, text="",
            font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_DARK,
            wraplength=400, justify="left",
        )
        self.algo_desc_label.pack(anchor="w", pady=(4, 0))
        self._update_algo_desc()
        self._update_algo_buttons()

        # ── Separator ──────────────────────────────────────────────────
        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(
            fill="x", pady=(10, 8),
        )

        # ── Duration entry ─────────────────────────────────────────────
        self.duration_label = self._section_label(parent, "TRAINING DURATION")

        dur_row = tk.Frame(parent, bg=BG_DARK)
        dur_row.pack(fill="x", pady=(3, 0))

        self.preset_combo = ttk.Combobox(
            dur_row, textvariable=self.preset_var,
            values=PRESET_NAMES, state="readonly",
            font=("Segoe UI", 11), width=15,
        )
        self.preset_combo.pack(side="left")
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_changed)

        self.duration_entry = tk.Entry(
            dur_row, textvariable=self.duration_var,
            font=("Segoe UI", 11), bg=BG_LIGHT, fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY, relief="flat", width=10,
        )
        self.duration_entry.pack(side="left", padx=(8, 0))

        self.duration_unit_label = tk.Label(
            dur_row, text="episodes",
            font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_DARK,
        )
        self.duration_unit_label.pack(side="left", padx=(6, 0))

        # Apply initial preset state
        self._on_preset_changed()
        self._update_duration_label()

    def _build_right_column(self, parent):
        """Right column: TRAINING OPTIONS + COMPUTE + LOAD MODEL."""

        # ── Training options ───────────────────────────────────────────
        self._section_label(parent, "TRAINING OPTIONS")

        _cb = dict(
            font=("Segoe UI", 10), fg=TEXT_PRIMARY, bg=BG_DARK,
            selectcolor=BG_MEDIUM, activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY, cursor="hand2",
        )
        tk.Checkbutton(
            parent, text="Show Live Dashboard",
            variable=self.visualize_var, **_cb,
        ).pack(anchor="w", pady=(4, 0))
        tk.Checkbutton(
            parent, text="Record Video",
            variable=self.record_var, **_cb,
        ).pack(anchor="w")
        tk.Checkbutton(
            parent, text="Evaluation Mode  (watch AI play, no training)",
            variable=self.eval_var, command=self._on_eval_toggle, **_cb,
        ).pack(anchor="w")
        tk.Checkbutton(
            parent, text="Auto-Advance Stages",
            variable=self.next_stage_var, **_cb,
        ).pack(anchor="w")
        tk.Checkbutton(
            parent, text="Whole Game  (curriculum learning)",
            variable=self.curriculum_var, **_cb,
        ).pack(anchor="w")

        # ── Opponent (board games) ───────────────────────────────────
        self.opponent_frame = tk.Frame(parent, bg=BG_DARK)
        # Initially hidden — shown only for board games via _update_opponent_visibility
        self._section_label(self.opponent_frame, "OPPONENT")

        opp_row = tk.Frame(self.opponent_frame, bg=BG_DARK)
        opp_row.pack(fill="x", pady=(4, 0))

        tk.Label(opp_row, text="Mode", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(side="left")
        self.opponent_combo = ttk.Combobox(
            opp_row, textvariable=self.opponent_var,
            values=["auto", "random", "minimax-easy", "minimax-medium",
                    "minimax-hard", "auto-difficulty", "model", "human", "human-vs-human"],
            state="readonly", width=14,
        )
        self.opponent_combo.pack(side="left", padx=(5, 10))
        self.opponent_combo.bind("<<ComboboxSelected>>",
                                 lambda e: self._update_opponent_options())

        # Depth label+combo (only visible for minimax)
        self.opp_depth_frame = tk.Frame(opp_row, bg=BG_DARK)
        tk.Label(self.opp_depth_frame, text="Depth", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(side="left")
        self.opponent_depth_spin = ttk.Combobox(
            self.opp_depth_frame, textvariable=self.opponent_depth_var,
            values=["1", "2", "3", "4", "5"],
            state="readonly", width=4,
        )
        self.opponent_depth_spin.pack(side="left", padx=(5, 0))

        # Description label
        self.opp_desc_label = tk.Label(
            self.opponent_frame,
            text="Auto: random opponent for training (default)",
            font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_DARK,
        )
        self.opp_desc_label.pack(anchor="w", pady=(2, 0))

        # ── Separator ──────────────────────────────────────────────────
        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(
            fill="x", pady=(10, 8),
        )

        # ── Compute ────────────────────────────────────────────────────
        self._section_label(parent, "COMPUTE")

        compute_row = tk.Frame(parent, bg=BG_DARK)
        compute_row.pack(fill="x", pady=(4, 0))

        dev_col = tk.Frame(compute_row, bg=BG_DARK)
        dev_col.pack(side="left", padx=(0, 20))
        tk.Label(dev_col, text="Device", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w")
        ttk.Combobox(
            dev_col, textvariable=self.device_var,
            values=["auto", "cuda", "mps", "cpu"],
            state="readonly", width=8,
        ).pack(anchor="w", pady=(2, 0))

        env_col = tk.Frame(compute_row, bg=BG_DARK)
        env_col.pack(side="left")
        tk.Label(env_col, text="Parallel Envs", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w")
        ttk.Combobox(
            env_col, textvariable=self.num_envs_var,
            values=["1", "2", "4", "8", "16"],
            state="readonly", width=6,
        ).pack(anchor="w", pady=(2, 0))

        # ── Separator ──────────────────────────────────────────────────
        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(
            fill="x", pady=(10, 8),
        )

        # ── Model loader ───────────────────────────────────────────────
        self._section_label(parent, "LOAD MODEL")

        model_row = tk.Frame(parent, bg=BG_DARK)
        model_row.pack(fill="x", pady=(4, 0))

        self.model_display = tk.Label(
            model_row, text="None",
            font=("Segoe UI", 10), fg=TEXT_DIM,
            bg=BG_LIGHT, anchor="w", padx=8, pady=4, relief="flat",
        )
        self.model_display.pack(side="left", fill="x", expand=True)

        tk.Button(
            model_row, text="Browse",
            font=("Segoe UI", 9), bg=BG_MEDIUM, fg=TEXT_PRIMARY,
            activebackground=BG_LIGHT, activeforeground=TEXT_PRIMARY,
            relief="flat", cursor="hand2", padx=12,
            command=self._browse_model,
        ).pack(side="left", padx=(5, 0))

        tk.Button(
            model_row, text="Clear",
            font=("Segoe UI", 9), bg=BG_MEDIUM, fg=TEXT_DIM,
            activebackground=BG_LIGHT, activeforeground=TEXT_PRIMARY,
            relief="flat", cursor="hand2", padx=8,
            command=self._clear_model,
        ).pack(side="left", padx=(3, 0))

    def _build_extras_section(self, parent):
        """Collapsible panel: streaming, music, ROM import, folder shortcuts."""
        # Toggle bar
        self.extras_toggle = tk.Frame(parent, bg=BG_MEDIUM, cursor="hand2")
        self.extras_toggle.pack(fill="x")

        self.extras_arrow = tk.Label(
            self.extras_toggle,
            text="▸  Streaming & Extras",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_DIM, bg=BG_MEDIUM,
            padx=20, pady=6,
        )
        self.extras_arrow.pack(anchor="w")

        # Make the whole bar clickable
        for widget in [self.extras_toggle, self.extras_arrow]:
            widget.bind("<Button-1>", lambda e: self._toggle_extras())

        # Content frame (hidden by default)
        self.extras_content = tk.Frame(parent, bg=BG_DARK)
        # Don't pack yet — starts hidden

        self._build_extras_content(self.extras_content)

    def _build_extras_content(self, parent):
        """Build the extras panel contents."""
        inner = tk.Frame(parent, bg=BG_DARK, padx=20, pady=8)
        inner.pack(fill="x")

        # Row with 3 sections side by side
        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)
        inner.columnconfigure(2, weight=1)

        # ── Streaming ──────────────────────────────────────────────────
        stream_col = tk.Frame(inner, bg=BG_DARK)
        stream_col.grid(row=0, column=0, sticky="nw", padx=(0, 15))

        tk.Label(
            stream_col, text="STREAMING",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w")

        tk.Checkbutton(
            stream_col, text="Enable Streaming",
            variable=self.stream_var,
            font=("Segoe UI", 9), fg=ACCENT_RED, bg=BG_DARK,
            selectcolor=BG_MEDIUM, activebackground=BG_DARK,
            activeforeground=ACCENT_RED, cursor="hand2",
        ).pack(anchor="w", pady=(2, 0))

        tk.Label(stream_col, text="Twitch Key:", font=("Segoe UI", 8),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w", pady=(4, 0))
        tk.Entry(
            stream_col, textvariable=self.twitch_key_var, show="*",
            font=("Segoe UI", 9), bg=BG_MEDIUM, fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY, relief="flat", width=20,
        ).pack(fill="x", pady=1)

        tk.Label(stream_col, text="YouTube Key:", font=("Segoe UI", 8),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w", pady=(3, 0))
        tk.Entry(
            stream_col, textvariable=self.youtube_key_var, show="*",
            font=("Segoe UI", 9), bg=BG_MEDIUM, fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY, relief="flat", width=20,
        ).pack(fill="x", pady=1)

        # ── Music + ROM ────────────────────────────────────────────────
        mid_col = tk.Frame(inner, bg=BG_DARK)
        mid_col.grid(row=0, column=1, sticky="nw", padx=15)

        tk.Label(
            mid_col, text="MUSIC & ROMs",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w")

        tk.Checkbutton(
            mid_col, text="Background Music",
            variable=self.music_var,
            font=("Segoe UI", 9), fg=TEXT_PRIMARY, bg=BG_DARK,
            selectcolor=BG_MEDIUM, activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY, cursor="hand2",
        ).pack(anchor="w", pady=(2, 0))

        tk.Button(
            mid_col, text="Import ROM Directory...",
            font=("Segoe UI", 9), bg=BG_MEDIUM, fg=TEXT_PRIMARY,
            activebackground=BG_LIGHT, activeforeground=TEXT_PRIMARY,
            relief="flat", cursor="hand2", padx=8,
            command=self._import_rom,
        ).pack(anchor="w", pady=(8, 0))

        self.rom_status = tk.Label(
            mid_col, text="",
            font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_DARK,
            wraplength=250, justify="left",
        )
        self.rom_status.pack(anchor="w", pady=(2, 0))

        # ── Tools & Features ──────────────────────────────────────────
        right_col = tk.Frame(inner, bg=BG_DARK)
        right_col.grid(row=0, column=2, sticky="nw", padx=(15, 0))

        tk.Label(
            right_col, text="TOOLS & FEATURES",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w")

        tools_grid = tk.Frame(right_col, bg=BG_DARK)
        tools_grid.pack(fill="x", pady=(4, 0))

        tool_items = [
            ("Tournament",      self._open_tournament,        ACCENT_BLUE),
            ("Agent Gallery",   self._open_agent_gallery,     ACCENT_GREEN),
            ("Export Agent",    self._export_agent,            TEXT_DIM),
            ("Import Agent",    self._import_agent,            TEXT_DIM),
            ("Highlight Reel",  self._generate_highlight_reel, ACCENT_ORANGE),
            ("Compare Runs",    self._open_comparison,         TEXT_DIM),
            ("Add Game...",     self._add_game_wizard,         TEXT_DIM),
        ]

        for i, (label, cmd, color) in enumerate(tool_items):
            row, col = divmod(i, 2)
            btn = tk.Button(
                tools_grid, text=label,
                font=("Segoe UI", 8), bg=BG_MEDIUM, fg=color,
                activebackground=BG_LIGHT, activeforeground=TEXT_PRIMARY,
                relief="flat", cursor="hand2", padx=6, pady=3, anchor="w",
                command=cmd,
            )
            btn.grid(row=row, column=col, sticky="ew", padx=2, pady=1)

        tools_grid.columnconfigure(0, weight=1)
        tools_grid.columnconfigure(1, weight=1)

        # Folder shortcuts (smaller, subtle)
        tk.Frame(right_col, bg=BORDER_COLOR, height=1).pack(fill="x", pady=(6, 4))
        folders_row = tk.Frame(right_col, bg=BG_DARK)
        folders_row.pack(fill="x")
        for label, folder in [("Models", MODELS_DIR), ("Recordings", RECORDINGS_DIR)]:
            tk.Button(
                folders_row, text=f"Open {label}",
                font=("Segoe UI", 7), bg=BG_DARK, fg=TEXT_DIM,
                activebackground=BG_MEDIUM, activeforeground=TEXT_PRIMARY,
                relief="flat", cursor="hand2", padx=4, pady=1,
                command=lambda f=folder: self._open_folder(f),
            ).pack(side="left", padx=(0, 6))

    def _toggle_extras(self, save=True):
        """Show/hide the extras panel."""
        if self._extras_expanded:
            self.extras_content.pack_forget()
            self.extras_arrow.configure(text="▸  Streaming & Extras")
            self._extras_expanded = False
        else:
            # Insert before the start area (which is packed last)
            self.extras_content.pack(fill="x", before=self._start_area_sep)
            self.extras_arrow.configure(text="▾  Streaming & Extras")
            self._extras_expanded = True

        if save:
            self._schedule_save()

    # ===================================================================
    # Training Schedule
    # ===================================================================

    def _build_schedule_section(self, parent):
        """Schedule button that opens the full scheduling window."""
        # Initialize the calendar store early
        try:
            from src.scheduler import CalendarStore
            self._calendar_store = CalendarStore()
        except Exception:
            self._calendar_store = None

        self._schedule_window = None

        sched_bar = tk.Frame(parent, bg=BG_MEDIUM, cursor="hand2")
        sched_bar.pack(fill="x")
        sched_label = tk.Label(
            sched_bar,
            text="\U0001f4c5  Open Training Schedule",
            font=("Segoe UI", 10, "bold"),
            fg=ACCENT_BLUE, bg=BG_MEDIUM,
            padx=20, pady=6,
        )
        sched_label.pack(anchor="w")
        for w in [sched_bar, sched_label]:
            w.bind("<Button-1>", lambda e: self._open_schedule_window())

    # ---------------------------------------------------------------
    # Schedule Window (Toplevel)
    # ---------------------------------------------------------------

    def _open_schedule_window(self):
        """Open (or focus) the full training-schedule management window."""
        if self._schedule_window is not None:
            try:
                self._schedule_window.lift()
                self._schedule_window.focus_force()
                return
            except tk.TclError:
                self._schedule_window = None

        win = tk.Toplevel(self.root)
        win.title("Training Schedule")
        win.configure(bg=BG_DARK)
        win.geometry("820x620")
        win.minsize(700, 450)
        win.transient(self.root)
        self._schedule_window = win
        win.protocol("WM_DELETE_WINDOW", self._close_schedule_window)

        # ── Header ────────────────────────────────────────────────
        hdr = tk.Frame(win, bg=BG_MEDIUM, padx=20, pady=10)
        hdr.pack(fill="x")
        tk.Label(
            hdr, text="Training Schedule",
            font=("Segoe UI", 14, "bold"), fg=TEXT_PRIMARY, bg=BG_MEDIUM,
        ).pack(anchor="w")
        tk.Label(
            hdr, text="Plan and automate your training sessions",
            font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_MEDIUM,
        ).pack(anchor="w")

        # ── Add-session form ──────────────────────────────────────
        form_outer = tk.LabelFrame(
            win, text="  Add Session  ", font=("Segoe UI", 9, "bold"),
            fg=TEXT_DIM, bg=BG_DARK, bd=1, relief="groove",
            padx=12, pady=8,
        )
        form_outer.pack(fill="x", padx=14, pady=(10, 4))

        self._sched_form = {}  # holds tk vars for the add-session form

        # Row 1: Game, Algorithm, Episodes
        r1 = tk.Frame(form_outer, bg=BG_DARK)
        r1.pack(fill="x", pady=(0, 6))

        tk.Label(r1, text="Game", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_DIM, bg=BG_DARK).grid(row=0, column=0, sticky="w")
        game_ids = [g.game_id for g in self.available_games]
        game_var = tk.StringVar(value=game_ids[0] if game_ids else "mario")
        game_cb = ttk.Combobox(r1, textvariable=game_var, values=game_ids,
                               state="readonly", width=14)
        game_cb.grid(row=1, column=0, sticky="w", padx=(0, 10))
        self._sched_form['game'] = game_var

        tk.Label(r1, text="Algorithm", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_DIM, bg=BG_DARK).grid(row=0, column=1, sticky="w")
        algo_names = list(ALGO_INFO.keys())
        algo_var = tk.StringVar(value="ppo")
        algo_cb = ttk.Combobox(r1, textvariable=algo_var, values=algo_names,
                               state="readonly", width=10)
        algo_cb.grid(row=1, column=1, sticky="w", padx=(0, 10))
        self._sched_form['algo'] = algo_var

        tk.Label(r1, text="Episodes", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_DIM, bg=BG_DARK).grid(row=0, column=2, sticky="w")
        ep_var = tk.StringVar(value="1000")
        ep_entry = tk.Entry(r1, textvariable=ep_var, width=10,
                            bg=BG_LIGHT, fg=TEXT_PRIMARY,
                            insertbackground=TEXT_PRIMARY, relief="flat")
        ep_entry.grid(row=1, column=2, sticky="w", padx=(0, 10))
        self._sched_form['episodes'] = ep_var

        # Stream checkbox
        stream_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            r1, text="Stream", variable=stream_var,
            font=("Segoe UI", 9), fg=TEXT_PRIMARY, bg=BG_DARK,
            selectcolor=BG_MEDIUM, activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY,
        ).grid(row=1, column=3, sticky="w", padx=(0, 10))
        self._sched_form['stream'] = stream_var

        # Row 2: Date/time pickers + immediate checkbox
        r2 = tk.Frame(form_outer, bg=BG_DARK)
        r2.pack(fill="x", pady=(0, 6))

        imm_var = tk.BooleanVar(value=True)
        self._sched_form['immediate'] = imm_var

        from datetime import datetime as _dt
        now = _dt.now()

        tk.Label(r2, text="Date", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_DIM, bg=BG_DARK).grid(row=0, column=0, sticky="w")
        dt_frame = tk.Frame(r2, bg=BG_DARK)
        dt_frame.grid(row=1, column=0, sticky="w", padx=(0, 10))

        years = [str(y) for y in range(2024, 2028)]
        months = [f'{m:02d}' for m in range(1, 13)]
        days_list = [f'{d:02d}' for d in range(1, 32)]
        hours_list = [f'{h:02d}' for h in range(24)]
        minutes_list = [f'{m:02d}' for m in range(0, 60, 5)]

        yr_var = tk.StringVar(value=str(now.year))
        mo_var = tk.StringVar(value=f'{now.month:02d}')
        dy_var = tk.StringVar(value=f'{now.day:02d}')
        hr_var = tk.StringVar(value=f'{now.hour:02d}')
        mn_var = tk.StringVar(value=f'{(now.minute // 5) * 5:02d}')

        self._sched_dt_widgets = []  # to enable/disable with immediate

        yr_cb = ttk.Combobox(dt_frame, textvariable=yr_var, values=years,
                             state="readonly", width=5)
        yr_cb.pack(side="left", padx=(0, 2))
        mo_cb = ttk.Combobox(dt_frame, textvariable=mo_var, values=months,
                             state="readonly", width=3)
        mo_cb.pack(side="left", padx=(0, 2))
        dy_cb = ttk.Combobox(dt_frame, textvariable=dy_var, values=days_list,
                             state="readonly", width=3)
        dy_cb.pack(side="left", padx=(0, 6))
        self._sched_dt_widgets.extend([yr_cb, mo_cb, dy_cb])

        tk.Label(r2, text="Time", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_DIM, bg=BG_DARK).grid(row=0, column=1, sticky="w")
        tm_frame = tk.Frame(r2, bg=BG_DARK)
        tm_frame.grid(row=1, column=1, sticky="w", padx=(0, 10))

        hr_cb = ttk.Combobox(tm_frame, textvariable=hr_var, values=hours_list,
                             state="readonly", width=3)
        hr_cb.pack(side="left", padx=(0, 1))
        tk.Label(tm_frame, text=":", fg=TEXT_DIM, bg=BG_DARK,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        mn_cb = ttk.Combobox(tm_frame, textvariable=mn_var, values=minutes_list,
                             state="readonly", width=3)
        mn_cb.pack(side="left")
        self._sched_dt_widgets.extend([hr_cb, mn_cb])

        self._sched_form['year'] = yr_var
        self._sched_form['month'] = mo_var
        self._sched_form['day'] = dy_var
        self._sched_form['hour'] = hr_var
        self._sched_form['minute'] = mn_var

        tk.Checkbutton(
            r2, text="Schedule Immediately", variable=imm_var,
            font=("Segoe UI", 9), fg=TEXT_PRIMARY, bg=BG_DARK,
            selectcolor=BG_MEDIUM, activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY,
            command=self._toggle_datetime_fields,
        ).grid(row=1, column=2, sticky="w", padx=(0, 10))

        # Initial state: immediate checked -> hide date/time
        self._toggle_datetime_fields()

        # Add session button
        tk.Button(
            r2, text="+ Add Session",
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT_GREEN, fg=BG_DARK,
            activebackground="#4bb569", activeforeground=BG_DARK,
            relief="flat", cursor="hand2", padx=14, pady=3,
            command=self._sched_add_from_form,
        ).grid(row=1, column=3, sticky="w")

        # ── Session queue (scrollable) ────────────────────────────
        queue_lbl_frame = tk.Frame(win, bg=BG_DARK, padx=14)
        queue_lbl_frame.pack(fill="x", pady=(8, 0))
        tk.Label(
            queue_lbl_frame, text="SESSION QUEUE",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w")

        # Header row
        hdr_frame = tk.Frame(win, bg=BG_MEDIUM, padx=14)
        hdr_frame.pack(fill="x", padx=14, pady=(4, 0))
        hdr_s = dict(font=("Segoe UI", 8, "bold"), fg=TEXT_DIM,
                     bg=BG_MEDIUM, pady=3)
        tk.Label(hdr_frame, text=" #", width=3, anchor="w", **hdr_s).pack(side="left")
        tk.Label(hdr_frame, text="Game", width=10, anchor="w", **hdr_s).pack(side="left")
        tk.Label(hdr_frame, text="Algorithm", width=9, anchor="w", **hdr_s).pack(side="left")
        tk.Label(hdr_frame, text="Episodes", width=8, anchor="w", **hdr_s).pack(side="left")
        tk.Label(hdr_frame, text="Scheduled", width=16, anchor="w", **hdr_s).pack(side="left")
        tk.Label(hdr_frame, text="Stream", width=6, anchor="w", **hdr_s).pack(side="left")
        tk.Label(hdr_frame, text="Status", width=10, anchor="w", **hdr_s).pack(side="left")
        tk.Label(hdr_frame, text="Actions", width=8, anchor="w", **hdr_s).pack(side="left")

        # Scrollable list container
        list_outer = tk.Frame(win, bg=BG_DARK, padx=14)
        list_outer.pack(fill="both", expand=True, padx=14)

        self._sw_canvas = tk.Canvas(list_outer, bg=BG_DARK,
                                    highlightthickness=0)
        self._sw_scrollbar = tk.Scrollbar(list_outer, orient="vertical",
                                          command=self._sw_canvas.yview)
        self._sw_inner = tk.Frame(self._sw_canvas, bg=BG_DARK)
        self._sw_inner.bind(
            "<Configure>",
            lambda e: self._sw_canvas.configure(
                scrollregion=self._sw_canvas.bbox("all")),
        )
        self._sw_canvas.create_window((0, 0), window=self._sw_inner,
                                      anchor="nw")
        self._sw_canvas.configure(yscrollcommand=self._sw_scrollbar.set)
        self._sw_canvas.pack(side="left", fill="both", expand=True)
        self._sw_scrollbar.pack(side="right", fill="y")

        # mousewheel scrolling
        def _on_mousewheel(event):
            self._sw_canvas.yview_scroll(
                int(-1 * (event.delta / 120)), "units")

        self._sw_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        win.bind("<Destroy>", lambda e: (
            self._sw_canvas.unbind_all("<MouseWheel>")
            if e.widget is win else None
        ))

        # ── Bottom controls ───────────────────────────────────────
        bot = tk.Frame(win, bg=BG_MEDIUM, padx=14, pady=8)
        bot.pack(fill="x", side="bottom")

        self._sw_status = tk.Label(
            bot, text="No sessions scheduled",
            font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_MEDIUM,
        )
        self._sw_status.pack(side="left")

        btn_s = dict(
            font=("Segoe UI", 9), relief="flat", cursor="hand2",
            padx=10, pady=3,
        )
        tk.Button(
            bot, text="Clear Completed",
            bg=BG_LIGHT, fg=TEXT_DIM,
            activebackground=BG_DARK, activeforeground=TEXT_DIM,
            command=self._sched_clear_completed, **btn_s,
        ).pack(side="right", padx=(4, 0))
        tk.Button(
            bot, text="DT Generalist Run",
            bg=BG_LIGHT, fg=ACCENT_ORANGE,
            activebackground=BG_DARK, activeforeground=ACCENT_ORANGE,
            command=self._sched_dt_run, **btn_s,
        ).pack(side="right", padx=(4, 0))
        tk.Button(
            bot, text="Overnight All Games",
            bg=BG_LIGHT, fg=ACCENT_BLUE,
            activebackground=BG_DARK, activeforeground=ACCENT_BLUE,
            command=self._sched_overnight, **btn_s,
        ).pack(side="right", padx=(4, 0))
        tk.Button(
            bot, text="\u25b6  Run All",
            bg=ACCENT_BLUE, fg=BG_DARK,
            activebackground="#3d8ee6", activeforeground=BG_DARK,
            font=("Segoe UI", 9, "bold"),
            command=self._sched_run_all, relief="flat",
            cursor="hand2", padx=12, pady=3,
        ).pack(side="right", padx=(4, 0))
        tk.Button(
            bot, text="\u25b6  Run Next",
            bg=ACCENT_GREEN, fg=BG_DARK,
            activebackground="#4bb569", activeforeground=BG_DARK,
            font=("Segoe UI", 9, "bold"),
            command=self._sched_run_next, relief="flat",
            cursor="hand2", padx=12, pady=3,
        ).pack(side="right", padx=(4, 0))

        # Populate the queue
        self._refresh_schedule_list()

    def _close_schedule_window(self):
        """Close the schedule window cleanly."""
        if self._schedule_window is not None:
            try:
                self._schedule_window.destroy()
            except tk.TclError:
                pass
            self._schedule_window = None

    def _toggle_datetime_fields(self):
        """Enable/disable date-time dropdowns based on 'immediate' checkbox."""
        is_imm = self._sched_form['immediate'].get()
        new_state = "disabled" if is_imm else "readonly"
        for w in self._sched_dt_widgets:
            w.configure(state=new_state)

    @staticmethod
    def _format_scheduled_time(dt_obj):
        """Format a datetime for display: 'Mar 20 at 14:30' or 'Immediate'."""
        if dt_obj is None:
            return "Immediate"
        return dt_obj.strftime("%b %d at %H:%M")

    # ---------------------------------------------------------------
    # Session queue refresh
    # ---------------------------------------------------------------

    def _refresh_schedule_list(self):
        """Refresh the session queue display from CalendarStore."""
        # Guard: the list widget may not exist yet
        if not hasattr(self, '_sw_inner'):
            return

        for widget in self._sw_inner.winfo_children():
            widget.destroy()

        if not self._calendar_store:
            tk.Label(
                self._sw_inner,
                text="  Schedule system unavailable",
                font=("Segoe UI", 9), fg=ACCENT_RED, bg=BG_DARK,
            ).pack(anchor="w")
            self._update_schedule_status()
            return

        sessions = self._calendar_store.get_all_sessions()
        if not sessions:
            tk.Label(
                self._sw_inner,
                text="  No sessions \u2014 add one above or use a quick preset",
                font=("Segoe UI", 9, "italic"), fg=TEXT_DIM, bg=BG_DARK,
            ).pack(anchor="w", pady=12)
            self._update_schedule_status()
            return

        status_colors = {
            'pending': TEXT_PRIMARY,
            'running': ACCENT_GREEN,
            'completed': TEXT_DIM,
            'failed': ACCENT_RED,
            'cancelled': TEXT_DIM,
        }

        for i, session in enumerate(sessions):
            row_bg = BG_DARK if i % 2 == 0 else BG_MEDIUM
            row = tk.Frame(self._sw_inner, bg=row_bg)
            row.pack(fill="x")

            rs = dict(font=("Segoe UI", 9), bg=row_bg, pady=2)
            tk.Label(row, text=f" {i + 1}", width=3, anchor="w",
                     fg=TEXT_DIM, **rs).pack(side="left")
            tk.Label(row, text=session.game_id, width=10, anchor="w",
                     fg=TEXT_PRIMARY, **rs).pack(side="left")
            tk.Label(
                row, text=session.algorithm.upper(), width=9, anchor="w",
                fg=ALGO_INFO.get(session.algorithm, {}).get(
                    'color', TEXT_PRIMARY),
                **rs,
            ).pack(side="left")
            tk.Label(row, text=str(session.episodes), width=8, anchor="w",
                     fg=TEXT_PRIMARY, **rs).pack(side="left")
            tk.Label(
                row,
                text=self._format_scheduled_time(session.scheduled_start),
                width=16, anchor="w", fg=TEXT_PRIMARY, **rs,
            ).pack(side="left")
            stream_txt = "Yes" if session.stream else "\u2014"
            tk.Label(row, text=stream_txt, width=6, anchor="w",
                     fg=ACCENT_RED if session.stream else TEXT_DIM,
                     **rs).pack(side="left")
            tk.Label(
                row, text=session.status.capitalize(), width=10, anchor="w",
                fg=status_colors.get(session.status, TEXT_DIM), **rs,
            ).pack(side="left")

            # Actions: Edit + Remove
            act = tk.Frame(row, bg=row_bg)
            act.pack(side="left")
            tk.Button(
                act, text="\u270e", font=("Segoe UI", 9),
                bg=row_bg, fg=ACCENT_BLUE,
                activebackground=row_bg, activeforeground="#3d8ee6",
                relief="flat", cursor="hand2", padx=3,
                command=lambda sid=session.session_id: self._open_session_editor(sid),
            ).pack(side="left")
            tk.Button(
                act, text="\u2715", font=("Segoe UI", 9),
                bg=row_bg, fg=ACCENT_RED,
                activebackground=row_bg, activeforeground="#ff6b7a",
                relief="flat", cursor="hand2", padx=3,
                command=lambda sid=session.session_id: self._sched_remove(sid),
            ).pack(side="left")

        self._update_schedule_status()

    def _update_schedule_status(self):
        """Update the status label at the bottom of the schedule window."""
        if not hasattr(self, '_sw_status'):
            return
        if not self._calendar_store:
            self._sw_status.config(text="Schedule system unavailable")
            return
        sessions = self._calendar_store.get_all_sessions()
        pending = sum(1 for s in sessions if s.status == 'pending')
        if pending:
            self._sw_status.config(
                text=f"{pending} session{'s' if pending != 1 else ''} pending")
        else:
            self._sw_status.config(text="No sessions scheduled")
        # Also update the header badge
        self._update_session_badge()

    # ---------------------------------------------------------------
    # Session Editor Dialog
    # ---------------------------------------------------------------

    def _open_session_editor(self, session_id):
        """Open a small Toplevel to edit an existing session's fields."""
        if not self._calendar_store:
            return
        session = self._calendar_store.get_session(session_id)
        if session is None:
            return

        ed = tk.Toplevel(self._schedule_window or self.root)
        ed.title("Edit Session")
        ed.configure(bg=BG_DARK)
        ed.geometry("340x310")
        ed.resizable(False, False)
        if self._schedule_window:
            ed.transient(self._schedule_window)

        pad = dict(padx=12, pady=(6, 0))
        lbl_s = dict(font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK)

        tk.Label(ed, text="Game", **lbl_s).pack(anchor="w", **pad)
        game_ids = [g.game_id for g in self.available_games]
        ed_game = tk.StringVar(value=session.game_id)
        ttk.Combobox(ed, textvariable=ed_game, values=game_ids,
                     state="readonly", width=20).pack(anchor="w", padx=12)

        tk.Label(ed, text="Algorithm", **lbl_s).pack(anchor="w", **pad)
        algo_names = list(ALGO_INFO.keys())
        ed_algo = tk.StringVar(value=session.algorithm)
        ttk.Combobox(ed, textvariable=ed_algo, values=algo_names,
                     state="readonly", width=20).pack(anchor="w", padx=12)

        tk.Label(ed, text="Episodes", **lbl_s).pack(anchor="w", **pad)
        ed_ep = tk.StringVar(value=str(session.episodes))
        tk.Entry(ed, textvariable=ed_ep, width=22,
                 bg=BG_LIGHT, fg=TEXT_PRIMARY,
                 insertbackground=TEXT_PRIMARY, relief="flat"
                 ).pack(anchor="w", padx=12)

        # Date / Time
        tk.Label(ed, text="Date / Time", **lbl_s).pack(anchor="w", **pad)
        dt_row = tk.Frame(ed, bg=BG_DARK)
        dt_row.pack(anchor="w", padx=12)

        from datetime import datetime as _dt
        ref = session.scheduled_start or _dt.now()

        years = [str(y) for y in range(2024, 2028)]
        months = [f'{m:02d}' for m in range(1, 13)]
        days_list = [f'{d:02d}' for d in range(1, 32)]
        hours_list = [f'{h:02d}' for h in range(24)]
        minutes_list = [f'{m:02d}' for m in range(0, 60, 5)]

        ed_yr = tk.StringVar(value=str(ref.year))
        ed_mo = tk.StringVar(value=f'{ref.month:02d}')
        ed_dy = tk.StringVar(value=f'{ref.day:02d}')
        ed_hr = tk.StringVar(value=f'{ref.hour:02d}')
        ed_mn = tk.StringVar(value=f'{(ref.minute // 5) * 5:02d}')

        ttk.Combobox(dt_row, textvariable=ed_yr, values=years,
                     state="readonly", width=5).pack(side="left", padx=(0, 2))
        ttk.Combobox(dt_row, textvariable=ed_mo, values=months,
                     state="readonly", width=3).pack(side="left", padx=(0, 2))
        ttk.Combobox(dt_row, textvariable=ed_dy, values=days_list,
                     state="readonly", width=3).pack(side="left", padx=(0, 6))
        ttk.Combobox(dt_row, textvariable=ed_hr, values=hours_list,
                     state="readonly", width=3).pack(side="left", padx=(0, 1))
        tk.Label(dt_row, text=":", fg=TEXT_DIM, bg=BG_DARK,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Combobox(dt_row, textvariable=ed_mn, values=minutes_list,
                     state="readonly", width=3).pack(side="left")

        # Stream
        ed_stream = tk.BooleanVar(value=session.stream)
        tk.Checkbutton(
            ed, text="Stream", variable=ed_stream,
            font=("Segoe UI", 9), fg=TEXT_PRIMARY, bg=BG_DARK,
            selectcolor=BG_MEDIUM, activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY,
        ).pack(anchor="w", padx=12, pady=(8, 0))

        # Buttons
        btn_row = tk.Frame(ed, bg=BG_DARK)
        btn_row.pack(fill="x", padx=12, pady=(12, 10))

        def _save_edit():
            try:
                new_start = _dt(
                    int(ed_yr.get()), int(ed_mo.get()), int(ed_dy.get()),
                    int(ed_hr.get()), int(ed_mn.get()),
                )
            except (ValueError, TypeError):
                new_start = None
            ep = ed_ep.get().strip()
            episodes = int(ep) if ep.isdigit() else session.episodes
            self._calendar_store.update_session(
                session_id,
                game_id=ed_game.get(),
                algorithm=ed_algo.get(),
                episodes=episodes,
                scheduled_start=new_start,
                stream=ed_stream.get(),
            )
            self._refresh_schedule_list()
            ed.destroy()

        tk.Button(
            btn_row, text="Save",
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT_GREEN, fg=BG_DARK,
            activebackground="#4bb569", activeforeground=BG_DARK,
            relief="flat", cursor="hand2", padx=14, pady=3,
            command=_save_edit,
        ).pack(side="left", padx=(0, 6))
        tk.Button(
            btn_row, text="Cancel",
            font=("Segoe UI", 9),
            bg=BG_MEDIUM, fg=TEXT_PRIMARY,
            activebackground=BG_LIGHT, activeforeground=TEXT_PRIMARY,
            relief="flat", cursor="hand2", padx=14, pady=3,
            command=ed.destroy,
        ).pack(side="left")

    # ---------------------------------------------------------------
    # Schedule actions
    # ---------------------------------------------------------------

    def _sched_add_from_form(self):
        """Add a session from the add-session form in the schedule window."""
        if not self._calendar_store:
            return
        try:
            from datetime import datetime as _dt
            from src.scheduler.session import create_session

            game_id = self._sched_form['game'].get()
            algo = self._sched_form['algo'].get()
            ep = self._sched_form['episodes'].get().strip()
            episodes = int(ep) if ep.isdigit() else 1000
            stream = self._sched_form['stream'].get()
            immediate = self._sched_form['immediate'].get()

            scheduled_start = None
            if not immediate:
                try:
                    scheduled_start = _dt(
                        int(self._sched_form['year'].get()),
                        int(self._sched_form['month'].get()),
                        int(self._sched_form['day'].get()),
                        int(self._sched_form['hour'].get()),
                        int(self._sched_form['minute'].get()),
                    )
                except (ValueError, TypeError):
                    pass

            session = create_session(
                game_id=game_id,
                algorithm=algo,
                episodes=episodes,
                stream=stream,
                scheduled_start=scheduled_start,
            )
            self._calendar_store.add_session(session)
            self._refresh_schedule_list()
        except Exception as e:
            self._set_status(f"Failed to add session: {e}", ACCENT_RED)

    def _sched_overnight(self):
        """Add overnight preset: all games with PPO."""
        if not self._calendar_store:
            return
        try:
            from src.scheduler.session import overnight_all_games
            sessions = overnight_all_games(episodes_per_game=500)
            self._calendar_store.add_preset(sessions)
            self._refresh_schedule_list()
        except Exception as e:
            self._set_status(f"Failed to add overnight preset: {e}", ACCENT_RED)

    def _sched_dt_run(self):
        """Add DT generalist preset: collect from all games + train DT."""
        if not self._calendar_store:
            return
        try:
            from src.scheduler.session import dt_generalist_run
            sessions = dt_generalist_run(episodes_per_game=100)
            self._calendar_store.add_preset(sessions)
            self._refresh_schedule_list()
        except Exception as e:
            self._set_status(f"Failed to add DT preset: {e}", ACCENT_RED)

    def _sched_remove(self, session_id):
        """Remove a session from the queue."""
        if self._calendar_store:
            self._calendar_store.remove_session(session_id)
            self._refresh_schedule_list()

    def _sched_clear_completed(self):
        """Remove all completed/failed/cancelled sessions."""
        if not self._calendar_store:
            return
        self._calendar_store.clear_completed()
        self._refresh_schedule_list()

    def _sched_apply_session(self, session):
        """Apply a session's settings to the launcher and start training."""
        self.game_var.set(session.game_id)
        for i, game_info in enumerate(self.available_games):
            if game_info.game_id == session.game_id:
                self.game_combo.current(i)
                self._on_game_changed()
                break
        self.selected_algo.set(session.algorithm)
        self._select_algorithm(session.algorithm)
        self.duration_var.set(str(session.episodes))
        self.stream_var.set(session.stream)

        from datetime import datetime
        self._calendar_store.update_session(
            session.session_id,
            status='running',
            actual_start=datetime.now(),
        )
        self._refresh_schedule_list()
        self._start_training()

    def _sched_run_next(self):
        """Run the next pending session from the schedule."""
        if not self._calendar_store:
            return
        next_session = self._calendar_store.get_next_session()
        if not next_session:
            pending = self._calendar_store.get_pending_sessions()
            if pending:
                next_session = pending[0]
        if not next_session:
            self._set_status("No pending sessions to run.", TEXT_DIM)
            return
        self._sched_apply_session(next_session)

    def _sched_run_all(self):
        """Run all pending sessions sequentially (starts the first one)."""
        if not self._calendar_store:
            return
        pending = self._calendar_store.get_pending_sessions()
        if not pending:
            self._set_status("No pending sessions to run.", TEXT_DIM)
            return
        # Start the first; the completion callback should chain the rest
        self._sched_apply_session(pending[0])

    def _build_start_area(self):
        """Always-visible START/STOP button and status bar at the bottom."""
        self._start_area_sep = tk.Frame(self.root, bg=BORDER_COLOR, height=1)
        self._start_area_sep.pack(fill="x", side="bottom")

        area = tk.Frame(self.root, bg=BG_DARK, padx=20, pady=10)
        area.pack(fill="x", side="bottom")

        # NL Command input
        cmd_frame = tk.Frame(area, bg=BG_DARK)
        cmd_frame.pack(fill="x", pady=(0, 6))

        tk.Label(cmd_frame, text="Quick Command:", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(side="left")
        self.nl_command_var = tk.StringVar()
        self.nl_command_entry = tk.Entry(
            cmd_frame, textvariable=self.nl_command_var,
            font=("Segoe UI", 10), bg=BG_MEDIUM, fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY, relief="flat",
        )
        self.nl_command_entry.pack(side="left", fill="x", expand=True, padx=(6, 6))
        self.nl_command_entry.bind('<Return>', lambda e: self._apply_nl_command())

        tk.Button(
            cmd_frame, text="Apply",
            font=("Segoe UI", 9), bg=ACCENT_BLUE, fg=BG_DARK,
            activebackground="#3d8ee6", relief="flat", cursor="hand2", padx=10,
            command=self._apply_nl_command,
        ).pack(side="left")

        # Placeholder text behaviour
        self.nl_command_entry.insert(0, 'e.g., "Train snake with PPO" or "Play chess against hard AI"')
        self.nl_command_entry.config(fg=TEXT_DIM)

        def _on_nl_focus_in(event):
            if self.nl_command_entry.get().startswith('e.g.,'):
                self.nl_command_entry.delete(0, 'end')
                self.nl_command_entry.config(fg=TEXT_PRIMARY)

        def _on_nl_focus_out(event):
            if not self.nl_command_entry.get():
                self.nl_command_entry.insert(0, 'e.g., "Train snake with PPO" or "Play chess against hard AI"')
                self.nl_command_entry.config(fg=TEXT_DIM)

        self.nl_command_entry.bind('<FocusIn>', _on_nl_focus_in)
        self.nl_command_entry.bind('<FocusOut>', _on_nl_focus_out)

        self.status_label = tk.Label(
            area,
            text="Ready — Pick a game and algorithm, then click START!",
            font=("Segoe UI", 9),
            fg=TEXT_DIM,
            bg=BG_DARK,
            anchor="w",
        )
        self.status_label.pack(fill="x", pady=(0, 4))

        self.start_btn = tk.Button(
            area,
            text="\u25B6   START TRAINING",
            font=("Segoe UI", 15, "bold"),
            bg=ACCENT_GREEN,
            fg="#0a0a1a",
            activebackground="#00b863",
            activeforeground="#0a0a1a",
            relief="flat",
            cursor="hand2",
            pady=12,
            command=self._on_start_stop,
        )
        self.start_btn.pack(fill="x")

        # Restart Training button — deletes checkpoint and starts fresh
        self.restart_btn = tk.Button(
            area,
            text="\u21BB   RESTART TRAINING (Fresh Start)",
            font=("Segoe UI", 10, "bold"),
            bg="#c87000",
            fg="#ffffff",
            activebackground="#e08800",
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            pady=6,
            command=self._restart_training,
        )
        self.restart_btn.pack(fill="x", pady=(4, 0))

    # ===================================================================
    # Helper Methods
    # ===================================================================

    def _section_label(self, parent, text, color=TEXT_DIM):
        """Create a section header with a subtle left accent bar."""
        frame = tk.Frame(parent, bg=BG_DARK)
        frame.pack(anchor="w", fill="x")
        tk.Frame(frame, bg=color, width=3, height=12).pack(side="left", padx=(0, 6))
        tk.Label(
            frame, text=text,
            font=("Segoe UI", 8, "bold"), fg=color, bg=BG_DARK,
        ).pack(side="left")
        return frame

    @staticmethod
    def _add_hover(widget, enter_bg, leave_bg):
        """Add hover color change to a widget (additive, won't replace other bindings)."""
        widget.bind('<Enter>', lambda e: widget.configure(bg=enter_bg), add='+')
        widget.bind('<Leave>', lambda e: widget.configure(bg=leave_bg), add='+')

    def _update_algo_desc(self):
        """Update the algorithm description text below the buttons."""
        algo = self.selected_algo.get()
        info = ALGO_INFO[algo]
        self.algo_desc_label.configure(
            text=info["short_desc"],
            fg=info["color"],
        )

    def _bind_tooltip(self, widget, text):
        """Bind hover tooltip to a widget using a floating label."""
        def show_tip(event):
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{event.x_root + 15}+{event.y_root + 10}")
            tip.configure(bg="#2a2a4a")
            label = tk.Label(
                tip,
                text=text,
                font=("Segoe UI", 9),
                fg=TEXT_PRIMARY,
                bg="#2a2a4a",
                justify="left",
                padx=10,
                pady=6,
                wraplength=350,
            )
            label.pack()
            widget._tooltip = tip

        def hide_tip(event):
            tip = getattr(widget, '_tooltip', None)
            if tip:
                tip.destroy()
                widget._tooltip = None

        widget.bind('<Enter>', show_tip)
        widget.bind('<Leave>', hide_tip)

    def _get_selected_adapter(self):
        """Return the game adapter for the currently selected game."""
        game_text = self.game_combo.get()
        game_id = game_text.rsplit("(", 1)[-1].rstrip(")").strip() if "(" in game_text else "mario"
        try:
            return self.game_registry.get_game(game_id)
        except KeyError:
            return None

    def _rebuild_game_options(self):
        """Rebuild the dynamic game-specific options widgets."""
        for widget in self.game_opts_frame.winfo_children():
            widget.destroy()
        self.game_opt_widgets.clear()

        adapter = self._get_selected_adapter()
        if not adapter:
            return

        opts = adapter.get_game_specific_options()
        if not opts:
            return

        tk.Label(
            self.game_opts_frame,
            text="Game Options",
            font=("Segoe UI", 9),
            fg=TEXT_DIM,
            bg=BG_DARK,
        ).pack(anchor="w", pady=(3, 2))

        for name, (opt_type, default, desc) in opts.items():
            row = tk.Frame(self.game_opts_frame, bg=BG_DARK)
            row.pack(fill="x", pady=1)

            tk.Label(
                row,
                text=f"{name}:",
                font=("Segoe UI", 9),
                fg=TEXT_PRIMARY,
                bg=BG_DARK,
                width=12,
                anchor="w",
            ).pack(side="left")

            if opt_type == bool:
                var = tk.BooleanVar(value=default)
                widget = tk.Checkbutton(
                    row, variable=var,
                    bg=BG_DARK, selectcolor=BG_MEDIUM,
                    activebackground=BG_DARK,
                )
            elif opt_type == int:
                var = tk.StringVar(value=str(default))
                widget = tk.Entry(
                    row, textvariable=var,
                    font=("Segoe UI", 9), bg=BG_LIGHT, fg=TEXT_PRIMARY,
                    insertbackground=TEXT_PRIMARY, relief="flat", width=8,
                )
            else:
                var = tk.StringVar(value=str(default))
                widget = tk.Entry(
                    row, textvariable=var,
                    font=("Segoe UI", 9), bg=BG_LIGHT, fg=TEXT_PRIMARY,
                    insertbackground=TEXT_PRIMARY, relief="flat", width=15,
                )

            widget.pack(side="left", padx=(3, 5))
            self.game_opt_widgets[name] = (var, opt_type)

            tk.Label(
                row,
                text=desc,
                font=("Segoe UI", 8),
                fg=TEXT_DIM,
                bg=BG_DARK,
            ).pack(side="left")

    def _get_game_options(self) -> dict:
        """Collect current game-specific option values."""
        result = {}
        for name, (var, opt_type) in self.game_opt_widgets.items():
            try:
                if opt_type == bool:
                    result[name] = var.get()
                elif opt_type == int:
                    result[name] = int(var.get())
                else:
                    result[name] = var.get()
            except (ValueError, tk.TclError):
                pass
        return result

    def _apply_nl_command(self):
        """Parse natural language command and apply to launcher settings."""
        text = self.nl_command_var.get().strip()
        if not text or text.startswith('e.g.,'):
            return

        try:
            from src.nl_commands.parser import parse_command
            cmd = parse_command(text)
        except Exception as e:
            self._set_status(f"Parse error: {e}", ACCENT_RED)
            return

        if cmd.confidence < 0.1:
            self._set_status(
                "Couldn't understand that command. Try: 'train snake with ppo'",
                TEXT_DIM,
            )
            return

        # Apply parsed fields to launcher
        if cmd.game_id:
            for i, game_info in enumerate(self.available_games):
                gid = getattr(game_info, 'game_id', None) or game_info.get('id', None)
                if gid == cmd.game_id:
                    self.game_combo.current(i)
                    self._on_game_changed()
                    break

        if cmd.algorithm:
            self.selected_algo.set(cmd.algorithm)
            self._select_algorithm(cmd.algorithm)

        if cmd.episodes:
            self.duration_var.set(str(cmd.episodes))
            self.preset_var.set("Custom")

        if cmd.opponent:
            self.opponent_var.set(cmd.opponent)

        if cmd.eval_mode:
            self.eval_var.set(True)

        if cmd.stream:
            self.stream_var.set(True)

        self._set_status(f"Applied: {cmd.description}", ACCENT_GREEN)
        self.nl_command_var.set("")

    def _on_game_changed(self, event=None):
        """Handle game selection change — update algorithm compatibility."""
        adapter = self._get_selected_adapter()
        if adapter:
            supported = adapter.supported_algorithms()
            current = self.selected_algo.get()
            if current not in supported and supported:
                self.selected_algo.set(supported[0])
                self._update_duration_label()
                self.model_path_var.set("")
                self._update_model_display()
        self._update_algo_buttons()
        self._update_algo_desc()
        self._rebuild_game_options()
        self._update_world_stage_visibility()
        self._update_opponent_visibility()

    def _update_world_stage_visibility(self):
        """Show world/stage selector only for games that have levels (Mario)."""
        game_text = self.game_combo.get()
        game_id = game_text.rsplit("(", 1)[-1].rstrip(")").strip() if "(" in game_text else "mario"
        # Only Mario (retro/NES) games have world/stage
        if game_id == "mario":
            self.ws_frame.pack(fill="x", pady=(8, 0))
        else:
            self.ws_frame.pack_forget()

    def _update_opponent_visibility(self):
        """Show opponent options only for board games."""
        game_text = self.game_combo.get()
        game_id = game_text.rsplit("(", 1)[-1].rstrip(")").strip() if "(" in game_text else "mario"
        board_games = {'chess', 'checkers', 'connect4', 'tictactoe'}
        if game_id in board_games:
            self.opponent_frame.pack(fill="x", pady=(8, 0))
        else:
            self.opponent_frame.pack_forget()
        self._update_opponent_options()

    def _update_opponent_options(self):
        """Show/hide depth selector and update description based on opponent mode."""
        mode = self.opponent_var.get()
        descriptions = {
            'auto': 'Auto: random opponent for training (default)',
            'random': 'Random: opponent picks random valid moves',
            'minimax-easy': 'Minimax Easy: looks 1 move ahead',
            'minimax-medium': 'Minimax Medium: looks 3 moves ahead',
            'minimax-hard': 'Minimax Hard: strong play (depth 5)',
            'auto-difficulty': 'Auto-Difficulty: starts easy, promotes as agent improves',
            'model': 'Model: play against a trained AI checkpoint',
            'human': 'Human: you play against the AI',
            'human-vs-human': 'Human vs Human: two players, no AI',
        }
        self.opp_desc_label.config(text=descriptions.get(mode, ''))
        # Only show depth for custom minimax (not presets)
        if mode.startswith('minimax') and mode not in ('minimax-easy', 'minimax-medium', 'minimax-hard'):
            self.opp_depth_frame.pack(side="left")
        else:
            self.opp_depth_frame.pack_forget()

    def _select_algorithm(self, algo):
        """Handle algorithm button click."""
        self.selected_algo.set(algo)
        self._update_algo_buttons()
        self._update_algo_desc()
        self._update_duration_label()
        self.model_path_var.set("")
        self._update_model_display()

    def _update_algo_buttons(self):
        """Restyle algorithm buttons to show which is active.

        Incompatible algorithms are grayed out and disabled.
        """
        current = self.selected_algo.get()
        adapter = self._get_selected_adapter()
        supported = adapter.supported_algorithms() if adapter else list(ALGO_INFO.keys())

        for algo, btn in self.algo_buttons.items():
            # Clear all previous bindings then re-establish
            btn.unbind('<Enter>')
            btn.unbind('<Leave>')
            info = ALGO_INFO[algo]
            if algo not in supported:
                btn.configure(
                    bg=BG_DARK, fg="#555555", activebackground=BG_DARK,
                    state="disabled",
                )
            elif algo == current:
                color = info["color"]
                btn.configure(
                    bg=color, fg=BG_DARK, activebackground=color,
                    state="normal",
                )
            else:
                btn.configure(
                    bg=BG_MEDIUM, fg=TEXT_DIM, activebackground=BG_LIGHT,
                    state="normal",
                )
                # Tooltip first (replaces), then hover (additive)
                self._bind_tooltip(btn, info["tooltip"])
                self._add_hover(btn, BG_LIGHT, BG_MEDIUM)
                continue
            self._bind_tooltip(btn, info["tooltip"])

    def _on_preset_changed(self, event=None):
        """Handle training preset selection change."""
        preset = self.preset_var.get()
        if preset == "Custom":
            # Enable manual entry, clear if it held a preset value
            self.duration_entry.configure(state="normal")
        else:
            value = TRAINING_PRESETS.get(preset, "")
            self.duration_var.set(value)
            self.duration_entry.configure(state="readonly")

    def _update_duration_label(self):
        """Update the duration unit label text when algorithm changes."""
        algo = self.selected_algo.get()
        label = ALGO_INFO[algo]["duration_label"]
        self.duration_unit_label.configure(text=label)

    def _on_eval_toggle(self):
        """Update UI when Evaluation Mode is toggled."""
        self._update_start_button_text()

    def _browse_model(self):
        """Open file dialog for selecting a saved model."""
        algo = self.selected_algo.get()
        info = ALGO_INFO[algo]
        initial_dir = os.path.join(MODELS_DIR, info["model_subdir"])
        os.makedirs(initial_dir, exist_ok=True)

        path = filedialog.askopenfilename(
            title=f"Select {algo.upper()} Model",
            initialdir=initial_dir,
            filetypes=info["file_ext"],
        )

        if path:
            self.model_path_var.set(path)
            self._update_model_display()

    def _clear_model(self):
        """Clear the selected model."""
        self.model_path_var.set("")
        self._update_model_display()

    def _update_model_display(self):
        """Update the model path display label."""
        path = self.model_path_var.get()
        if path:
            display = os.path.basename(path)
            self.model_display.configure(text=display, fg=TEXT_PRIMARY)
        else:
            self.model_display.configure(text="None", fg=TEXT_DIM)

    def _update_start_button_text(self):
        """Update button text based on eval mode and running state."""
        if self.process is not None:
            self.start_btn.configure(
                text="\u23F9   STOP",
                bg=ACCENT_RED,
                activebackground="#e63946",
            )
        else:
            if self.eval_var.get():
                self.start_btn.configure(
                    text="\u25B6   START EVALUATION",
                    bg=ACCENT_BLUE,
                    activebackground="#3d8ee0",
                )
            else:
                algo = self.selected_algo.get()
                color = ALGO_INFO[algo]["color"]
                self.start_btn.configure(
                    text="\u25B6   START TRAINING",
                    bg=color,
                    activebackground=color,
                )

    def _open_agent_gallery(self):
        """Open the Agent Gallery dialog showing personality profiles."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Agent Gallery")
        dialog.configure(bg=BG_DARK)
        dialog.geometry("700x500")
        dialog.transient(self.root)

        tk.Label(
            dialog, text="Agent Gallery",
            font=("Segoe UI", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_DARK,
        ).pack(pady=(15, 5))
        tk.Label(
            dialog, text="Browse your trained agents and their personalities",
            font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_DARK,
        ).pack()

        # Scrollable area
        canvas = tk.Canvas(dialog, bg=BG_DARK, highlightthickness=0)
        scrollbar = tk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=BG_DARK)
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=20, pady=10)
        scrollbar.pack(side="right", fill="y")

        try:
            from src.achievements.profile import scan_all_profiles
            profiles = scan_all_profiles()
        except Exception as e:
            tk.Label(
                scroll_frame, text=f"Error loading profiles: {e}",
                fg=ACCENT_RED, bg=BG_DARK,
            ).pack()
            return

        if not profiles:
            tk.Label(
                scroll_frame,
                text="No trained agents found.\nTrain some agents first!",
                font=("Segoe UI", 11), fg=TEXT_DIM, bg=BG_DARK,
            ).pack(pady=20)
            return

        for profile in profiles:
            card = tk.Frame(scroll_frame, bg=BG_MEDIUM, padx=12, pady=8)
            card.pack(fill="x", pady=4)

            # Header: name + style badge
            header = tk.Frame(card, bg=BG_MEDIUM)
            header.pack(fill="x")
            algo_color = ALGO_INFO.get(
                profile.algorithm, {},
            ).get("color", TEXT_PRIMARY)
            tk.Label(
                header, text=profile.display_name,
                font=("Segoe UI", 11, "bold"), fg=algo_color, bg=BG_MEDIUM,
            ).pack(side="left")
            tk.Label(
                header, text=f"  [{profile.play_style}]",
                font=("Segoe UI", 9, "italic"), fg=TEXT_DIM, bg=BG_MEDIUM,
            ).pack(side="left")

            # Stats row
            stats = (
                f"{profile.game_id} | {profile.algorithm.upper()} | "
                f"{profile.total_episodes} eps | Best: {profile.best_reward:.1f}"
            )
            tk.Label(
                card, text=stats,
                font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_MEDIUM,
            ).pack(anchor="w")

            # Dimension bars
            dims_frame = tk.Frame(card, bg=BG_MEDIUM)
            dims_frame.pack(fill="x", pady=(4, 0))

            for dim_name, dim_val in [
                ("Consistency", profile.consistency),
                ("Exploration", profile.exploration),
                ("Speed", profile.speed),
                ("Resilience", profile.resilience),
                ("Peak", profile.peak_performance),
            ]:
                dim_row = tk.Frame(dims_frame, bg=BG_MEDIUM)
                dim_row.pack(fill="x")
                tk.Label(
                    dim_row, text=f"{dim_name:>12}:",
                    font=("Consolas", 8), fg=TEXT_DIM, bg=BG_MEDIUM,
                    width=13, anchor="e",
                ).pack(side="left")
                filled = int(dim_val / 10)
                bar = "\u2588" * filled + "\u2591" * (10 - filled)
                tk.Label(
                    dim_row, text=f" {bar} {dim_val}",
                    font=("Consolas", 8), fg=algo_color, bg=BG_MEDIUM,
                ).pack(side="left")

            # Badges
            if profile.badges:
                badges_text = "  ".join(f"[{b}]" for b in profile.badges[:5])
                tk.Label(
                    card, text=badges_text,
                    font=("Segoe UI", 8), fg=ACCENT_GREEN, bg=BG_MEDIUM,
                ).pack(anchor="w", pady=(2, 0))

    # ── Model Zoo: Export / Import ──────────────────────────────────

    def _export_agent(self):
        """Export a trained agent as a .agent package."""
        model_dir = filedialog.askdirectory(
            title="Select Model Directory to Export",
            initialdir=MODELS_DIR,
        )
        if not model_dir:
            return
        try:
            from src.model_zoo.agent_package import export_agent
            output = export_agent(model_dir)
            self._set_status(f"Agent exported: {os.path.basename(output)}", ACCENT_GREEN)
            messagebox.showinfo("Export Successful", f"Agent saved to:\n{output}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))

    def _import_agent(self):
        """Import a .agent package."""
        agent_path = filedialog.askopenfilename(
            title="Select Agent Package",
            filetypes=[("Agent Package", "*.agent"), ("All Files", "*.*")],
        )
        if not agent_path:
            return
        try:
            from src.model_zoo.agent_package import import_agent
            dest = import_agent(agent_path)
            self._set_status(f"Agent imported to: {dest}", ACCENT_GREEN)
            messagebox.showinfo("Import Successful", f"Agent imported to:\n{dest}")
        except Exception as e:
            messagebox.showerror("Import Failed", str(e))

    def _generate_highlight_reel(self):
        """Generate highlight reel from a recording + markers file."""
        recording = filedialog.askopenfilename(
            title="Select Recording",
            initialdir=RECORDINGS_DIR,
            filetypes=[("Video Files", "*.mp4 *.avi"), ("All Files", "*.*")],
        )
        if not recording:
            return

        # Look for markers sidecar
        base = os.path.splitext(recording)[0]
        markers_path = f"{base}_markers.json"
        if not os.path.isfile(markers_path):
            messagebox.showwarning("No Markers",
                "No milestone markers found for this recording.\n"
                "Train with achievements enabled to generate markers.")
            return

        try:
            from src.recording.highlight_reel import HighlightReel
            reel = HighlightReel()
            output = reel.generate_reel(recording, markers_path)
            self._set_status(f"Highlight reel saved: {os.path.basename(output)}", ACCENT_GREEN)
            messagebox.showinfo("Highlight Reel", f"Reel saved to:\n{output}")
        except Exception as e:
            messagebox.showerror("Reel Error", str(e))

    def _open_tournament(self):
        """Open tournament setup dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Tournament Mode")
        dialog.configure(bg=BG_DARK)
        dialog.geometry("600x550")
        dialog.resizable(True, True)
        dialog.transient(self.root)

        # Header
        tk.Label(
            dialog, text="Tournament Mode",
            font=("Segoe UI", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_DARK,
        ).pack(pady=(15, 5))
        tk.Label(
            dialog, text="Pit agents against each other in board games",
            font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_DARK,
        ).pack()

        # Settings frame
        settings = tk.Frame(dialog, bg=BG_DARK, padx=20)
        settings.pack(fill="x", pady=(10, 0))

        # Game selector
        game_row = tk.Frame(settings, bg=BG_DARK)
        game_row.pack(fill="x", pady=4)
        tk.Label(game_row, text="Game:", font=("Segoe UI", 10),
                 fg=TEXT_PRIMARY, bg=BG_DARK, width=12, anchor="w").pack(side="left")
        tourney_game = tk.StringVar(value="tictactoe")
        ttk.Combobox(
            game_row, textvariable=tourney_game,
            values=["tictactoe", "connect4", "checkers", "chess"],
            state="readonly", width=15,
        ).pack(side="left")

        # Games per match
        gpm_row = tk.Frame(settings, bg=BG_DARK)
        gpm_row.pack(fill="x", pady=4)
        tk.Label(gpm_row, text="Games/Match:", font=("Segoe UI", 10),
                 fg=TEXT_PRIMARY, bg=BG_DARK, width=12, anchor="w").pack(side="left")
        gpm_var = tk.StringVar(value="10")
        ttk.Combobox(
            gpm_row, textvariable=gpm_var,
            values=["2", "4", "6", "10", "20"],
            state="readonly", width=6,
        ).pack(side="left")

        # Separator
        tk.Frame(settings, bg=BORDER_COLOR, height=1).pack(fill="x", pady=8)

        # Participants section
        tk.Label(settings, text="PARTICIPANTS", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w")

        # Participant list
        participants_list = []

        list_frame = tk.Frame(settings, bg=BG_MEDIUM, padx=5, pady=5)
        list_frame.pack(fill="x", pady=4)

        participants_display = tk.Frame(list_frame, bg=BG_MEDIUM)
        participants_display.pack(fill="x")

        def refresh_participants():
            for w in participants_display.winfo_children():
                w.destroy()
            if not participants_list:
                tk.Label(participants_display, text="  No participants added yet",
                         font=("Segoe UI", 9, "italic"), fg=TEXT_DIM, bg=BG_MEDIUM).pack(anchor="w")
            for i, p in enumerate(participants_list):
                row = tk.Frame(participants_display, bg=BG_MEDIUM)
                row.pack(fill="x")
                tk.Label(row, text=f"  {i+1}. {p['name']} ({p['type']})",
                         font=("Segoe UI", 9), fg=TEXT_PRIMARY, bg=BG_MEDIUM).pack(side="left")
                tk.Button(row, text="\u2715", font=("Segoe UI", 8),
                          bg=BG_MEDIUM, fg=ACCENT_RED, relief="flat", cursor="hand2",
                          command=lambda idx=i: (participants_list.pop(idx), refresh_participants()),
                          ).pack(side="right", padx=5)

        # Add participant controls
        add_row = tk.Frame(settings, bg=BG_DARK)
        add_row.pack(fill="x", pady=4)

        name_var = tk.StringVar(value="")
        type_var = tk.StringVar(value="random")

        tk.Label(add_row, text="Name:", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(side="left")
        name_entry = tk.Entry(add_row, textvariable=name_var, width=12,
                              font=("Segoe UI", 9), bg=BG_MEDIUM, fg=TEXT_PRIMARY,
                              insertbackground=TEXT_PRIMARY, relief="flat")
        name_entry.pack(side="left", padx=(3, 8))

        tk.Label(add_row, text="Type:", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(side="left")
        ttk.Combobox(
            add_row, textvariable=type_var,
            values=["random", "minimax-easy", "minimax-medium", "minimax-hard"],
            state="readonly", width=14,
        ).pack(side="left", padx=(3, 8))

        def add_participant():
            name = name_var.get().strip()
            ptype = type_var.get()
            if not name:
                name = f"{ptype.replace('-', ' ').title()} {len(participants_list)+1}"
            participants_list.append({'name': name, 'type': ptype})
            name_var.set("")
            refresh_participants()

        tk.Button(add_row, text="+ Add", font=("Segoe UI", 9, "bold"),
                  bg=ACCENT_GREEN, fg=BG_DARK, relief="flat", cursor="hand2",
                  padx=8, command=add_participant).pack(side="left")

        # Quick-add presets
        preset_row = tk.Frame(settings, bg=BG_DARK)
        preset_row.pack(fill="x", pady=2)
        tk.Label(preset_row, text="Quick:", font=("Segoe UI", 8),
                 fg=TEXT_DIM, bg=BG_DARK).pack(side="left")

        def quick_add_3():
            participants_list.clear()
            participants_list.extend([
                {'name': 'Random Bot', 'type': 'random'},
                {'name': 'Minimax Easy', 'type': 'minimax-easy'},
                {'name': 'Minimax Hard', 'type': 'minimax-hard'},
            ])
            refresh_participants()

        def quick_add_4():
            participants_list.clear()
            participants_list.extend([
                {'name': 'Random Bot', 'type': 'random'},
                {'name': 'Minimax Easy', 'type': 'minimax-easy'},
                {'name': 'Minimax Medium', 'type': 'minimax-medium'},
                {'name': 'Minimax Hard', 'type': 'minimax-hard'},
            ])
            refresh_participants()

        qbtn = dict(font=("Segoe UI", 8), bg=BG_MEDIUM, fg=TEXT_DIM,
                    relief="flat", cursor="hand2", padx=6)
        tk.Button(preset_row, text="3 Bots", command=quick_add_3, **qbtn).pack(side="left", padx=3)
        tk.Button(preset_row, text="4 Bots", command=quick_add_4, **qbtn).pack(side="left", padx=3)

        refresh_participants()

        # Separator
        tk.Frame(settings, bg=BORDER_COLOR, height=1).pack(fill="x", pady=8)

        # Results area
        results_frame = tk.Frame(settings, bg=BG_DARK)
        results_frame.pack(fill="both", expand=True)

        results_text = tk.Text(
            results_frame, height=8, font=("Consolas", 9),
            bg=BG_MEDIUM, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
            relief="flat", state="disabled",
        )
        results_text.pack(fill="both", expand=True, pady=4)

        # Start button
        def run_tournament():
            if len(participants_list) < 2:
                results_text.config(state="normal")
                results_text.delete("1.0", "end")
                results_text.insert("end", "Need at least 2 participants!")
                results_text.config(state="disabled")
                return

            # Map types to TournamentParticipant format
            type_map = {
                'random': ('random', {}),
                'minimax-easy': ('minimax', {'depth': 1}),
                'minimax-medium': ('minimax', {'depth': 3}),
                'minimax-hard': ('minimax', {'depth': 5}),
            }

            try:
                from src.tournament import TournamentEngine, TournamentParticipant
                parts = []
                for p in participants_list:
                    ptype, pconfig = type_map.get(p['type'], ('random', {}))
                    parts.append(TournamentParticipant(
                        name=p['name'],
                        participant_type=ptype,
                        config=pconfig,
                    ))

                engine = TournamentEngine(
                    game_id=tourney_game.get(),
                    participants=parts,
                    games_per_match=int(gpm_var.get()),
                    format='round_robin',
                )

                results_text.config(state="normal")
                results_text.delete("1.0", "end")
                results_text.insert("end", "Running tournament...\n\n")
                results_text.config(state="disabled")
                dialog.update()

                def on_progress(match_num, total, result):
                    results_text.config(state="normal")
                    results_text.insert("end",
                        f"  Match {match_num}/{total}: {result.player1} vs {result.player2} — "
                        f"{result.wins_p1}-{result.wins_p2} (draws: {result.draws})\n")
                    results_text.config(state="disabled")
                    dialog.update()

                result = engine.run_tournament(progress_callback=on_progress)

                # Show standings
                results_text.config(state="normal")
                results_text.insert("end", "\n" + "=" * 50 + "\n")
                results_text.insert("end", f"  STANDINGS — {tourney_game.get().upper()}\n")
                results_text.insert("end", "=" * 50 + "\n\n")
                results_text.insert("end", f"  {'Name':<20} {'W':>3} {'L':>3} {'D':>3} {'Pts':>5}\n")
                results_text.insert("end", "  " + "-" * 36 + "\n")
                for s in result.standings:
                    results_text.insert("end",
                        f"  {s['name']:<20} {s['wins']:>3} {s['losses']:>3} {s['draws']:>3} {s['points']:>5.0f}\n")
                results_text.insert("end", f"\n  Champion: {result.champion}\n")
                results_text.config(state="disabled")
                results_text.see("end")

            except Exception as e:
                results_text.config(state="normal")
                results_text.delete("1.0", "end")
                results_text.insert("end", f"Tournament error: {e}")
                results_text.config(state="disabled")

        tk.Button(
            settings, text="Start Tournament",
            font=("Segoe UI", 11, "bold"),
            bg=ACCENT_BLUE, fg=BG_DARK,
            activebackground="#3d8ee6", activeforeground=BG_DARK,
            relief="flat", cursor="hand2", pady=8,
            command=run_tournament,
        ).pack(fill="x", pady=(4, 10))

    def _open_comparison(self):
        """Open the cross-game training comparison panel."""
        try:
            from src.visualization.comparison_panel import ComparisonPanel
            ComparisonPanel(
                logs_dir=os.path.join(PROJECT_ROOT, 'logs'),
                parent=self.root,
            )
        except ImportError as e:
            self._set_status(f"Comparison panel error: {e}", ACCENT_RED)

    def _add_game_wizard(self):
        """Open a dialog to create a new user game from the template."""
        raw_name = simpledialog.askstring(
            "Add Game",
            "Enter a name for your new game:\n"
            "(e.g. 'My Platformer', 'Space Invaders')",
            parent=self.root,
        )
        if not raw_name or not raw_name.strip():
            return

        # Sanitize: lowercase, underscores, no special chars
        sanitized = re.sub(r'[^a-z0-9_]', '_', raw_name.strip().lower())
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        if not sanitized:
            messagebox.showerror("Invalid Name", "Could not create a valid game name.")
            return

        template_dir = os.path.join(PROJECT_ROOT, "games", "user", "_template")
        dest_dir = os.path.join(PROJECT_ROOT, "games", "user", sanitized)

        if os.path.exists(dest_dir):
            messagebox.showwarning(
                "Already Exists",
                f"A game called '{sanitized}' already exists at:\n{dest_dir}",
            )
            return

        # Copy template to new directory
        try:
            shutil.copytree(template_dir, dest_dir)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create game directory:\n{e}")
            return

        # Open the new folder in file explorer
        try:
            if sys.platform == "win32":
                os.startfile(dest_dir)
            elif sys.platform == "darwin":
                subprocess.run(["open", dest_dir])
            else:
                subprocess.run(["xdg-open", dest_dir])
        except Exception:
            pass

        messagebox.showinfo(
            "Game Created",
            f"New game '{sanitized}' created!\n\n"
            f"Location: {dest_dir}\n\n"
            "Next steps:\n"
            "1. Edit game.py to implement your game logic\n"
            "2. Edit adapter.py to configure the adapter\n"
            "3. Update game_id to match your folder name\n"
            "4. Restart the launcher to see your game",
        )

    def _import_rom(self):
        """Open a directory picker and run retro.import to register ROMs."""
        rom_dir = filedialog.askdirectory(
            title="Select Directory Containing ROM Files",
            initialdir=os.path.expanduser("~"),
        )
        if not rom_dir:
            return

        self.rom_status.configure(
            text="Importing ROMs...", fg=ACCENT_BLUE,
        )
        self.root.update_idletasks()

        try:
            result = subprocess.run(
                [VENV_PYTHON, "-m", "retro.import", rom_dir],
                capture_output=True, text=True, timeout=30,
                cwd=PROJECT_ROOT,
            )
            if result.returncode == 0:
                output = result.stdout.strip() or "Import complete."
                self.rom_status.configure(text=output, fg=ACCENT_GREEN)
                self.game_registry.discover()
                self.available_games = self.game_registry.list_games()
                game_names = [
                    f"{g.name} ({g.game_id})" for g in self.available_games
                ]
                self.game_combo.configure(values=game_names)
            else:
                err = result.stderr.strip() or "Import failed."
                if "No module named" in err:
                    err = (
                        "stable-retro not installed. "
                        "Run: pip install stable-retro"
                    )
                self.rom_status.configure(text=err, fg=ACCENT_RED)
        except subprocess.TimeoutExpired:
            self.rom_status.configure(
                text="Import timed out.", fg=ACCENT_RED,
            )
        except FileNotFoundError:
            self.rom_status.configure(
                text="Python not found. Check venv.", fg=ACCENT_RED,
            )

    # ===================================================================
    # Actions
    # ===================================================================

    def _on_start_stop(self):
        """Handle the START / STOP button click."""
        if self.process is not None:
            self._stop_training()
        else:
            self._start_training()

    def _restart_training(self):
        """Delete existing checkpoint and start training from scratch."""
        from tkinter import messagebox

        algo = self.selected_algo.get()
        game_text = self.game_combo.get()
        game_id = game_text.rsplit("(", 1)[-1].rstrip(")").strip() if "(" in game_text else "mario"

        model_dir = os.path.join('models', algo)

        # Check if any checkpoint files exist
        checkpoint_files = []
        for pattern in ['final.zip', 'final.pt', 'final_best_genome.pkl', 'metadata.json']:
            path = os.path.join(model_dir, pattern)
            if os.path.isfile(path):
                checkpoint_files.append(path)

        if not checkpoint_files:
            messagebox.showinfo(
                "No Existing Model",
                f"No existing {algo.upper()} model found.\n"
                "Training will start fresh."
            )
            self._start_training()
            return

        # Confirm deletion
        result = messagebox.askyesno(
            "Restart Training?",
            f"This will DELETE the existing {algo.upper()} model for {game_id}:\n\n"
            + "\n".join(f"  \u2022 {os.path.basename(f)}" for f in checkpoint_files)
            + "\n\nTraining will start completely from scratch.\n"
            "This cannot be undone. Continue?",
            icon='warning'
        )

        if not result:
            self._set_status("Restart cancelled.", TEXT_DIM)
            return

        # Delete checkpoint files
        for f in checkpoint_files:
            try:
                os.remove(f)
            except OSError as e:
                print(f"  Warning: Could not delete {f}: {e}")

        self._set_status(f"Old {algo.upper()} model deleted. Starting fresh...", ACCENT_GREEN)
        self._start_training()

    def _start_training(self):
        """Build the CLI command and launch main.py as a subprocess."""
        algo = self.selected_algo.get()
        world = self.world_var.get()
        stage = self.stage_var.get()
        duration = self.duration_var.get().strip()
        model_path = self.model_path_var.get()
        is_eval = self.eval_var.get()

        # Validation: eval mode requires a loaded model
        if is_eval and not model_path:
            self._set_status("Please load a model for evaluation.", ACCENT_RED)
            return

        # Extract game_id from combo text: "Snake (snake)" -> "snake"
        game_text = self.game_combo.get()
        game_id = game_text.rsplit("(", 1)[-1].rstrip(")").strip() if "(" in game_text else "mario"

        # Check if the selected game has all dependencies installed
        adapter = self._get_selected_adapter()
        if adapter and hasattr(adapter, 'is_available') and not adapter.is_available():
            game_display = game_text.split("(")[0].strip() if "(" in game_text else game_id
            self._set_status(
                f"{game_display} is not available. Check required dependencies.",
                ACCENT_RED,
            )
            # Show a helpful message box with install instructions
            try:
                from tkinter import messagebox
                if game_id == "chess":
                    messagebox.showwarning(
                        "Missing Dependency",
                        "The Chess game requires python-chess.\n\n"
                        "Install it with:\n  pip install python-chess",
                    )
                elif game_id in ("pokemon", "sonic"):
                    messagebox.showwarning(
                        "Missing Dependency",
                        f"The {game_display} game requires stable-retro.\n\n"
                        "Install it with:\n  pip install stable-retro\n\n"
                        "You also need the game ROM imported.",
                    )
                else:
                    messagebox.showwarning(
                        "Missing Dependency",
                        f"{game_display} is not available.\n"
                        "Check the game's documentation for required packages.",
                    )
            except Exception:
                pass
            return

        # Build the command
        cmd = [
            VENV_PYTHON,
            MAIN_SCRIPT,
            "--algorithm", algo,
            "--game", game_id,
        ]

        # Only pass world/stage for Mario (has levels)
        if game_id == "mario":
            cmd.extend(["--world", world, "--stage", stage])

        if self.visualize_var.get():
            cmd.append("--visualize")
        if self.record_var.get():
            cmd.append("--record")
        if is_eval:
            cmd.append("--eval")
        if self.next_stage_var.get() and not is_eval:
            cmd.append("--next-stage")
        if self.curriculum_var.get() and not is_eval:
            cmd.append("--curriculum")

        num_envs = self.num_envs_var.get()
        if num_envs and int(num_envs) > 1:
            cmd.extend(["--num-envs", num_envs])

        if model_path:
            cmd.extend(["--load", model_path])

        if duration:
            try:
                val = int(duration)
                if val <= 0:
                    raise ValueError
                if algo in ("ppo", "a2c", "dt"):
                    val = val * 1000
                cmd.extend(["--episodes", str(val)])
            except ValueError:
                self._set_status("Invalid number for duration.", ACCENT_RED)
                return

        if self.stream_var.get():
            twitch_key = self.twitch_key_var.get().strip()
            youtube_key = self.youtube_key_var.get().strip()
            if not twitch_key and not youtube_key:
                self._set_status(
                    "Streaming enabled but no keys provided.", ACCENT_RED
                )
                return
            if twitch_key:
                cmd.extend(["--stream-twitch", twitch_key])
            if youtube_key:
                cmd.extend(["--stream-youtube", youtube_key])

        if self.music_var.get():
            music_dir = self.music_dir_var.get().strip()
            if not music_dir or not os.path.isdir(music_dir):
                print(f'[Music] Directory not found: {music_dir!r}, skipping music')
            else:
                cmd.extend(["--music", music_dir])

        device = self.device_var.get()
        if device and device != 'auto':
            cmd.extend(["--device", device])

        # Opponent config for board games
        opp_type = self.opponent_var.get()
        if opp_type and opp_type != "auto":
            # Map preset difficulties to minimax with fixed depths
            opp_map = {
                'minimax-easy': ('minimax', '1'),
                'minimax-medium': ('minimax', '3'),
                'minimax-hard': ('minimax', '5'),
            }
            if opp_type in opp_map:
                opp_algo, opp_depth = opp_map[opp_type]
                cmd.extend(["--opponent", opp_algo,
                            "--opponent-depth", opp_depth])
            else:
                cmd.extend(["--opponent", opp_type])
                if opp_type == "model":
                    opp_model = self.opponent_model_var.get()
                    if opp_model:
                        cmd.extend(["--opponent-model", opp_model])

        game_opts = self._get_game_options()
        if game_opts:
            cmd.append("--game-opts")
            for key, value in game_opts.items():
                cmd.append(f"{key}={value}")

        # Launch the subprocess
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        except Exception as e:
            self._set_status(f"Failed to start: {e}", ACCENT_RED)
            return

        mode = "Evaluating" if is_eval else "Training"
        # Game-aware status message
        game_display = game_text.split("(")[0].strip() if "(" in game_text else game_id
        if game_id == "mario":
            status_msg = f"{mode} {algo.upper()} on {game_display} World {world}-{stage}..."
        else:
            status_msg = f"{mode} {algo.upper()} on {game_display}..."
        self._set_status(status_msg, ALGO_INFO[algo]["color"])
        self._update_start_button_text()
        self._disable_controls(True)

        # Save settings before training (captures current state)
        self._save_settings()

        self.root.after(500, self._check_process)

    def _stop_training(self):
        """Stop the running training subprocess with graduated shutdown.

        Graduated approach:
        1. Send CTRL_BREAK_EVENT (maps to SIGBREAK — the trainer has a handler)
        2. Poll for 10 seconds (model save + stream stop can take time)
        3. Escalate to terminate()
        4. After 3 more seconds → kill entire process tree
        """
        if self.process is None:
            return

        self._set_status("Stopping... (saving model)", ACCENT_ORANGE)

        # Step 1: Send CTRL_BREAK_EVENT (triggers SIGBREAK handler in trainer)
        # NOTE: CTRL_C_EVENT is DISABLED for processes created with
        # CREATE_NEW_PROCESS_GROUP. Only CTRL_BREAK_EVENT can reach them.
        try:
            os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
        except (OSError, PermissionError):
            pass

        # Step 2: Poll for up to 10 seconds (20 x 500ms)
        # Needs longer than before because model save + ffmpeg stream
        # shutdown can take several seconds.
        self._graduated_shutdown_poll(attempts=20)

    def _graduated_shutdown_poll(self, attempts):
        """Poll subprocess during graduated shutdown."""
        if self.process is None:
            return

        retcode = self.process.poll()
        if retcode is not None:
            # Process exited cleanly
            self._on_process_finished(retcode)
            return

        if attempts > 0:
            # Still running — check again in 500ms
            self.root.after(
                500,
                lambda: self._graduated_shutdown_poll(attempts - 1),
            )
            return

        # Step 3: Escalate to terminate()
        self._set_status("Force stopping...", ACCENT_RED)
        try:
            self.process.terminate()
        except OSError:
            pass

        # Step 4: Wait 3 more seconds then kill entire process tree
        self.root.after(3000, self._force_kill_if_running)

    def _force_kill_if_running(self):
        """Final escalation: kill the entire process tree.

        Uses taskkill /T on Windows to kill the process and all its
        children (including ffmpeg). This prevents orphan processes
        that keep streaming after training stops.
        """
        if self.process is None:
            return

        retcode = self.process.poll()
        if retcode is not None:
            self._on_process_finished(retcode)
            return

        # Kill entire process tree (catches child ffmpeg processes)
        try:
            if sys.platform == 'win32':
                subprocess.call(
                    ['taskkill', '/F', '/T', '/PID', str(self.process.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                self.process.kill()
        except OSError:
            pass

        self.process = None
        self._disable_controls(False)
        self._update_start_button_text()
        self._set_status("Process killed.", ACCENT_RED)

    def _check_process(self):
        """Poll the subprocess to see if it's still running."""
        if self.process is None:
            return

        retcode = self.process.poll()
        if retcode is None:
            self.root.after(500, self._check_process)
        else:
            self._on_process_finished(retcode)

    def _on_process_finished(self, retcode):
        """Handle subprocess completion."""
        self.process = None
        self._disable_controls(False)
        self._update_start_button_text()

        if retcode == 0:
            self._set_status(
                "Finished. Check models/ for saved checkpoints.",
                ACCENT_GREEN,
            )
        else:
            self._set_status(
                f"Process exited with code {retcode}.",
                ACCENT_ORANGE,
            )

    def _set_status(self, text, color=TEXT_DIM):
        """Update the status bar text and color."""
        self.status_label.configure(text=text, fg=color)

    def _disable_controls(self, disabled):
        """Enable/disable controls while training is running."""
        if disabled:
            for btn in self.algo_buttons.values():
                btn.configure(state="disabled")
        else:
            self._update_algo_buttons()

    def _open_folder(self, path):
        """Open a folder in the file manager."""
        os.makedirs(path, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])

    def _on_close(self):
        """Handle window close — stop training if running, save settings."""
        # Always save settings on close
        self._save_settings()

        if self.process is None:
            self.root.destroy()
            return

        # Graduated shutdown on window close
        self._set_status("Closing... saving model", ACCENT_ORANGE)

        # Send CTRL_BREAK_EVENT (not CTRL_C_EVENT — that is disabled
        # for processes created with CREATE_NEW_PROCESS_GROUP)
        try:
            os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
        except (OSError, PermissionError):
            pass

        # Poll for up to 10 seconds (model save + stream stop)
        self._on_close_poll(attempts=20)

    def _on_close_poll(self, attempts):
        """Poll during window-close shutdown."""
        if self.process is None or self.process.poll() is not None:
            self.process = None
            self.root.destroy()
            return

        if attempts > 0:
            self.root.after(500, lambda: self._on_close_poll(attempts - 1))
            return

        # Escalate: terminate
        try:
            self.process.terminate()
        except OSError:
            pass

        # Wait 2 more seconds then force destroy
        self.root.after(2000, self._on_close_force)

    def _on_close_force(self):
        """Final window close — kill entire process tree and destroy."""
        if self.process is not None:
            retcode = self.process.poll()
            if retcode is None:
                try:
                    if sys.platform == 'win32':
                        subprocess.call(
                            ['taskkill', '/F', '/T', '/PID',
                             str(self.process.pid)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    else:
                        self.process.kill()
                except OSError:
                    pass
            self.process = None
        self.root.destroy()

    # ===================================================================
    # Run
    # ===================================================================

    def run(self):
        """Start the tkinter main loop."""
        self.root.mainloop()

# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    app = MarioLauncher()
    app.run()
