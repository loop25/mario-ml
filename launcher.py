"""
Super Mario Bros ML — Launcher GUI.

A simple, clean graphical launcher for the Mario ML training system.
Instead of typing terminal commands with --flags, just click buttons.

This launcher builds the appropriate command and runs main.py as a
subprocess, so the training/pygame dashboard runs independently.

Usage:
    1. Activate the virtual environment:
       C:\\Projects\\mario-ml\\venv\\Scripts\\activate
    2. Run the launcher:
       python launcher.py
    3. Pick your algorithm, options, and click START.

Requirements:
    - Python 3.11 with tkinter (included by default)
    - All project dependencies installed in the venv
"""

import os
import sys
import signal
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog

from games.registry import GameRegistry

# ---------------------------------------------------------------------------
# Path setup — figure out where the project lives
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "main.py")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "recordings")
ROMS_DIR = os.path.join(PROJECT_ROOT, "roms")

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
        "badge": None,  # No special badge
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

class MarioLauncher:
    """
    Main launcher window.

    Creates a tkinter GUI with controls for algorithm selection,
    world/stage, training options, and a START/STOP button.
    Spawns main.py as a subprocess when START is clicked.
    """

    def __init__(self):
        # ---------------------------------------------------------------
        # Window setup
        # ---------------------------------------------------------------
        self.root = tk.Tk()
        self.root.title("Game AI Training Studio")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        # Track the training subprocess (None when idle)
        self.process = None

        # State variables
        self.selected_algo = tk.StringVar(value="neat")
        self.world_var = tk.StringVar(value="1")
        self.stage_var = tk.StringVar(value="1")
        self.duration_var = tk.StringVar(value="")
        self.visualize_var = tk.BooleanVar(value=True)
        self.record_var = tk.BooleanVar(value=False)
        self.eval_var = tk.BooleanVar(value=False)
        self.next_stage_var = tk.BooleanVar(value=False)
        self.curriculum_var = tk.BooleanVar(value=False)
        self.num_envs_var = tk.StringVar(value="1")
        self.model_path_var = tk.StringVar(value="")

        self.device_var = tk.StringVar(value="auto")

        # Streaming
        self.twitch_key_var = tk.StringVar()
        self.youtube_key_var = tk.StringVar()
        self.stream_var = tk.BooleanVar(value=False)

        # Music
        self.music_var = tk.BooleanVar(value=True)
        self.music_dir_var = tk.StringVar(
            value=os.path.join(PROJECT_ROOT, 'assets', 'music')
        )

        # Discover games
        self.game_registry = GameRegistry()
        self.game_registry.discover()
        self.available_games = self.game_registry.list_games()
        self.game_var = tk.StringVar(value='mario')

        # Build UI — tabbed layout
        self._build_header()
        self._build_notebook()
        self._build_start_area()

        # Center the window on screen
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ===================================================================
    # GUI Building Methods
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

    def _build_notebook(self):
        """Build the ttk.Notebook with 3 tabs."""
        style = ttk.Style()
        style.configure(
            'Dark.TNotebook',
            background=BG_DARK,
            borderwidth=0,
            tabmargins=[0, 0, 0, 0],
        )
        style.configure(
            'Dark.TNotebook.Tab',
            background=BG_MEDIUM,
            foreground=TEXT_DIM,
            padding=[14, 7],
            font=("Segoe UI", 10),
        )
        style.map(
            'Dark.TNotebook.Tab',
            background=[('selected', BG_LIGHT)],
            foreground=[('selected', TEXT_PRIMARY)],
        )

        self.notebook = ttk.Notebook(self.root, style='Dark.TNotebook')
        self.notebook.pack(fill="both", expand=True)

        tab_train = tk.Frame(self.notebook, bg=BG_DARK)
        tab_settings = tk.Frame(self.notebook, bg=BG_DARK)
        tab_stream = tk.Frame(self.notebook, bg=BG_DARK)

        self.notebook.add(tab_train,    text='  ▶  Train  ')
        self.notebook.add(tab_settings, text='  ⚙  Settings  ')
        self.notebook.add(tab_stream,   text='  📡  More  ')

        self._build_train_tab(tab_train)
        self._build_settings_tab(tab_settings)
        self._build_stream_tab(tab_stream)

    def _build_train_tab(self, parent):
        """Tab 1: Game selector, algorithm picker, duration, quick options."""
        pad = {'padx': 20}

        # ── Game selector ──────────────────────────────────────────────
        game_sec = tk.Frame(parent, bg=BG_DARK, pady=12, **pad)
        game_sec.pack(fill="x")

        tk.Label(
            game_sec, text="GAME",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w")

        game_names = [f"{g.name} ({g.game_id})" for g in self.available_games]
        if not game_names:
            game_names = ["Super Mario Bros (mario)"]

        self.game_combo = ttk.Combobox(
            game_sec, textvariable=self.game_var,
            values=game_names, state="readonly",
            font=("Segoe UI", 11),
        )
        self.game_combo.pack(fill="x", pady=(4, 0))
        self.game_combo.set(game_names[0])
        self.game_combo.bind("<<ComboboxSelected>>", self._on_game_changed)

        # Dynamic game-specific options container
        self.game_opts_frame = tk.Frame(game_sec, bg=BG_DARK)
        self.game_opts_frame.pack(fill="x", pady=(5, 0))
        self.game_opt_widgets = {}
        self._rebuild_game_options()

        # ── Separator ──────────────────────────────────────────────────
        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(fill="x", padx=20)

        # ── Algorithm selector ─────────────────────────────────────────
        algo_sec = tk.Frame(parent, bg=BG_DARK, pady=12, **pad)
        algo_sec.pack(fill="x")

        hdr = tk.Frame(algo_sec, bg=BG_DARK)
        hdr.pack(fill="x")
        tk.Label(
            hdr, text="ALGORITHM",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(side="left")
        tk.Label(
            hdr, text="(hover for details)",
            font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(side="left", padx=(8, 0))

        # Top row: 5 single-game algorithms
        btn_frame = tk.Frame(algo_sec, bg=BG_DARK)
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
        dt_frame = tk.Frame(algo_sec, bg=BG_DARK)
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
            algo_sec, text="",
            font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_DARK,
            wraplength=480, justify="left",
        )
        self.algo_desc_label.pack(anchor="w", pady=(4, 0))
        self._update_algo_desc()
        self._update_algo_buttons()

        # ── Separator ──────────────────────────────────────────────────
        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(fill="x", padx=20)

        # ── Duration + quick checkboxes (two-column row) ───────────────
        bottom = tk.Frame(parent, bg=BG_DARK, pady=12, **pad)
        bottom.pack(fill="x")

        left_col = tk.Frame(bottom, bg=BG_DARK)
        left_col.pack(side="left", fill="x", expand=True)

        right_col = tk.Frame(bottom, bg=BG_DARK, padx=20)
        right_col.pack(side="left", anchor="n")

        # Duration entry
        self.duration_label = tk.Label(
            left_col, text="Generations",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        )
        self.duration_label.pack(anchor="w")

        dur_row = tk.Frame(left_col, bg=BG_DARK)
        dur_row.pack(fill="x", pady=(3, 0))

        self.duration_entry = tk.Entry(
            dur_row, textvariable=self.duration_var,
            font=("Segoe UI", 11), bg=BG_LIGHT, fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY, relief="flat", width=12,
        )
        self.duration_entry.pack(side="left")
        tk.Label(
            dur_row, text="(blank = default)",
            font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(side="left", padx=(8, 0))

        # Quick option checkboxes
        tk.Label(
            right_col, text="OPTIONS",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w")

        _cb = dict(
            font=("Segoe UI", 10), fg=TEXT_PRIMARY, bg=BG_DARK,
            selectcolor=BG_MEDIUM, activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY, cursor="hand2",
        )
        tk.Checkbutton(
            right_col, text="Show Live Dashboard",
            variable=self.visualize_var, **_cb,
        ).pack(anchor="w")
        tk.Checkbutton(
            right_col, text="Record Video",
            variable=self.record_var, **_cb,
        ).pack(anchor="w")

        self._update_duration_label()

    def _build_settings_tab(self, parent):
        """Tab 2: World/Stage, training options, compute, model loader."""
        pad = {'padx': 20}

        # ── World / Stage ──────────────────────────────────────────────
        ws_sec = tk.Frame(parent, bg=BG_DARK, pady=12, **pad)
        ws_sec.pack(fill="x")

        tk.Label(
            ws_sec, text="WORLD & STAGE  (Mario / multi-level games)",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w")

        ws_row = tk.Frame(ws_sec, bg=BG_DARK)
        ws_row.pack(fill="x", pady=(4, 0))

        world_col = tk.Frame(ws_row, bg=BG_DARK)
        world_col.pack(side="left", padx=(0, 20))
        tk.Label(world_col, text="World", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w")
        ttk.Combobox(
            world_col, textvariable=self.world_var,
            values=[str(i) for i in range(1, 9)],
            state="readonly", width=6,
        ).pack(anchor="w", pady=(2, 0))

        stage_col = tk.Frame(ws_row, bg=BG_DARK)
        stage_col.pack(side="left")
        tk.Label(stage_col, text="Stage", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w")
        ttk.Combobox(
            stage_col, textvariable=self.stage_var,
            values=[str(i) for i in range(1, 5)],
            state="readonly", width=6,
        ).pack(anchor="w", pady=(2, 0))

        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(fill="x", padx=20)

        # ── Training options ───────────────────────────────────────────
        opt_sec = tk.Frame(parent, bg=BG_DARK, pady=12, **pad)
        opt_sec.pack(fill="x")

        tk.Label(
            opt_sec, text="TRAINING OPTIONS",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w")

        _cb = dict(
            font=("Segoe UI", 10), fg=TEXT_PRIMARY, bg=BG_DARK,
            selectcolor=BG_MEDIUM, activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY, cursor="hand2",
        )
        tk.Checkbutton(
            opt_sec, text="Evaluation Mode  (watch AI play, no training)",
            variable=self.eval_var, command=self._on_eval_toggle, **_cb,
        ).pack(anchor="w", pady=(4, 0))
        tk.Checkbutton(
            opt_sec, text="Auto-Advance Stages  (transfer weights to next stage)",
            variable=self.next_stage_var, **_cb,
        ).pack(anchor="w")
        tk.Checkbutton(
            opt_sec, text="Whole Game  (curriculum learning across all 32 stages)",
            variable=self.curriculum_var, **_cb,
        ).pack(anchor="w")

        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(fill="x", padx=20)

        # ── Compute ────────────────────────────────────────────────────
        compute_sec = tk.Frame(parent, bg=BG_DARK, pady=12, **pad)
        compute_sec.pack(fill="x")

        tk.Label(
            compute_sec, text="COMPUTE",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w")

        compute_row = tk.Frame(compute_sec, bg=BG_DARK)
        compute_row.pack(fill="x", pady=(4, 0))

        env_col = tk.Frame(compute_row, bg=BG_DARK)
        env_col.pack(side="left", padx=(0, 30))
        tk.Label(env_col, text="Parallel Envs", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w")
        ttk.Combobox(
            env_col, textvariable=self.num_envs_var,
            values=["1", "2", "4", "8", "16"],
            state="readonly", width=6,
        ).pack(anchor="w", pady=(2, 0))

        dev_col = tk.Frame(compute_row, bg=BG_DARK)
        dev_col.pack(side="left")
        tk.Label(dev_col, text="Device  (auto = CUDA→MPS→CPU)", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w")
        ttk.Combobox(
            dev_col, textvariable=self.device_var,
            values=["auto", "cuda", "mps", "cpu"],
            state="readonly", width=8,
        ).pack(anchor="w", pady=(2, 0))

        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(fill="x", padx=20)

        # ── Model loader ───────────────────────────────────────────────
        model_sec = tk.Frame(parent, bg=BG_DARK, pady=12, **pad)
        model_sec.pack(fill="x")

        tk.Label(
            model_sec, text="LOAD MODEL",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w")

        model_row = tk.Frame(model_sec, bg=BG_DARK)
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

    def _build_stream_tab(self, parent):
        """Tab 3: Streaming keys, music, ROM import, folder shortcuts."""
        pad = {'padx': 20}

        # ── Live streaming ─────────────────────────────────────────────
        stream_sec = tk.Frame(parent, bg=BG_DARK, pady=12, **pad)
        stream_sec.pack(fill="x")

        tk.Label(
            stream_sec, text="LIVE STREAMING",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w")

        tk.Checkbutton(
            stream_sec, text="Enable Live Streaming  (Twitch / YouTube)",
            variable=self.stream_var,
            font=("Segoe UI", 10), fg=ACCENT_RED, bg=BG_DARK,
            selectcolor=BG_MEDIUM, activebackground=BG_DARK,
            activeforeground=ACCENT_RED, cursor="hand2",
        ).pack(anchor="w", pady=(4, 0))

        tk.Label(stream_sec, text="Twitch Stream Key:", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w", pady=(6, 0))
        tk.Entry(
            stream_sec, textvariable=self.twitch_key_var, show="*",
            font=("Segoe UI", 10), bg=BG_MEDIUM, fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY, relief="flat",
        ).pack(fill="x", pady=2)

        tk.Label(stream_sec, text="YouTube Stream Key:", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w", pady=(5, 0))
        tk.Entry(
            stream_sec, textvariable=self.youtube_key_var, show="*",
            font=("Segoe UI", 10), bg=BG_MEDIUM, fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY, relief="flat",
        ).pack(fill="x", pady=2)

        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(fill="x", padx=20)

        # ── Music ──────────────────────────────────────────────────────
        music_sec = tk.Frame(parent, bg=BG_DARK, pady=12, **pad)
        music_sec.pack(fill="x")

        tk.Label(
            music_sec, text="MUSIC",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w")
        tk.Checkbutton(
            music_sec, text="Play Background Music during training",
            variable=self.music_var,
            font=("Segoe UI", 10), fg=TEXT_PRIMARY, bg=BG_DARK,
            selectcolor=BG_MEDIUM, activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY, cursor="hand2",
        ).pack(anchor="w", pady=(4, 0))

        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(fill="x", padx=20)

        # ── ROM import ─────────────────────────────────────────────────
        rom_sec = tk.Frame(parent, bg=BG_DARK, pady=12, **pad)
        rom_sec.pack(fill="x")

        tk.Label(
            rom_sec, text="ROM IMPORT  (stable-retro)",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w")
        tk.Label(
            rom_sec, text="Import legally obtained ROMs for retro games",
            font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w", pady=(2, 4))
        tk.Button(
            rom_sec, text="Import ROM Directory...",
            font=("Segoe UI", 10), bg=BG_MEDIUM, fg=TEXT_PRIMARY,
            activebackground=BG_LIGHT, activeforeground=TEXT_PRIMARY,
            relief="flat", cursor="hand2", padx=12,
            command=self._import_rom,
        ).pack(anchor="w")
        self.rom_status = tk.Label(
            rom_sec, text="",
            font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_DARK,
            wraplength=400, justify="left",
        )
        self.rom_status.pack(anchor="w", pady=(4, 0))

        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(fill="x", padx=20)

        # ── Quick-access folder buttons ────────────────────────────────
        folder_sec = tk.Frame(parent, bg=BG_DARK, pady=12, **pad)
        folder_sec.pack(fill="x")

        tk.Label(
            folder_sec, text="QUICK ACCESS",
            font=("Segoe UI", 8, "bold"), fg=TEXT_DIM, bg=BG_DARK,
        ).pack(anchor="w")

        btn_row = tk.Frame(folder_sec, bg=BG_DARK)
        btn_row.pack(fill="x", pady=(4, 0))

        for label, cmd in [
            ("Open Models Folder",      lambda: self._open_folder(MODELS_DIR)),
            ("Open Recordings Folder",  lambda: self._open_folder(RECORDINGS_DIR)),
            ("Compare Runs",            self._open_comparison),
        ]:
            tk.Button(
                btn_row, text=label,
                font=("Segoe UI", 9), bg=BG_MEDIUM, fg=TEXT_DIM,
                activebackground=BG_LIGHT, activeforeground=TEXT_PRIMARY,
                relief="flat", cursor="hand2", padx=10,
                command=cmd,
            ).pack(side="left", expand=True, fill="x", padx=2)

    def _build_start_area(self):
        """Always-visible START/STOP button and status bar at the bottom."""
        tk.Frame(self.root, bg=BORDER_COLOR, height=1).pack(fill="x")

        area = tk.Frame(self.root, bg=BG_DARK, padx=20, pady=10)
        area.pack(fill="x")

        self.start_btn = tk.Button(
            area,
            text="\u25B6   START TRAINING",
            font=("Segoe UI", 14, "bold"),
            bg=ACCENT_GREEN,
            fg=BG_DARK,
            activebackground="#00b863",
            activeforeground=BG_DARK,
            relief="flat",
            cursor="hand2",
            pady=10,
            command=self._on_start_stop,
        )
        self.start_btn.pack(fill="x")

        self.status_label = tk.Label(
            area,
            text="Ready — Pick a game and algorithm, then click START to begin training!",
            font=("Segoe UI", 9),
            fg=TEXT_DIM,
            bg=BG_DARK,
            anchor="w",
        )
        self.status_label.pack(fill="x", pady=(4, 0))

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
        # Clear existing widgets
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

    def _on_game_changed(self, event=None):
        """Handle game selection change — update algorithm compatibility."""
        adapter = self._get_selected_adapter()
        if adapter:
            supported = adapter.supported_algorithms()
            current = self.selected_algo.get()
            # If current algo is no longer supported, switch to first supported
            if current not in supported and supported:
                self.selected_algo.set(supported[0])
                self._update_duration_label()
                self.model_path_var.set("")
                self._update_model_display()
        self._update_algo_buttons()
        self._update_algo_desc()
        self._rebuild_game_options()

    def _select_algorithm(self, algo):
        """Handle algorithm button click."""
        self.selected_algo.set(algo)
        self._update_algo_buttons()
        self._update_algo_desc()
        self._update_duration_label()
        # Clear loaded model when switching algorithms
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
            if algo not in supported:
                # Incompatible — gray out and disable
                btn.configure(
                    bg=BG_DARK, fg="#555555", activebackground=BG_DARK,
                    state="disabled",
                )
            elif algo == current:
                color = ALGO_INFO[algo]["color"]
                btn.configure(
                    bg=color, fg=BG_DARK, activebackground=color,
                    state="normal",
                )
            else:
                btn.configure(
                    bg=BG_MEDIUM, fg=TEXT_DIM, activebackground=BG_LIGHT,
                    state="normal",
                )

    def _update_duration_label(self):
        """Update the duration label text when algorithm changes."""
        algo = self.selected_algo.get()
        self.duration_label.configure(text=ALGO_INFO[algo]["duration_label"])

    def _on_eval_toggle(self):
        """Update UI when Evaluation Mode is toggled."""
        self._update_start_button_text()

    def _browse_model(self):
        """Open file dialog for selecting a saved model."""
        algo = self.selected_algo.get()
        info = ALGO_INFO[algo]
        initial_dir = os.path.join(MODELS_DIR, info["model_subdir"])

        # Create the directory if it doesn't exist
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
            # Show just the filename for cleanliness
            display = os.path.basename(path)
            self.model_display.configure(text=display, fg=TEXT_PRIMARY)
        else:
            self.model_display.configure(text="None", fg=TEXT_DIM)

    def _update_start_button_text(self):
        """Update button text based on eval mode and running state."""
        if self.process is not None:
            # Training/eval is running — show STOP
            self.start_btn.configure(
                text="\u23F9   STOP",
                bg=ACCENT_RED,
                activebackground="#e63946",
            )
        else:
            # Idle — show START TRAINING or START EVALUATION
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
                # Refresh game registry to pick up newly available games
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
            # Currently running — stop it
            self._stop_training()
        else:
            # Idle — start training
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

        # Build the command — use venv Python so all deps are available
        cmd = [
            VENV_PYTHON,
            MAIN_SCRIPT,
            "--algorithm", algo,
            "--game", game_id,
            "--world", world,
            "--stage", stage,
        ]

        # Add visualization flag
        if self.visualize_var.get():
            cmd.append("--visualize")

        # Add record flag
        if self.record_var.get():
            cmd.append("--record")

        # Add eval flag
        if is_eval:
            cmd.append("--eval")

        # Add stage progression flag
        if self.next_stage_var.get() and not is_eval:
            cmd.append("--next-stage")

        # Add curriculum flag
        if self.curriculum_var.get() and not is_eval:
            cmd.append("--curriculum")

        # Add parallel environments
        num_envs = self.num_envs_var.get()
        if num_envs and int(num_envs) > 1:
            cmd.extend(["--num-envs", num_envs])

        # Add model loading
        if model_path:
            cmd.extend(["--load", model_path])

        # Add duration (episodes/generations/timesteps)
        if duration:
            try:
                val = int(duration)
                if val <= 0:
                    raise ValueError
                # For PPO/A2C/DT, the GUI shows "×1000", so multiply
                if algo in ("ppo", "a2c", "dt"):
                    val = val * 1000
                cmd.extend(["--episodes", str(val)])
            except ValueError:
                self._set_status("Invalid number for duration.", ACCENT_RED)
                return

        # Add streaming flags
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

        # Add music flag
        if self.music_var.get():
            music_dir = self.music_dir_var.get().strip()
            if not music_dir or not os.path.isdir(music_dir):
                print(f'[Music] Directory not found: {music_dir!r}, skipping music')
            else:
                cmd.extend(["--music", music_dir])

        # Add device preference
        device = self.device_var.get()
        if device and device != 'auto':
            cmd.extend(["--device", device])

        # Add game-specific options from the dynamic config panel
        game_opts = self._get_game_options()
        if game_opts:
            cmd.append("--game-opts")
            for key, value in game_opts.items():
                cmd.append(f"{key}={value}")

        # Launch the subprocess
        try:
            # CREATE_NEW_PROCESS_GROUP allows sending CTRL_BREAK_EVENT
            # for graceful shutdown on Windows
            self.process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        except Exception as e:
            self._set_status(f"Failed to start: {e}", ACCENT_RED)
            return

        # Update UI
        mode = "Evaluating" if is_eval else "Training"
        self._set_status(
            f"{mode} {algo.upper()} on World {world}-{stage}...",
            ALGO_INFO[algo]["color"],
        )
        self._update_start_button_text()
        self._disable_controls(True)

        # Start polling the process
        self.root.after(500, self._check_process)

    def _stop_training(self):
        """Stop the running training subprocess gracefully."""
        if self.process is not None:
            try:
                # Send CTRL_BREAK_EVENT on Windows — this triggers the
                # signal handler in base_trainer.py which saves the model
                # before shutting down gracefully.
                os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
                self._set_status("Stopping... (saving model)", ACCENT_ORANGE)
            except Exception:
                # If graceful shutdown fails, force terminate
                self.process.terminate()
                self._set_status("Force stopped.", ACCENT_RED)

    def _check_process(self):
        """Poll the subprocess to see if it's still running."""
        if self.process is None:
            return

        retcode = self.process.poll()
        if retcode is None:
            # Still running — check again in 500ms
            self.root.after(500, self._check_process)
        else:
            # Process finished
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
            # Re-enable only supported algorithms
            self._update_algo_buttons()

    def _open_folder(self, path):
        """Open a folder in Windows Explorer."""
        os.makedirs(path, exist_ok=True)
        # os.startfile is Windows-specific
        os.startfile(path)

    def _on_close(self):
        """Handle window close — stop training if running."""
        if self.process is not None:
            self._stop_training()
            # Give it a moment to save, then close
            self.root.after(2000, self.root.destroy)
        else:
            self.root.destroy()

    # ===================================================================
    # Run
    # ===================================================================

    def run(self):
        """Start the tkinter main loop."""
        self.root.mainloop()

# -----------------------------------------------------------------------
# Apply a dark theme to ttk widgets (comboboxes, etc.)
# -----------------------------------------------------------------------
def apply_dark_theme(root):
    """
    Configure ttk Style for a dark theme.

    tkinter's ttk widgets use a separate theming system. This sets up
    dark colors so comboboxes etc. match the rest of the GUI.
    """
    style = ttk.Style(root)

    # Use 'clam' theme as base — it's the most customizable
    style.theme_use("clam")

    # Combobox styling
    style.configure(
        "TCombobox",
        fieldbackground=BG_LIGHT,
        background=BG_MEDIUM,
        foreground=TEXT_PRIMARY,
        arrowcolor=TEXT_PRIMARY,
        borderwidth=0,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", BG_LIGHT)],
        foreground=[("readonly", TEXT_PRIMARY)],
        selectbackground=[("readonly", BG_LIGHT)],
        selectforeground=[("readonly", TEXT_PRIMARY)],
    )

    # General frame styling
    style.configure("TFrame", background=BG_DARK)
    style.configure("TLabel", background=BG_DARK, foreground=TEXT_PRIMARY)

# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    app = MarioLauncher()
    apply_dark_theme(app.root)
    app.run()
