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

# ---------------------------------------------------------------------------
# Path setup — figure out where the project lives
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "main.py")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "recordings")

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

# Algorithm-specific accent colors and file extensions
ALGO_INFO = {
    "neat": {
        "color": ACCENT_GREEN,
        "duration_label": "Generations",
        "duration_default": "100",
        "file_ext": [("NEAT Genome", "*.pkl"), ("All Files", "*.*")],
        "model_subdir": "neat",
    },
    "ppo": {
        "color": ACCENT_BLUE,
        "duration_label": "Timesteps (×1000)",
        "duration_default": "1000",
        "file_ext": [("SB3 Model", "*.zip"), ("All Files", "*.*")],
        "model_subdir": "ppo",
    },
    "dqn": {
        "color": ACCENT_ORANGE,
        "duration_label": "Episodes",
        "duration_default": "5000",
        "file_ext": [("PyTorch Model", "*.pt"), ("All Files", "*.*")],
        "model_subdir": "dqn",
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
        self.root.title("Mario ML Launcher")
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

        # Streaming
        self.twitch_key_var = tk.StringVar()
        self.youtube_key_var = tk.StringVar()
        self.stream_var = tk.BooleanVar(value=False)

        # Music
        self.music_var = tk.BooleanVar(value=True)
        self.music_dir_var = tk.StringVar(
            value=os.path.join(PROJECT_ROOT, 'assets', 'music')
        )

        # Build all GUI sections
        self._build_header()
        self._build_algorithm_selector()
        self._build_world_stage()
        self._build_duration()
        self._build_options()
        self._build_streaming_section()
        self._build_music_section()
        self._build_model_loader()
        self._build_start_button()
        self._build_status_bar()
        self._build_folder_buttons()

        # Add some bottom padding
        spacer = tk.Frame(self.root, bg=BG_DARK, height=10)
        spacer.pack(fill="x")

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
        """Title section at the top of the window."""
        header = tk.Frame(self.root, bg=BG_MEDIUM, pady=15)
        header.pack(fill="x")

        title = tk.Label(
            header,
            text="Super Mario Bros ML",
            font=("Segoe UI", 20, "bold"),
            fg=ACCENT_GREEN,
            bg=BG_MEDIUM,
        )
        title.pack()

        subtitle = tk.Label(
            header,
            text="Training Launcher",
            font=("Segoe UI", 11),
            fg=TEXT_DIM,
            bg=BG_MEDIUM,
        )
        subtitle.pack()

    def _build_algorithm_selector(self):
        """Three toggle buttons for NEAT / PPO / DQN."""
        section = tk.Frame(self.root, bg=BG_DARK, pady=10, padx=25)
        section.pack(fill="x")

        label = tk.Label(
            section,
            text="Algorithm",
            font=("Segoe UI", 10),
            fg=TEXT_DIM,
            bg=BG_DARK,
        )
        label.pack(anchor="w")

        btn_frame = tk.Frame(section, bg=BG_DARK)
        btn_frame.pack(fill="x", pady=(5, 0))

        # Store button references so we can restyle them on selection
        self.algo_buttons = {}

        for algo in ["neat", "ppo", "dqn"]:
            btn = tk.Button(
                btn_frame,
                text=algo.upper(),
                font=("Segoe UI", 12, "bold"),
                width=10,
                cursor="hand2",
                relief="flat",
                bd=0,
                command=lambda a=algo: self._select_algorithm(a),
            )
            btn.pack(side="left", expand=True, fill="x", padx=3)
            self.algo_buttons[algo] = btn

        # Apply initial styling
        self._update_algo_buttons()

    def _select_algorithm(self, algo):
        """Handle algorithm button click."""
        self.selected_algo.set(algo)
        self._update_algo_buttons()
        self._update_duration_label()
        # Clear loaded model when switching algorithms
        self.model_path_var.set("")
        self._update_model_display()

    def _update_algo_buttons(self):
        """Restyle algorithm buttons to show which is active."""
        current = self.selected_algo.get()
        for algo, btn in self.algo_buttons.items():
            if algo == current:
                color = ALGO_INFO[algo]["color"]
                btn.configure(bg=color, fg=BG_DARK, activebackground=color)
            else:
                btn.configure(
                    bg=BG_MEDIUM, fg=TEXT_DIM, activebackground=BG_LIGHT
                )

    def _build_world_stage(self):
        """World and Stage dropdown selectors."""
        section = tk.Frame(self.root, bg=BG_DARK, padx=25, pady=5)
        section.pack(fill="x")

        # World
        world_frame = tk.Frame(section, bg=BG_DARK)
        world_frame.pack(side="left", expand=True, fill="x", padx=(0, 10))

        tk.Label(
            world_frame,
            text="World",
            font=("Segoe UI", 10),
            fg=TEXT_DIM,
            bg=BG_DARK,
        ).pack(anchor="w")

        world_combo = ttk.Combobox(
            world_frame,
            textvariable=self.world_var,
            values=[str(i) for i in range(1, 9)],
            state="readonly",
            width=8,
        )
        world_combo.pack(anchor="w", pady=(3, 0))

        # Stage
        stage_frame = tk.Frame(section, bg=BG_DARK)
        stage_frame.pack(side="left", expand=True, fill="x")

        tk.Label(
            stage_frame,
            text="Stage",
            font=("Segoe UI", 10),
            fg=TEXT_DIM,
            bg=BG_DARK,
        ).pack(anchor="w")

        stage_combo = ttk.Combobox(
            stage_frame,
            textvariable=self.stage_var,
            values=[str(i) for i in range(1, 5)],
            state="readonly",
            width=8,
        )
        stage_combo.pack(anchor="w", pady=(3, 0))

    def _build_duration(self):
        """Episodes / Generations / Timesteps input field."""
        section = tk.Frame(self.root, bg=BG_DARK, padx=25, pady=5)
        section.pack(fill="x")

        # Label that changes based on selected algorithm
        self.duration_label = tk.Label(
            section,
            text="Generations",
            font=("Segoe UI", 10),
            fg=TEXT_DIM,
            bg=BG_DARK,
        )
        self.duration_label.pack(anchor="w")

        entry_frame = tk.Frame(section, bg=BG_DARK)
        entry_frame.pack(fill="x", pady=(3, 0))

        self.duration_entry = tk.Entry(
            entry_frame,
            textvariable=self.duration_var,
            font=("Segoe UI", 11),
            bg=BG_LIGHT,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="flat",
            width=15,
        )
        self.duration_entry.pack(side="left")

        # Placeholder hint
        self.duration_hint = tk.Label(
            entry_frame,
            text="(blank = use config default)",
            font=("Segoe UI", 9),
            fg=TEXT_DIM,
            bg=BG_DARK,
        )
        self.duration_hint.pack(side="left", padx=(10, 0))

    def _update_duration_label(self):
        """Update the duration label text when algorithm changes."""
        algo = self.selected_algo.get()
        self.duration_label.configure(text=ALGO_INFO[algo]["duration_label"])

    def _build_options(self):
        """Checkboxes for visualization, recording, eval mode."""
        section = tk.LabelFrame(
            self.root,
            text=" Options ",
            font=("Segoe UI", 10),
            fg=TEXT_DIM,
            bg=BG_DARK,
            bd=1,
            relief="groove",
            highlightbackground=BORDER_COLOR,
            padx=15,
            pady=8,
        )
        section.pack(fill="x", padx=25, pady=8)

        # Visualize checkbox
        viz_cb = tk.Checkbutton(
            section,
            text="Show Live Visualization",
            variable=self.visualize_var,
            font=("Segoe UI", 10),
            fg=TEXT_PRIMARY,
            bg=BG_DARK,
            selectcolor=BG_MEDIUM,
            activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY,
            cursor="hand2",
        )
        viz_cb.pack(anchor="w")

        # Record checkbox
        rec_cb = tk.Checkbutton(
            section,
            text="Record Training Video",
            variable=self.record_var,
            font=("Segoe UI", 10),
            fg=TEXT_PRIMARY,
            bg=BG_DARK,
            selectcolor=BG_MEDIUM,
            activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY,
            cursor="hand2",
        )
        rec_cb.pack(anchor="w")

        # Eval mode checkbox
        eval_cb = tk.Checkbutton(
            section,
            text="Evaluation Mode  (watch AI play, no training)",
            variable=self.eval_var,
            font=("Segoe UI", 10),
            fg=TEXT_PRIMARY,
            bg=BG_DARK,
            selectcolor=BG_MEDIUM,
            activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY,
            command=self._on_eval_toggle,
            cursor="hand2",
        )
        eval_cb.pack(anchor="w")

        # Stage progression checkbox
        stage_cb = tk.Checkbutton(
            section,
            text="Auto-Advance Stages  (transfer weights to next stage)",
            variable=self.next_stage_var,
            font=("Segoe UI", 10),
            fg=TEXT_PRIMARY,
            bg=BG_DARK,
            selectcolor=BG_MEDIUM,
            activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY,
            cursor="hand2",
        )
        stage_cb.pack(anchor="w")

        # Curriculum (whole game) checkbox
        curriculum_cb = tk.Checkbutton(
            section,
            text="Whole Game  (curriculum learning across all 32 stages)",
            variable=self.curriculum_var,
            font=("Segoe UI", 10),
            fg=TEXT_PRIMARY,
            bg=BG_DARK,
            selectcolor=BG_MEDIUM,
            activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY,
            cursor="hand2",
        )
        curriculum_cb.pack(anchor="w")

        # Parallel environments selector
        envs_frame = tk.Frame(section, bg=BG_DARK)
        envs_frame.pack(anchor="w", pady=(6, 0))

        tk.Label(
            envs_frame,
            text="Parallel Envs:",
            font=("Segoe UI", 10),
            fg=TEXT_PRIMARY,
            bg=BG_DARK,
        ).pack(side="left")

        envs_combo = ttk.Combobox(
            envs_frame,
            textvariable=self.num_envs_var,
            values=["1", "2", "4", "8", "16"],
            state="readonly",
            width=4,
        )
        envs_combo.pack(side="left", padx=(6, 0))

        tk.Label(
            envs_frame,
            text="(multi-Mario grid display)",
            font=("Segoe UI", 9),
            fg=TEXT_DIM,
            bg=BG_DARK,
        ).pack(side="left", padx=(8, 0))

    def _on_eval_toggle(self):
        """Update UI when Evaluation Mode is toggled."""
        self._update_start_button_text()

    def _build_streaming_section(self):
        """Build the streaming configuration section."""
        section = tk.LabelFrame(
            self.root,
            text="  Streaming  ",
            font=("Segoe UI", 10),
            fg=TEXT_DIM,
            bg=BG_DARK,
            bd=1,
            relief="groove",
            highlightbackground=BORDER_COLOR,
            padx=15,
            pady=8,
        )
        section.pack(fill="x", padx=25, pady=8)

        # Enable streaming checkbox
        stream_cb = tk.Checkbutton(
            section,
            text="Enable Live Streaming",
            variable=self.stream_var,
            font=("Segoe UI", 10),
            fg=ACCENT_RED,
            bg=BG_DARK,
            selectcolor=BG_MEDIUM,
            activebackground=BG_DARK,
            activeforeground=ACCENT_RED,
            cursor="hand2",
        )
        stream_cb.pack(anchor="w")

        # Twitch key
        tk.Label(section, text="Twitch Stream Key:", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w", pady=(5, 0))
        twitch_entry = tk.Entry(
            section, textvariable=self.twitch_key_var, show="*",
            font=("Segoe UI", 10), bg=BG_MEDIUM, fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY, relief="flat",
        )
        twitch_entry.pack(fill="x", pady=2)

        # YouTube key
        tk.Label(section, text="YouTube Stream Key:", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w", pady=(5, 0))
        youtube_entry = tk.Entry(
            section, textvariable=self.youtube_key_var, show="*",
            font=("Segoe UI", 10), bg=BG_MEDIUM, fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY, relief="flat",
        )
        youtube_entry.pack(fill="x", pady=2)

    def _build_music_section(self):
        """Build the music configuration section."""
        section = tk.LabelFrame(
            self.root,
            text="  Music  ",
            font=("Segoe UI", 10),
            fg=TEXT_DIM,
            bg=BG_DARK,
            bd=1,
            relief="groove",
            highlightbackground=BORDER_COLOR,
            padx=15,
            pady=8,
        )
        section.pack(fill="x", padx=25, pady=8)

        music_cb = tk.Checkbutton(
            section,
            text="Play Background Music",
            variable=self.music_var,
            font=("Segoe UI", 10),
            fg=TEXT_PRIMARY,
            bg=BG_DARK,
            selectcolor=BG_MEDIUM,
            activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY,
            cursor="hand2",
        )
        music_cb.pack(anchor="w")

    def _build_model_loader(self):
        """File browser for loading a saved model."""
        section = tk.Frame(self.root, bg=BG_DARK, padx=25, pady=5)
        section.pack(fill="x")

        tk.Label(
            section,
            text="Load Model",
            font=("Segoe UI", 10),
            fg=TEXT_DIM,
            bg=BG_DARK,
        ).pack(anchor="w")

        row = tk.Frame(section, bg=BG_DARK)
        row.pack(fill="x", pady=(3, 0))

        # Display selected model path
        self.model_display = tk.Label(
            row,
            text="None",
            font=("Segoe UI", 10),
            fg=TEXT_DIM,
            bg=BG_LIGHT,
            anchor="w",
            padx=8,
            pady=4,
            relief="flat",
        )
        self.model_display.pack(side="left", fill="x", expand=True)

        # Browse button
        browse_btn = tk.Button(
            row,
            text="Browse",
            font=("Segoe UI", 9),
            bg=BG_MEDIUM,
            fg=TEXT_PRIMARY,
            activebackground=BG_LIGHT,
            activeforeground=TEXT_PRIMARY,
            relief="flat",
            cursor="hand2",
            padx=12,
            command=self._browse_model,
        )
        browse_btn.pack(side="left", padx=(5, 0))

        # Clear button
        clear_btn = tk.Button(
            row,
            text="Clear",
            font=("Segoe UI", 9),
            bg=BG_MEDIUM,
            fg=TEXT_DIM,
            activebackground=BG_LIGHT,
            activeforeground=TEXT_PRIMARY,
            relief="flat",
            cursor="hand2",
            padx=8,
            command=self._clear_model,
        )
        clear_btn.pack(side="left", padx=(3, 0))

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

    def _build_start_button(self):
        """Large START / STOP button."""
        section = tk.Frame(self.root, bg=BG_DARK, padx=25, pady=10)
        section.pack(fill="x")

        self.start_btn = tk.Button(
            section,
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

    def _build_status_bar(self):
        """Status message at the bottom."""
        section = tk.Frame(self.root, bg=BG_DARK, padx=25, pady=0)
        section.pack(fill="x")

        self.status_label = tk.Label(
            section,
            text="Ready. Select an algorithm and click START.",
            font=("Segoe UI", 9),
            fg=TEXT_DIM,
            bg=BG_DARK,
            anchor="w",
        )
        self.status_label.pack(fill="x")

    def _build_folder_buttons(self):
        """Quick-access buttons to open models and recordings folders."""
        section = tk.Frame(self.root, bg=BG_DARK, padx=25, pady=8)
        section.pack(fill="x")

        models_btn = tk.Button(
            section,
            text="Open Models Folder",
            font=("Segoe UI", 9),
            bg=BG_MEDIUM,
            fg=TEXT_DIM,
            activebackground=BG_LIGHT,
            activeforeground=TEXT_PRIMARY,
            relief="flat",
            cursor="hand2",
            padx=10,
            command=lambda: self._open_folder(MODELS_DIR),
        )
        models_btn.pack(side="left", expand=True, fill="x", padx=(0, 3))

        rec_btn = tk.Button(
            section,
            text="Open Recordings Folder",
            font=("Segoe UI", 9),
            bg=BG_MEDIUM,
            fg=TEXT_DIM,
            activebackground=BG_LIGHT,
            activeforeground=TEXT_PRIMARY,
            relief="flat",
            cursor="hand2",
            padx=10,
            command=lambda: self._open_folder(RECORDINGS_DIR),
        )
        rec_btn.pack(side="left", expand=True, fill="x", padx=(3, 0))

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

        # Build the command
        cmd = [
            sys.executable,
            MAIN_SCRIPT,
            "--algorithm", algo,
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
                # For PPO, the GUI shows "×1000", so multiply
                if algo == "ppo":
                    val = val * 1000
                cmd.extend(["--episodes", str(val)])
            except ValueError:
                self._set_status("Invalid number for duration.", ACCENT_RED)
                return

        # Add streaming flags
        if self.stream_var.get():
            twitch_key = self.twitch_key_var.get().strip()
            youtube_key = self.youtube_key_var.get().strip()
            if twitch_key:
                cmd.extend(["--stream-twitch", twitch_key])
            if youtube_key:
                cmd.extend(["--stream-youtube", youtube_key])

        # Add music flag
        if self.music_var.get():
            music_dir = self.music_dir_var.get().strip()
            if music_dir and os.path.isdir(music_dir):
                cmd.extend(["--music", music_dir])

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
        state = "disabled" if disabled else "normal"
        for btn in self.algo_buttons.values():
            btn.configure(state=state)

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
