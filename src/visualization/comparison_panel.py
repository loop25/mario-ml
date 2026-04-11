"""
Cross-game training comparison panel.

Reads timestamped run logs from logs/runs/ and displays overlaid
training curves so users can compare performance across different
game+algorithm combinations.

Each run log is a JSON file named {game}_{algo}_{timestamp}.json
containing the full metrics history (reward, distance, etc.).

Usage:
    From the launcher: Click "Compare Runs" button.
    Standalone: python -m src.visualization.comparison_panel
"""

import json
import os
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# --- Colors matching the launcher/dashboard dark theme ---
BG_DARK = "#1a1a2e"
BG_MEDIUM = "#16213e"
BG_LIGHT = "#0f3460"
TEXT_PRIMARY = "#e0e0e0"
TEXT_DIM = "#8888aa"
ACCENT_GREEN = "#00d474"
BORDER_COLOR = "#2a2a4a"

# Distinct colors for up to 10 overlaid runs
RUN_COLORS = [
    '#4da6ff',  # Blue
    '#ff8c42',  # Orange
    '#00d474',  # Green
    '#ff4757',  # Red
    '#9B59B6',  # Purple
    '#F1C40F',  # Yellow
    '#1ABC9C',  # Teal
    '#E74C3C',  # Crimson
    '#3498DB',  # Light blue
    '#E67E22',  # Dark orange
]


def _parse_run_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parse a run log filename into components.

    Expected format: {game}_{algo}_{YYYYMMDD}_{HHMMSS}.json
    Returns dict with game, algo, date, time keys, or None if invalid.
    """
    if not filename.endswith('.json'):
        return None
    stem = filename[:-5]  # Remove .json
    parts = stem.rsplit('_', 2)
    if len(parts) < 3:
        return None
    # game_algo might itself contain underscores, so we split carefully
    # The last two parts are date and time
    time_part = parts[-1]
    date_part = parts[-2]
    game_algo = parts[0] if len(parts) == 3 else '_'.join(parts[:-2])

    # Now split game_algo — algo is the last segment
    ga_parts = game_algo.rsplit('_', 1)
    if len(ga_parts) != 2:
        return None

    return {
        'game': ga_parts[0],
        'algo': ga_parts[1],
        'date': date_part,
        'time': time_part,
        'filename': filename,
    }


def load_run_logs(logs_dir: str = 'logs') -> List[Dict]:
    """Load all run logs from the runs directory.

    Returns list of dicts with run metadata and loaded history data.
    """
    runs_dir = os.path.join(logs_dir, 'runs')
    if not os.path.isdir(runs_dir):
        return []

    runs = []
    for fname in sorted(os.listdir(runs_dir)):
        meta = _parse_run_filename(fname)
        if meta is None:
            continue
        filepath = os.path.join(runs_dir, fname)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            meta['data'] = data
            meta['label'] = f"{meta['game']} / {meta['algo'].upper()} ({meta['date']})"
            runs.append(meta)
        except (json.JSONDecodeError, IOError):
            continue

    return runs


class ComparisonPanel:
    """Tkinter window for comparing training runs across games."""

    def __init__(self, logs_dir: str = 'logs', parent: tk.Tk = None):
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for the comparison panel. "
                "Install it with: pip install matplotlib"
            )

        self.logs_dir = logs_dir
        self.runs = load_run_logs(logs_dir)

        # Create window
        if parent:
            self.window = tk.Toplevel(parent)
        else:
            self.window = tk.Tk()
        self.window.title("Training Run Comparison")
        self.window.configure(bg=BG_DARK)
        self.window.geometry("900x650")

        # Selected runs (indices into self.runs)
        self.selected_indices: List[int] = []

        self._build_ui()

    def _build_ui(self):
        """Build the comparison panel UI."""
        # --- Top: run selector ---
        selector_frame = tk.Frame(self.window, bg=BG_DARK, padx=10, pady=10)
        selector_frame.pack(fill="x")

        tk.Label(
            selector_frame,
            text="Select Runs to Compare",
            font=("Segoe UI", 12, "bold"),
            fg=ACCENT_GREEN,
            bg=BG_DARK,
        ).pack(anchor="w")

        if not self.runs:
            tk.Label(
                selector_frame,
                text="No training runs found in logs/runs/. "
                     "Complete a training session first.",
                font=("Segoe UI", 10),
                fg=TEXT_DIM,
                bg=BG_DARK,
            ).pack(anchor="w", pady=(5, 0))
            return

        # Listbox with checkbutton-style multi-select
        list_frame = tk.Frame(selector_frame, bg=BG_MEDIUM)
        list_frame.pack(fill="x", pady=(5, 0))

        self.run_vars = []
        for i, run in enumerate(self.runs):
            var = tk.BooleanVar(value=i < 2)  # Auto-select first 2
            self.run_vars.append(var)
            cb = tk.Checkbutton(
                list_frame,
                text=run['label'],
                variable=var,
                font=("Segoe UI", 10),
                fg=TEXT_PRIMARY,
                bg=BG_MEDIUM,
                selectcolor=BG_DARK,
                activebackground=BG_MEDIUM,
                activeforeground=TEXT_PRIMARY,
                command=self._on_selection_changed,
            )
            cb.pack(anchor="w", padx=5, pady=1)

        # Metric selector
        metric_frame = tk.Frame(selector_frame, bg=BG_DARK, pady=5)
        metric_frame.pack(fill="x")

        tk.Label(
            metric_frame,
            text="Metric:",
            font=("Segoe UI", 10),
            fg=TEXT_DIM,
            bg=BG_DARK,
        ).pack(side="left")

        self.metric_var = tk.StringVar(value="reward")
        metrics = self._collect_available_metrics()
        self.metric_combo = ttk.Combobox(
            metric_frame,
            textvariable=self.metric_var,
            values=metrics,
            state="readonly",
            width=20,
        )
        self.metric_combo.pack(side="left", padx=(5, 0))
        self.metric_combo.bind("<<ComboboxSelected>>", self._on_selection_changed)

        # Smoothing slider
        tk.Label(
            metric_frame,
            text="  Smoothing:",
            font=("Segoe UI", 10),
            fg=TEXT_DIM,
            bg=BG_DARK,
        ).pack(side="left", padx=(15, 0))

        self.smooth_var = tk.IntVar(value=10)
        smooth_scale = tk.Scale(
            metric_frame,
            from_=1, to=100,
            variable=self.smooth_var,
            orient="horizontal",
            length=120,
            bg=BG_DARK,
            fg=TEXT_PRIMARY,
            highlightthickness=0,
            troughcolor=BG_MEDIUM,
            command=lambda _: self._on_selection_changed(),
        )
        smooth_scale.pack(side="left", padx=(5, 0))

        # --- Bottom: matplotlib chart ---
        chart_frame = tk.Frame(self.window, bg=BG_DARK)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.fig = Figure(figsize=(8, 4), dpi=100, facecolor=BG_DARK)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Initial plot
        self._update_chart()

    def _collect_available_metrics(self) -> List[str]:
        """Gather all metric names found across loaded runs."""
        metrics = set()
        for run in self.runs:
            history = run.get('data', {}).get('history', {})
            for key, vals in history.items():
                # Skip non-numeric metrics like action_distribution
                if vals and isinstance(vals[0], (int, float)):
                    metrics.add(key)
        # Put 'reward' first if present
        result = sorted(metrics)
        if 'reward' in result:
            result.remove('reward')
            result.insert(0, 'reward')
        return result if result else ['reward']

    def _on_selection_changed(self, event=None):
        """Redraw chart when run selection or metric changes."""
        self._update_chart()

    def _smooth(self, values: List[float], window: int) -> List[float]:
        """Apply rolling average smoothing."""
        if window <= 1 or not values:
            return values
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            chunk = values[start:i + 1]
            result.append(sum(chunk) / len(chunk))
        return result

    def _update_chart(self):
        """Redraw the comparison chart with selected runs."""
        self.ax.clear()

        metric = self.metric_var.get()
        window = self.smooth_var.get()

        # Style the axes
        self.ax.set_facecolor(BG_MEDIUM)
        self.ax.tick_params(colors=TEXT_DIM, labelsize=8)
        self.ax.spines['bottom'].set_color(BORDER_COLOR)
        self.ax.spines['left'].set_color(BORDER_COLOR)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.set_xlabel('Episode', color=TEXT_DIM, fontsize=9)
        self.ax.set_ylabel(metric.replace('_', ' ').title(), color=TEXT_DIM, fontsize=9)
        self.ax.set_title(
            f'{metric.replace("_", " ").title()} Comparison',
            color=TEXT_PRIMARY, fontsize=11, pad=10,
        )

        color_idx = 0
        for i, run in enumerate(self.runs):
            if not self.run_vars[i].get():
                continue

            history = run.get('data', {}).get('history', {})
            values = history.get(metric, [])
            if not values or not isinstance(values[0], (int, float)):
                continue

            smoothed = self._smooth(values, window)
            episodes = list(range(1, len(smoothed) + 1))
            color = RUN_COLORS[color_idx % len(RUN_COLORS)]
            color_idx += 1

            # Plot smoothed line
            self.ax.plot(
                episodes, smoothed,
                color=color, linewidth=1.5, alpha=0.9,
                label=run['label'],
            )
            # Plot raw values as faint background
            self.ax.plot(
                episodes, values,
                color=color, linewidth=0.3, alpha=0.2,
            )

        if color_idx > 0:
            self.ax.legend(
                loc='upper left', fontsize=8,
                facecolor=BG_DARK, edgecolor=BORDER_COLOR,
                labelcolor=TEXT_PRIMARY,
            )

        self.fig.tight_layout()
        self.canvas.draw()

    def run(self):
        """Start the tkinter main loop (standalone mode)."""
        self.window.mainloop()


if __name__ == '__main__':
    panel = ComparisonPanel()
    panel.run()
