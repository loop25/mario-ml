"""
Overlay Manager for stream-ready frame compositing.

Renders a polished streaming overlay on top of the dashboard capture.
The overlay appears whenever a stream is live and includes:
- Top bar: game name, algorithm, training progress
- Bottom ticker: episode count, reward, elapsed time, best reward
- Corner accent lines for visual framing

All drawing is done with OpenCV — no external image assets needed.
"""
import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np


# ── Color Palette (BGR for OpenCV) ───────────────────────────────────
PANEL_BG = (30, 30, 30)          # Near-black panel fill
ACCENT = (225, 160, 50)          # Warm amber accent
ACCENT_DIM = (150, 100, 30)      # Dimmer amber for secondary elements
TEXT_WHITE = (240, 240, 240)      # Bright white text
TEXT_DIM = (170, 170, 170)        # Subdued label text
RED_LIVE = (60, 60, 230)         # Red for live dot (BGR)
PROGRESS_GREEN = (80, 200, 80)   # Green for progress bar
SEPARATOR = (80, 80, 80)         # Thin separator lines


class OverlayManager:
    """
    Composites dashboard frames with a polished streaming overlay.

    The overlay is designed to be informative without obscuring the
    dashboard content — thin bars at top and bottom with high contrast
    text and subtle transparency.

    Args:
        resolution: Output resolution (width, height). Default 1280x720.
    """

    TOP_BAR_HEIGHT = 38
    BOTTOM_BAR_HEIGHT = 34

    def __init__(self, resolution: Tuple[int, int] = (1280, 720)):
        self.width, self.height = resolution
        self._start_time = time.time()

    def compose(
        self,
        frame: np.ndarray,
        is_live: bool = False,
        algorithm: Optional[str] = None,
        stage: Optional[str] = None,
        episode: Optional[int] = None,
        reward: Optional[float] = None,
        best_reward: Optional[float] = None,
        elapsed_time: Optional[float] = None,
        game_name: Optional[str] = None,
        training_target: Optional[int] = None,
        num_envs: int = 1,
    ) -> np.ndarray:
        """
        Compose the final stream frame with overlays.

        Args:
            frame: Input frame (H, W, 3) uint8 RGB from dashboard.
            is_live: Whether currently streaming.
            algorithm: Current algorithm name.
            stage: Current world-stage string (e.g., '1-1').
            episode: Current episode number.
            reward: Latest episode reward.
            best_reward: Best reward achieved so far.
            elapsed_time: Training elapsed time in seconds.
            game_name: Current game name.
            training_target: Total target episodes (for progress bar).

        Returns:
            Composited frame at target resolution (H, W, 3) uint8.
        """
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        else:
            frame = frame.copy()

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if is_live:
            self._draw_top_bar(frame_bgr, algorithm, stage, game_name,
                               episode, training_target)
            self._draw_bottom_bar(frame_bgr, episode, reward, best_reward,
                                  elapsed_time, num_envs)
            self._draw_corner_accents(frame_bgr)

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ── Top Bar ──────────────────────────────────────────────────────

    def _draw_top_bar(
        self, f: np.ndarray, algorithm: Optional[str],
        stage: Optional[str], game_name: Optional[str],
        episode: Optional[int], training_target: Optional[int],
    ) -> None:
        h = self.TOP_BAR_HEIGHT

        # Solid opaque background — no bleed-through from dashboard text
        cv2.rectangle(f, (0, 0), (self.width, h), PANEL_BG, -1)

        # Bottom edge line
        cv2.line(f, (0, h), (self.width, h), ACCENT_DIM, 1)

        # ─ Left: pulsing recording dot + game & algorithm ─
        elapsed = time.time() - self._start_time
        pulse = 0.5 + 0.5 * math.sin(elapsed * 3.0)
        dot_r = int(5 + 2 * pulse)
        dot_color = (
            int(60 + 30 * pulse),
            int(60 + 30 * pulse),
            int(180 + 75 * pulse),
        )
        dot_cx, dot_cy = 18, h // 2
        cv2.circle(f, (dot_cx, dot_cy), dot_r, dot_color, -1)
        # Outer glow ring
        glow_alpha = int(80 * pulse)
        cv2.circle(f, (dot_cx, dot_cy), dot_r + 3,
                   (0, 0, min(255, 140 + glow_alpha)), 1)

        # Game name
        x_cursor = 34
        if game_name:
            self._put_text(f, game_name, (x_cursor, h // 2 + 5),
                           scale=0.50, color=TEXT_WHITE, thickness=1)
            tw = cv2.getTextSize(game_name, cv2.FONT_HERSHEY_SIMPLEX,
                                 0.50, 1)[0][0]
            x_cursor += tw + 12

        # Separator dot
        if game_name and algorithm:
            cv2.circle(f, (x_cursor, h // 2), 2, SEPARATOR, -1)
            x_cursor += 12

        # Algorithm badge
        if algorithm:
            label = algorithm.upper()
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                 0.45, 1)[0]
            badge_w = ts[0] + 14
            badge_h = ts[1] + 10
            bx = x_cursor
            by = h // 2 - badge_h // 2
            cv2.rectangle(f, (bx, by), (bx + badge_w, by + badge_h),
                          ACCENT_DIM, -1)
            cv2.rectangle(f, (bx, by), (bx + badge_w, by + badge_h),
                          ACCENT, 1)
            cv2.putText(f, label, (bx + 7, by + badge_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_WHITE, 1,
                        cv2.LINE_AA)
            x_cursor += badge_w + 12

        # Stage if present
        if stage:
            self._put_text(f, f'W{stage}', (x_cursor, h // 2 + 5),
                           scale=0.42, color=TEXT_DIM, thickness=1)

        # ─ Right: training progress ─
        if episode is not None and training_target and training_target > 0:
            progress = min(1.0, episode / training_target)
            pct_text = f'{progress:.0%}'
            bar_w = 120
            bar_h = 8
            bar_x = self.width - bar_w - 60
            bar_y = h // 2 - bar_h // 2

            # Background track
            cv2.rectangle(f, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h),
                          (60, 60, 60), -1)
            # Fill
            fill_w = int(bar_w * progress)
            if fill_w > 0:
                bar_color = PROGRESS_GREEN if progress < 1.0 else ACCENT
                cv2.rectangle(f, (bar_x, bar_y),
                              (bar_x + fill_w, bar_y + bar_h),
                              bar_color, -1)
            # Percentage label
            self._put_text(f, pct_text,
                           (bar_x + bar_w + 8, h // 2 + 5),
                           scale=0.40, color=TEXT_DIM, thickness=1)

    # ── Bottom Bar ───────────────────────────────────────────────────

    def _draw_bottom_bar(
        self, f: np.ndarray, episode: Optional[int],
        reward: Optional[float], best_reward: Optional[float],
        elapsed_time: Optional[float], num_envs: int = 1,
    ) -> None:
        h = self.BOTTOM_BAR_HEIGHT
        y0 = self.height - h

        # Solid opaque background — no bleed-through from dashboard text
        cv2.rectangle(f, (0, y0), (self.width, self.height), PANEL_BG, -1)

        # Top edge line
        cv2.line(f, (0, y0), (self.width, y0), ACCENT_DIM, 1)

        text_y = y0 + h // 2 + 5
        x = 16

        # Episode count (with multi-env context if applicable)
        if episode is not None:
            self._put_text(f, 'EP', (x, text_y),
                           scale=0.38, color=TEXT_DIM, thickness=1)
            x += 26
            self._put_text(f, f'{episode:,}', (x, text_y),
                           scale=0.48, color=TEXT_WHITE, thickness=1)
            tw = cv2.getTextSize(f'{episode:,}', cv2.FONT_HERSHEY_SIMPLEX,
                                 0.48, 1)[0][0]
            x += tw + 4
            if num_envs > 1:
                env_label = f'({num_envs} envs)'
                self._put_text(f, env_label, (x, text_y),
                               scale=0.32, color=TEXT_DIM, thickness=1)
                tw2 = cv2.getTextSize(env_label, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.32, 1)[0][0]
                x += tw2 + 10
            else:
                x += 16

        # Separator
        if episode is not None and reward is not None:
            cv2.line(f, (x, y0 + 8), (x, self.height - 8), SEPARATOR, 1)
            x += 16

        # Reward
        if reward is not None:
            self._put_text(f, 'RWD', (x, text_y),
                           scale=0.38, color=TEXT_DIM, thickness=1)
            x += 36
            self._put_text(f, f'{reward:,.0f}', (x, text_y),
                           scale=0.48, color=ACCENT, thickness=1)
            tw = cv2.getTextSize(f'{reward:,.0f}', cv2.FONT_HERSHEY_SIMPLEX,
                                 0.48, 1)[0][0]
            x += tw + 20

        # Best reward
        if best_reward is not None:
            cv2.line(f, (x, y0 + 8), (x, self.height - 8), SEPARATOR, 1)
            x += 16
            self._put_text(f, 'BEST', (x, text_y),
                           scale=0.38, color=TEXT_DIM, thickness=1)
            x += 40
            self._put_text(f, f'{best_reward:,.0f}', (x, text_y),
                           scale=0.48, color=PROGRESS_GREEN, thickness=1)

        # Elapsed time (right-aligned)
        if elapsed_time is not None:
            hours = int(elapsed_time // 3600)
            mins = int((elapsed_time % 3600) // 60)
            secs = int(elapsed_time % 60)
            if hours > 0:
                time_str = f'{hours}:{mins:02d}:{secs:02d}'
            else:
                time_str = f'{mins}:{secs:02d}'
            tw = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX,
                                 0.48, 1)[0][0]
            tx = self.width - tw - 16
            self._put_text(f, time_str, (tx, text_y),
                           scale=0.48, color=TEXT_WHITE, thickness=1)
            # Clock icon (small circle)
            lx = tx - 16
            cv2.circle(f, (lx, y0 + h // 2), 5, TEXT_DIM, 1)
            cv2.line(f, (lx, y0 + h // 2 - 3), (lx, y0 + h // 2), TEXT_DIM, 1)
            cv2.line(f, (lx, y0 + h // 2), (lx + 3, y0 + h // 2 + 1),
                     TEXT_DIM, 1)

    # ── Corner Accents ───────────────────────────────────────────────

    def _draw_corner_accents(self, f: np.ndarray) -> None:
        """Draw subtle L-shaped accent lines in corners for visual framing."""
        accent_len = 30
        t = self.TOP_BAR_HEIGHT + 4
        b = self.height - self.BOTTOM_BAR_HEIGHT - 4

        # Top-left
        cv2.line(f, (4, t), (4, t + accent_len), ACCENT_DIM, 1)
        cv2.line(f, (4, t), (4 + accent_len, t), ACCENT_DIM, 1)

        # Top-right
        cv2.line(f, (self.width - 5, t),
                 (self.width - 5, t + accent_len), ACCENT_DIM, 1)
        cv2.line(f, (self.width - 5, t),
                 (self.width - 5 - accent_len, t), ACCENT_DIM, 1)

        # Bottom-left
        cv2.line(f, (4, b), (4, b - accent_len), ACCENT_DIM, 1)
        cv2.line(f, (4, b), (4 + accent_len, b), ACCENT_DIM, 1)

        # Bottom-right
        cv2.line(f, (self.width - 5, b),
                 (self.width - 5, b - accent_len), ACCENT_DIM, 1)
        cv2.line(f, (self.width - 5, b),
                 (self.width - 5 - accent_len, b), ACCENT_DIM, 1)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _put_text(
        f: np.ndarray, text: str, pos: Tuple[int, int],
        scale: float = 0.5, color: Tuple[int, int, int] = TEXT_WHITE,
        thickness: int = 1,
    ) -> None:
        """Draw text with a subtle dark shadow for readability."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = pos
        cv2.putText(f, text, (x + 1, y + 1), font, scale, (0, 0, 0),
                    thickness + 1, cv2.LINE_AA)
        cv2.putText(f, text, (x, y), font, scale, color, thickness,
                    cv2.LINE_AA)
