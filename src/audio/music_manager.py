"""
Background music manager for training sessions.

Uses pygame.mixer.music for playlist-based playback.
Supports .ogg, .mp3, and .wav files dropped into a music directory.
"""
import os
import random
from typing import List, Optional

import pygame


SUPPORTED_EXTENSIONS = {'.ogg', '.mp3', '.wav', '.flac'}
MUSIC_END_EVENT = pygame.USEREVENT + 99


class MusicManager:
    """
    Manages background music playback during training.

    Scans a directory for audio files, shuffles them into a playlist,
    and loops playback continuously.

    Args:
        music_dir: Path to directory containing music files.
        volume: Initial volume (0.0 to 1.0). Default 0.5.
    """

    def __init__(self, music_dir: str, volume: float = 0.5):
        self.music_dir = music_dir
        self._volume = max(0.0, min(1.0, volume))
        self._pre_mute_volume = self._volume
        self.is_muted = False
        self.is_playing = False
        self.current_track: Optional[str] = None
        self._track_index = 0

        self.playlist: List[str] = self._scan_directory()
        if self.playlist:
            random.shuffle(self.playlist)

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            pygame.mixer.music.set_volume(self._volume)
        except pygame.error:
            pass

    def _scan_directory(self) -> List[str]:
        if not os.path.isdir(self.music_dir):
            return []
        tracks = []
        for f in sorted(os.listdir(self.music_dir)):
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                tracks.append(os.path.join(self.music_dir, f))
        return tracks

    def play(self) -> None:
        if not self.playlist:
            return
        try:
            self.current_track = self.playlist[self._track_index]
            pygame.mixer.music.load(self.current_track)
            pygame.mixer.music.play()
            pygame.mixer.music.set_endevent(MUSIC_END_EVENT)
            self.is_playing = True
        except pygame.error as e:
            print(f'Music playback error: {e}')

    def stop(self) -> None:
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        self.is_playing = False

    def pause(self) -> None:
        if self.is_playing:
            try:
                pygame.mixer.music.pause()
            except pygame.error:
                pass

    def resume(self) -> None:
        if self.is_playing:
            try:
                pygame.mixer.music.unpause()
            except pygame.error:
                pass

    def next_track(self) -> None:
        if not self.playlist:
            return
        self._track_index = (self._track_index + 1) % len(self.playlist)
        if self.is_playing:
            self.play()

    def handle_music_end_event(self) -> None:
        self.next_track()

    @property
    def volume(self) -> float:
        return self._volume

    def set_volume(self, vol: float) -> None:
        self._volume = max(0.0, min(1.0, vol))
        if not self.is_muted:
            try:
                pygame.mixer.music.set_volume(self._volume)
            except pygame.error:
                pass

    def volume_up(self, step: float = 0.05) -> None:
        self.set_volume(self._volume + step)

    def volume_down(self, step: float = 0.05) -> None:
        self.set_volume(self._volume - step)

    def toggle_mute(self) -> None:
        if self.is_muted:
            self.is_muted = False
            self.set_volume(self._pre_mute_volume)
        else:
            self._pre_mute_volume = self._volume
            self.is_muted = True
            try:
                pygame.mixer.music.set_volume(0.0)
            except pygame.error:
                pass

    @property
    def current_track_name(self) -> str:
        if self.current_track:
            return os.path.splitext(os.path.basename(self.current_track))[0]
        return "No music"

    @property
    def track_count(self) -> int:
        return len(self.playlist)
