"""
Generates highlight reels from recorded training videos.

Reads marker sidecar files (created by MilestoneRecorder) and produces
a compilation video of the best moments using OpenCV.
"""

import json
import os
from typing import List

import cv2


class HighlightReel:
    """Generates highlight reels from recorded training videos."""

    def __init__(self, clip_duration: float = 5.0, max_clips: int = 10,
                 output_dir: str = 'recordings/highlights'):
        self.clip_duration = clip_duration    # Seconds around each marker
        self.max_clips = max_clips
        self.output_dir = output_dir

    def load_markers(self, markers_path: str) -> List[dict]:
        """Load markers from a JSON sidecar file."""
        with open(markers_path, 'r') as f:
            data = json.load(f)
        return data.get('markers', [])

    def select_highlights(self, markers: List[dict]) -> List[dict]:
        """Select the best markers for the reel.

        Priority: new_best_reward > achievement > episode_milestone
        Limit to max_clips, sorted by time.
        """
        priority = {'new_best_reward': 3, 'achievement': 2, 'episode_milestone': 1}
        scored = [(m, priority.get(m.get('type', ''), 0)) for m in markers]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [m for m, _ in scored[:self.max_clips]]
        selected.sort(key=lambda m: m.get('time', 0))
        return selected

    def generate_reel_metadata(self, recording_path: str, markers_path: str) -> dict:
        """Generate metadata for a highlight reel without actually cutting video.

        Returns dict with clip definitions that could be used by video
        processing.
        """
        markers = self.load_markers(markers_path)
        highlights = self.select_highlights(markers)

        clips = []
        for marker in highlights:
            start_time = max(0, marker['time'] - self.clip_duration / 2)
            end_time = marker['time'] + self.clip_duration / 2
            clips.append({
                'start_time': start_time,
                'end_time': end_time,
                'marker_type': marker.get('type', ''),
                'description': marker.get('description', ''),
                'value': marker.get('value'),
            })

        return {
            'source_recording': recording_path,
            'source_markers': markers_path,
            'total_clips': len(clips),
            'clip_duration': self.clip_duration,
            'clips': clips,
        }

    def extract_clips_from_video(self, recording_path: str, reel_metadata: dict,
                                 output_path: str = None) -> str:
        """Extract clips from a video file and concatenate into a highlight reel.

        Uses OpenCV VideoCapture/VideoWriter.
        Returns output file path.
        """
        if output_path is None:
            os.makedirs(self.output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(recording_path))[0]
            output_path = os.path.join(self.output_dir, f'{base}_highlights.mp4')

        clips = reel_metadata.get('clips', [])
        if not clips:
            return output_path

        cap = cv2.VideoCapture(recording_path)
        if not cap.isOpened():
            raise IOError(f'Cannot open video: {recording_path}')

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for clip in clips:
            start_frame = int(clip['start_time'] * fps)
            end_frame = int(clip['end_time'] * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break

                # Add title card overlay for first 30 frames (~1 second at 30fps)
                if frame_idx - start_frame < 30:
                    desc = clip.get('description', '')
                    if desc:
                        cv2.putText(frame, desc, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 210, 255), 2)

                writer.write(frame)

        writer.release()
        cap.release()
        return output_path

    def generate_reel(self, recording_path: str, markers_path: str,
                      output_path: str = None) -> str:
        """Full pipeline: load markers -> select highlights -> cut video -> save.

        Returns output path.
        """
        metadata = self.generate_reel_metadata(recording_path, markers_path)
        return self.extract_clips_from_video(recording_path, metadata, output_path)
