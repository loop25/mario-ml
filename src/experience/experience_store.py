"""
Shared Experience Store for the Generalist Agent.

Unified buffer that collects tokenized trajectories from all games
and supports efficient sampling for Decision Transformer training.

The store persists trajectories to disk so they survive across sessions.
It supports:
    - Adding trajectories from any game
    - Sampling random trajectory segments (context windows) for training
    - Filtering by game or by minimum return
    - Persistence via numpy .npz files

Storage layout:
    models/generalist/experience_pool/
        index.json              — metadata index of all stored trajectories
        traj_000000.npz         — individual trajectory files
        traj_000001.npz
        ...

Usage:
    store = ExperienceStore('models/generalist/experience_pool')
    store.add_trajectory(tokenized_trajectory)

    # Sample a batch of context windows for DT training
    batch = store.sample_batch(batch_size=32, context_length=20)
"""

import json
import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from src.experience.tokenizer import TokenizedTrajectory


@dataclass
class TrajectoryMeta:
    """Metadata for a stored trajectory (written to index.json)."""
    traj_id: int
    game_token_id: int
    total_return: float
    length: int
    filename: str


class ExperienceStore:
    """Unified experience buffer across all games.

    Args:
        store_dir: Directory to persist trajectories.
        max_trajectories: Maximum number of trajectories to keep.
            When exceeded, lowest-return trajectories are evicted.
    """

    def __init__(
        self,
        store_dir: str = 'models/generalist/experience_pool',
        max_trajectories: int = 10000,
    ):
        self.store_dir = store_dir
        self.max_trajectories = max_trajectories
        self._next_id = 0
        self._index: List[TrajectoryMeta] = []

        os.makedirs(store_dir, exist_ok=True)
        self._load_index()

    # ── Persistence ────────────────────────────────────────────────

    def _index_path(self) -> str:
        return os.path.join(self.store_dir, 'index.json')

    def _load_index(self):
        """Load the trajectory index from disk."""
        path = self._index_path()
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self._index = [TrajectoryMeta(**m) for m in data.get('trajectories', [])]
            self._next_id = data.get('next_id', 0)
        else:
            self._index = []
            self._next_id = 0

    def _save_index(self):
        """Save the trajectory index to disk."""
        data = {
            'next_id': self._next_id,
            'trajectories': [asdict(m) for m in self._index],
        }
        with open(self._index_path(), 'w') as f:
            json.dump(data, f, indent=2)

    # ── Core API ───────────────────────────────────────────────────

    def add_trajectory(self, trajectory: TokenizedTrajectory) -> int:
        """Add a tokenized trajectory to the store.

        Args:
            trajectory: TokenizedTrajectory from the Tokenizer.

        Returns:
            The assigned trajectory ID.
        """
        traj_id = self._next_id
        self._next_id += 1

        filename = f'traj_{traj_id:06d}.npz'
        filepath = os.path.join(self.store_dir, filename)

        # Save trajectory data as compressed numpy archive
        np.savez_compressed(
            filepath,
            observations=trajectory.observations,
            actions=trajectory.actions,
            rewards=trajectory.rewards,
            returns_to_go=trajectory.returns_to_go,
            timesteps=trajectory.timesteps,
            dones=trajectory.dones,
        )

        meta = TrajectoryMeta(
            traj_id=traj_id,
            game_token_id=trajectory.game_token_id,
            total_return=trajectory.total_return,
            length=trajectory.length,
            filename=filename,
        )
        self._index.append(meta)

        # Evict lowest-return trajectories if over capacity
        if len(self._index) > self.max_trajectories:
            self._evict()

        self._save_index()
        return traj_id

    def _evict(self):
        """Remove lowest-return trajectories to stay under max capacity."""
        # Sort by return (ascending) and remove the worst ones
        self._index.sort(key=lambda m: m.total_return)
        num_to_remove = len(self._index) - self.max_trajectories
        removed = self._index[:num_to_remove]
        self._index = self._index[num_to_remove:]

        for meta in removed:
            filepath = os.path.join(self.store_dir, meta.filename)
            if os.path.exists(filepath):
                os.remove(filepath)

    def load_trajectory(self, meta: TrajectoryMeta) -> TokenizedTrajectory:
        """Load a full trajectory from disk.

        Args:
            meta: TrajectoryMeta from the index.

        Returns:
            TokenizedTrajectory with all data loaded.
        """
        filepath = os.path.join(self.store_dir, meta.filename)
        data = np.load(filepath)
        return TokenizedTrajectory(
            game_token_id=meta.game_token_id,
            observations=data['observations'],
            actions=data['actions'],
            rewards=data['rewards'],
            returns_to_go=data['returns_to_go'],
            timesteps=data['timesteps'],
            dones=data['dones'],
            total_return=meta.total_return,
            length=meta.length,
        )

    def sample_batch(
        self,
        batch_size: int,
        context_length: int,
        game_filter: Optional[int] = None,
        min_return: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Sample a batch of context windows for DT training.

        Each sample is a contiguous segment of length `context_length`
        from a random trajectory. Shorter trajectories are zero-padded
        and a mask is provided.

        Args:
            batch_size: Number of context windows to sample.
            context_length: Length of each context window (K in DT paper).
            game_filter: If set, only sample from this game_token_id.
            min_return: If set, only sample from trajectories with
                total_return >= this value.

        Returns:
            Dict with keys:
                'observations': (B, K, C, H, W) float32
                'actions': (B, K) int64
                'returns_to_go': (B, K) float32
                'timesteps': (B, K) int64
                'game_tokens': (B,) int64
                'mask': (B, K) bool — True for valid positions
        """
        # Filter eligible trajectories
        eligible = self._index
        if game_filter is not None:
            eligible = [m for m in eligible if m.game_token_id == game_filter]
        if min_return is not None:
            eligible = [m for m in eligible if m.total_return >= min_return]

        if not eligible:
            raise ValueError(
                "No eligible trajectories found. "
                "Add trajectories to the store first."
            )

        # Sample trajectories (with replacement if needed)
        chosen_indices = np.random.randint(0, len(eligible), size=batch_size)

        # Determine obs shape from first trajectory
        first_traj = self.load_trajectory(eligible[chosen_indices[0]])
        obs_shape = first_traj.observations.shape[1:]  # (C, H, W)

        # Pre-allocate batch arrays
        obs_batch = np.zeros(
            (batch_size, context_length, *obs_shape), dtype=np.float32,
        )
        act_batch = np.zeros((batch_size, context_length), dtype=np.int64)
        rtg_batch = np.zeros((batch_size, context_length), dtype=np.float32)
        ts_batch = np.zeros((batch_size, context_length), dtype=np.int64)
        game_batch = np.zeros(batch_size, dtype=np.int64)
        mask_batch = np.zeros((batch_size, context_length), dtype=bool)

        for i, idx in enumerate(chosen_indices):
            meta = eligible[idx]
            traj = self.load_trajectory(meta) if i > 0 or idx != chosen_indices[0] else first_traj

            T = traj.length
            if T <= context_length:
                # Use entire trajectory, zero-pad the rest
                seg_len = T
                start = 0
            else:
                # Random start position
                start = np.random.randint(0, T - context_length + 1)
                seg_len = context_length

            end = start + seg_len

            # Normalize observations to float32 [0, 1]
            obs_batch[i, :seg_len] = (
                traj.observations[start:end].astype(np.float32) / 255.0
            )
            act_batch[i, :seg_len] = traj.actions[start:end]
            rtg_batch[i, :seg_len] = traj.returns_to_go[start:end]
            ts_batch[i, :seg_len] = traj.timesteps[start:end]
            game_batch[i] = traj.game_token_id
            mask_batch[i, :seg_len] = True

        return {
            'observations': obs_batch,
            'actions': act_batch,
            'returns_to_go': rtg_batch,
            'timesteps': ts_batch,
            'game_tokens': game_batch,
            'mask': mask_batch,
        }

    # ── Query API ──────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of trajectories in the store."""
        return len(self._index)

    def get_stats(self) -> Dict:
        """Get summary statistics about the stored experience."""
        if not self._index:
            return {
                'total_trajectories': 0,
                'total_timesteps': 0,
                'games': {},
            }

        games: Dict[int, Dict] = {}
        total_timesteps = 0

        for meta in self._index:
            total_timesteps += meta.length
            gid = meta.game_token_id
            if gid not in games:
                games[gid] = {
                    'count': 0,
                    'total_timesteps': 0,
                    'avg_return': 0.0,
                    'max_return': float('-inf'),
                    'min_return': float('inf'),
                }
            g = games[gid]
            g['count'] += 1
            g['total_timesteps'] += meta.length
            g['max_return'] = max(g['max_return'], meta.total_return)
            g['min_return'] = min(g['min_return'], meta.total_return)

        # Compute averages
        for gid, g in games.items():
            returns = [
                m.total_return for m in self._index
                if m.game_token_id == gid
            ]
            g['avg_return'] = float(np.mean(returns))

        return {
            'total_trajectories': len(self._index),
            'total_timesteps': total_timesteps,
            'games': games,
        }

    def get_game_ids(self) -> List[int]:
        """Get all unique game_token_ids in the store."""
        return list(set(m.game_token_id for m in self._index))

    def clear(self):
        """Remove all trajectories from the store."""
        for meta in self._index:
            filepath = os.path.join(self.store_dir, meta.filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        self._index = []
        self._next_id = 0
        self._save_index()
