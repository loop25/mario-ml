"""
PyTorch Dataset for Decision Transformer training.

Wraps the ExperienceStore to provide random context-window samples
compatible with PyTorch's DataLoader.

Each sample is a contiguous segment of `context_length` timesteps
from a trajectory, containing interleaved (return, obs, action) data.

Usage:
    from src.experience.experience_store import ExperienceStore
    from src.algorithms.decision_transformer.trajectory_dataset import TrajectoryDataset
    from torch.utils.data import DataLoader

    store = ExperienceStore('models/generalist/experience_pool')
    dataset = TrajectoryDataset(store, context_length=20)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional

from src.experience.experience_store import ExperienceStore


class TrajectoryDataset(Dataset):
    """PyTorch Dataset that samples context windows from the Experience Store.

    Each __getitem__ call returns a random context window from a random
    trajectory. The dataset length is the total number of valid starting
    positions across all trajectories.

    Args:
        store: ExperienceStore with tokenized trajectories.
        context_length: Number of timesteps per sample (K in the DT paper).
        game_filter: If set, only use trajectories from this game.
        min_return_pct: If set, only use top (1-min_return_pct) percentile
            of trajectories by return (e.g., 0.0 = all, 0.5 = top half).
    """

    def __init__(
        self,
        store: ExperienceStore,
        context_length: int = 20,
        game_filter: Optional[int] = None,
        min_return_pct: float = 0.0,
    ):
        self.store = store
        self.context_length = context_length

        # Build index of (trajectory_meta, start_position) pairs
        eligible = store._index
        if game_filter is not None:
            eligible = [m for m in eligible if m.game_token_id == game_filter]

        # Filter by return percentile
        if min_return_pct > 0.0 and eligible:
            returns = sorted([m.total_return for m in eligible])
            threshold = returns[int(len(returns) * min_return_pct)]
            eligible = [m for m in eligible if m.total_return >= threshold]

        self._eligible = eligible

        # Build flat index: each entry is (meta_index, start_pos)
        self._samples = []
        for i, meta in enumerate(self._eligible):
            if meta.length <= context_length:
                # Short trajectory: one sample with padding
                self._samples.append((i, 0))
            else:
                # Multiple possible starting positions
                for start in range(meta.length - context_length + 1):
                    self._samples.append((i, start))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta_idx, start = self._samples[idx]
        meta = self._eligible[meta_idx]
        traj = self.store.load_trajectory(meta)

        K = self.context_length
        T = traj.length
        seg_len = min(K, T - start)

        # Determine obs shape
        obs_shape = traj.observations.shape[1:]  # (C, H, W)

        # Allocate arrays (zero-padded)
        obs = np.zeros((K, *obs_shape), dtype=np.float32)
        actions = np.zeros(K, dtype=np.int64)
        rtg = np.zeros(K, dtype=np.float32)
        timesteps = np.zeros(K, dtype=np.int64)
        mask = np.zeros(K, dtype=bool)

        # Fill valid portion
        obs[:seg_len] = traj.observations[start:start + seg_len].astype(np.float32) / 255.0
        actions[:seg_len] = traj.actions[start:start + seg_len]
        rtg[:seg_len] = traj.returns_to_go[start:start + seg_len]
        timesteps[:seg_len] = traj.timesteps[start:start + seg_len]
        mask[:seg_len] = True

        return {
            'observations': torch.from_numpy(obs),
            'actions': torch.from_numpy(actions),
            'returns_to_go': torch.from_numpy(rtg),
            'timesteps': torch.from_numpy(timesteps),
            'game_tokens': torch.tensor(traj.game_token_id, dtype=torch.long),
            'mask': torch.from_numpy(mask),
        }
