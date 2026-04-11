"""Model opponent — uses a trained RL checkpoint as an opponent."""

import os
import random

import numpy as np

from .base_opponent import BaseOpponent


class ModelOpponent(BaseOpponent):
    """Uses a trained RL model checkpoint as an opponent.

    Supports Stable Baselines 3 checkpoints (``.zip``) and raw PyTorch
    checkpoints (``.pt`` / ``.pth``).  Falls back to random action
    selection when the model cannot be loaded or inference fails.

    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint file.
    device : str
        Device for inference (default ``'cpu'``).
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        self._path = checkpoint_path
        self._device = device
        self._basename = os.path.basename(checkpoint_path)
        self._model = None
        self._model_type = None  # 'sb3' or 'torch'
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._path.endswith('.zip'):
            self._model_type = 'sb3'
            self._load_sb3()
        elif self._path.endswith('.pt') or self._path.endswith('.pth'):
            self._model_type = 'torch'
            self._load_torch()

    def _load_sb3(self):
        try:
            from stable_baselines3 import PPO, DQN, A2C
        except ImportError:
            print('  [ModelOpponent] stable_baselines3 not available')
            return
        for algo_cls in [PPO, DQN, A2C]:
            try:
                self._model = algo_cls.load(self._path, device=self._device)
                return
            except Exception:
                continue
        print(f'  [ModelOpponent] Could not load SB3 model: {self._path}')

    def _load_torch(self):
        try:
            import torch
            self._model = torch.load(
                self._path, map_location=self._device, weights_only=False,
            )
        except Exception as e:
            print(f'  [ModelOpponent] Could not load torch model: {e}')

    # ------------------------------------------------------------------
    # BaseOpponent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f'Model ({self._basename})'

    @property
    def difficulty_tier(self) -> int:
        return 5 if 'best' in self._basename.lower() else 4

    def pick_action(self, board_state: dict) -> int:
        valid = board_state.get('valid_actions', [])
        if not valid:
            return 0

        if self._model is None:
            return random.choice(valid)

        try:
            board = board_state['board']
            if self._model_type == 'sb3':
                return self._predict_sb3(board, valid)
            if self._model_type == 'torch':
                return self._predict_torch(board, valid)
        except Exception as e:
            print(f'  [ModelOpponent] Inference error, falling back to random: {e}')

        return random.choice(valid)

    def reset(self) -> None:
        """No-op — model is stateless per action."""

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _predict_sb3(self, board, valid):
        obs = np.array(board, dtype=np.float32)
        if obs.ndim == 2:
            obs = obs.flatten()
        action, _ = self._model.predict(obs, deterministic=True)
        action = int(action)
        if action in valid:
            return action
        return random.choice(valid)

    def _predict_torch(self, board, valid):
        import torch

        obs = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            if hasattr(self._model, 'forward'):
                logits = self._model(obs)
            elif isinstance(self._model, dict) and 'policy' in self._model:
                logits = self._model['policy'](obs)
            else:
                return random.choice(valid)
            action = int(logits.argmax(dim=-1).item())
            if action in valid:
                return action
        return random.choice(valid)
