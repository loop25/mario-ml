"""
Decision Transformer Network Architecture.

GPT-2 style decoder-only transformer that predicts actions given
sequences of (return-to-go, observation, action) tuples.

Input sequence per context window of K timesteps:
    [R_1, s_1, a_1, R_2, s_2, a_2, ..., R_K, s_K, a_K]
    where R = return-to-go, s = state/observation, a = action

Each modality has its own embedding head:
    - Returns: Linear projection to embed_dim
    - Observations: CNN encoder + Linear projection
    - Actions: Embedding lookup (discrete actions)
    - Game token: Embedding lookup (identifies which game)
    - Timestep: Learned positional embedding

The transformer processes 3K tokens (3 per timestep: R, s, a)
and predicts actions at the action positions.

Reference:
    Chen et al. "Decision Transformer" (NeurIPS 2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (no future peeking).

    Implements the standard attention mechanism with a causal mask
    so each position can only attend to earlier positions.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, H, T, T)

        # Causal mask: prevent attending to future positions
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1,
        )
        attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Optional padding mask
        if mask is not None:
            # mask shape: (B, T) — True means valid
            # Expand to (B, 1, 1, T) for broadcasting
            pad_mask = ~mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    """Single transformer block: attention + FFN with residual connections."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


class ObservationEncoder(nn.Module):
    """CNN encoder for observations.

    Uses the same conv backbone pattern as the DQN/Rainbow networks
    but outputs an embedding vector rather than Q-values.

    Supports variable input channels (1 for grayscale, 3 for RGB).
    """

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flattened size dynamically
        self._dummy_input = torch.zeros(1, in_channels, 84, 84)
        with torch.no_grad():
            conv_out_size = self.conv(self._dummy_input).shape[1]
        del self._dummy_input

        self.projection = nn.Linear(conv_out_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations.

        Args:
            x: (B, C, H, W) float32 observations in [0, 1].

        Returns:
            (B, embed_dim) observation embeddings.
        """
        return self.projection(self.conv(x))


class DecisionTransformer(nn.Module):
    """Decision Transformer for multi-game RL.

    Architecture overview:
        1. Each timestep contributes 3 tokens: (return, obs, action)
        2. Each modality has its own embedding head
        3. Game identity is added to all embeddings
        4. GPT-2 style causal transformer processes the sequence
        5. Action predictions come from the observation token positions

    Args:
        obs_channels: Number of observation channels (1=gray, 3=RGB).
        max_action_vocab: Maximum action vocabulary size across all games.
        max_game_tokens: Maximum number of distinct games (for game embedding).
        embed_dim: Transformer embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        context_length: Maximum context window (K timesteps = 3K tokens).
        max_timestep: Maximum episode length (for positional embedding).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        obs_channels: int = 1,
        max_action_vocab: int = 64,
        max_game_tokens: int = 1024,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        context_length: int = 20,
        max_timestep: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.max_action_vocab = max_action_vocab

        # ── Modality Embeddings ──────────────────────────────────

        # Observation encoder (CNN → embed_dim)
        self.obs_encoder = ObservationEncoder(obs_channels, embed_dim)

        # Return-to-go embedding (scalar → embed_dim)
        self.return_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Tanh(),
        )

        # Action embedding (discrete index → embed_dim)
        self.action_embed = nn.Embedding(max_action_vocab, embed_dim)

        # Game identity embedding
        self.game_embed = nn.Embedding(max_game_tokens, embed_dim)

        # Learned timestep (positional) embedding
        self.timestep_embed = nn.Embedding(max_timestep, embed_dim)

        # ── Transformer ──────────────────────────────────────────

        self.embed_ln = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # ── Output Head ──────────────────────────────────────────

        # Predict actions from observation positions
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, max_action_vocab),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        game_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the Decision Transformer.

        Args:
            observations: (B, K, C, H, W) float32 observations.
            actions: (B, K) int64 action indices.
            returns_to_go: (B, K) float32 return-to-go values.
            timesteps: (B, K) int64 timestep positions.
            game_tokens: (B,) int64 game identity tokens.
            mask: (B, K) bool mask — True for valid positions.

        Returns:
            action_logits: (B, K, max_action_vocab) logits for action prediction.
        """
        B, K = actions.shape

        # ── Embed each modality ──────────────────────────────────

        # Observations: (B, K, C, H, W) → (B, K, embed_dim)
        obs_flat = observations.reshape(B * K, *observations.shape[2:])
        obs_emb = self.obs_encoder(obs_flat).reshape(B, K, self.embed_dim)

        # Returns-to-go: (B, K) → (B, K, embed_dim)
        rtg_emb = self.return_embed(returns_to_go.unsqueeze(-1))

        # Actions: (B, K) → (B, K, embed_dim)
        act_emb = self.action_embed(actions.clamp(0, self.max_action_vocab - 1))

        # Game identity: (B,) → (B, 1, embed_dim) → broadcast
        game_emb = self.game_embed(game_tokens).unsqueeze(1)  # (B, 1, E)

        # Timestep positional encoding: (B, K) → (B, K, embed_dim)
        ts_emb = self.timestep_embed(
            timesteps.clamp(0, self.timestep_embed.num_embeddings - 1)
        )

        # ── Combine embeddings ───────────────────────────────────

        # Add positional and game identity to each modality
        rtg_emb = rtg_emb + ts_emb + game_emb
        obs_emb = obs_emb + ts_emb + game_emb
        act_emb = act_emb + ts_emb + game_emb

        # Interleave: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
        # Shape: (B, 3*K, embed_dim)
        stacked = torch.stack([rtg_emb, obs_emb, act_emb], dim=2)
        sequence = stacked.reshape(B, 3 * K, self.embed_dim)

        # Create interleaved mask if provided
        if mask is not None:
            # Expand mask from (B, K) to (B, 3K)
            interleaved_mask = mask.unsqueeze(2).expand(-1, -1, 3)
            interleaved_mask = interleaved_mask.reshape(B, 3 * K)
        else:
            interleaved_mask = None

        # ── Transformer forward ──────────────────────────────────

        x = self.embed_ln(sequence)
        for block in self.blocks:
            x = block(x, mask=interleaved_mask)

        # ── Extract action predictions ───────────────────────────

        # Actions are predicted at observation positions (index 1, 4, 7, ...)
        # In the interleaved sequence: R=0, s=1, a=2, R=3, s=4, a=5, ...
        obs_positions = torch.arange(1, 3 * K, 3, device=x.device)
        obs_hidden = x[:, obs_positions, :]  # (B, K, embed_dim)

        action_logits = self.action_head(obs_hidden)  # (B, K, max_action_vocab)
        return action_logits

    def get_action(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        game_token: int,
    ) -> int:
        """Get a single action for inference (no batch dimension needed).

        Args:
            observations: (K, C, H, W) recent observations.
            actions: (K,) recent actions (last one can be dummy).
            returns_to_go: (K,) return-to-go values.
            timesteps: (K,) timestep positions.
            game_token: Integer game identity token.

        Returns:
            Predicted action index.
        """
        self.eval()
        device = next(self.parameters()).device

        # Add batch dimension
        obs = observations.unsqueeze(0).to(device)
        act = actions.unsqueeze(0).to(device)
        rtg = returns_to_go.unsqueeze(0).to(device)
        ts = timesteps.unsqueeze(0).to(device)
        gt = torch.tensor([game_token], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = self.forward(obs, act, rtg, ts, gt)
            # Take the last valid timestep's prediction
            action = logits[0, -1].argmax().item()

        return action

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
