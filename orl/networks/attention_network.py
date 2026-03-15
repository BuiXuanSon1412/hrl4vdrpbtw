"""
networks/attention_network.py
------------------------------
Transformer encoder + context-pointer decoder policy/value network.

Architecture
------------
  Encoder : Multi-head self-attention over node features  → node embeddings
  Decoder : Mean-pooled context → Linear → logits over action space
  Critic  : Separate MLP on mean-pooled context

For graph-structured problems (VRP, TSP) use PointerNetwork instead;
this network works with any flat or 2D observation.

Key correctness fix vs original
---------------------------------
- PolicyNetwork now inherits BaseNetwork (was only nn.Module).
- The decoder for graph problems should be PointerNetwork, not this class.
  This class is for problems where action_space_size is FIXED and known
  at construction time (e.g. Knapsack: 2 actions).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_network import BaseNetwork
from configs import NetworkConfig


# ---------------------------------------------------------------------------
# Attention sub-modules
# ---------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.head_dim
        Q = self.q(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.k(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v(x).view(B, T, H, Dh).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = (
            torch.matmul(attn, V)
            .transpose(1, 2)
            .contiguous()
            .view(B, T, self.embed_dim)
        )
        return self.out(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.attn(x)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ---------------------------------------------------------------------------
# AttentionNetwork
# ---------------------------------------------------------------------------


class AttentionNetwork(BaseNetwork):
    """
    Encoder-decoder attention network for combinatorial RL.

    Parameters
    ----------
    obs_shape        : From problem.observation_shape.
    action_space_size: From problem.action_space_size.  MUST be fixed.
    cfg              : NetworkConfig (architecture hyperparameters only).
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_space_size: int,
        cfg: NetworkConfig,
    ):
        super().__init__()
        self._obs_shape = obs_shape
        self._action_space_size = action_space_size
        self.cfg = cfg
        D = cfg.embed_dim

        if cfg.network_type == "mlp" or len(obs_shape) == 1:
            # Flat observation — MLP encoder
            flat_dim = int(np.prod(obs_shape))
            self.encoder = nn.Sequential(
                nn.Linear(flat_dim, D),
                nn.ReLU(),
                nn.Linear(D, D),
                nn.ReLU(),
            )
            self._use_attention = False
        else:
            # Graph observation (N, feat_dim) — Transformer encoder
            feat_dim = obs_shape[-1]
            self.node_embed = nn.Linear(feat_dim, D)
            self.encoder_layers = nn.ModuleList(
                [
                    TransformerEncoderLayer(D, cfg.n_heads, cfg.dropout)
                    for _ in range(cfg.n_encoder_layers)
                ]
            )
            self.context_proj = nn.Linear(D, D, bias=False)
            self._use_attention = True

        # Policy head: context (D,) → logits (action_space_size,)
        self.policy_head = nn.Linear(D, action_space_size)

        # Value head: separate from policy to reduce gradient interference
        # sg(context) prevents value loss from flowing into the shared encoder
        self.value_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.Tanh(),
            nn.Linear(D // 2, 1),
        )

        if cfg.ortho_init:
            self._ortho_init(self)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode obs → context vector (B, D)."""
        if self._use_attention:
            h = self.node_embed(obs)
            for layer in self.encoder_layers:
                h = layer(h)
            return self.context_proj(h.mean(dim=1))  # mean pooling → (B, D)
        else:
            B = obs.shape[0]
            return self.encoder(obs.view(B, -1))

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        context=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = self._encode(obs)  # (B, D)
        logits = torch.tanh(self.policy_head(context)) * self.cfg.clip_logits
        logits = self._apply_mask(logits, action_mask)
        # Stop-gradient on value input: prevents value loss from corrupting encoder
        value = self.value_head(context.detach()).squeeze(-1)  # (B,)
        return logits, value

    def get_action_and_log_prob(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        context=None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        context=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()
