"""
networks/pointer_network.py
----------------------------
Pointer Network with Transformer encoder for variable-size graph problems.

Key design
----------
- Output size is DYNAMIC: logits.shape == (B, N_nodes_in_input).
  No fixed Linear(D → N) decoder.  Works for N=5, N=50 with the same weights.
- Inherits BaseNetwork correctly.
- action_space_size = max_nodes (used for buffer allocation only; actual
  logits are always padded/trimmed to match the input).

Architecture
------------
  Encoder : Transformer over node features    → (B, N, D)
  Context : Learned projection of mean pooling → (B, D)
  Decoder : Dot-product attention context × nodes → (B, N) logits  (pointer)
  Value   : MLP on context (stop-gradient)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base_network import BaseNetwork
from configs import NetworkConfig


# ---------------------------------------------------------------------------
# Encoder sub-modules  (shared with AttentionNetwork)
# ---------------------------------------------------------------------------


class _MHA(nn.Module):
    def __init__(self, D: int, H: int, dropout: float):
        super().__init__()
        assert D % H == 0
        self.H, self.Dh, self.D = H, D // H, D
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        def proj(lin, t):
            return lin(t).view(B, T, self.H, self.Dh).transpose(1, 2)

        Q, K, V = proj(self.q, x), proj(self.k, x), proj(self.v, x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.Dh)
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, self.D)
        return self.out(out)


class _EncoderLayer(nn.Module):
    def __init__(self, D: int, H: int, dropout: float):
        super().__init__()
        self.attn = _MHA(D, H, dropout)
        self.ff = nn.Sequential(
            nn.Linear(D, D * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(D * 4, D)
        )
        self.n1 = nn.LayerNorm(D)
        self.n2 = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.n1(x + self.drop(self.attn(x)))
        x = self.n2(x + self.drop(self.ff(x)))
        return x


# ---------------------------------------------------------------------------
# Pointer Network
# ---------------------------------------------------------------------------


class PointerNetwork(BaseNetwork):
    """
    Variable-size pointer network.

    Parameters
    ----------
    feat_dim         : Node feature dimension (from problem.observation_shape[-1]).
    max_nodes        : Upper bound on N; used for buffer allocation.
    cfg              : NetworkConfig.
    """

    def __init__(
        self,
        feat_dim: int,
        cfg: NetworkConfig,
    ):
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dim

        # Encoder
        self.node_embed = nn.Linear(feat_dim, D)
        self.encoder_layers = nn.ModuleList(
            [
                _EncoderLayer(D, cfg.n_heads, cfg.dropout)
                for _ in range(cfg.n_encoder_layers)
            ]
        )

        # Pointer decoder: query from context, keys from node embeddings
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, D, bias=False)
        self._scale = math.sqrt(D)

        # Value head (stop-gradient on input)
        self.value_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.Tanh(),
            nn.Linear(D // 2, 1),
        )

        if cfg.ortho_init:
            self._ortho_init(self)

    def _encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs: (B, N, feat_dim) → node_emb (B, N, D), context (B, D)
        N is variable; no size constraint.
        """
        h = self.node_embed(obs)
        for layer in self.encoder_layers:
            h = layer(h)
        context = h.mean(dim=1)  # (B, D)
        return h, context

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        context=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs         : (B, N, feat_dim)
        action_mask : (B, N) bool

        Returns
        -------
        logits : (B, N)  — dynamic, matches input N
        value  : (B,)
        """
        node_emb, context = self._encode(obs)

        # Pointer attention
        query = self.W_q(context).unsqueeze(1)  # (B, 1, D)
        keys = self.W_k(node_emb)  # (B, N, D)
        logits = (
            torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / self._scale
        )  # (B, N)
        logits = torch.tanh(logits) * self.cfg.clip_logits
        logits = self._apply_mask(logits, action_mask)

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
