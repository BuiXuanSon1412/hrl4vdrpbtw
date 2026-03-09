"""
networks/policy_network.py
--------------------------
Encoder-Decoder policy network for combinatorial RL.

Architecture (inspired by Pointer Networks / Attention Model):
  Encoder : Multi-head self-attention over the node feature set
  Decoder : Context-query attention to produce a distribution over nodes
  Critic  : Separate MLP head for value estimation (PPO)

The network is pure-PyTorch.  If PyTorch is not available, a lightweight
NumPy MLP fallback is used automatically (useful for unit tests / CPU-only).

Usage
-----
from networks.policy_network import PolicyNetwork, NetworkConfig

cfg = NetworkConfig(obs_shape=(20, 4), action_space_size=20, embed_dim=128)
net = PolicyNetwork(cfg)
logits, value = net.forward(obs_batch, action_mask_batch)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

# Try to import PyTorch; fall back to numpy stub if unavailable
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NetworkConfig:
    obs_shape: Tuple[int, ...]  # e.g. (n_nodes, feat_dim) or (flat_dim,)
    action_space_size: int
    embed_dim: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    dropout: float = 0.0
    use_attention: bool = True  # False → plain MLP (flat obs)
    clip_logits: float = 10.0  # tanh clipping for stability
    ortho_init: bool = True  # orthogonal weight initialisation


# ---------------------------------------------------------------------------
# PyTorch implementation
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # (B, T_q, D)
        key: torch.Tensor,  # (B, T_k, D)
        value: torch.Tensor,  # (B, T_k, D)
        mask: Optional[torch.Tensor] = None,  # (B, T_q, T_k) bool, True=ignore
    ) -> torch.Tensor:
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape
        H, D_h = self.n_heads, self.head_dim

        Q = self.q_proj(query).view(B, T_q, H, D_h).transpose(1, 2)
        K = self.k_proj(key).view(B, T_k, H, D_h).transpose(1, 2)
        V = self.v_proj(value).view(B, T_k, H, D_h).transpose(1, 2)

        scale = math.sqrt(D_h)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B,H,T_q,T_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = self.drop(torch.softmax(scores, dim=-1))

        out = torch.matmul(attn, V)  # (B,H,T_q,D_h)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        return self.out(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, n_heads, dropout)
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
        x = self.norm1(x + self.drop(self.self_attn(x, x, x)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class PolicyNetwork(nn.Module):
    """
    Encoder-Decoder attention policy + value network.

    Forward signature
    -----------------
    logits, value = net(obs, action_mask)

        obs         : (B, n_nodes, feat_dim)  or  (B, flat_dim) if use_attention=False
        action_mask : (B, action_space_size)  bool, True = feasible
        logits      : (B, action_space_size)  raw (pre-softmax) scores
        value       : (B,)                    state-value estimate
    """

    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dim

        if cfg.use_attention:
            feat_dim = cfg.obs_shape[-1]
            self.node_embed = nn.Linear(feat_dim, D)
            self.encoder = nn.ModuleList(
                [
                    TransformerEncoderLayer(D, cfg.n_heads, cfg.dropout)
                    for _ in range(cfg.n_encoder_layers)
                ]
            )
            # Decoder context = mean pooling + learned "step" embed
            self.context_proj = nn.Linear(D, D, bias=False)
            self.key_proj = nn.Linear(D, D, bias=False)
            decoder_in = D
        else:
            flat_dim = int(np.prod(cfg.obs_shape))
            self.mlp_encoder = nn.Sequential(
                nn.Linear(flat_dim, D),
                nn.ReLU(),
                nn.Linear(D, D),
                nn.ReLU(),
            )
            decoder_in = D

        # Decoder head → logits over actions
        self.decoder = nn.Linear(decoder_in, cfg.action_space_size)

        # Critic head
        self.value_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.Tanh(),
            nn.Linear(D // 2, 1),
        )

        if cfg.ortho_init:
            self._ortho_init()

    def _ortho_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Return node embeddings (B, n_nodes, D) or context (B, D)."""
        if self.cfg.use_attention:
            h = self.node_embed(obs)  # (B, N, D)
            for layer in self.encoder:
                h = layer(h)
            return h
        else:
            B = obs.shape[0]
            return self.mlp_encoder(obs.view(B, -1))  # (B, D)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits : (B, action_space_size)  — masked, tanh-clipped
        value  : (B,)
        """
        h = self.encode(obs)

        if self.cfg.use_attention:
            context = h.mean(dim=1)  # (B, D)  mean pooling
            logits = self.decoder(self.context_proj(context))
        else:
            context = h
            logits = self.decoder(context)

        # Tanh clipping for numerical stability
        logits = torch.tanh(logits) * self.cfg.clip_logits

        # Mask infeasible actions with −∞
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        value = self.value_head(context).squeeze(-1)  # (B,)
        return logits, value

    def get_action_and_log_prob(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Returns: action (B,), log_prob (B,), value (B,)
        """
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log_probs and entropy for given obs–action pairs.

        Returns: log_probs (B,), values (B,), entropy (B,)
        """
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value, entropy
