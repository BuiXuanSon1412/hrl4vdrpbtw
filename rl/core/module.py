"""
core/module.py
--------------
Abstract base class for all policy + value networks, plus the
shared neural network sub-modules reused across architectures.

What lives here
---------------
  BaseNetwork          — abstract interface agents call (forward,
                         get_action_and_log_prob, evaluate_actions)
  _MHA                 — multi-head attention (self + cross)
  _FF                  — position-wise feed-forward block
  _InstanceNormWrapper — InstanceNorm1d with the sequence-dim swap
  _make_norm           — factory: InstanceNorm or LayerNorm by flag

What does NOT live here
-----------------------
  Concrete network architectures (PolicyNetwork / HACN) → networks/hacn.py
  Problem-specific constants (NODE_FEAT_DIM etc.)       → problems/vrpbtw.py
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared sub-modules
# ---------------------------------------------------------------------------


def _make_norm(use_instance_norm: bool, dim: int) -> nn.Module:
    """Return InstanceNorm or LayerNorm depending on the flag."""
    return _InstanceNormWrapper(dim) if use_instance_norm else nn.LayerNorm(dim)


class _InstanceNormWrapper(nn.Module):
    """InstanceNorm1d that works on (B, T, D) tensors."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(dim, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) → transpose → norm → transpose back
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _MHA(nn.Module):
    """
    Multi-head attention supporting both self-attention and cross-attention.

    Parameters
    ----------
    D       : embedding dimension
    H       : number of heads (must divide D)
    dropout : dropout on attention weights
    """

    def __init__(self, D: int, H: int, dropout: float = 0.0):
        super().__init__()
        assert D % H == 0, f"embed_dim {D} must be divisible by n_heads {H}"
        self.H = H
        self.Dh = D // H
        self.D = D
        self.Wq = nn.Linear(D, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self.Wv = nn.Linear(D, D, bias=False)
        self.Wo = nn.Linear(D, D, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,  # (B, Tq, D)
        k: torch.Tensor,  # (B, Tk, D)
        v: torch.Tensor,  # (B, Tk, D)
        mask: Optional[torch.Tensor] = None,  # (B, Tq, Tk) bool True=ignore
    ) -> torch.Tensor:
        B, Tq, _ = q.shape
        Tk = k.shape[1]
        H, Dh = self.H, self.Dh

        def reshape(t: torch.Tensor, T: int) -> torch.Tensor:
            return t.view(B, T, H, Dh).transpose(1, 2)

        Q = reshape(self.Wq(q), Tq)
        K = reshape(self.Wk(k), Tk)
        V = reshape(self.Wv(v), Tk)
        sc = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)
        if mask is not None:
            sc = sc.masked_fill(mask.unsqueeze(1), float("-inf"))
        at = self.drop(torch.softmax(sc, dim=-1))
        at = torch.nan_to_num(at, nan=0.0)
        out = torch.matmul(at, V).transpose(1, 2).contiguous().view(B, Tq, self.D)
        return self.Wo(out)


class _FF(nn.Module):
    """Position-wise feed-forward: Linear → ReLU → Dropout → Linear."""

    def __init__(self, D: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(D * 4, D),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# BaseNetwork  (abstract)
# ---------------------------------------------------------------------------


class BaseNetwork(nn.Module, ABC):
    """
    Abstract base for all policy + value networks.

    Agents hold a BaseNetwork reference and call only these three methods.
    Concrete architectures must subclass both nn.Module and BaseNetwork.

    Abstract methods
    ----------------
    forward               → (logits, value)
    get_action_and_log_prob → (action, log_prob, value)
    evaluate_actions      → (log_probs, values, entropy)
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(
        self,
        obs,
        action_mask: Optional[torch.Tensor] = None,
        context=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits : (B, action_space_size)   raw masked scores
        value  : (B,)                     critic estimate
        """
        ...

    @abstractmethod
    def get_action_and_log_prob(
        self,
        obs,
        action_mask: Optional[torch.Tensor] = None,
        context=None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        action   : (B,) int64
        log_prob : (B,) float32
        value    : (B,) float32
        """
        ...

    @abstractmethod
    def evaluate_actions(
        self,
        obs,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        context=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used in PPO update.

        Returns
        -------
        log_probs : (B,) float32
        values    : (B,) float32
        entropy   : (B,) float32
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers available to all subclasses
    # ------------------------------------------------------------------

    def to_device(self, device: str) -> "BaseNetwork":
        return self.to(torch.device(device))

    @staticmethod
    def _apply_mask(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    @staticmethod
    def _ortho_init(module: nn.Module, gain: float = 1.414) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
