"""
networks/base_network.py
------------------------
Abstract network interface that all policy networks must implement.

Design principles
-----------------
- BaseNetwork defines the contract that agents call.  Agents never import
  concrete network classes; they hold a BaseNetwork reference.
- obs_shape and action_space_size are NOT properties of the network.
  They belong to the problem and are used to build the network, not stored on it.
- The three abstract methods (forward, get_action_and_log_prob,
  evaluate_actions) are the ONLY interface agents use.
- Device management is centralised here so concrete classes don't repeat it.

Concrete networks must inherit BOTH nn.Module and BaseNetwork.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseNetwork(nn.Module, ABC):
    """
    Abstract base for all policy + value networks.

    Subclassing
    -----------
    class MyNetwork(BaseNetwork):
        def __init__(self, obs_shape, action_space_size, cfg):
            super().__init__()
            # build layers here

        def forward(self, obs, action_mask=None):
            ...
            return logits, value

        def get_action_and_log_prob(self, obs, action_mask=None, deterministic=False):
            ...
            return action, log_prob, value

        def evaluate_actions(self, obs, actions, action_mask=None):
            ...
            return log_probs, values, entropy
    """

    # ------------------------------------------------------------------
    # Abstract methods — the agent/algorithm interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(
        self,
        obs,
        action_mask: Optional[torch.Tensor] = None,
        context=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        obs         : (B, *obs_shape) float32
        action_mask : (B, action_space_size) bool, True=feasible.  None = no masking.

        Returns
        -------
        logits : (B, action_space_size) — raw, masked, tanh-clipped scores
        value  : (B,) — critic estimate
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
        Sample or deterministically select an action.

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
        Evaluate stored actions under the current policy.  Used in PPO update.

        Returns
        -------
        log_probs : (B,) float32
        values    : (B,) float32
        entropy   : (B,) float32
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers (available to all subclasses)
    # ------------------------------------------------------------------

    def to_device(self, device: str) -> "BaseNetwork":
        """Move network to device and return self for chaining."""
        return self.to(torch.device(device))

    @staticmethod
    def _apply_mask(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Zero-out infeasible logits with -inf."""
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    @staticmethod
    def _ortho_init(module: nn.Module, gain: float = 1.414) -> None:
        """Apply orthogonal initialisation to all Linear layers."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
