"""
core/agent.py
-------------
Agent abstraction: policy only.

An agent maps observations to actions.  It has no opinion on how it is
trained — that is entirely the trainer's responsibility.

  BaseAgent    — abstract interface (network, prepare_obs, select_action,
                                     save, load, clone)
  PolicyAgent  — single concrete implementation; works with any trainer

"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from core.buffer import RolloutBuffer
from core.policy import BasePolicy


# ---------------------------------------------------------------------------
# BaseAgent  — policy interface
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """
    Policy contract: obs → action.

    An agent holds a policy network and knows how to:
      - select an action given an observation and mask (select_action)
      - persist and restore the network weights (save / load)
      - produce an independent copy of itself (clone)

    Anything related to *training* — optimizers, loss functions, rollout
    collection, gradient updates — belongs to the Trainer, not the Agent.
    """

    @property
    @abstractmethod
    def policy(self) -> BasePolicy:
        """The policy network.  Always non-None for concrete agents."""
        ...

    @property
    def estimator(self) -> Optional[Any]:
        """The loss estimator (optional, used for training)."""
        return None

    @classmethod
    @abstractmethod
    def from_config(
        cls, cfg: dict, policy: BasePolicy, estimator: Optional[Any], opt_policy: Optional[optim.Optimizer] = None
    ) -> "BaseAgent":
        """Factory method: instantiate agent from config, policy, estimator, and optimizer."""
        ...

    @abstractmethod
    def select_action(
        self,
        obs: Any,
        action_mask: np.ndarray,
        training: bool = True,
    ) -> Tuple[int, float, float]:
        """Return (action, log_prob, value)."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist network weights to *path*."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore network weights from *path*."""
        ...

    def clone(self) -> "BaseAgent":
        """Deep copy — used by MetaTrainer for inner-loop fast agents."""
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# PolicyAgent  — the one concrete agent
# ---------------------------------------------------------------------------


class PolicyAgent(BaseAgent):
    """
    Policy-only agent for RL training.

    Properties:
      policy      — policy network (π_θ)
      estimator   — loss computation (PPO, A2C, REINFORCE, etc.)
      opt_policy  — optimizer for policy

    Methods:
      select_action(obs, mask, training) — sample or greedy action
      update(buffer) — perform gradient updates using the estimator
    """

    def __init__(
        self,
        policy: BasePolicy,
        estimator: Optional[Any] = None,
        opt_policy: Optional[optim.Optimizer] = None,
        device: str = "cpu",
    ):
        self._policy = policy
        self._estimator = estimator
        self._opt_policy = opt_policy
        self.device = device

    @property
    def policy(self) -> BasePolicy:
        return self._policy

    @property
    def estimator(self) -> Optional[Any]:
        return self._estimator

    @property
    def opt_policy(self) -> Optional[optim.Optimizer]:
        return self._opt_policy

    @classmethod
    def from_config(
        cls, cfg: dict, policy: BasePolicy, estimator: Optional[Any], opt_policy: Optional[optim.Optimizer] = None
    ) -> "PolicyAgent":
        """Factory method: instantiate PolicyAgent from config, policy, estimator, and optimizer.

        Config keys:
        - device: device string (cpu/cuda)
        """
        device = cfg.get("device", "cpu")
        return cls(policy=policy, estimator=estimator, opt_policy=opt_policy, device=device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: Any,
        action_mask: np.ndarray,
        training: bool = True,
    ) -> Tuple[int, float, float]:
        from core.utils import obs_to_tensor

        with torch.no_grad():
            obs_t = obs_to_tensor(obs, device=self.device)
            mask_t = torch.tensor(
                action_mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0)
            action_t, lp_t, val_t = self._policy.get_action_and_log_prob(
                obs_t, mask_t, deterministic=not training
            )
        return int(action_t.item()), float(lp_t.item()), float(val_t.item())

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def update(self, buffer: RolloutBuffer) -> float:
        """
        Perform one gradient update using the collected buffer.

        Args:
            buffer: RolloutBuffer with transitions, advantages, returns.

        Returns:
            scalar loss value.
        """
        if self._estimator is None:
            raise ValueError("PolicyAgent.update() requires an estimator")
        if self._opt_policy is None:
            raise ValueError("PolicyAgent.update() requires opt_policy")

        loss = self._estimator.compute_loss(self._policy, buffer)
        self._opt_policy.zero_grad()
        loss.backward()
        self._opt_policy.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Persistence  (network weights only — trainers save optimizer state)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save network weights.  Trainers may write additional keys to the
        same file (optimizer state, training counters) via torch.save."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"network_state": self._policy.state_dict()}, path)

    def load(self, path: str) -> None:
        """Load network weights.  Ignores any extra keys written by the
        trainer so that evaluate.py can load training checkpoints directly."""
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._policy.load_state_dict(ckpt["network_state"])

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self._policy.parameters())
        return (
            f"PolicyAgent(network={type(self._policy).__name__}, "
            f"params={n_params:,}, device={self.device!r})"
        )
