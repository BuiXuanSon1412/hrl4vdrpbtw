"""
core/agent.py
-------------
Agent abstraction: policy only.

An agent maps observations to actions.  It has no opinion on how it is
trained — that is entirely the trainer's responsibility.

  BaseAgent    — abstract interface (network, prepare_obs, select_action,
                                     save, load, clone)
  PolicyAgent  — single concrete implementation; works with any trainer

Module-level helpers
--------------------
  _obs_to_tensor   — convert a single obs dict to a B=1 tensor

This helper is used by estimators to tensorize observations.
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
      - tensorise an observation (prepare_obs)
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

    @abstractmethod
    def prepare_obs(self, obs: Any) -> Any:
        """Add batch dimension and tensorize a single obs dict."""
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


class Agent(BaseAgent):
    """
    Policy agent with learnable policy and value functions.

    Properties:
      policy      — policy network (π_θ)
      value_fn    — value function network (V_ϕ, if separate from policy)
      estimator   — loss computation (PPO, A2C, etc.)
      opt_policy  — optimizer for policy
      opt_value   — optimizer for value function

    Methods:
      select_action(obs, mask, training) — sample or greedy action
      update(buffer) — perform gradient updates using the estimator
    """

    def __init__(
        self,
        policy: BasePolicy,
        estimator: Optional[Any] = None,
        opt_policy: Optional[optim.Optimizer] = None,
        opt_value: Optional[optim.Optimizer] = None,
        value_fn: Optional[BasePolicy] = None,
        device: str = "cpu",
    ):
        self._policy = policy
        self._value_fn = value_fn or policy
        self._estimator = estimator
        self._opt_policy = opt_policy
        self._opt_value = opt_value
        self.device = device

    @property
    def policy(self) -> BasePolicy:
        return self._policy

    @property
    def value_fn(self) -> BasePolicy:
        return self._value_fn

    @property
    def estimator(self) -> Optional[Any]:
        return self._estimator

    @property
    def opt_policy(self) -> Optional[optim.Optimizer]:
        return self._opt_policy

    @property
    def opt_value(self) -> Optional[optim.Optimizer]:
        return self._opt_value

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def prepare_obs(self, obs: Any) -> Any:
        """Add batch dimension (B=1) and move to device."""
        if isinstance(obs, dict):
            result: dict = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    dtype = torch.long if "edge_index" in k else torch.float32
                    result[k] = torch.tensor(
                        v, dtype=dtype, device=self.device
                    ).unsqueeze(0)
                elif isinstance(v, torch.Tensor):
                    result[k] = v.to(self.device).unsqueeze(0)
                else:
                    result[k] = v
            return result
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: Any,
        action_mask: np.ndarray,
        training: bool = True,
    ) -> Tuple[int, float, float]:
        with torch.no_grad():
            obs_t = self.prepare_obs(obs)
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
            raise ValueError("Agent.update() requires an estimator")
        if self._opt_policy is None:
            raise ValueError("Agent.update() requires opt_policy")

        loss = self._estimator.compute_loss(self._policy, buffer)
        self._opt_policy.zero_grad()
        loss.backward()
        self._opt_policy.step()

        if self._opt_value is not None and self._value_fn is not self._policy:
            loss_value = self._estimator.compute_loss(self._value_fn, buffer)
            self._opt_value.zero_grad()
            loss_value.backward()
            self._opt_value.step()

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


# ---------------------------------------------------------------------------
# Module-level helpers  (used by trainers)
# ---------------------------------------------------------------------------


def _obs_to_tensor(obs: Any, device: str) -> Any:
    """Convert a single obs dict to a B=1 tensor on *device*.  No padding."""
    if isinstance(obs, dict):
        result: dict = {}
        for k, v in obs.items():
            if not isinstance(v, np.ndarray):
                continue
            t = torch.from_numpy(v[None]).to(device)
            result[k] = t.long() if "edge_index" in k else t.float()
        return result
    return torch.from_numpy(obs[None]).float().to(device)
