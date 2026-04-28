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
from typing import Any, Optional, Tuple, Dict

import numpy as np
import torch
import torch.optim as optim

import globals
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

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        cfg: dict,
        policy: BasePolicy,
        opt_policy: Optional[optim.Optimizer] = None,
    ) -> "BaseAgent":
        """Factory method: instantiate agent from config, policy, and optimizer."""
        ...

    @abstractmethod
    def select_action(
        self,
        obs: Any,
        action_mask: np.ndarray,
        training: bool = True,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Return (action, log_prob, value).

        Returns:
            action: int, the selected action
            log_prob: torch.Tensor (scalar), log probability with gradients
            value: torch.Tensor (scalar), value estimate with gradients
        """
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

    @abstractmethod
    def clone_policy(self, source: "BaseAgent") -> None:
        """Clone policy from source agent into self.

        Replaces self's policy with a cloned copy of source's policy.
        Preserves all other configurations of self (optimizer, rollout_length, etc).

        Args:
            source: agent to clone policy from
        """
        ...

    @abstractmethod
    def update(self, loss: torch.Tensor) -> float:
        """Update policy network from computed loss.

        Args:
            loss: scalar loss tensor with gradient graph attached
        """
        ...

    @abstractmethod
    def collect(self, env: Any) -> RolloutBuffer:
        """Collect complete episodes into buffer.

        Args:
            env: environment with reset/step interface (already initialized for a task)

        Returns:
            RolloutBuffer with collected transitions
        """
        ...

    @abstractmethod
    def action_to_log_prob(
        self,
        obs: Any,
        action_mask: np.ndarray,
    ) -> Dict[int, torch.Tensor]:
        """Compute log probabilities for all feasible actions.

        Args:
            obs: observation from environment
            action_mask: binary mask of valid actions

        Returns:
            dict mapping action_id -> log_prob (torch.Tensor with requires_grad=True)
        """
        ...


# ---------------------------------------------------------------------------
# PolicyAgent  — the one concrete agent
# ---------------------------------------------------------------------------


class PolicyAgent(BaseAgent):
    """
    Policy-only agent for RL training.

    Properties:
      policy      — policy network (π_θ)
      opt_policy  — optimizer for policy (optional)

    Methods:
      select_action(obs, mask, training) — sample or greedy action
      update(loss) — perform gradient updates from computed loss
    """

    def __init__(
        self,
        policy: BasePolicy,
        opt_policy: Optional[optim.Optimizer] = None,
        rollout_length: int = 256,
        max_grad_norm: float = 0.5,
    ):
        self._policy = policy
        self._optimizer = opt_policy
        self.rollout_length = rollout_length
        self.max_grad_norm = max_grad_norm

    @property
    def policy(self) -> BasePolicy:
        return self._policy

    @property
    def optimizer(self) -> Optional[optim.Optimizer]:
        return self._optimizer

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        policy: BasePolicy,
        opt_policy: Optional[optim.Optimizer] = None,
    ) -> "PolicyAgent":
        """Factory method: instantiate PolicyAgent from agent config, policy, and optimizer.

        Agent config keys:
        - rollout_length: length of rollout buffer
        - max_grad_norm: gradient clipping threshold
        """
        rollout_length = cfg.get("rollout_length", 256)
        max_grad_norm = cfg.get("max_grad_norm", 0.5)
        return cls(
            policy=policy,
            opt_policy=opt_policy,
            rollout_length=rollout_length,
            max_grad_norm=max_grad_norm,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: Any,
        action_mask: np.ndarray,
        training: bool = True,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        from core.utils import obs_to_tensor

        obs_t = obs_to_tensor(obs, device=globals.DEVICE)
        mask_t = torch.tensor(
            action_mask, dtype=torch.bool, device=globals.DEVICE
        ).unsqueeze(0)
        action_t, lp_t, val_t = self._policy.get_action_and_log_prob(
            obs_t, mask_t, deterministic=not training
        )
        return int(action_t.item()), lp_t, val_t

    def action_to_log_prob(
        self,
        obs: Any,
        action_mask: np.ndarray,
    ) -> Dict[int, torch.Tensor]:
        """Compute log probabilities for all feasible actions.

        Args:
            obs: observation from environment
            action_mask: binary mask of valid actions

        Returns:
            dict mapping action_id -> log_prob (torch.Tensor with requires_grad=True)
        """
        from core.utils import obs_to_tensor

        obs_t = obs_to_tensor(obs, device=globals.DEVICE)
        mask_t = torch.tensor(
            action_mask, dtype=torch.bool, device=globals.DEVICE
        ).unsqueeze(0)

        # Get feasible action indices
        feasible_actions = np.where(action_mask)[0].tolist()

        if not feasible_actions:
            return {}

        # Compute log_probs for all feasible actions in one forward pass
        log_probs_dict = {}
        for action in feasible_actions:
            act_t = torch.tensor([action], dtype=torch.long, device=globals.DEVICE)
            log_prob, _, _ = self._policy.evaluate_actions(obs_t, act_t, mask_t)
            log_probs_dict[int(action)] = log_prob  # Keep as tensor with gradients

        return log_probs_dict

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect(self, env: Any) -> RolloutBuffer:
        """Collect complete episodes into buffer.

        Assumes env is already initialized with a task (trainer called reset(task_id)).
        Collects transitions until buffer is full.

        Args:
            env: environment with reset/step interface (already initialized for a task)

        Returns:
            RolloutBuffer with collected transitions
        """
        buffer = RolloutBuffer(capacity=self.rollout_length)
        rollout_len = buffer.capacity

        obs, info = env.reset()
        action_mask = info["action_mask"]

        while buffer._ptr < rollout_len and not buffer.is_full:
            action, lp, val = self.select_action(obs, action_mask, training=True)

            if not action_mask[action]:
                feasible = np.where(action_mask)[0]
                if len(feasible) > 0:
                    action = int(np.random.choice(feasible))
                    lp = torch.tensor(0.0, device=globals.DEVICE, requires_grad=True)
                else:
                    obs, info = env.get_obs_info()
                    action_mask = info["action_mask"]
                    continue

            next_obs, reward, terminated, truncated, info = env.step(action)

            buffer.add(
                obs=obs,
                action=action,
                done=(terminated or truncated),
                log_prob=lp,
                action_mask=action_mask,
                reward=reward,
                value=val,
            )

            obs = next_obs
            action_mask = info["action_mask"]

            if terminated or truncated or not action_mask.any():
                obs, info = env.get_obs_info()
                action_mask = info["action_mask"]

        return buffer

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def update(self, loss: torch.Tensor) -> float:
        """
        Update policy network from computed loss.

        Args:
            loss: scalar loss tensor with gradient graph attached

        Returns:
            grad_norm: L2 norm of gradients after clipping (or -1.0 if no update)

        Uses optimizer if available, otherwise performs manual gradient update.
        """
        grad_norm = -1.0
        if self._optimizer is not None:
            self._optimizer.zero_grad()
            loss.backward()

            if self.max_grad_norm > 0:
                unclipped = torch.nn.utils.clip_grad_norm_(
                    self._policy.parameters(), self.max_grad_norm
                ).item()
                grad_norm = min(unclipped, self.max_grad_norm)

            self._optimizer.step()
        else:
            loss.backward()

            if self.max_grad_norm > 0:
                unclipped = torch.nn.utils.clip_grad_norm_(
                    self._policy.parameters(), self.max_grad_norm
                ).item()
                grad_norm = min(unclipped, self.max_grad_norm)

            for param in self._policy.parameters():
                if param.grad is not None:
                    param.data -= param.grad

        return grad_norm

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
            ckpt = torch.load(path, map_location=globals.DEVICE, weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location=globals.DEVICE, weights_only=False)
        self._policy.load_state_dict(ckpt["network_state"])

    def clone_policy(self, source: "BaseAgent") -> None:
        """Clone policy from source agent into self.

        Creates independent copy of source's policy parameters.
        Updates to self won't affect source.
        """
        self._policy.load_state_dict(source.policy.state_dict())

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self._policy.parameters())
        return (
            f"PolicyAgent(network={type(self._policy).__name__}, "
            f"params={n_params:,}, device={globals.DEVICE!r})"
        )
