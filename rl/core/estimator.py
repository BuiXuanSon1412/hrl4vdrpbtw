"""
core/estimator.py
-----------------
Loss computation abstraction for policy gradient methods.

Estimators take a policy network and a buffer of rollout data, then
compute the total loss for a parameter update.

  BaseEstimator  — abstract interface (compute_loss)
  PPOEstimator   — PPO loss with clipping and entropy regularization
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from core.buffer import RolloutBuffer
from core.policy import BasePolicy
from core.utils import obs_to_tensor


class BaseEstimator(ABC):
    """
    Abstract estimator: computes loss given policy and buffer.

    Implementations compute gradients for parameter updates.
    """

    def __init__(self, device: str = "cpu", **kwargs: Any):
        self.device = device

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "BaseEstimator":
        """Factory method: instantiate estimator from config dict."""
        ...

    @abstractmethod
    def compute_loss(self, policy: BasePolicy, buffer: RolloutBuffer) -> torch.Tensor:
        """
        Compute total loss over the buffer.

        Args:
            policy: the policy network (nn.Module).
            buffer: rollout buffer with collected transitions.

        Returns:
            scalar loss tensor for backpropagation.
        """
        ...

    def compute_entropy(self, agent: Any, buffer: RolloutBuffer) -> float:
        """
        Compute policy entropy for curriculum monitoring (optional).

        Default: returns 0.0 (no entropy computation).
        Override in subclasses for problem-specific entropy metrics.

        Args:
            agent: BaseAgent with policy and related components.
            buffer: rollout buffer with collected transitions.

        Returns:
            scalar entropy value.
        """
        return 0.0


class REINFORCEEstimator(BaseEstimator):
    """
    REINFORCE with baseline: -log π(a|s) * (G_t - baseline)

    Loss = -1/N * sum_t(log_prob_t * advantage_t)
    where advantage_t = return_t - baseline (or value_t if using critic)
    """

    def __init__(
        self,
        device: str = "cpu",
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        baseline_mode: str = "mean",
    ):
        """
        Args:
            device: device string (cpu/cuda)
            gamma: discount factor for return computation
            entropy_coef: weight for entropy bonus
            baseline_mode: "mean" (mean return) or "value" (use critic value)
        """
        super().__init__(device=device)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.baseline_mode = baseline_mode

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "REINFORCEEstimator":
        """Factory method: instantiate REINFORCE estimator from config dict.

        Config keys:
        - device: device string (cpu/cuda)
        - estimator.hparams: dict with gamma, entropy_coef, baseline_mode
        """
        device = cfg.get("device", "cpu")
        # Read from estimator.hparams
        estimator_cfg = cfg.get("estimator", {})
        obj_cfg = estimator_cfg.get("hparams", {}) if isinstance(estimator_cfg, dict) else {}

        return cls(
            device=device,
            gamma=obj_cfg.get("gamma", 0.99),
            entropy_coef=obj_cfg.get("entropy_coef", 0.01),
            baseline_mode=obj_cfg.get("baseline_mode", "mean"),
        )

    def compute_loss(self, policy: BasePolicy, buffer: RolloutBuffer) -> torch.Tensor:
        """
        Compute REINFORCE loss over all transitions in the buffer.

        For each transition:
          - Compute return: discounted sum of future rewards
          - Compute baseline: mean return or critic value
          - Advantage = return - baseline
          - Loss = -log_prob * advantage - entropy_bonus
        """
        n = buffer._ptr
        if n == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Compute returns (discounted cumulative rewards)
        returns = self._compute_returns(buffer)

        # Compute baseline
        if self.baseline_mode == "mean":
            baseline = float(np.mean(returns))
        else:
            baseline = 0.0

        total_loss = None
        for i, tr in enumerate(buffer._data[:n]):
            obs_t = obs_to_tensor(tr.obs, self.device)
            act_t = torch.tensor([tr.action], dtype=torch.long, device=self.device)
            mask_t = torch.tensor(
                tr.action_mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0)

            # Evaluate current policy
            log_prob, value, entropy = policy.evaluate_actions(obs_t, act_t, mask_t)

            # Compute advantage
            ret = torch.tensor([returns[i]], dtype=torch.float32, device=self.device)
            if self.baseline_mode == "value":
                advantage = ret - value
            else:
                advantage = ret - baseline

            # REINFORCE loss: -log_prob * advantage
            policy_loss = -log_prob * advantage

            # Entropy bonus
            entropy_loss = -entropy

            loss_i = (
                policy_loss + self.entropy_coef * entropy_loss
            ) / n
            total_loss = loss_i if total_loss is None else total_loss + loss_i

        return (
            total_loss
            if total_loss is not None
            else torch.tensor(0.0, device=self.device, requires_grad=True)
        )

    def _compute_returns(self, buffer: RolloutBuffer) -> list:
        """
        Compute discounted cumulative returns for each transition.

        Returns[t] = sum_{k=0}^{n-t-1} gamma^k * reward[t+k]
        """
        n = buffer._ptr
        returns = [0.0] * n
        cumsum = 0.0

        for t in reversed(range(n)):
            tr = buffer._data[t]
            cumsum = tr.reward + self.gamma * cumsum * (1.0 - float(tr.done))
            returns[t] = cumsum

        return returns


class PPOEstimator(BaseEstimator):
    """
    PPO loss with clipped objective, value loss, and entropy regularization.

    Loss = -policy_loss + value_coef * value_loss - entropy_coef * entropy
    """

    def __init__(
        self,
        device: str = "cpu",
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        clip_ratio: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        super().__init__(device=device)
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "PPOEstimator":
        """Factory method: instantiate PPO estimator from config dict.

        Config keys:
        - device: device string (cpu/cuda)
        - estimator.hparams: dict with gamma, gae_lambda, value_coefficient, entropy_coefficient, clip_eps
        """
        device = cfg.get("device", "cpu")
        # Read from estimator.hparams
        estimator_cfg = cfg.get("estimator", {})
        obj_cfg = estimator_cfg.get("hparams", {}) if isinstance(estimator_cfg, dict) else {}

        return cls(
            device=device,
            gamma=obj_cfg.get("gamma", 0.99),
            gae_lambda=obj_cfg.get("gae_lambda", 0.95),
            value_coef=obj_cfg.get("value_coefficient", 0.5),
            entropy_coef=obj_cfg.get("entropy_coefficient", 0.01),
            clip_ratio=obj_cfg.get("clip_eps", 0.2),
        )

    def compute_loss(self, policy: BasePolicy, buffer: RolloutBuffer) -> torch.Tensor:
        """
        Compute PPO loss over all transitions in the buffer.

        Computes advantages and returns on-the-fly using GAE-λ, then:
          - Compute log_prob, value, entropy from the current policy
          - Ratio = exp(log_prob - old_log_prob)
          - Clipped advantage = min(ratio * adv, clip(ratio, 1-clip_ratio, 1+clip_ratio) * adv)
          - Total loss = -clipped_adv + value_loss + entropy_loss
        """
        n = buffer._ptr
        if n == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Compute GAE advantages and returns on-the-fly
        advantages = self._compute_advantages(buffer)
        returns = self._compute_returns(buffer, advantages)

        total_loss = None
        for i, tr in enumerate(buffer._data[:n]):
            obs_t = obs_to_tensor(tr.obs, self.device)
            act_t = torch.tensor([tr.action], dtype=torch.long, device=self.device)
            mask_t = torch.tensor(
                tr.action_mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0)
            adv = torch.tensor([advantages[i]], dtype=torch.float32, device=self.device)
            ret = torch.tensor([returns[i]], dtype=torch.float32, device=self.device)
            old_log_prob = torch.tensor(
                [tr.log_prob], dtype=torch.float32, device=self.device
            )

            # Evaluate current policy
            log_prob, value, entropy = policy.evaluate_actions(obs_t, act_t, mask_t)

            # PPO clipped objective
            ratio = torch.exp(log_prob - old_log_prob)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2)

            # Value loss
            value_loss = 0.5 * F.mse_loss(value, ret)

            # Entropy bonus
            entropy_loss = -entropy

            loss_i = (
                policy_loss
                + self.value_coef * value_loss
                + self.entropy_coef * entropy_loss
            ) / n
            total_loss = loss_i if total_loss is None else total_loss + loss_i

        return (
            total_loss
            if total_loss is not None
            else torch.tensor(0.0, device=self.device, requires_grad=True)
        )

    def _compute_advantages(self, buffer: RolloutBuffer) -> list:
        """
        Compute GAE-λ advantages for all transitions in the buffer.

        Bootstraps from next state value, properly handling episode boundaries
        (done=1 prevents GAE flow across episodes). At buffer end, bootstraps
        from the last state's value estimate if episode continues (not_done=1).
        """
        n = buffer._ptr
        gae = 0.0
        advantages = [0.0] * n

        for t in reversed(range(n)):
            tr = buffer._data[t]
            not_done = 1.0 - float(tr.done)

            if t < n - 1:
                next_val = buffer._data[t + 1].value * not_done
            else:
                next_val = buffer._data[t].value * not_done

            delta = tr.reward + self.gamma * next_val - tr.value
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae

        # Normalize advantages
        adv_array = np.array(advantages, dtype=np.float32)
        mean, std = adv_array.mean(), adv_array.std() + 1e-8
        advantages = [(a - mean) / std for a in advantages]

        return advantages

    def _compute_returns(self, buffer: RolloutBuffer, advantages: list) -> list:
        """Compute value targets as return = V_old(s_t) + advantage."""
        returns = [buffer._data[i].value + advantages[i] for i in range(buffer._ptr)]
        return returns
