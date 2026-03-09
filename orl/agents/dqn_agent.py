"""
agents/dqn_agent.py
--------------------
Deep Q-Network (DQN) agent with:
  - Double DQN  (target network)
  - Dueling architecture  (optional)
  - Prioritised Experience Replay  (optional)
  - ε-greedy exploration with linear/exponential decay
  - Action masking for infeasible actions
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from core.buffers import ReplayBuffer, PrioritizedReplayBuffer, Transition, Batch
from networks.policy_network import PolicyNetwork, NetworkConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DQNConfig:
    # Network
    embed_dim: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    use_attention: bool = True
    clip_logits: float = 10.0

    # DQN
    lr: float = 1e-4
    gamma: float = 0.99
    buffer_capacity: int = 100_000
    batch_size: int = 64
    target_update_freq: int = 500  # steps between hard target updates
    tau: float = 1.0  # 1.0 = hard copy; <1 = soft (Polyak)
    train_freq: int = 4  # update every N env steps
    learning_starts: int = 1_000  # steps before first update

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000

    # PER
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4

    device: str = "cpu"


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------


class DQNAgent:
    """
    Off-policy DQN agent with Double DQN + optional PER.

    Full RL cycle
    -------------
    For each env step:
      1. ``select_action(obs, mask)``      → ε-greedy action
      2. Execute action, observe (r, s', done)
      3. ``store_transition(transition)``  → push to replay
      4. ``update()``                      → one gradient step (if ready)

    Inference
    ---------
      action = agent.select_action(obs, mask, training=False)
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_space_size: int,
        cfg: Optional[DQNConfig] = None,
    ):
        self.obs_shape = obs_shape
        self.action_space_size = action_space_size
        self.cfg = cfg or DQNConfig()

        net_cfg = NetworkConfig(
            obs_shape=obs_shape,
            action_space_size=action_space_size,
            embed_dim=self.cfg.embed_dim,
            n_heads=self.cfg.n_heads,
            n_encoder_layers=self.cfg.n_encoder_layers,
            use_attention=self.cfg.use_attention,
            clip_logits=self.cfg.clip_logits,
        )

        self.q_network = PolicyNetwork(net_cfg)
        self.target_network = PolicyNetwork(net_cfg)

        self.q_network.to(self.cfg.device)
        self.target_network.to(self.cfg.device)
        self._hard_update()  # sync weights
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.cfg.lr)

        # Replay buffer
        if self.cfg.use_per:
            self.buffer = PrioritizedReplayBuffer(
                capacity=self.cfg.buffer_capacity,
                obs_shape=obs_shape,
                action_space_size=action_space_size,
                alpha=self.cfg.per_alpha,
                beta_start=self.cfg.per_beta_start,
            )
        else:
            self.buffer = ReplayBuffer(
                capacity=self.cfg.buffer_capacity,
                obs_shape=obs_shape,
                action_space_size=action_space_size,
            )

        self._step_count = 0
        self._update_count = 0
        self.metrics_log: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Exploration schedule
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Linearly decayed ε."""
        frac = min(self._step_count / max(self.cfg.eps_decay_steps, 1), 1.0)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        training: bool = True,
    ) -> int:
        """
        ε-greedy action selection with action masking.

        Args:
            obs:         Observation, shape obs_shape.
            action_mask: Bool array (action_space_size,), True = feasible.
            training:    If False, always greedy.

        Returns:
            action index.
        """
        feasible = np.where(action_mask)[0]
        assert len(feasible) > 0, "No feasible actions available."

        if training and np.random.random() < self.epsilon:
            return int(np.random.choice(feasible))

        self.q_network.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.cfg.device)
            mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(self.cfg.device)
            q_vals, _ = self.q_network.forward(obs_t, mask_t)
        q_vals = q_vals.cpu().numpy()[0]

        return int(np.argmax(q_vals))

    # ------------------------------------------------------------------
    # Transition storage
    # ------------------------------------------------------------------

    def store_transition(self, transition: Transition) -> None:
        """Push a transition into the replay buffer."""
        self.buffer.add(transition)
        self._step_count += 1

    # ------------------------------------------------------------------
    # DQN update step
    # ------------------------------------------------------------------

    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform one gradient update step if conditions are met.

        Conditions: buffer has enough samples AND it's an update step.

        Returns:
            metrics dict or None if no update was performed.
        """
        if (
            self._step_count < self.cfg.learning_starts
            or len(self.buffer) < self.cfg.batch_size
            or self._step_count % self.cfg.train_freq != 0
        ):
            return None

        cfg = self.cfg
        batch = self.buffer.sample(cfg.batch_size)

        obs_t = torch.FloatTensor(batch.obs).to(cfg.device)
        next_obs_t = torch.FloatTensor(batch.next_obs).to(cfg.device)
        acts_t = torch.LongTensor(batch.actions).to(cfg.device)
        rew_t = torch.FloatTensor(batch.rewards).to(cfg.device)
        done_t = torch.FloatTensor(batch.dones).to(cfg.device)
        mask_t = torch.BoolTensor(batch.action_masks).to(cfg.device)

        # Current Q-values
        self.q_network.train()
        q_vals, _ = self.q_network.forward(obs_t)  # (B, A)
        q_vals = q_vals.gather(1, acts_t.unsqueeze(1)).squeeze(1)  # (B,)

        # Double DQN target
        with torch.no_grad():
            next_q_online, _ = self.q_network.forward(next_obs_t, mask_t)
            next_acts = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target, _ = self.target_network.forward(next_obs_t, mask_t)
            next_q = next_q_target.gather(1, next_acts).squeeze(1)
            targets = rew_t + cfg.gamma * next_q * (1.0 - done_t)

        td_errors = targets - q_vals

        # PER importance weights
        if batch.weights is not None:
            weights = torch.FloatTensor(batch.weights).to(cfg.device)
            loss = (weights * td_errors.pow(2)).mean()
            # Update priorities
            self.buffer.update_priorities(
                batch.indices, td_errors.detach().cpu().numpy()
            )
        else:
            loss = td_errors.pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Target network sync
        if cfg.tau < 1.0:
            self._soft_update()
        elif self._update_count % cfg.target_update_freq == 0:
            self._hard_update()

        self._update_count += 1
        metrics = {
            "train/td_loss": loss.item(),
            "train/mean_td_error": td_errors.abs().mean().item(),
            "train/epsilon": self.epsilon,
            "train/update_count": self._update_count,
            "train/buffer_size": len(self.buffer),
        }
        self.metrics_log.append(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Target network updates
    # ------------------------------------------------------------------

    def _hard_update(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _soft_update(self) -> None:
        tau = self.cfg.tau
        for tp, qp in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            tp.data.copy_(tau * qp.data + (1.0 - tau) * tp.data)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self._step_count,
                "update_count": self._update_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_network.load_state_dict(ckpt["target_network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step_count = ckpt.get("step_count", 0)
        self._update_count = ckpt.get("update_count", 0)

    def __repr__(self) -> str:
        return (
            f"DQNAgent(obs={self.obs_shape}, "
            f"actions={self.action_space_size}, "
            f"eps={self.epsilon:.3f}, "
            f"buffer={len(self.buffer)})"
        )
