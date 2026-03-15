"""
agents/dqn_agent.py
--------------------
DQN agent: owns state (networks, buffer, optimizer, epsilon)
and delegates algorithm math to algorithms/dqn.dqn_update().

Key fixes vs original
----------------------
- Inherits BaseAgent correctly with matching select_action signature.
- select_action returns (action, 0.0, 0.0) — consistent with BaseAgent interface.
- Network is always reset to train() mode after eval(); no mode leak.
- DQNConfig has no network fields.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from .base_agent import BaseAgent
from algorithms.dqn import dqn_update
from buffers import ReplayBuffer, PrioritizedReplayBuffer, Transition
from configs import DQNConfig
from networks.base_network import BaseNetwork


class DQNAgent(BaseAgent):
    """
    Off-policy Double DQN agent with optional PER.

    Parameters
    ----------
    network          : Online Q-network.
    obs_shape        : From problem.observation_shape.
    action_space_size: From problem.action_space_size.
    cfg              : DQNConfig — algorithm hyperparameters only.
    device           : Compute device.
    """

    def __init__(
        self,
        network: BaseNetwork,
        obs_shape: Tuple[int, ...],
        action_space_size: int,
        cfg: DQNConfig,
        device: str = "cpu",
    ):
        self.network = network
        self.target_network = copy.deepcopy(network)
        self.target_network.eval()

        self.obs_shape = obs_shape
        self.action_space_size = action_space_size
        self.cfg = cfg
        self.device = device

        self.optimizer = optim.Adam(self.network.parameters(), lr=cfg.lr)

        self.buffer: ReplayBuffer = (
            PrioritizedReplayBuffer(
                capacity=cfg.buffer_capacity,
                obs_shape=obs_shape,
                action_space_size=action_space_size,
                alpha=cfg.per_alpha,
                beta_start=cfg.per_beta_start,
                beta_end=cfg.per_beta_end,
                beta_steps=cfg.per_beta_steps,
            )
            if cfg.use_per
            else ReplayBuffer(
                capacity=cfg.buffer_capacity,
                obs_shape=obs_shape,
                action_space_size=action_space_size,
            )
        )

        self._step_count = 0
        self._update_count = 0

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        frac = min(self._step_count / max(self.cfg.eps_decay_steps, 1), 1.0)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        training: bool = True,
    ) -> Tuple[int, float, float]:
        """
        ε-greedy action selection.

        Returns (action, 0.0, 0.0) — log_prob and value are unused for DQN
        but the signature is consistent with BaseAgent.
        """
        feasible = np.where(action_mask)[0]
        assert len(feasible) > 0, "No feasible actions."

        if training and np.random.random() < self.epsilon:
            return int(np.random.choice(feasible)), 0.0, 0.0

        self.network.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
            q_vals, _ = self.network.forward(obs_t, mask_t)
        self.network.train()  # always restore — no mode leak

        action = int(q_vals.squeeze(0).argmax().item())
        return action, 0.0, 0.0

    # ------------------------------------------------------------------
    # Experience collection
    # ------------------------------------------------------------------

    def collect(self, env: Any, instance_generator: Callable) -> Dict[str, float]:
        """Run one episode and store transitions in the replay buffer."""
        raw = instance_generator()
        obs, info = env.reset(raw)
        action_mask = info["action_mask"]

        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _, _ = self.select_action(obs, action_mask, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_mask = info["action_mask"]
            done = terminated or truncated

            self.buffer.add(
                Transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                    action_mask=next_mask,
                )
            )

            self._step_count += 1
            total_reward += reward
            steps += 1
            obs = next_obs
            action_mask = next_mask

        return {
            "rollout/mean_reward": total_reward,
            "rollout/mean_ep_length": float(steps),
            "rollout/steps": float(steps),
            "train/epsilon": self.epsilon,
            "train/buffer_size": float(len(self.buffer)),
        }

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def update(self) -> Optional[Dict[str, float]]:
        """One DQN gradient step if conditions are met."""
        if (
            self._step_count < self.cfg.learning_starts
            or len(self.buffer) < self.cfg.batch_size
            or self._step_count % self.cfg.train_freq != 0
        ):
            return None

        self.network.train()
        batch = self.buffer.sample(self.cfg.batch_size)
        metrics = dqn_update(
            q_network=self.network,
            target_network=self.target_network,
            optimizer=self.optimizer,
            batch=batch,
            cfg=self.cfg,
            device=self.device,
            buffer=self.buffer,
        )

        # Target network sync
        if self.cfg.tau < 1.0:
            self._soft_update()
        elif self._update_count % self.cfg.target_update_freq == 0:
            self._hard_update()

        self._update_count += 1
        metrics["train/update_count"] = self._update_count
        metrics["train/epsilon"] = self.epsilon
        return metrics

    def _hard_update(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())

    def _soft_update(self) -> None:
        tau = self.cfg.tau
        for tp, qp in zip(self.target_network.parameters(), self.network.parameters()):
            tp.data.copy_(tau * qp.data + (1.0 - tau) * tp.data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, extra: Optional[Dict] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "network_state": self.network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "dqn_cfg": dataclasses.asdict(self.cfg),
            "obs_shape": self.obs_shape,
            "action_space_size": self.action_space_size,
            "step_count": self._step_count,
            "update_count": self._update_count,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network_state"])
        self.target_network.load_state_dict(ckpt["target_network_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._step_count = ckpt.get("step_count", 0)
        self._update_count = ckpt.get("update_count", 0)

    def __repr__(self) -> str:
        return (
            f"DQNAgent(obs={self.obs_shape}, actions={self.action_space_size}, "
            f"eps={self.epsilon:.3f}, buffer={len(self.buffer)}, "
            f"device={self.device!r})"
        )
