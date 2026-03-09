"""
agents/ppo_agent.py
--------------------
Proximal Policy Optimisation (PPO-clip) agent for combinatorial RL.

Covers the full RL procedure:
  1. Rollout collection (actor inference with action masking)
  2. GAE advantage estimation  (in RolloutBuffer)
  3. Multiple epochs of clipped surrogate + value + entropy loss
  4. Gradient clipping
  5. Adaptive KL early-stopping
  6. Checkpoint save / load
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from core.buffers import RolloutBuffer, Batch
from networks.policy_network import PolicyNetwork, NetworkConfig


# ---------------------------------------------------------------------------
# Hyper-parameter dataclass
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    # Network
    embed_dim: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    dropout: float = 0.0
    use_attention: bool = True
    clip_logits: float = 10.0

    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    mini_batch_size: int = 256
    rollout_len: int = 2048
    target_kl: Optional[float] = 0.015  # None to disable early-stop

    # Misc
    device: str = "cpu"
    normalize_rewards: bool = True


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------


class PPOAgent:
    """
    On-policy PPO agent with clipped surrogate objective.

    Full RL cycle
    -------------
    For each training iteration:
      1. ``collect_rollout(env, n_steps)``   → fill RolloutBuffer
      2. ``update(buffer)``                  → gradient updates, return metrics
      3. ``save / load``                     → checkpointing

    Inference
    ---------
      action = agent.select_action(obs, action_mask, training=False)
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_space_size: int,
        cfg: Optional[PPOConfig] = None,
    ):
        self.obs_shape = obs_shape
        self.action_space_size = action_space_size
        self.cfg = cfg or PPOConfig()

        net_cfg = NetworkConfig(
            obs_shape=obs_shape,
            action_space_size=action_space_size,
            embed_dim=self.cfg.embed_dim,
            n_heads=self.cfg.n_heads,
            n_encoder_layers=self.cfg.n_encoder_layers,
            dropout=self.cfg.dropout,
            use_attention=self.cfg.use_attention,
            clip_logits=self.cfg.clip_logits,
        )
        self.network = PolicyNetwork(net_cfg)

        self.network.to(self.cfg.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.cfg.lr)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda _: 1.0
        )

        self.rollout_buffer = RolloutBuffer(
            capacity=self.cfg.rollout_len,
            obs_shape=obs_shape,
            action_space_size=action_space_size,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        # Running reward normalizer
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0

        # Metrics
        self._update_count = 0
        self.metrics_log: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        training: bool = True,
    ) -> Tuple[int, float, float]:
        """
        Sample (or greedily pick) an action.

        Args:
            obs:         Observation array, shape obs_shape.
            action_mask: Boolean array, shape (action_space_size,).
            training:    If False, use deterministic greedy decoding.

        Returns:
            action:   Integer action index.
            log_prob: Log-probability of the chosen action.
            value:    Critic's value estimate.
        """
        self.network.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.cfg.device)
            mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(self.cfg.device)
            action_t, lp_t, val_t = self.network.get_action_and_log_prob(
                obs_t, mask_t, deterministic=not training
            )
        action = int(action_t.item())
        log_prob = float(lp_t.item())
        value = float(val_t.item())

        return action, log_prob, value

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self, env, instance_generator) -> Dict[str, float]:
        """
        Collect a full rollout of ``cfg.rollout_len`` steps.

        Args:
            env:                CombinatorialEnv instance.
            instance_generator: Callable[] → raw_instance; called per episode.

        Returns:
            Rollout statistics (mean reward, mean episode length, …).
        """
        self.network.train()
        self.rollout_buffer.reset()

        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        ep_reward = 0.0
        ep_length = 0

        raw_instance = instance_generator()
        obs, info = env.reset(raw_instance)
        action_mask = info["action_mask"]

        steps_collected = 0
        while not self.rollout_buffer.is_full:
            action, log_prob, value = self.select_action(
                obs, action_mask, training=True
            )
            next_obs, reward, terminated, truncated, info = env.step(action)

            if self.cfg.normalize_rewards:
                reward = self._normalize_reward(reward)

            self.rollout_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=terminated or truncated,
                log_prob=log_prob,
                value=value,
                action_mask=action_mask,
            )

            ep_reward += reward
            ep_length += 1
            steps_collected += 1

            obs = next_obs
            action_mask = info["action_mask"]

            if terminated or truncated:
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                ep_reward = 0.0
                ep_length = 0
                raw_instance = instance_generator()
                obs, info = env.reset(raw_instance)
                action_mask = info["action_mask"]

        # Bootstrap last value
        _, _, last_value = self.select_action(obs, action_mask, training=False)
        self.rollout_buffer.compute_returns(last_value=last_value)

        stats = {
            "rollout/mean_reward": float(np.mean(episode_rewards))
            if episode_rewards
            else 0.0,
            "rollout/mean_ep_length": float(np.mean(episode_lengths))
            if episode_lengths
            else 0.0,
            "rollout/n_episodes": len(episode_rewards),
            "rollout/steps_collected": steps_collected,
        }
        return stats

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """
        Run ``cfg.n_epochs`` of PPO updates on the current rollout buffer.

        Returns metrics dict with policy/value/entropy losses and KL.
        """

        self.network.train()
        cfg = self.cfg

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_kl = 0.0
        n_updates = 0
        early_stop = False

        for batch in self.rollout_buffer.iter_batches(
            cfg.mini_batch_size, n_epochs=cfg.n_epochs
        ):
            if early_stop:
                break

            obs_t = torch.FloatTensor(batch.obs).to(cfg.device)
            acts_t = torch.LongTensor(batch.actions).to(cfg.device)
            masks_t = torch.BoolTensor(batch.action_masks).to(cfg.device)
            adv_t = torch.FloatTensor(batch.advantages).to(cfg.device)
            ret_t = torch.FloatTensor(batch.returns).to(cfg.device)
            old_lp_t = torch.FloatTensor(batch.log_probs).to(cfg.device)

            new_lp, values, entropy = self.network.evaluate_actions(
                obs_t, acts_t, masks_t
            )

            # Clipped surrogate loss
            ratio = torch.exp(new_lp - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (clipped)
            value_loss = 0.5 * F.mse_loss(values, ret_t)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                + cfg.entropy_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

            # Approximate KL for early stopping
            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_kl += approx_kl
            n_updates += 1

            if cfg.target_kl is not None and approx_kl > 1.5 * cfg.target_kl:
                early_stop = True

        self._update_count += 1
        n = max(n_updates, 1)
        metrics = {
            "train/policy_loss": total_policy_loss / n,
            "train/value_loss": total_value_loss / n,
            "train/entropy": -total_entropy_loss / n,
            "train/approx_kl": total_kl / n,
            "train/update_count": self._update_count,
            "train/early_stop": float(early_stop),
        }
        self.metrics_log.append(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Reward normalisation (running Welford)
    # ------------------------------------------------------------------

    def _normalize_reward(self, reward: float) -> float:
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        self._reward_var += delta * (reward - self._reward_mean)
        std = max(math.sqrt(self._reward_var / max(self._reward_count - 1, 1)), 1e-8)
        return reward / std

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist network weights and optimiser state."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_count": self._update_count,
                "reward_mean": self._reward_mean,
                "reward_var": self._reward_var,
                "reward_count": self._reward_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore network and optimiser from checkpoint."""
        file_path = Path(path)
        ckpt = torch.load(file_path, map_location=self.cfg.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._update_count = ckpt.get("update_count", 0)
        self._reward_mean = ckpt.get("reward_mean", 0.0)
        self._reward_var = ckpt.get("reward_var", 1.0)
        self._reward_count = ckpt.get("reward_count", 0)

    def __repr__(self) -> str:
        return (
            f"PPOAgent(obs={self.obs_shape}, "
            f"actions={self.action_space_size}, "
            f"updates={self._update_count})"
        )


# import needed for reward normalisation fallback
import math

try:
    import torch.nn.functional as F
except ImportError:
    pass
