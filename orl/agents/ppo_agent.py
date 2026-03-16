"""
agents/ppo_agent.py
--------------------
PPO agent — supports both flat array observations (Knapsack)
and dict observations (VRPBTW with HACN network).

Key changes vs original
------------------------
- select_action accepts obs as either np.ndarray or dict
- If obs is a dict (VRPBTW), batches both node_features and vehicle_features
  and passes the dict to the network
- collect() reads node_features / vehicle_features from info when available
- The rollout buffer stores the FLAT obs for Knapsack; for VRPBTW it stores
  node_features as the primary obs (vehicle_features are recomputed at
  update time from the stored obs — NOT stored separately, to keep buffer simple)
- update() calls network.train() (mode management fix)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.optim as optim

from .base_agent import BaseAgent
from algorithms.ppo import ppo_update
from buffers import RolloutBuffer
from configs import PPOConfig
from networks.base_network import BaseNetwork
from utils.normalizer import RunningNormalizer


# ---------------------------------------------------------------------------
# Padding helpers for variable-size curriculum instances
# ---------------------------------------------------------------------------


def _pad_to(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Zero-pad arr to target_shape along the first axis.

    Used when curriculum generates N_small < N_max nodes so that
    node_features (N_small+1, feat_dim) fits into a buffer slot
    pre-allocated for (N_max+1, feat_dim).

    Padding rows are all zeros — the served flag (col 7) is 0, which the
    network interprets as unserved, but padded rows are also masked out
    in the action mask so they are never selected.
    """
    if arr.shape == target_shape:
        return arr
    out = np.zeros(target_shape, dtype=arr.dtype)
    slices = tuple(slice(0, s) for s in arr.shape)
    out[slices] = arr
    return out


def _pad_mask(mask: np.ndarray, target_size: int) -> np.ndarray:
    """
    Zero-pad a boolean action mask to target_size.

    Padded entries are False (infeasible) — correct because those
    (fleet, vehicle, node) combinations do not exist for smaller instances.
    """
    if mask.shape[0] == target_size:
        return mask
    out = np.zeros(target_size, dtype=bool)
    out[: mask.shape[0]] = mask
    return out


class PPOAgent(BaseAgent):
    def __init__(
        self,
        network: BaseNetwork,
        obs_shape: Tuple[int, ...],
        action_space_size: int,
        cfg: PPOConfig,
        device: str = "cpu",
    ):
        self.network = network
        self.obs_shape = obs_shape
        self.action_space_size = action_space_size
        self.cfg = cfg
        self.device = device

        self.optimizer = optim.Adam(self.network.parameters(), lr=cfg.lr)

        self.rollout_buffer = RolloutBuffer(
            capacity=cfg.rollout_len,
            obs_shape=obs_shape,
            action_space_size=action_space_size,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )
        self.rollout_buffer.normalize_advantages_flag = cfg.normalize_advantages

        self.reward_normalizer = RunningNormalizer() if cfg.normalize_rewards else None

        self._update_count = 0
        self._step_count = 0

    # ------------------------------------------------------------------
    # Helpers — obs conversion
    # ------------------------------------------------------------------

    def _to_tensor_obs(self, obs: Union[np.ndarray, Dict]) -> Any:
        """
        Convert obs (array or dict) to the format the network expects.
        - array: (B, *obs_shape) FloatTensor
        - dict:  {key: (B, ...) FloatTensor}  — vehicle_features come from
                 obs dict directly or from the rollout buffer extra arrays.
        """
        if isinstance(obs, dict):
            return {k: torch.FloatTensor(v).to(self.device) for k, v in obs.items()}
        return torch.FloatTensor(obs).to(self.device)

    def _batch_obs(self, obs: Union[np.ndarray, Dict]) -> Any:
        """Add batch dimension (B=1) to obs."""
        if isinstance(obs, dict):
            return {k: v[None] for k, v in obs.items()}
        return obs[None]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: Union[np.ndarray, Dict],
        action_mask: np.ndarray,
        training: bool = True,
    ) -> Tuple[int, float, float]:
        with torch.no_grad():
            obs_t = self._to_tensor_obs(self._batch_obs(obs))
            mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
            action_t, lp_t, val_t = self.network.get_action_and_log_prob(
                obs_t, mask_t, deterministic=not training
            )
        return int(action_t.item()), float(lp_t.item()), float(val_t.item())

    # ------------------------------------------------------------------
    # Experience collection
    # ------------------------------------------------------------------

    def collect(self, env: Any, instance_generator: Callable) -> Dict[str, float]:
        self.rollout_buffer.reset()

        ep_rewards: List[float] = []
        ep_lengths: List[int] = []
        ep_reward, ep_length = 0.0, 0

        raw = instance_generator()
        obs, info = env.reset(raw)
        action_mask = info["action_mask"]

        while not self.rollout_buffer.is_full:
            action, log_prob, value = self.select_action(
                obs, action_mask, training=True
            )
            next_obs, reward, terminated, truncated, info = env.step(action)

            if self.reward_normalizer is not None:
                reward = self.reward_normalizer.normalise(reward)

            # For dict obs (VRPBTW), store node_features as primary obs
            # and vehicle_features separately for the HACN decoder context.
            # Pad to buffer's pre-allocated shapes when curriculum generates
            # smaller instances than the maximum problem size.
            if isinstance(obs, dict):
                obs_to_store = _pad_to(obs["node_features"], self.obs_shape)
                veh_to_store = obs["vehicle_features"]  # (2K, 7) — fixed size
            else:
                obs_to_store = obs
                veh_to_store = None

            # Pad action_mask to buffer's action_space_size
            mask_to_store = _pad_mask(action_mask, self.action_space_size)

            self.rollout_buffer.add(
                obs=obs_to_store,
                action=action,
                reward=reward,
                done=(terminated or truncated),
                log_prob=log_prob,
                value=value,
                action_mask=mask_to_store,
                vehicle_features=veh_to_store,
            )

            self._step_count += 1
            ep_reward += reward
            ep_length += 1
            obs = next_obs
            action_mask = info["action_mask"]

            if terminated or truncated:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_length)
                ep_reward, ep_length = 0.0, 0
                raw = instance_generator()
                obs, info = env.reset(raw)
                action_mask = info["action_mask"]

        # Bootstrap value for GAE
        _, _, last_value = self.select_action(obs, action_mask, training=False)
        self.rollout_buffer.compute_returns_and_advantages(last_value=last_value)

        return {
            "rollout/mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "rollout/mean_ep_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
            "rollout/n_episodes": len(ep_rewards),
            "rollout/steps": self.rollout_buffer.capacity,
        }

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def update(self) -> Optional[Dict[str, float]]:
        self.network.train()  # set training mode once, right here
        metrics = ppo_update(
            network=self.network,
            optimizer=self.optimizer,
            buffer=self.rollout_buffer,
            cfg=self.cfg,
            device=self.device,
        )
        self._update_count += 1
        metrics["train/update_count"] = self._update_count
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, extra: Optional[Dict] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "ppo_cfg": dataclasses.asdict(self.cfg),
            "obs_shape": self.obs_shape,
            "action_space_size": self.action_space_size,
            "update_count": self._update_count,
            "step_count": self._step_count,
            "reward_norm": (
                self.reward_normalizer.state_dict() if self.reward_normalizer else None
            ),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._update_count = ckpt.get("update_count", 0)
        self._step_count = ckpt.get("step_count", 0)
        if self.reward_normalizer and ckpt.get("reward_norm"):
            self.reward_normalizer.load_state_dict(ckpt["reward_norm"])

    def __repr__(self) -> str:
        return (
            f"PPOAgent(obs={self.obs_shape}, actions={self.action_space_size}, "
            f"updates={self._update_count}, device={self.device!r})"
        )
