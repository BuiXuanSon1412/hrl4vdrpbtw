"""
core/agent.py
-------------
Agent abstractions and implementations.

Classes
-------
  BaseAgent  — abstract interface (select_action, collect, update, save, load)
  PPOAgent   — on-policy PPO agent

Algorithm functions (pure compute, no state)
--------------------------------------------
  _build_obs   — reconstruct network input from a Batch
  ppo_update   — one round of PPO-clip gradient updates

Design rules
------------
- Agents hold a network (injected, not built internally).
- Algorithm math (ppo_update) lives alongside the agent that uses it
  because it is tightly coupled to PPOAgent's buffer and config.
  If a new algorithm like MAML is introduced, add MAMLAgent here and
  its update function below it.
- Agents are problem-agnostic: they work with numpy obs arrays and
  integer actions regardless of the underlying problem.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.buffer import Batch, RolloutBuffer
from core.module import BaseNetwork
from core.utils import RunningNormalizer


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """
    Minimal interface all RL agents must implement.

    Constructor contract
    --------------------
    Every concrete agent accepts:
        network           : BaseNetwork  — injected, not built here
        obs_shape         : Tuple        — from problem.observation_shape
        action_space_size : int          — from problem.action_space_size
        cfg               : algorithm-specific config dataclass
    """

    @property
    def network(self) -> Optional["BaseNetwork"]:
        """
        Return the policy network if this agent has one, else None.
        Override in concrete agents that hold a network.
        Default returns None so evaluator can check cleanly.
        """
        return None

    @abstractmethod
    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        training: bool = True,
    ) -> Tuple[int, float, float]:
        """
        Returns (action, log_prob, value).
        log_prob and value are 0.0 for off-policy agents.
        """
        ...

    @abstractmethod
    def collect(self, env: Any, instance_generator: Any) -> Dict[str, float]:
        """Collect experience; return rollout statistics."""
        ...

    @abstractmethod
    def update(self) -> Optional[Dict[str, float]]:
        """Perform one learning update; return training metrics."""
        ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...


# ---------------------------------------------------------------------------
# Padding helpers  (PPOAgent internal)
# ---------------------------------------------------------------------------


def _pad_to(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    out = np.zeros(target_shape, dtype=arr.dtype)
    slices = tuple(slice(0, s) for s in arr.shape)
    out[slices] = arr
    return out


def _pad_mask(mask: np.ndarray, target_size: int) -> np.ndarray:
    if mask.shape[0] == target_size:
        return mask
    out = np.zeros(target_size, dtype=bool)
    out[: mask.shape[0]] = mask
    return out


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------


class PPOAgent(BaseAgent):
    """
    On-policy PPO agent.

    Works with flat array obs (e.g. Knapsack) and dict obs with optional
    graph fields (e.g. VRPBTW with GNN vehicle embedder).
    """

    def __init__(
        self,
        network: BaseNetwork,
        obs_shape: Tuple[int, ...],
        action_space_size: int,
        cfg: Any,  # PPOConfig
        device: str = "cpu",
    ):
        self._network = network
        self.obs_shape = obs_shape
        self.action_space_size = action_space_size
        self.cfg = cfg
        self.device = device

        self.optimizer = optim.Adam(self._network.parameters(), lr=cfg.lr)

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
    # Obs helpers
    # ------------------------------------------------------------------

    def _to_tensor_obs(self, obs: Union[np.ndarray, Dict]) -> Any:
        if isinstance(obs, dict):
            return {
                k: torch.FloatTensor(v).to(self.device)
                for k, v in obs.items()
                if isinstance(v, np.ndarray)
            }
        return torch.FloatTensor(obs).to(self.device)

    def _batch_obs(self, obs: Union[np.ndarray, Dict]) -> Any:
        if isinstance(obs, dict):
            batched = {k: v[None] for k, v in obs.items()}
            # Pad node_features to the fixed obs_shape
            if "node_features" in batched:
                nf = batched["node_features"]  # (1, n_cur+1, 5)
                target = self.obs_shape[0]  # N_max+1
                if nf.shape[1] < target:
                    pad = np.zeros(
                        (1, target - nf.shape[1], nf.shape[2]), dtype=nf.dtype
                    )
                    batched["node_features"] = np.concatenate([nf, pad], axis=1)
            return batched
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
        # Always ensure mask matches full action space size
        if action_mask.shape[0] != self.action_space_size:
            action_mask = _pad_mask(action_mask, self.action_space_size)
        with torch.no_grad():
            obs_t = self._to_tensor_obs(self._batch_obs(obs))
            mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
            action_t, lp_t, val_t = self._network.get_action_and_log_prob(
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

        def _get_fresh_episode():
            """Keep sampling new instances until we get one with a non-empty mask."""
            for _ in range(100):
                raw = instance_generator()
                obs, info = env.reset(raw)
                mask = info["action_mask"]
                if mask.any():
                    return obs, mask
            raise RuntimeError(
                "instance_generator produced 100 consecutive dead-start instances."
            )

        obs, action_mask = _get_fresh_episode()

        while not self.rollout_buffer.is_full:
            padded_mask = _pad_mask(action_mask, self.action_space_size)

            action, log_prob, value = self.select_action(
                obs, padded_mask, training=True
            )

            # Re-map if network selected an out-of-range or infeasible action
            if action >= len(action_mask) or not action_mask[action]:
                feasible = np.where(action_mask)[0]
                if len(feasible) > 0:
                    action = int(np.random.choice(feasible))
                else:
                    # Dead-end state — record episode as done and start fresh
                    ep_rewards.append(ep_reward)
                    ep_lengths.append(ep_length)
                    ep_reward, ep_length = 0.0, 0
                    obs, action_mask = _get_fresh_episode()
                    continue
                log_prob = 0.0

            next_obs, reward, terminated, truncated, info = env.step(action)

            if self.reward_normalizer is not None:
                reward = self.reward_normalizer.normalise(reward)

            if isinstance(obs, dict):
                obs_to_store = _pad_to(obs["node_features"], self.obs_shape)
                veh_to_store = obs.get("vehicle_features")
                graph_to_store: Optional[Dict] = None
                if "edge_index" in obs:
                    graph_to_store = {
                        "edge_index": obs["edge_index"].copy(),
                        "edge_attr": obs["edge_attr"].copy(),
                        "edge_fleet": obs["edge_fleet"].copy(),
                    }
            else:
                obs_to_store = obs
                veh_to_store = None
                graph_to_store = None

            self.rollout_buffer.add(
                obs=obs_to_store,
                action=action,
                reward=reward,
                done=(terminated or truncated),
                log_prob=log_prob,
                value=value,
                action_mask=padded_mask,
                vehicle_features=veh_to_store,
                graph_data=graph_to_store,
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
                obs, action_mask = _get_fresh_episode()
            elif not action_mask.any():
                # Dead-end reached mid-episode (mask emptied without termination)
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_length)
                ep_reward, ep_length = 0.0, 0
                obs, action_mask = _get_fresh_episode()

        _, _, last_value = self.select_action(
            obs, _pad_mask(action_mask, self.action_space_size), training=False
        )
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
        self._network.train()
        metrics = ppo_update(
            network=self._network,
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
            "network_state": self._network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
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
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            # PyTorch 2.6+: numpy scalars in checkpoint require weights_only=False
            # Safe here because we only save our own checkpoints
            import torch.serialization as _ts
            import numpy._core.multiarray as _npcm

            try:
                with _ts.safe_globals([_npcm.scalar]):
                    ckpt = torch.load(path, map_location=self.device, weights_only=True)
            except Exception:
                ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self._network.load_state_dict(ckpt["network_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._update_count = ckpt.get("update_count", 0)
        self._step_count = ckpt.get("step_count", 0)
        if self.reward_normalizer and ckpt.get("reward_norm"):
            self.reward_normalizer.load_state_dict(ckpt["reward_norm"])

    def __repr__(self) -> str:
        return (
            f"PPOAgent(obs={self.obs_shape}, "
            f"actions={self.action_space_size}, "
            f"updates={self._update_count}, "
            f"device={self.device!r})"
        )

    @property
    def network(self) -> "BaseNetwork":
        return self._network


# ---------------------------------------------------------------------------
# PPO algorithm  (pure compute — no agent state, no env)
# ---------------------------------------------------------------------------


def _build_obs(batch: Batch, device: str) -> Union[torch.Tensor, Dict]:
    """
    Reconstruct the network input from a stored Batch.

    Flat obs (Knapsack)  → FloatTensor
    Dict obs (VRPBTW)    → dict with node_features, vehicle_features,
                           and optionally edge_index / edge_attr / edge_fleet
    """
    if batch.vehicle_features is None and batch.graph_data is None:
        return torch.FloatTensor(batch.obs).to(device)

    obs: Dict = {
        "node_features": torch.FloatTensor(batch.obs).to(device),
    }
    if batch.vehicle_features is not None:
        obs["vehicle_features"] = torch.FloatTensor(batch.vehicle_features).to(device)

    if batch.graph_data is not None:
        obs["edge_index"] = [
            gd["edge_index"] if gd is not None else np.zeros((2, 0), dtype=np.int32)
            for gd in batch.graph_data
        ]
        obs["edge_attr"] = [
            gd["edge_attr"] if gd is not None else np.zeros((0, 6), dtype=np.float32)
            for gd in batch.graph_data
        ]
        obs["edge_fleet"] = [
            gd["edge_fleet"] if gd is not None else np.zeros(0, dtype=np.int32)
            for gd in batch.graph_data
        ]
    return obs


def ppo_update(
    network: BaseNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    cfg: Any,  # PPOConfig
    device: str,
) -> Dict[str, float]:
    """
    Run cfg.n_epochs of PPO-clip updates on the current rollout buffer.
    Returns a dict of averaged training metrics.
    """
    network.train()

    totals = dict(
        policy_loss=0.0,
        value_loss=0.0,
        entropy=0.0,
        approx_kl=0.0,
        grad_norm=0.0,
        explained_var=0.0,
    )
    n_updates = 0
    early_stop = False

    for batch in buffer.iter_batches(cfg.mini_batch_size, n_epochs=cfg.n_epochs):
        if early_stop:
            break

        obs_t = _build_obs(batch, device)
        acts_t = torch.LongTensor(batch.actions).to(device)
        masks_t = torch.BoolTensor(batch.action_masks).to(device)
        adv_t = torch.FloatTensor(batch.advantages).to(device)
        ret_t = torch.FloatTensor(batch.returns).to(device)
        old_lp_t = torch.FloatTensor(batch.log_probs).to(device)

        new_lp, values, entropy = network.evaluate_actions(obs_t, acts_t, masks_t)

        ratio = torch.exp(new_lp - old_lp_t)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * F.mse_loss(values, ret_t)
        entropy_loss = -entropy.mean()

        loss = (
            policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(network.parameters(), cfg.max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

        totals["policy_loss"] += policy_loss.item()
        totals["value_loss"] += value_loss.item()
        totals["entropy"] += (-entropy_loss).item()
        totals["approx_kl"] += approx_kl
        totals["grad_norm"] += float(grad_norm)
        n_updates += 1

        if cfg.target_kl is not None and approx_kl > 1.5 * cfg.target_kl:
            early_stop = True

    # Explained variance from full buffer
    n = buffer._ptr
    if n > 0:
        y_true = buffer._returns[:n]
        y_pred = buffer._values[:n]
        var_y = np.var(y_true)
        totals["explained_var"] = float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8))

    d = max(n_updates, 1)
    return {
        "train/policy_loss": totals["policy_loss"] / d,
        "train/value_loss": totals["value_loss"] / d,
        "train/entropy": totals["entropy"] / d,
        "train/approx_kl": totals["approx_kl"] / d,
        "train/grad_norm": totals["grad_norm"] / d,
        "train/explained_var": totals["explained_var"],
        "train/early_stop": float(early_stop),
        "train/n_updates": float(n_updates),
    }
