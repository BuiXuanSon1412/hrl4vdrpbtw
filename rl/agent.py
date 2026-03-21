"""
agents/base_agent.py
--------------------
Abstract agent interface.

Design principles
-----------------
- Agents hold a network (injected, not built internally).
- Shapes (obs_shape, action_space_size) come from the problem, passed at
  construction time — not from the network.
- Both on-policy (PPO) and off-policy (DQN) agents expose collect() and
  update() so the Trainer never needs isinstance checks.
- Algorithm logic lives in algorithms/ not here.  Agents are thin shells
  that own state (network, buffer, optimizer) and delegate compute to
  algorithm objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import dataclasses
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.optim as optim

from config import PPOConfig
from module import BaseNetwork
from utils import RunningNormalizer

import torch.nn as nn
import torch.nn.functional as F

from core import RolloutBuffer, Batch


class BaseAgent(ABC):
    """
    Minimal interface all RL agents must implement.

    Constructor contract
    --------------------
    Every concrete agent must accept:
        network          : BaseNetwork  — injected, not built internally
        obs_shape        : Tuple        — from problem.observation_shape
        action_space_size: int          — from problem.action_space_size
        cfg              : algorithm-specific config dataclass

    The agent should NOT build the network, normalise rewards itself
    (delegate to RunningNormalizer), or know about problem instances.
    """

    # ------------------------------------------------------------------
    # Action selection (inference interface)
    # ------------------------------------------------------------------

    @abstractmethod
    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        training: bool = True,
    ) -> Tuple[int, float, float]:
        """
        Select an action for the current state.

        Parameters
        ----------
        obs         : Observation array, shape obs_shape.
        action_mask : Boolean array (action_space_size,), True=feasible.
        training    : If False, always deterministic.

        Returns
        -------
        action   : int   — selected action index
        log_prob : float — log-probability (0.0 for DQN)
        value    : float — critic estimate (0.0 for DQN)
        """
        ...

    # ------------------------------------------------------------------
    # Experience collection  (training interface)
    # ------------------------------------------------------------------

    @abstractmethod
    def collect(self, env: Any, instance_generator: Any) -> Dict[str, float]:
        """
        Collect experience from the environment.

        On-policy  (PPO): fills rollout buffer, returns rollout statistics.
        Off-policy (DQN): runs one episode, stores transitions, returns stats.

        Returns
        -------
        stats : Dict of metric name → value for logging.
        """
        ...

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    @abstractmethod
    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform one learning update.

        On-policy  (PPO): runs n_epochs over rollout buffer.
        Off-policy (DQN): samples batch from replay and does one gradient step.

        Returns
        -------
        metrics : Dict of training metrics, or None if no update was performed.
        """
        ...

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save all state needed to fully resume training.

        Must include:
          - network weights
          - optimizer state
          - any running statistics (reward normalizer, etc.)
          - step count
          - the NetworkConfig (so the checkpoint is self-contained)
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore agent state from a checkpoint."""
        ...


"""
agents/ppo_agent.py
--------------------
PPO agent — supports flat array obs (Knapsack), dict obs (VRPBTW without
graph), and dict obs with graph fields (VRPBTW with GNN vehicle embedder).

Key changes vs original
------------------------
- collect(): extracts edge_index/edge_attr/edge_fleet from info when
  present and stores them as graph_data in the rollout buffer.
- update() / _build_obs_for_update(): reconstructs the full obs dict
  (including graph fields) from the batch before calling ppo_update.
- select_action(): obs dict is passed directly to the network as-is;
  graph fields are already present in the dict from state_to_obs.
- _pad_to / _pad_mask helpers unchanged.
- vehicle_features handling unchanged.
"""


# ---------------------------------------------------------------------------
# Padding helpers (unchanged)
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

            # ── obs storage ──────────────────────────────────────────
            if isinstance(obs, dict):
                obs_to_store = _pad_to(obs["node_features"], self.obs_shape)
                veh_to_store = obs.get("vehicle_features")

                # graph fields — variable length, stored as dict
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
                raw = instance_generator()
                obs, info = env.reset(raw)
                action_mask = info["action_mask"]

        # bootstrap value for GAE
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
        self.network.train()
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


"""
algorithms/ppo.py
-----------------
PPO-clip algorithm: pure compute, no state, no env, no agent coupling.

Changes vs original
-------------------
- _build_obs(): new helper that reconstructs the full obs dict from a batch,
  including graph fields from batch.graph_data when present.
  For VRPBTW: obs = {
      "node_features":    (B, N+1, D)   from batch.obs
      "vehicle_features": (B, 2K, Dv)   from batch.vehicle_features
      "edge_index":       list[B] of (2, E_b)   from batch.graph_data
      "edge_attr":        list[B] of (E_b, 6)
      "edge_fleet":       list[B] of (E_b,)
  }
  For flat obs (Knapsack): obs = batch.obs tensor directly.
"""


# ---------------------------------------------------------------------------
# Obs reconstruction helper
# ---------------------------------------------------------------------------


def _build_obs(batch: Batch, device: str) -> Union[torch.Tensor, Dict]:
    """
    Reconstruct the obs format the network expects from a stored batch.

    - Flat obs (no vehicle_features, no graph_data): return FloatTensor
    - Dict obs without graph: {"node_features", "vehicle_features"}
    - Dict obs with graph:    {"node_features", "vehicle_features",
                                "edge_index", "edge_attr", "edge_fleet"}
      graph fields stay as Python lists of numpy arrays — the network's
      _prep_batch() handles list-of-arrays correctly.
    """
    if batch.vehicle_features is None and batch.graph_data is None:
        # plain flat obs (Knapsack)
        return torch.FloatTensor(batch.obs).to(device)

    obs: Dict = {
        "node_features": torch.FloatTensor(batch.obs).to(device),
    }

    if batch.vehicle_features is not None:
        obs["vehicle_features"] = torch.FloatTensor(batch.vehicle_features).to(device)

    if batch.graph_data is not None:
        # keep as list — network's _prep_batch handles it
        obs["edge_index"] = [
            (gd["edge_index"] if gd is not None else np.zeros((2, 0), dtype=np.int32))
            for gd in batch.graph_data
        ]
        obs["edge_attr"] = [
            (gd["edge_attr"] if gd is not None else np.zeros((0, 6), dtype=np.float32))
            for gd in batch.graph_data
        ]
        obs["edge_fleet"] = [
            (gd["edge_fleet"] if gd is not None else np.zeros(0, dtype=np.int32))
            for gd in batch.graph_data
        ]

    return obs


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------


def ppo_update(
    network: BaseNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    cfg: PPOConfig,
    device: str,
) -> Dict[str, float]:
    """
    Run cfg.n_epochs of PPO-clip updates on the current rollout buffer.
    """
    network.train()

    total = dict(
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

        # clipped surrogate objective
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

        total["policy_loss"] += policy_loss.item()
        total["value_loss"] += value_loss.item()
        total["entropy"] += (-entropy_loss).item()
        total["approx_kl"] += approx_kl
        total["grad_norm"] += float(grad_norm)
        n_updates += 1

        if cfg.target_kl is not None and approx_kl > 1.5 * cfg.target_kl:
            early_stop = True

    # explained variance from full buffer
    n_steps = buffer._ptr
    if n_steps > 0:
        y_true = buffer._returns[:n_steps]
        y_pred = buffer._values[:n_steps]
        var_y = np.var(y_true)
        ev = float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8))
        total["explained_var"] = ev

    n = max(n_updates, 1)
    return {
        "train/policy_loss": total["policy_loss"] / n,
        "train/value_loss": total["value_loss"] / n,
        "train/entropy": total["entropy"] / n,
        "train/approx_kl": total["approx_kl"] / n,
        "train/grad_norm": total["grad_norm"] / n,
        "train/explained_var": total.get("explained_var", float("nan")),
        "train/early_stop": float(early_stop),
        "train/n_updates": float(n_updates),
    }
