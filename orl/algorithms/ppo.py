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

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import PPOConfig
from networks.base_network import BaseNetwork
from buffers import RolloutBuffer, Batch


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
