"""
algorithms/ppo.py
-----------------
PPO-clip algorithm: pure compute, no state, no env, no agent coupling.

This module contains ONLY the mathematical operations of the PPO update.
It is a stateless function, not a class.  The PPOAgent owns optimizer,
buffer, and network; it calls ppo_update() to do the math.

Separating algorithm math from agent state is the standard research-lab
pattern (see CleanRL, Stable Baselines 3, RLlib).  It makes:
  - unit-testing the algorithm trivial (no env needed)
  - swapping PPO for another on-policy algo a one-line change in the agent

Fixes vs original
-----------------
- explained_variance is now logged (key PPO diagnostic)
- gradient norm is logged BEFORE clipping (shows if clipping is active)
- KL approximation uses the numerically stable formula
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import PPOConfig
from networks.base_network import BaseNetwork
from buffers import RolloutBuffer


def ppo_update(
    network: BaseNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    cfg: PPOConfig,
    device: str,
) -> Dict[str, float]:
    """
    Run cfg.n_epochs of PPO-clip updates on the current rollout buffer.

    Parameters
    ----------
    network   : Policy + value network (modified in-place).
    optimizer : Adam / etc. for the network.
    buffer    : Filled RolloutBuffer with returns already computed.
    cfg       : PPOConfig.
    device    : "cpu" or "cuda".

    Returns
    -------
    metrics : Dict of aggregated training metrics for logging.
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

        # Reconstruct dict obs for HACN when vehicle_features are stored
        if batch.vehicle_features is not None:
            obs_t = {
                "node_features": torch.FloatTensor(batch.obs).to(device),
                "vehicle_features": torch.FloatTensor(batch.vehicle_features).to(
                    device
                ),
            }
        elif isinstance(batch.obs, dict):
            obs_t = {k: torch.FloatTensor(v).to(device) for k, v in batch.obs.items()}
        else:
            obs_t = torch.FloatTensor(batch.obs).to(device)
        acts_t = torch.LongTensor(batch.actions).to(device)
        masks_t = torch.BoolTensor(batch.action_masks).to(device)
        adv_t = torch.FloatTensor(batch.advantages).to(device)
        ret_t = torch.FloatTensor(batch.returns).to(device)
        old_lp_t = torch.FloatTensor(batch.log_probs).to(device)

        new_lp, values, entropy = network.evaluate_actions(obs_t, acts_t, masks_t)

        # Clipped surrogate objective
        ratio = torch.exp(new_lp - old_lp_t)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = 0.5 * F.mse_loss(values, ret_t)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        loss = (
            policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss
        )

        optimizer.zero_grad()
        loss.backward()

        # Log gradient norm BEFORE clipping
        grad_norm = nn.utils.clip_grad_norm_(network.parameters(), cfg.max_grad_norm)

        optimizer.step()

        # Approximate KL  (numerically stable formula)
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

    # Explained variance (key PPO diagnostic: should trend toward 1.0)
    # Computed from the full buffer — not from the last mini-batch — to avoid
    # the "batch possibly unbound" error and to get a more stable estimate.
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
