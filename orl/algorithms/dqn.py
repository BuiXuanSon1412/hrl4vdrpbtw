"""
algorithms/dqn.py
-----------------
Double DQN update step: pure compute, no agent/env coupling.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from configs import DQNConfig
from networks.base_network import BaseNetwork
from buffers import ReplayBuffer, PrioritizedReplayBuffer, Batch


def dqn_update(
    q_network: BaseNetwork,
    target_network: BaseNetwork,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    cfg: DQNConfig,
    device: str,
    buffer: Optional[ReplayBuffer] = None,
) -> Dict[str, float]:
    """
    One Double DQN gradient step.

    Parameters
    ----------
    q_network      : Online Q-network (modified in-place).
    target_network : Target Q-network (not modified here; sync separately).
    optimizer      : Optimizer for q_network.
    batch          : Sampled batch from the replay buffer.
    cfg            : DQNConfig.
    device         : Compute device.
    buffer         : Replay buffer (needed only for PER priority update).

    Returns
    -------
    metrics : Dict of training metrics.
    """
    q_network.train()

    obs_t = torch.FloatTensor(batch.obs).to(device)
    next_obs_t = torch.FloatTensor(batch.next_obs).to(device)
    acts_t = torch.LongTensor(batch.actions).to(device)
    rew_t = torch.FloatTensor(batch.rewards).to(device)
    done_t = torch.FloatTensor(batch.dones).to(device)
    mask_t = torch.BoolTensor(batch.action_masks).to(device)

    # Current Q-values
    q_vals, _ = q_network.forward(obs_t)  # (B, A)
    q_vals = q_vals.gather(1, acts_t.unsqueeze(1)).squeeze(1)  # (B,)

    # Double DQN target
    with torch.no_grad():
        next_q_online, _ = q_network.forward(next_obs_t, mask_t)
        next_acts = next_q_online.argmax(dim=1, keepdim=True)
        next_q_target, _ = target_network.forward(next_obs_t, mask_t)
        next_q = next_q_target.gather(1, next_acts).squeeze(1)
        targets = rew_t + cfg.gamma * next_q * (1.0 - done_t)

    td_errors = targets - q_vals

    # PER importance-sampling weights
    if batch.weights is not None and isinstance(buffer, PrioritizedReplayBuffer):
        weights = torch.FloatTensor(batch.weights).to(device)
        loss = (weights * td_errors.pow(2)).mean()
        if batch.indices is not None:
            buffer.update_priorities(batch.indices, td_errors.detach().cpu().numpy())
    else:
        loss = td_errors.pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)
    optimizer.step()

    return {
        "train/td_loss": loss.item(),
        "train/mean_td_error": td_errors.abs().mean().item(),
        "train/grad_norm": float(grad_norm),
    }
