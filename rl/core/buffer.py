"""
core/buffer.py
--------------
Experience storage for RL algorithms.

Classes
-------
  Batch                   — mini-batch container (on-policy and off-policy)
  ReplayBuffer            — uniform circular replay buffer (DQN)
  PrioritizedReplayBuffer — PER buffer (Schaul et al. 2016)
  RolloutBuffer           — fixed-length on-policy rollout buffer (PPO / A2C)

Design rules
------------
- Buffers are problem-agnostic: they store numpy arrays of fixed shape
  plus optional side-channels (vehicle_features, graph_data) for problems
  that need them.
- No agent, network, or environment imports.
- graph_data is stored as a Python list of dicts because edge counts are
  variable per step and cannot be pre-allocated into a fixed numpy array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


@dataclass
class Batch:
    """
    Mini-batch drawn from either buffer type.

    PPO-only fields  (log_probs, values, advantages, returns) are None
    for off-policy batches.
    PER-only fields  (weights, indices) are None for uniform batches.
    Graph fields     (graph_data) are None for non-graph problems.
    """

    obs: np.ndarray  # (B, *obs_shape)
    actions: np.ndarray  # (B,)
    rewards: np.ndarray  # (B,)
    next_obs: np.ndarray  # (B, *obs_shape)
    dones: np.ndarray  # (B,) bool
    action_masks: np.ndarray  # (B, action_space_size) bool

    # On-policy (PPO / A2C)
    log_probs: Optional[np.ndarray] = None  # (B,)
    values: Optional[np.ndarray] = None  # (B,)
    advantages: Optional[np.ndarray] = None  # (B,)
    returns: Optional[np.ndarray] = None  # (B,)

    # Prioritised replay
    weights: Optional[np.ndarray] = None  # (B,) importance weights
    indices: Optional[np.ndarray] = None  # (B,) buffer indices

    # Side-channels for structured observations
    vehicle_features: Optional[np.ndarray] = None  # (B, 2K, VEH_DIM)
    graph_data: Optional[List[Dict]] = None  # per-step edge dicts


# ---------------------------------------------------------------------------
# Replay Buffer  (uniform, off-policy)
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Circular replay buffer with uniform sampling."""

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        action_space_size: int,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_space_size = action_space_size
        self._ptr = 0
        self._size = 0

        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=bool)
        self._action_masks = np.ones((capacity, action_space_size), dtype=bool)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
    ) -> None:
        i = self._ptr
        self._obs[i] = obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_obs[i] = next_obs
        self._dones[i] = done
        self._action_masks[i] = action_mask
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        assert self._size >= batch_size, "Not enough samples in buffer."
        idx = np.random.randint(0, self._size, size=batch_size)
        return Batch(
            obs=self._obs[idx],
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            next_obs=self._next_obs[idx],
            dones=self._dones[idx],
            action_masks=self._action_masks[idx],
        )

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Prioritised Replay Buffer
# ---------------------------------------------------------------------------


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritised Experience Replay (Schaul et al. 2016)."""

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        action_space_size: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100_000,
        epsilon: float = 1e-6,
    ):
        super().__init__(capacity, obs_shape, action_space_size)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.epsilon = epsilon
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._beta_step = 0

    def add(  # type: ignore[override]
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        priority: Optional[float] = None,
    ) -> None:
        max_p = self._priorities[: self._size].max() if self._size > 0 else 1.0
        self._priorities[self._ptr] = priority if priority is not None else max_p
        super().add(obs, action, reward, next_obs, done, action_mask)

    def sample(self, batch_size: int) -> Batch:
        priorities = self._priorities[: self._size] ** self.alpha
        probs = priorities / priorities.sum()
        idx = np.random.choice(self._size, size=batch_size, replace=False, p=probs)

        beta = min(
            self.beta_end,
            self.beta_start
            + self._beta_step * (self.beta_end - self.beta_start) / self.beta_steps,
        )
        self._beta_step += 1
        weights = (self._size * probs[idx]) ** (-beta)
        weights /= weights.max()

        return Batch(
            obs=self._obs[idx],
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            next_obs=self._next_obs[idx],
            dones=self._dones[idx],
            action_masks=self._action_masks[idx],
            weights=weights.astype(np.float32),
            indices=idx,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        self._priorities[indices] = np.abs(td_errors) + self.epsilon


# ---------------------------------------------------------------------------
# Rollout Buffer  (on-policy, PPO / A2C)
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """
    Fixed-length rollout buffer for on-policy algorithms.

    Variable-length graph fields (edge_index, edge_attr, edge_fleet) are
    stored in a Python list rather than a pre-allocated array because edge
    count E varies per step.  Everything else is a pre-allocated numpy array.
    """

    normalize_advantages_flag: bool = True

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        action_space_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._ptr = 0

        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=bool)
        self._log_probs = np.zeros(capacity, dtype=np.float32)
        self._values = np.zeros(capacity, dtype=np.float32)
        self._action_masks = np.ones((capacity, action_space_size), dtype=bool)
        self._advantages = np.zeros(capacity, dtype=np.float32)
        self._returns = np.zeros(capacity, dtype=np.float32)

        # Optional side-channels (allocated lazily on first use)
        self._vehicle_features: Optional[np.ndarray] = None
        self._graph_data: List[Optional[Dict]] = [None] * capacity

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        action_mask: np.ndarray,
        vehicle_features: Optional[np.ndarray] = None,
        graph_data: Optional[Dict] = None,
    ) -> None:
        assert self._ptr < self.capacity, "RolloutBuffer is full; call reset()."
        i = self._ptr
        self._obs[i] = obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = done
        self._log_probs[i] = log_prob
        self._values[i] = value
        self._action_masks[i] = action_mask

        if vehicle_features is not None:
            if self._vehicle_features is None:
                self._vehicle_features = np.zeros(
                    (self.capacity, *vehicle_features.shape), dtype=np.float32
                )
            self._vehicle_features[i] = vehicle_features

        self._graph_data[i] = (
            {k: v.copy() if hasattr(v, "copy") else v for k, v in graph_data.items()}
            if graph_data is not None
            else None
        )
        self._ptr += 1

    # ------------------------------------------------------------------
    # GAE-λ advantage computation
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(self, last_value: float = 0.0) -> None:
        """GAE-λ.  done[t] correctly masks the bootstrap at step t."""
        gae = 0.0
        n = self._ptr
        for t in reversed(range(n)):
            not_done = 1.0 - float(self._dones[t])
            next_value = (
                self._values[t + 1] * not_done if t < n - 1 else last_value * not_done
            )
            delta = self._rewards[t] + self.gamma * next_value - self._values[t]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            self._advantages[t] = gae

        self._returns[:n] = self._advantages[:n] + self._values[:n]

        if self.normalize_advantages_flag:
            adv = self._advantages[:n]
            self._advantages[:n] = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ------------------------------------------------------------------
    # Mini-batch iterator
    # ------------------------------------------------------------------

    def iter_batches(self, mini_batch_size: int, n_epochs: int = 1) -> Iterator[Batch]:
        """Yield randomly shuffled mini-batches for n_epochs passes."""
        n = self._ptr
        has_graph = any(self._graph_data[i] is not None for i in range(n))

        for _ in range(n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, mini_batch_size):
                i = idx[start : start + mini_batch_size]
                vf = (
                    self._vehicle_features[i]
                    if self._vehicle_features is not None
                    else None
                )
                gd = [self._graph_data[j] for j in i] if has_graph else None

                yield Batch(
                    obs=self._obs[i],
                    actions=self._actions[i],
                    rewards=self._rewards[i],
                    next_obs=self._obs[i],  # unused in on-policy
                    dones=self._dones[i],
                    action_masks=self._action_masks[i],
                    log_probs=self._log_probs[i],
                    values=self._values[i],
                    advantages=self._advantages[i],
                    returns=self._returns[i],
                    vehicle_features=vf,
                    graph_data=gd,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._ptr = 0
        self._vehicle_features = None
        self._graph_data = [None] * self.capacity

    @property
    def is_full(self) -> bool:
        return self._ptr >= self.capacity

    def __len__(self) -> int:
        return self._ptr
