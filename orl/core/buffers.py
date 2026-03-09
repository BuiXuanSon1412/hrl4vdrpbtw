"""
core/buffers.py
---------------
Replay buffer (off-policy DQN) and Rollout buffer (on-policy PPO).
Both are NumPy-based and framework-agnostic.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple


# ---------------------------------------------------------------------------
# Transition container
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs:          np.ndarray   # (obs_dim,) or (n_nodes, feat_dim)
    action:       int
    reward:       float
    next_obs:     np.ndarray
    done:         bool
    action_mask:  np.ndarray   # (action_space_size,) bool — next state mask


@dataclass
class Batch:
    """A mini-batch drawn from either buffer type."""
    obs:          np.ndarray   # (B, *obs_shape)
    actions:      np.ndarray   # (B,)
    rewards:      np.ndarray   # (B,)
    next_obs:     np.ndarray   # (B, *obs_shape)
    dones:        np.ndarray   # (B,) bool
    action_masks: np.ndarray   # (B, action_space_size)
    # PPO extras (None for DQN)
    log_probs:    Optional[np.ndarray] = None   # (B,)
    values:       Optional[np.ndarray] = None   # (B,)
    advantages:   Optional[np.ndarray] = None   # (B,)
    returns:      Optional[np.ndarray] = None   # (B,)
    # PER extras (None for uniform)
    weights:      Optional[np.ndarray] = None   # (B,) importance weights
    indices:      Optional[np.ndarray] = None   # (B,) buffer indices


# ---------------------------------------------------------------------------
# Replay Buffer  (uniform, off-policy)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Circular replay buffer with uniform sampling.

    Usage
    -----
    buf = ReplayBuffer(capacity=100_000, obs_shape=(20, 4), action_space_size=20)
    buf.add(transition)
    batch = buf.sample(batch_size=64)
    """

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

        self._obs         = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions     = np.zeros(capacity, dtype=np.int64)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._next_obs    = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._dones       = np.zeros(capacity, dtype=bool)
        self._action_masks = np.ones((capacity, action_space_size), dtype=bool)

    def add(self, transition: Transition) -> None:
        i = self._ptr
        self._obs[i]          = transition.obs
        self._actions[i]      = transition.action
        self._rewards[i]      = transition.reward
        self._next_obs[i]     = transition.next_obs
        self._dones[i]        = transition.done
        self._action_masks[i] = transition.action_mask
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        assert self._size >= batch_size, (
            f"Buffer has only {self._size} transitions; need {batch_size}."
        )
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

    @property
    def is_ready(self) -> bool:
        return self._size > 0


# ---------------------------------------------------------------------------
# Prioritised Replay Buffer  (PER, off-policy)
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritised Experience Replay (Schaul et al. 2016).

    Samples transitions proportional to |TD-error|^alpha.
    Importance-sampling weights correct for the bias.
    """

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
        self._step = 0

    def add(self, transition: Transition, priority: Optional[float] = None) -> None:
        max_p = self._priorities[:self._size].max() if self._size > 0 else 1.0
        self._priorities[self._ptr] = (priority or max_p)
        super().add(transition)

    def sample(self, batch_size: int) -> Batch:
        priorities = self._priorities[:self._size] ** self.alpha
        probs = priorities / priorities.sum()
        idx = np.random.choice(self._size, size=batch_size, replace=False, p=probs)

        beta = min(
            self.beta_end,
            self.beta_start + self._step * (self.beta_end - self.beta_start) / self.beta_steps,
        )
        self._step += 1

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
# Rollout Buffer  (on-policy, PPO)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Fixed-length rollout buffer for on-policy algorithms (PPO, A2C).

    Stores one full rollout, computes GAE advantages, then yields
    mini-batches for multiple epochs of gradient updates.

    Usage
    -----
    buf = RolloutBuffer(capacity=2048, obs_shape=(20,4), action_space_size=20)
    buf.add(obs, action, reward, done, log_prob, value, action_mask)
    ...
    buf.compute_returns(last_value=0.0)
    for batch in buf.iter_batches(mini_batch_size=256, n_epochs=4):
        agent.update(batch)
    buf.reset()
    """

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

        self._obs          = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions      = np.zeros(capacity, dtype=np.int64)
        self._rewards      = np.zeros(capacity, dtype=np.float32)
        self._dones        = np.zeros(capacity, dtype=bool)
        self._log_probs    = np.zeros(capacity, dtype=np.float32)
        self._values       = np.zeros(capacity, dtype=np.float32)
        self._action_masks = np.ones((capacity, action_space_size), dtype=bool)
        self._advantages   = np.zeros(capacity, dtype=np.float32)
        self._returns      = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        action_mask: np.ndarray,
    ) -> None:
        assert self._ptr < self.capacity, "RolloutBuffer is full; call reset()."
        i = self._ptr
        self._obs[i]          = obs
        self._actions[i]      = action
        self._rewards[i]      = reward
        self._dones[i]        = done
        self._log_probs[i]    = log_prob
        self._values[i]       = value
        self._action_masks[i] = action_mask
        self._ptr += 1

    def compute_returns(self, last_value: float = 0.0) -> None:
        """Compute GAE-λ advantages and discounted returns in-place."""
        gae = 0.0
        for t in reversed(range(self._ptr)):
            next_val   = last_value if t == self._ptr - 1 else self._values[t + 1]
            next_done  = 0.0 if t == self._ptr - 1 else float(self._dones[t + 1])
            delta      = (self._rewards[t]
                          + self.gamma * next_val * (1.0 - next_done)
                          - self._values[t])
            gae        = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae
            self._advantages[t] = gae
        self._returns[:self._ptr] = (self._advantages[:self._ptr]
                                     + self._values[:self._ptr])
        # Normalise advantages
        adv = self._advantages[:self._ptr]
        self._advantages[:self._ptr] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def iter_batches(
        self, mini_batch_size: int, n_epochs: int = 1
    ) -> Iterator[Batch]:
        """Yield shuffled mini-batches for gradient updates."""
        n = self._ptr
        for _ in range(n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, mini_batch_size):
                i = idx[start: start + mini_batch_size]
                yield Batch(
                    obs=self._obs[i],
                    actions=self._actions[i],
                    rewards=self._rewards[i],
                    next_obs=self._obs[i],       # unused for on-policy
                    dones=self._dones[i],
                    action_masks=self._action_masks[i],
                    log_probs=self._log_probs[i],
                    values=self._values[i],
                    advantages=self._advantages[i],
                    returns=self._returns[i],
                )

    def reset(self) -> None:
        self._ptr = 0

    @property
    def is_full(self) -> bool:
        return self._ptr >= self.capacity

    def __len__(self) -> int:
        return self._ptr
