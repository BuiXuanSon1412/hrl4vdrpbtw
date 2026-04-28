"""
core/buffer.py
--------------
Experience storage for on-policy RL.

Classes
-------
  Transition    — one environment step, stored at its exact problem size
  RolloutBuffer — fixed-capacity buffer backed by a plain Python list

Design
------
Observations are stored as raw dicts of numpy arrays with no padding.
Each Transition keeps whatever shape state_to_obs produced for that
instance — (N_i+1, NODE_FEAT_DIM) for N_i customers, (2K, VEH_FEAT_DIM)
for K fleets, (2, E_i) for edge indices.

iter_batches yields List[Transition].  The PPO update iterates over the
list and processes each transition at B=1 (gradient accumulation), so no
padding is ever needed.  Buffer size is bounded by capacity × max(obs_bytes)
which for typical VRPBTW instances is well under 10 MB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    """
    Single environment step stored at its natural problem size.

    obs holds the full obs dict returned by state_to_obs — all arrays
    are copies so there is no aliasing with the live environment state.
    """

    obs: Any                          # dict of numpy arrays, exact problem size
    action: int
    done: bool
    log_prob: Any                     # torch.Tensor or float (for compatibility)
    action_mask: np.ndarray           # (action_space_size,) bool, exact size
    reward: Optional[float] = None    # float or None (optional)
    value: Optional[Any] = None       # torch.Tensor or float (optional, for compatibility)


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """
    Fixed-capacity on-policy rollout buffer using list storage.

    Observations and masks are stored at their exact problem sizes.
    No padding is applied at storage time — padding (if ever needed)
    is a caller responsibility at mini-batch formation time.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._ptr: int = 0
        self._data: List[Transition] = []

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(
        self,
        obs: Any,
        action: int,
        done: bool,
        log_prob: Any,                 # torch.Tensor or float
        action_mask: np.ndarray,
        reward: Optional[float] = None,  # float or None (optional)
        value: Optional[Any] = None,     # torch.Tensor or float (optional)
    ) -> None:
        assert self._ptr < self.capacity, "RolloutBuffer is full."

        # Deep-copy obs arrays so the buffer owns its data
        if isinstance(obs, dict):
            obs_stored = {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in obs.items()
            }
        else:
            obs_stored = obs.copy() if isinstance(obs, np.ndarray) else obs

        tr = Transition(
            obs=obs_stored,
            action=int(action),
            done=bool(done),
            log_prob=log_prob,            # Keep as-is (tensor or float)
            action_mask=action_mask.copy(),
            reward=float(reward) if reward is not None else None,  # Convert to float or None
            value=value,                  # Keep as-is (tensor or float, or None)
        )

        if self._ptr < len(self._data):
            self._data[self._ptr] = tr
        else:
            self._data.append(tr)

        self._ptr += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear the buffer (reset write pointer)."""
        self._ptr = 0

    @property
    def is_full(self) -> bool:
        return self._ptr >= self.capacity

    def __len__(self) -> int:
        return self._ptr
