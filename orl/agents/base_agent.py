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
from typing import Any, Dict, Optional, Tuple

import numpy as np


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
