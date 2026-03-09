"""
src/agents/base.py
─────────────────────────────────────────────────────────────────────────────
Abstract base class every agent must implement.
Provides a uniform API for training, evaluation, saving/loading,
and serialisation — used by all scripts.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict


class BaseAgent(abc.ABC):
    """
    Minimal interface every agent must satisfy.

    Lifecycle
    ─────────
        agent = SomeAgent(env, cfg)
        for episode in range(cfg.training.num_episodes):
            result  = agent.rollout(training=True)      # collect experience
            if should_train:
                losses  = agent.train_step()            # update weights
            agent.update_exploration(episode)           # decay ε / τ

        agent.save(checkpoint_dir / "best.pt")

        # Later:
        agent.load(checkpoint_dir / "best.pt")
        result = agent.rollout(training=False)          # greedy inference
    """

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg

    # ── Must implement ─────────────────────────────────────────────────────

    @abc.abstractmethod
    def rollout(self, training: bool = True) -> Dict[str, Any]:
        """
        Run one complete episode.

        Returns a dict with at least:
            total_reward    : float
            total_cost      : float
            max_tardiness : float
            service_rate    : float
            customers_served: int
            steps           : int
        """

    @abc.abstractmethod
    def train_step(self) -> Dict[str, float]:
        """
        Sample from the internal experience buffer and perform one
        gradient update.  Returns a dict of named scalar losses.
        """

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Persist model weights and optimiser state."""

    @abc.abstractmethod
    def load(self, path: Path) -> None:
        """Restore model weights and optimiser state."""

    # ── May override ───────────────────────────────────────────────────────

    def update_exploration(self, episode: int) -> None:
        """Decay ε-greedy and/or temperature. Override as needed."""

    def clear_buffers(self) -> None:
        """Flush experience replay buffer(s). Override as needed."""

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters across all models."""
        return 0
