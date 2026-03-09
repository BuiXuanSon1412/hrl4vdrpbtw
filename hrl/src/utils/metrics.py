"""
src/utils/metrics.py
─────────────────────────────────────────────────────────────────────────────
Shared metric computation for training, evaluation, and comparison scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EpisodeResult:
    """Result of a single rollout episode."""

    total_reward: float
    total_cost: float
    max_tardiness: float
    service_rate: float
    customers_served: int
    steps: int


@dataclass
class EvalStats:
    """Aggregate statistics over multiple evaluation episodes."""

    n_episodes: int

    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_cost: float = 0.0
    std_cost: float = 0.0
    mean_tardiness: float = 0.0
    std_tardiness: float = 0.0
    mean_service_rate: float = 0.0
    std_service_rate: float = 0.0
    min_service_rate: float = 0.0
    max_service_rate: float = 0.0
    pct_full_service: float = 0.0  # fraction of episodes with 100% service

    @classmethod
    def from_results(cls, results: List[EpisodeResult]) -> "EvalStats":
        rewards = [r.total_reward for r in results]
        costs = [r.total_cost for r in results]
        tards = [r.max_tardiness for r in results]
        srates = [r.service_rate for r in results]

        return cls(
            n_episodes=len(results),
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            mean_cost=float(np.mean(costs)),
            std_cost=float(np.std(costs)),
            mean_tardiness=float(np.mean(tards)),
            std_tardiness=float(np.std(tards)),
            mean_service_rate=float(np.mean(srates)),
            std_service_rate=float(np.std(srates)),
            min_service_rate=float(np.min(srates)),
            max_service_rate=float(np.max(srates)),
            pct_full_service=float(np.mean([r >= 1.0 for r in srates])),
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"EvalStats ({self.n_episodes} eps) | "
            f"ServiceRate={self.mean_service_rate * 100:.1f}±{self.std_service_rate * 100:.1f}% | "
            f"Cost={self.mean_cost:.1f}±{self.std_cost:.1f} | "
            f"Tardiness={self.mean_tardiness:.3f}±{self.std_tardiness:.3f} | "
            f"100%Service={self.pct_full_service * 100:.1f}%"
        )


def compute_gap(
    agent_cost: float,
    baseline_cost: float,
) -> float:
    """Optimality gap relative to a baseline (e.g. OR-Tools solution)."""
    if baseline_cost == 0:
        return float("inf")
    return (agent_cost - baseline_cost) / baseline_cost * 100.0
