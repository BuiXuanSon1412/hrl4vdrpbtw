"""
src/rewards/default.py
5────────────────────────────────────────────────────────────────────────────
Reward functions are fully decoupled from the environment.
Swap the reward function without touching any other code.

All reward functions implement the RewardFn protocol:
    travel(cost, tardiness)  -> float
    invalid_action(reason)      -> float
    unserved_penalty(n)         -> float
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class RewardFn(Protocol):
    def travel(self, cost: float, tardiness: float) -> float: ...
    def invalid_action(self, reason: str) -> float: ...
    def unserved_penalty(self, n_unserved: int) -> float: ...


# ─────────────────────────────────────────────────────────────────────────────
# Concrete implementations
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DefaultRewardFn:
    """
    Normalised multi-objective reward.
    Weights cost minimisation and customer tardiness equally.
    """

    cost_weight: float = 0.5
    tardiness_weight: float = 0.5
    max_possible_cost: float = 28284.0  # sqrt(2)*100*100*2  (worst case)
    invalid_action_penalty: float = -0.5
    unserved_penalty_per: float = -1.0

    def travel(self, cost: float, tardiness: float) -> float:
        norm_cost = 1.0 - min(cost / self.max_possible_cost, 1.0)
        return self.cost_weight * norm_cost + self.tardiness_weight * tardiness

    def invalid_action(self, reason: str) -> float:
        return self.invalid_action_penalty

    def unserved_penalty(self, n_unserved: int) -> float:
        return self.unserved_penalty_per * n_unserved


@dataclass
class ParetoRewardFn:
    """
    Same structure but with configurable weights — used for Pareto-front sweeps.
    """

    cost_weight: float = 0.5
    tardiness_weight: float = 0.5
    max_possible_cost: float = 28284.0

    def __post_init__(self):
        assert abs(self.cost_weight + self.tardiness_weight - 1.0) < 1e-6, (
            "Weights must sum to 1.0"
        )

    def travel(self, cost: float, tardiness: float) -> float:
        norm_cost = 1.0 - min(cost / self.max_possible_cost, 1.0)
        return self.cost_weight * norm_cost + self.tardiness_weight * tardiness

    def invalid_action(self, reason: str) -> float:
        return -0.5

    def unserved_penalty(self, n_unserved: int) -> float:
        return -1.0 * n_unserved


@dataclass
class SparseRewardFn:
    """
    Only gives a reward at episode end.
    Useful for studying credit-assignment difficulty.
    """

    cost_weight: float = 0.5
    tardines_weight: float = 0.5

    def travel(self, cost: float, tardiness: float) -> float:
        return 0.0  # no intermediate reward

    def invalid_action(self, reason: str) -> float:
        return -0.5

    def unserved_penalty(self, n_unserved: int) -> float:
        return -float(n_unserved)
