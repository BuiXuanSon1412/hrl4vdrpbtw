"""
problems/knapsack.py
-----------------------------
0/1 Knapsack: select items to maximise value subject to a capacity constraint.

This is the minimal example of Problem subclassing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.problem import Problem, ActionMask, StepResult
from core.solution import Solution


@dataclass
class KnapsackState:
    selected: List[int]
    remaining_cap: float
    step: int


class KnapsackProblem(Problem):
    """
    0/1 Knapsack problem.

    Instance format
    ---------------
    {"weights": [...], "values": [...], "capacity": C}

    Action space: {0=skip, 1=take} at each step (binary).
    Observation:  5-dim vector per step.
    """

    def __init__(self, n_items: int = 20):
        super().__init__(name="Knapsack")
        self.n_items = n_items
        self._weights: np.ndarray = np.array([])
        self._values: np.ndarray = np.array([])
        self._capacity: float = 0.0
        self._max_w: float = 1.0
        self._max_v: float = 1.0

    def encode_instance(self, raw_instance: Dict) -> None:
        self._weights = np.array(raw_instance["weights"], dtype=np.float32)
        self._values = np.array(raw_instance["values"], dtype=np.float32)
        self._capacity = float(raw_instance["capacity"])
        self.n_items = len(self._weights)
        self._max_w = max(self._weights.max(), 1e-6)
        self._max_v = max(self._values.max(), 1e-6)

    def initial_state(self) -> KnapsackState:
        return KnapsackState(selected=[], remaining_cap=self._capacity, step=0)

    def get_action_mask(self, state: KnapsackState) -> ActionMask:
        if state.step >= self.n_items:
            return ActionMask.all_valid(2)
        w = self._weights[state.step]
        return ActionMask.from_bool_array(np.array([True, w <= state.remaining_cap]))

    def apply_action(self, state: KnapsackState, action: int) -> StepResult:
        item = state.step
        selected = list(state.selected)
        cap = state.remaining_cap

        if action == 1:
            selected.append(item)
            cap -= self._weights[item]

        next_state = KnapsackState(selected=selected, remaining_cap=cap, step=item + 1)
        terminated = next_state.step >= self.n_items
        reward = float(self._values[item]) if action == 1 else 0.0
        next_mask = (
            self.get_action_mask(next_state)
            if not terminated
            else ActionMask.all_valid(2)
        )
        return StepResult(
            next_state,
            reward,
            terminated,
            False,
            next_mask,
            info={"item": item, "action": action},
        )

    def state_to_obs(self, state: KnapsackState) -> np.ndarray:
        if state.step >= self.n_items:
            return np.zeros(5, dtype=np.float32)
        item = state.step
        w = self._weights[item] / self._max_w
        v = self._values[item] / self._max_v
        cap = state.remaining_cap / max(self._capacity, 1e-6)
        dens = (self._values[item] / max(self._weights[item], 1e-6)) / (
            self._max_v / self._max_w
        )
        prog = item / max(self.n_items - 1, 1)
        return np.array([cap, w, v, dens, prog], dtype=np.float32)

    def evaluate(self, state: KnapsackState) -> float:
        return (
            float(self._values[list(state.selected)].sum()) if state.selected else 0.0
        )

    def is_complete(self, state: KnapsackState) -> bool:
        return state.step >= self.n_items

    @property
    def action_space_size(self) -> int:
        return 2

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (5,)

    def decode_solution(self, state: KnapsackState) -> Solution:
        w = float(self._weights[list(state.selected)].sum()) if state.selected else 0.0
        return Solution(
            problem_name=self.name,
            raw_state=state,
            objective=self.evaluate(state),
            decision_sequence=list(state.selected),
            metadata={
                "selected": list(state.selected),
                "total_weight": w,
                "capacity": self._capacity,
            },
        )

    def heuristic_solution(self) -> Optional[float]:
        """Greedy fractional upper bound."""
        order = np.argsort(self._values / np.maximum(self._weights, 1e-6))[::-1]
        cap, total = self._capacity, 0.0
        for i in order:
            if self._weights[i] <= cap:
                total += self._values[i]
                cap -= self._weights[i]
            else:
                total += self._values[i] * (cap / self._weights[i])
                break
        return float(total)


def generate_knapsack(
    n_items: int = 20,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> Dict:
    """
    Generate a random Knapsack instance.

    Parameters
    ----------
    n_items : Number of items.
    rng     : numpy Generator — caller provides for reproducibility.
              If None, a fresh default_rng() is used (non-reproducible).
    """
    if rng is None:
        rng = np.random.default_rng()
    weights = rng.uniform(1, 10, size=n_items).astype(np.float32)
    values = rng.uniform(1, 10, size=n_items).astype(np.float32)
    capacity = float(weights.sum() * 0.5)
    return {
        "weights": weights.tolist(),
        "values": values.tolist(),
        "capacity": capacity,
    }
