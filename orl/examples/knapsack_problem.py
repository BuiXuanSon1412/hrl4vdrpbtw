"""
examples/knapsack_problem.py
-----------------------------
End-to-end example: 0/1 Knapsack solved with the RL framework.

Demonstrates the MINIMAL subclassing contract:
  - encode_instance
  - initial_state
  - get_action_mask
  - apply_action
  - state_to_obs
  - evaluate
  - is_complete
  - action_space_size / observation_shape

Run:
    python -m examples.knapsack_problem
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.problem import CombinatorialProblem, ActionMask, StepResult
from core.solution import Solution


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------


@dataclass
class KnapsackState:
    selected: List[int]  # indices of selected items
    remaining_cap: float  # remaining weight capacity
    step: int  # current decision index (0..n_items)


# ---------------------------------------------------------------------------
# Knapsack problem
# ---------------------------------------------------------------------------


class KnapsackProblem(CombinatorialProblem):
    """
    0/1 Knapsack: select a subset of items to maximise total value
    subject to a weight capacity constraint.

    Instance format (dict):
        {
          "weights":  [w0, w1, ..., wN-1],
          "values":   [v0, v1, ..., vN-1],
          "capacity": C,
        }

    State: KnapsackState (which items picked, remaining capacity)
    Action: 0 = skip item, 1 = take item  (binary at each step)
    Observation: [remaining_cap/C, item_weight/max_w, item_value/max_v,
                  fractional_value_density, step/n_items]  (5-dim vector)
    """

    def __init__(self, n_items: int = 20):
        super().__init__(name="Knapsack")
        self.n_items = n_items
        # These are set in encode_instance
        self._weights: np.ndarray = np.array([])
        self._values: np.ndarray = np.array([])
        self._capacity: float = 0.0
        self._max_w: float = 1.0
        self._max_v: float = 1.0

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def encode_instance(self, raw_instance: Dict) -> None:
        self._weights = np.array(raw_instance["weights"], dtype=np.float32)
        self._values = np.array(raw_instance["values"], dtype=np.float32)
        self._capacity = float(raw_instance["capacity"])
        self.n_items = len(self._weights)
        self._max_w = max(self._weights.max(), 1e-6)
        self._max_v = max(self._values.max(), 1e-6)

    def initial_state(self) -> KnapsackState:
        return KnapsackState(
            selected=[],
            remaining_cap=self._capacity,
            step=0,
        )

    def get_action_mask(self, state: KnapsackState) -> ActionMask:
        """
        At each step we decide on item[state.step].
        Action 0 (skip) is always feasible.
        Action 1 (take) is feasible only if item fits.
        """
        if state.step >= self.n_items:
            return ActionMask.all_valid(2)  # dummy (is_complete handles this)

        w = self._weights[state.step]
        mask = np.array([True, w <= state.remaining_cap])
        return ActionMask.from_bool_array(mask)

    def apply_action(self, state: KnapsackState, action: int) -> StepResult:
        """action=0: skip, action=1: take."""
        item = state.step
        selected = list(state.selected)
        remaining = state.remaining_cap

        if action == 1:
            selected.append(item)
            remaining -= self._weights[item]

        next_state = KnapsackState(
            selected=selected,
            remaining_cap=remaining,
            step=item + 1,
        )
        terminated = next_state.step >= self.n_items

        # Dense reward: incremental value gained
        reward = float(self._values[item]) if action == 1 else 0.0

        next_mask = (
            self.get_action_mask(next_state)
            if not terminated
            else ActionMask.all_valid(2)
        )

        return StepResult(
            next_state=next_state,
            reward=reward,
            terminated=terminated,
            truncated=False,
            action_mask=next_mask,
            info={"item": item, "action": action},
        )

    def state_to_obs(self, state: KnapsackState) -> np.ndarray:
        """5-dimensional feature vector for the current decision item."""
        if state.step >= self.n_items:
            return np.zeros(self.observation_shape, dtype=np.float32)

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
        return float(self._values[state.selected].sum())

    def is_complete(self, state: KnapsackState) -> bool:
        return state.step >= self.n_items

    @property
    def action_space_size(self) -> int:
        return 2  # binary: skip or take

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (5,)

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def decode_solution(self, state: KnapsackState) -> Solution:
        return Solution(
            problem_name=self.name,
            raw_state=state,
            objective=self.evaluate(state),
            decision_sequence=state.selected,
            metadata={
                "selected_items": state.selected,
                "total_weight": float(self._weights[state.selected].sum())
                if state.selected
                else 0.0,
                "capacity": self._capacity,
            },
        )

    def heuristic_solution(self) -> Optional[float]:
        """Greedy fractional (upper-bound approximation for dense shaping)."""
        order = np.argsort(self._values / np.maximum(self._weights, 1e-6))[::-1]
        cap = self._capacity
        total = 0.0
        for i in order:
            if self._weights[i] <= cap:
                total += self._values[i]
                cap -= self._weights[i]
            else:
                total += self._values[i] * (cap / self._weights[i])
                break
        return float(total)


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------


def generate_knapsack(n_items: int = 20, seed: Optional[int] = None) -> Dict:
    rng = np.random.default_rng(seed)
    weights = rng.uniform(1, 10, size=n_items).astype(np.float32)
    values = rng.uniform(1, 10, size=n_items).astype(np.float32)
    capacity = float(weights.sum() * 0.5)
    return {
        "weights": weights.tolist(),
        "values": values.tolist(),
        "capacity": capacity,
    }


# ---------------------------------------------------------------------------
# Self-contained demo (no PyTorch needed)
# ---------------------------------------------------------------------------


def demo():
    """Quick sanity check: random rollout on a Knapsack instance."""
    from environments.combinatorial_env import CombinatorialEnv

    problem = KnapsackProblem(n_items=10)
    env = CombinatorialEnv(problem, max_steps=20, dense_shaping=True)

    raw = generate_knapsack(n_items=10, seed=42)
    obs, info = env.reset(raw)
    print("Knapsack Demo — Random Policy")
    print(f"  Instance capacity : {raw['capacity']:.1f}")
    print(f"  Obs shape         : {obs.shape}")

    total_reward = 0.0
    done = False
    while not done:
        mask = info["action_mask"]
        feasible = info["feasible_actions"]
        action = int(np.random.choice(feasible))
        obs, r, terminated, truncated, info = env.step(action)
        total_reward += r
        done = terminated or truncated

    sol = env.decode_current_solution()
    print(f"  Selected items    : {sol.decision_sequence}")
    print(f"  Objective (value) : {sol.objective:.2f}")
    print(f"  Episode reward    : {total_reward:.2f}")
    print(f"  {sol.summary()}")


if __name__ == "__main__":
    demo()
