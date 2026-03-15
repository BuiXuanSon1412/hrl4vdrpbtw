"""
core/problem.py
---------------
Abstract base class encoding any combinatorial optimisation problem as an MDP.

Users subclass Problem to plug in a new problem.
The framework never imports concrete problem classes; it only uses this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .solution import Solution


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ActionMask:
    """
    Boolean mask over the discrete action space.

    True  → action is feasible at the current state.
    False → infeasible / already used.
    """

    mask: np.ndarray  # shape (n_actions,), dtype bool
    action_indices: np.ndarray  # indices of feasible actions

    @classmethod
    def all_valid(cls, n: int) -> "ActionMask":
        m = np.ones(n, dtype=bool)
        return cls(mask=m, action_indices=np.arange(n))

    @classmethod
    def from_bool_array(cls, arr: np.ndarray) -> "ActionMask":
        arr = arr.astype(bool)
        return cls(mask=arr, action_indices=np.where(arr)[0])

    def is_empty(self) -> bool:
        return len(self.action_indices) == 0


@dataclass
class StepResult:
    """Everything returned by Problem.apply_action."""

    next_state: Any
    reward: float
    terminated: bool  # natural construction end
    truncated: bool  # external step-limit hit
    action_mask: ActionMask
    info: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract problem
# ---------------------------------------------------------------------------


class Problem(ABC):
    """
    MDP definition for a combinatorial optimisation problem.

    Minimal contract (7 abstract members)
    ----------------------------------------
    encode_instance   – parse raw input, build internal structures
    initial_state     – empty / trivial starting state
    get_action_mask   – legal actions at the current state
    apply_action      – apply one decision, return StepResult
    state_to_obs      – state → numpy array for the policy network
    evaluate          – scalar objective of a complete solution
    is_complete       – True when no more decisions are needed
    action_space_size – total discrete actions (property)
    observation_shape – obs array shape (property)
    """

    def __init__(self, name: str = "Problem"):
        self.name = name
        self._n_steps: int = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def encode_instance(self, raw_instance: Any) -> None: ...

    @abstractmethod
    def initial_state(self) -> Any: ...

    @abstractmethod
    def get_action_mask(self, state: Any) -> ActionMask: ...

    @abstractmethod
    def apply_action(self, state: Any, action: int) -> StepResult: ...

    @abstractmethod
    def state_to_obs(self, state: Any) -> np.ndarray: ...

    @abstractmethod
    def evaluate(self, state: Any) -> float: ...

    @abstractmethod
    def is_complete(self, state: Any) -> bool: ...

    @property
    @abstractmethod
    def action_space_size(self) -> int: ...

    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]: ...

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def decode_solution(self, state: Any) -> Solution:
        return Solution(
            problem_name=self.name,
            raw_state=state,
            objective=self.evaluate(state)
            if self.is_complete(state)
            else float("-inf"),
        )

    def heuristic_solution(self) -> Optional[float]:
        """Return a heuristic baseline objective (for reward shaping)."""
        return None

    def augment_instance(self, raw_instance: Any) -> Any:
        return raw_instance

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset(self, raw_instance: Any) -> Any:
        """Encode instance and return initial state."""
        self.encode_instance(raw_instance)
        self._n_steps = 0
        return self.initial_state()

    def step(self, state: Any, action: int) -> StepResult:
        """Validate action then apply it."""
        mask = self.get_action_mask(state)
        if not mask.mask[action]:
            raise ValueError(
                f"Action {action} is infeasible. "
                f"Feasible: {mask.action_indices.tolist()}"
            )
        self._n_steps += 1
        result = self.apply_action(state, action)
        result.info["n_steps"] = self._n_steps
        return result

    @property
    def n_steps(self) -> int:
        return self._n_steps

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
