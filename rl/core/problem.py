"""
core/problem.py
---------------
Abstract base class for combinatorial optimisation problems,
plus the data containers every problem shares.

Any new problem (TSP, CVRP, Knapsack, …) must subclass Problem and
implement the seven abstract members.  All other core/ classes depend
only on this ABC — never on a concrete problem.

Containers defined here
-----------------------
  ActionMask    — boolean feasibility mask over the action space
  StepResult    — everything returned by apply_action
  Solution      — decoded solution with objective and metadata
  SolutionPool  — fixed-capacity best-solution keeper
  Problem       — abstract MDP definition
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# ActionMask
# ---------------------------------------------------------------------------


@dataclass
class ActionMask:
    """
    Boolean mask over the discrete action space.
    True  → action is feasible at the current state.
    False → infeasible.
    """

    mask: np.ndarray  # (n_actions,) bool
    action_indices: np.ndarray  # indices where mask is True

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


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


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
# Solution
# ---------------------------------------------------------------------------


@dataclass
class Solution:
    """Decoded combinatorial solution with objective and metadata."""

    problem_name: str
    raw_state: Any
    objective: float
    decision_sequence: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __lt__(self, other: "Solution") -> bool:
        return self.objective < other.objective

    def is_better_than(
        self, other: Optional["Solution"], minimise: bool = False
    ) -> bool:
        if other is None:
            return True
        return (
            self.objective < other.objective
            if minimise
            else self.objective > other.objective
        )

    def summary(self) -> str:
        seq_preview = self.decision_sequence[:20]
        tail = "…" if len(self.decision_sequence) > 20 else ""
        lines = [
            f"Solution [{self.problem_name}]",
            f"  Objective  : {self.objective:.6f}",
            f"  # Steps    : {len(self.decision_sequence)}",
            f"  Sequence   : {seq_preview}{tail}",
        ]
        if self.metadata:
            lines.append(f"  Metadata   : {self.metadata}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Solution(problem={self.problem_name!r}, "
            f"objective={self.objective:.4f}, "
            f"steps={len(self.decision_sequence)})"
        )


# ---------------------------------------------------------------------------
# SolutionPool
# ---------------------------------------------------------------------------


@dataclass
class SolutionPool:
    """Fixed-capacity pool that keeps the best solutions found so far."""

    capacity: int = 10
    minimise: bool = False
    _solutions: List[Solution] = field(default_factory=list, init=False)

    def add(self, sol: Solution) -> bool:
        self._solutions.append(sol)
        self._solutions.sort(reverse=not self.minimise, key=lambda s: s.objective)
        if len(self._solutions) > self.capacity:
            self._solutions = self._solutions[: self.capacity]
        return sol in self._solutions

    @property
    def best(self) -> Optional[Solution]:
        return self._solutions[0] if self._solutions else None

    @property
    def all(self) -> List[Solution]:
        return list(self._solutions)

    def __len__(self) -> int:
        return len(self._solutions)


# ---------------------------------------------------------------------------
# Problem  (abstract base)
# ---------------------------------------------------------------------------


class Problem(ABC):
    """
    Abstract MDP definition for a combinatorial optimisation problem.

    Subclassing contract
    --------------------
    Implement all seven abstract members.  Never import concrete agent,
    network, or buffer classes — the problem layer is dependency-free
    with respect to the rest of the framework.

    Abstract members
    ----------------
    encode_instance   parse raw input, build internal structures
    initial_state     return the empty / trivial starting state
    get_action_mask   return the ActionMask for the current state
    apply_action      apply one decision, return StepResult
    state_to_obs      state → numpy array (or dict of arrays)
    evaluate          scalar (or tuple) objective of a complete state
    is_complete       True when no more decisions can be made

    Properties
    ----------
    action_space_size  total number of discrete actions
    observation_shape  shape of the obs array returned by state_to_obs
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
    def state_to_obs(self, state: Any) -> Union[np.ndarray, Dict[str, np.ndarray]]: ...

    @abstractmethod
    def evaluate(self, state: Any) -> Union[float, Tuple[float, ...]]: ...

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

    def scalar_objective(self, state: Any) -> float:
        """
        Return a single scalar objective for ranking solutions.
        Default: use evaluate() directly if it already returns a float,
        or the first element if it returns a tuple.
        Override whenever evaluate() returns a multi-objective tuple.
        """
        result = self.evaluate(state)
        return float(result) if not isinstance(result, tuple) else float(result[0])

    def decode_solution(self, state: Any) -> Solution:
        obj = self.scalar_objective(state) if self.is_complete(state) else float("-inf")
        return Solution(problem_name=self.name, raw_state=state, objective=obj)

    def heuristic_solution(self) -> Optional[float]:
        """Optional baseline objective (used for reward shaping)."""
        return None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset(self, raw_instance: Any) -> Any:
        """Encode instance and return initial state."""
        self.encode_instance(raw_instance)
        self._n_steps = 0
        return self.initial_state()

    def step(self, state: Any, action: int) -> StepResult:
        """Validate action then delegate to apply_action."""
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
