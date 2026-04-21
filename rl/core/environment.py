"""
core/environment.py
---------------
Abstract base class for combinatorial optimisation problems,
plus the data containers every problem shares.

Any new problem (TSP, CVRP, Knapsack, …) must subclass Environment and
implement the abstract members.  All other core/ classes depend
only on this ABC — never on a concrete problem.

Containers defined here
-----------------------
  ActionMask    — boolean feasibility mask over the action space
  StepResult    — everything returned by apply_action
  Solution      — decoded solution with objective and metadata
  SolutionPool  — fixed-capacity best-solution keeper
  State         — abstract state representation (see core/state.py)
  Environment   — abstract MDP definition
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


"""
-------------
Abstract state representation for MDP/environment abstraction.

All environments track state internally and transition through states.
A State encapsulates the complete problem state at a point in time.

Design principle:
  - Problem holds no mutable state
  - State is immutable or at least semantically isolated
  - Episode management (stateful reset/step) is separate from Problem (stateless)
"""


class State(ABC):
    """
    Abstract state representation for any MDP.

    A state encapsulates:
    - Decision variables (which node served, vehicle positions, etc.)
    - Constraints (time windows, capacities, etc.)
    - Derived quantities (cost, feasibility, etc.)

    Implementations should be immutable or at least thread-safe.
    """

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """True if this state is terminal (no more actions possible)."""
        ...

    @property
    @abstractmethod
    def is_feasible(self) -> bool:
        """True if solution represented by this state is feasible."""
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dict for checkpointing, logging, etc."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> State:
        """Deserialize state from dict."""
        ...


# Example concrete implementation (defined in problems/vrpbtw.py):
# class VRPBTWState(State):
#     truck_node: np.ndarray  # (K,)
#     truck_time: np.ndarray  # (K,)
#     drone_node: np.ndarray  # (K,)
#     ...
#
#     @property
#     def is_terminal(self) -> bool:
#         return all_nodes_served()
#
#     @property
#     def is_feasible(self) -> bool:
#         return all_time_windows_met()


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
    """Everything returned by Environment.apply_action."""

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
# Environment  (abstract base)
# ---------------------------------------------------------------------------


class Environment(ABC):
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

    def __init__(self, name: str = "Environment"):
        self.name = name
        self._n_steps: int = 0
        self._current_state: Any = None

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

    def get_candidate_starts(self) -> List[Tuple[Any, Dict[str, Any], Dict[str, Any]]]:
        """
        Return a list of candidate starting states for POMO or other multi-start algorithms.

        Called AFTER encode_instance(). Each tuple is (state, obs, info) where info contains
        'action_mask' and 'feasible_actions'.

        Default: raises NotImplementedError. Concrete envs override this for algorithms
        that benefit from multiple starting points (POMO, symmetry breaking, etc.).

        Returns
        -------
        list of (state, obs, info) tuples for each candidate start configuration.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_candidate_starts() for POMO"
        )

    @property
    def n_steps(self) -> int:
        return self._n_steps

    # ------------------------------------------------------------------
    # Stateful episode interface  (replaces Environment)
    # ------------------------------------------------------------------

    def reset(
        self, raw_instance: Any
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], Dict[str, Any]]:
        """
        Encode a new instance and return the first observation.

        Returns
        -------
        obs  : observation dict/array from state_to_obs
        info : dict with 'action_mask' (bool array) and 'feasible_actions' (indices)
        """
        self.encode_instance(raw_instance)
        self._current_state = self.initial_state()
        self._n_steps = 0
        mask = self.get_action_mask(self._current_state)
        obs = self.state_to_obs(self._current_state)
        return obs, {
            "action_mask": mask.mask,
            "feasible_actions": mask.action_indices,
        }

    def step(
        self, action: int
    ) -> Tuple[
        Union[np.ndarray, Dict[str, np.ndarray]], float, bool, bool, Dict[str, Any]
    ]:
        """
        Apply action to the current episode state.

        Returns
        -------
        obs        : next observation
        reward     : float
        terminated : bool — episode ended naturally
        truncated  : bool — episode ended by external limit (unused; always False)
        info       : dict with 'action_mask', 'feasible_actions', plus any result.info keys
        """
        result = self.apply_action(self._current_state, action)
        self._current_state = result.next_state
        obs = self.state_to_obs(self._current_state)
        mask = result.action_mask
        info: Dict[str, Any] = {
            "action_mask": mask.mask,
            "feasible_actions": mask.action_indices,
            **result.info,
        }
        return obs, result.reward, result.terminated, result.truncated, info

    def current_solution(self) -> Solution:
        """Decode the current episode state into a Solution."""
        return self.decode_solution(self._current_state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
