"""
core/problem.py
---------------
Abstract base class that encodes ANY combinatorial optimization problem
as an MDP.  Users subclass this one file to plug in a new problem.

Terminology
-----------
  instance   – a concrete problem input  (e.g. a TSP graph)
  state      – partial solution + context at decision step t
  action     – one construction / modification decision
  solution   – complete assignment of decision variables
  objective  – scalar quality of a solution (higher = better by convention)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .solution import Solution
import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ActionMask:
    """Boolean mask over the discrete action space.

    True  → action is *feasible* at the current state.
    False → action is infeasible / already used.
    """

    mask: np.ndarray  # shape (n_actions,), dtype bool
    action_indices: np.ndarray  # indices of *feasible* actions

    @classmethod
    def all_valid(cls, n: int) -> "ActionMask":
        m = np.ones(n, dtype=bool)
        return cls(mask=m, action_indices=np.arange(n))

    @classmethod
    def from_bool_array(cls, mask: np.ndarray) -> "ActionMask":
        return cls(mask=mask.astype(bool), action_indices=np.where(mask)[0])

    def is_empty(self) -> bool:
        return len(self.action_indices) == 0


@dataclass
class StepResult:
    """Everything returned by CombinatorialProblem.apply_action."""

    next_state: Any
    reward: float
    terminated: bool  # natural end of construction
    truncated: bool  # external step-limit hit
    action_mask: ActionMask
    info: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract problem
# ---------------------------------------------------------------------------


class CombinatorialProblem(ABC):
    """
    Abstract interface for combinatorial optimisation problems.

    Minimal subclassing contract
    ----------------------------
    1. ``encode_instance``   – turn raw input data into internal structures
    2. ``initial_state``     – empty / trivial starting state
    3. ``get_action_mask``   – which actions are legal right now
    4. ``apply_action``      – apply one decision, return StepResult
    5. ``state_to_obs``      – convert state → numpy observation for the net
    6. ``evaluate``          – objective value of a *complete* solution
    7. ``is_complete``       – True when no more decisions are needed
    8. ``action_space_size`` – total number of possible discrete actions

    Optional overrides
    ------------------
    - ``decode_solution``    – convert final state → human-readable Solution
    - ``constraint_penalty`` – shaped reward for constraint violations
    - ``augment_instance``   – data-augmentation for training variety
    - ``heuristic_solution`` – warm-start / baseline comparator
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, name: str = "CombinatorialProblem"):
        self.name = name
        self._instance: Any = None
        self._n_steps: int = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def encode_instance(self, raw_instance: Any) -> None:
        """
        Ingest raw problem data and build any internal data structures
        (distance matrices, graphs, capacity arrays, …).

        Args:
            raw_instance: Problem-specific data (dict, numpy array, graph, …)
        """

    @abstractmethod
    def initial_state(self) -> Any:
        """
        Return the starting state for a new episode.
        The state can be any Python object; it will be passed back to
        ``apply_action`` and ``state_to_obs`` unchanged.
        """

    @abstractmethod
    def get_action_mask(self, state: Any) -> ActionMask:
        """
        Return a boolean mask of which actions are currently feasible.

        Args:
            state: Current partial-solution state.

        Returns:
            ActionMask with feasible action indices populated.
        """

    @abstractmethod
    def apply_action(self, state: Any, action: int) -> StepResult:
        """
        Apply ``action`` to ``state`` and return the transition.

        Reward design guidelines
        ------------------------
        * Dense rewards: shaped signal at every step (e.g. –Δcost).
        * Sparse rewards: 0 until termination, then objective value.
        * Mix:  small dense bonus + large terminal reward.

        Args:
            state:  Current state.
            action: Integer action index.

        Returns:
            StepResult containing next_state, reward, flags, mask, info.
        """

    @abstractmethod
    def state_to_obs(self, state: Any) -> np.ndarray:
        """
        Flatten/encode the state into a numpy array for the policy network.

        The shape must be consistent across calls for a fixed instance size.
        Typically returns a 2-D array (n_nodes × feature_dim) for attention
        networks, or a 1-D vector for MLP-based policies.

        Args:
            state: Current state.

        Returns:
            obs: numpy float32 array.
        """

    @abstractmethod
    def evaluate(self, state: Any) -> float:
        """
        Compute the objective value of a (complete) solution state.
        Higher is better by convention; negate costs if minimising.

        Args:
            state: Terminal state representing a complete solution.

        Returns:
            objective: Scalar float.
        """

    @abstractmethod
    def is_complete(self, state: Any) -> bool:
        """Return True when the construction is finished."""

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Total number of discrete actions (fixed for the problem class)."""

    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        """
        Shape of the observation array returned by ``state_to_obs``.
        Used by networks to build their input layers.
        """

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def decode_solution(self, state: Any) -> Solution:
        """
        Convert a terminal state into a Solution object.
        Default implementation stores the raw state; override for richer output.
        """
        from .solution import Solution

        return Solution(
            problem_name=self.name,
            raw_state=state,
            objective=self.evaluate(state)
            if self.is_complete(state)
            else float("-inf"),
        )

    def constraint_penalty(self, state: Any, action: int) -> float:
        """
        Optional shaped penalty for soft-constraint violations.
        Return 0.0 if all constraints are hard (handled via action mask).
        """
        return 0.0

    def augment_instance(self, raw_instance: Any) -> Any:
        """
        Data augmentation: return a transformed version of the raw instance.
        E.g. rotate / reflect a TSP instance, renumber nodes, etc.
        Default: identity (no augmentation).
        """
        return raw_instance

    def heuristic_solution(self) -> Optional[float]:
        """
        Return a heuristic objective value for the current instance,
        used as a baseline in normalised reward shaping.
        Return None to disable normalisation.
        """
        return None

    def instance_features(self) -> np.ndarray:
        """
        Optional global instance-level feature vector (appended to observations).
        Shape: (feature_dim,).  Default: empty array.
        """
        return np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reset(self, raw_instance: Any, seed: Optional[int] = None) -> Any:
        """
        Convenience: encode instance and return initial state.
        Equivalent to calling encode_instance then initial_state.
        """
        if seed is not None:
            np.random.seed(seed)
        self.encode_instance(raw_instance)
        self._n_steps = 0
        return self.initial_state()

    def step(self, state: Any, action: int) -> StepResult:
        """
        Convenience wrapper: validates action then calls apply_action.
        Raises ValueError on infeasible actions.
        """
        mask = self.get_action_mask(state)
        if not mask.mask[action]:
            raise ValueError(
                f"Action {action} is infeasible at the current state. "
                f"Feasible actions: {mask.action_indices.tolist()}"
            )
        self._n_steps += 1
        result = self.apply_action(state, action)
        # Inject step counter into info
        result.info["step"] = self._n_steps
        return result

    @property
    def n_steps(self) -> int:
        """Steps taken since last reset."""
        return self._n_steps

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
