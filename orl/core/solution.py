"""
core/solution.py
----------------
Container for a decoded combinatorial solution with metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Solution:
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
        lines = [
            f"Solution [{self.problem_name}]",
            f"  Objective  : {self.objective:.6f}",
            f"  # Steps    : {len(self.decision_sequence)}",
            f"  Sequence   : {self.decision_sequence[:20]}"
            + ("…" if len(self.decision_sequence) > 20 else ""),
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


@dataclass
class SolutionPool:
    """Fixed-capacity pool of best solutions found so far."""

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
