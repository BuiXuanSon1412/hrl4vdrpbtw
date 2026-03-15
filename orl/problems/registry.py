"""
problems/registry.py
--------------------
Registry for problem classes.

Register new problems here; nothing else changes.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from core.problem import Problem
from configs import EnvConfig


# ── Problem constructors ────────────────────────────────────────────────────


def _make_knapsack(kwargs) -> Problem:
    from problems.knapsack import KnapsackProblem

    return KnapsackProblem(**kwargs)


def _make_vrpbtw(kwargs) -> Problem:
    from problems.vrpbtw import VRPBTWProblem

    return VRPBTWProblem(**kwargs)


_PROBLEM_REGISTRY: Dict[str, Callable] = {
    "knapsack": _make_knapsack,
    "vrpbtw": _make_vrpbtw,
}

_GENERATOR_REGISTRY: Dict[str, Callable] = {}


def register_problem(name: str, constructor: Callable, generator: Callable) -> None:
    """Register a new problem type at runtime."""
    _PROBLEM_REGISTRY[name] = constructor
    _GENERATOR_REGISTRY[name] = generator


def build_problem(cfg: EnvConfig) -> Problem:
    name = cfg.problem_name.lower()
    if name not in _PROBLEM_REGISTRY:
        raise ValueError(
            f"Unknown problem={name!r}. Available: {list(_PROBLEM_REGISTRY)}"
        )
    return _PROBLEM_REGISTRY[name](cfg.problem_kwargs)


def get_generator(cfg: EnvConfig) -> Callable:
    """Return the instance generator function for a problem."""
    name = cfg.problem_name.lower()
    if name == "knapsack":
        from problems.knapsack import generate_knapsack

        return generate_knapsack
    if name == "vrpbtw":
        from problems.vrpbtw import generate_vrpbtw

        return generate_vrpbtw
    raise ValueError(f"No generator registered for problem={name!r}.")
