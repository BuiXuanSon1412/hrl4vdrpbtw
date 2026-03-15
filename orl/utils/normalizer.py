"""
utils/normalizer.py
-------------------
Welford online mean/variance estimator for reward normalisation.

Fixes two bugs from the original code:
  1. Normalisation subtracts mean AND divides by std  (original only divided).
  2. The normaliser can be reset on curriculum advancement.
  3. Named so its purpose is clear at the call site.

Usage
-----
norm = RunningNormalizer()
normalised = norm.normalise(raw_reward)
norm.reset()  # e.g. when curriculum advances
"""

from __future__ import annotations

import math
from typing import Optional


class RunningNormalizer:
    """
    Streaming mean/std estimator using Welford's online algorithm.

    Thread-unsafe (single-process use only).
    """

    def __init__(self, eps: float = 1e-8, clip: Optional[float] = 10.0):
        """
        Parameters
        ----------
        eps  : Small constant added to std to prevent division by zero.
        clip : If set, clip normalised values to [-clip, clip].
        """
        self.eps = eps
        self.clip = clip
        self._count = 0
        self._mean = 0.0
        self._M2 = 0.0    # sum of squared deviations (Welford accumulator)

    # ------------------------------------------------------------------
    # Welford update
    # ------------------------------------------------------------------

    def update(self, value: float) -> None:
        """Update running statistics with a new value."""
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        self._M2 += delta * (value - self._mean)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        if self._count < 2:
            return 1.0
        return self._M2 / (self._count - 1)

    @property
    def std(self) -> float:
        return max(math.sqrt(self.variance), self.eps)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalise(self, value: float) -> float:
        """
        Normalise value and update statistics.

        Returns (value - mean) / std — a proper z-score.
        Statistics are updated BEFORE normalisation so the first value
        always has std=1 (avoids NaN on first step).
        """
        self.update(value)
        z = (value - self.mean) / self.std
        if self.clip is not None:
            z = max(-self.clip, min(self.clip, z))
        return z

    def reset(self) -> None:
        """Reset statistics — call when the distribution shifts (e.g. curriculum)."""
        self._count = 0
        self._mean = 0.0
        self._M2 = 0.0

    def state_dict(self) -> dict:
        return {"count": self._count, "mean": self._mean, "M2": self._M2}

    def load_state_dict(self, d: dict) -> None:
        self._count = d["count"]
        self._mean = d["mean"]
        self._M2 = d["M2"]

    def __repr__(self) -> str:
        return (f"RunningNormalizer(n={self._count}, "
                f"mean={self._mean:.4f}, std={self.std:.4f})")
