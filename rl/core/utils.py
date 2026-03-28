"""
core/utils.py
-------------
General-purpose utilities for RL research projects.

Classes
-------
  RunningNormalizer  — Welford online mean/std for reward normalisation
  SeedManager        — centralised seed management for full reproducibility
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# RunningNormalizer
# ---------------------------------------------------------------------------


class RunningNormalizer:
    """
    Streaming mean/std estimator using Welford's online algorithm.

    Normalises a stream of scalar values to zero mean / unit variance.
    Call normalise(x) at every step; it updates statistics then returns
    the z-score.

    Parameters
    ----------
    eps  : Added to std to prevent division by zero.
    clip : If set, clip z-scores to [-clip, clip].
    """

    def __init__(self, eps: float = 1e-8, clip: Optional[float] = 10.0):
        self.eps = eps
        self.clip = clip
        self._count = 0
        self._mean = 0.0
        self._M2 = 0.0  # Welford sum-of-squared-deviations accumulator

    def update(self, value: float) -> None:
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        self._M2 += delta * (value - self._mean)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._M2 / (self._count - 1) if self._count >= 2 else 1.0

    @property
    def std(self) -> float:
        return max(math.sqrt(self.variance), self.eps)

    def normalise(self, value: float) -> float:
        """Update statistics then return the z-score of value."""
        self.update(value)
        z = (value - self.mean) / self.std
        if self.clip is not None:
            z = max(-self.clip, min(self.clip, z))
        return z

    def reset(self) -> None:
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
        return (
            f"RunningNormalizer(n={self._count}, "
            f"mean={self._mean:.4f}, std={self.std:.4f})"
        )


# ---------------------------------------------------------------------------
# SeedManager
# ---------------------------------------------------------------------------


def _derive_seed(base: int, tag: str) -> int:
    """Deterministically derive a child seed from a base seed and a tag."""
    h = hashlib.md5(f"{base}:{tag}".encode()).hexdigest()
    return int(h, 16) % (2**31)


class SeedManager:
    """
    Manages all sources of randomness for a single experiment.

    Calling seed_everything() seeds Python, NumPy, and PyTorch from
    global_seed.  Child RNGs for the environment and data pipeline are
    derived deterministically so they are independent yet reproducible.

    Parameters
    ----------
    global_seed : Master seed.
    env_seed    : Override for the environment RNG (None → derived).
    data_seed   : Override for the data/instance RNG (None → derived).
    """

    def __init__(
        self,
        global_seed: int = 42,
        env_seed: Optional[int] = None,
        data_seed: Optional[int] = None,
    ):
        self.global_seed = global_seed
        self.env_seed = (
            env_seed if env_seed is not None else _derive_seed(global_seed, "env")
        )
        self.data_seed = (
            data_seed if data_seed is not None else _derive_seed(global_seed, "data")
        )

    def seed_everything(self) -> None:
        """Seed Python, NumPy, and PyTorch (if available)."""
        random.seed(self.global_seed)
        np.random.seed(self.global_seed)
        try:
            import torch

            torch.manual_seed(self.global_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.global_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

    def make_env_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.env_seed)

    def make_data_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.data_seed)

    def make_eval_rng(self) -> np.random.Generator:
        """Fixed RNG for evaluation — identical across runs."""
        return np.random.default_rng(_derive_seed(self.global_seed, "eval"))

    def __repr__(self) -> str:
        return (
            f"SeedManager(global={self.global_seed}, "
            f"env={self.env_seed}, data={self.data_seed})"
        )
