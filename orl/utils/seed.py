"""
utils/seed.py
-------------
Centralised seed management for full reproducibility.

Every source of randomness — Python, NumPy, PyTorch, CUDA — is seeded
from a single integer.  Child seeds for the environment and data pipeline
are derived deterministically so they are independent yet reproducible.

Usage
-----
from utils.seed import SeedManager

sm = SeedManager(global_seed=42)
sm.seed_everything()
env_rng  = sm.make_env_rng()
data_rng = sm.make_data_rng()
"""

from __future__ import annotations

import hashlib
import random
from typing import Optional

import numpy as np


def _derive_seed(base: int, tag: str) -> int:
    """Deterministically derive a child seed from a base seed and a tag."""
    h = hashlib.md5(f"{base}:{tag}".encode()).hexdigest()
    return int(h, 16) % (2**31)


class SeedManager:
    """
    Manages all sources of randomness for a single experiment.

    Parameters
    ----------
    global_seed  : Master seed.
    env_seed     : Override for environment RNG (default: derived).
    data_seed    : Override for data pipeline RNG (default: derived).
    """

    def __init__(
        self,
        global_seed: int = 42,
        env_seed: Optional[int] = None,
        data_seed: Optional[int] = None,
    ):
        self.global_seed = global_seed
        self.env_seed = env_seed if env_seed is not None else _derive_seed(global_seed, "env")
        self.data_seed = data_seed if data_seed is not None else _derive_seed(global_seed, "data")

    def seed_everything(self) -> None:
        """Seed Python, NumPy, and PyTorch (if available)."""
        random.seed(self.global_seed)
        np.random.seed(self.global_seed)
        try:
            import torch
            torch.manual_seed(self.global_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.global_seed)
            # Deterministic ops (at some perf cost)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

    def make_env_rng(self) -> np.random.Generator:
        """Return a dedicated NumPy RNG for the environment."""
        return np.random.default_rng(self.env_seed)

    def make_data_rng(self) -> np.random.Generator:
        """Return a dedicated NumPy RNG for data/instance generation."""
        return np.random.default_rng(self.data_seed)

    def make_eval_rng(self) -> np.random.Generator:
        """Return a dedicated NumPy RNG for evaluation (fixed across runs)."""
        return np.random.default_rng(_derive_seed(self.global_seed, "eval"))

    def __repr__(self) -> str:
        return (
            f"SeedManager(global={self.global_seed}, "
            f"env={self.env_seed}, data={self.data_seed})"
        )
