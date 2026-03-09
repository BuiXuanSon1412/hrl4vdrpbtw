"""
utils/helpers.py
-----------------
Utility helpers used across the framework:
  - MetricsTracker   – running mean/std/min/max for any scalar metric
  - EpsilonScheduler – linear/exponential exploration decay
  - LRScheduler      – learning-rate warm-up + cosine annealing
  - set_global_seed  – reproducibility helper
  - explained_variance – PPO diagnostic
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and (if available) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Tracks running statistics for a collection of named scalar metrics.

    Usage
    -----
    tracker = MetricsTracker(window=100)
    tracker.update({"loss": 0.5, "reward": 12.3})
    print(tracker.summary())
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self._buffers[k].append(float(v))

    def mean(self, key: str) -> float:
        buf = self._buffers.get(key)
        return float(np.mean(buf)) if buf else 0.0

    def std(self, key: str) -> float:
        buf = self._buffers.get(key)
        return float(np.std(buf)) if buf else 0.0

    def latest(self, key: str) -> float:
        buf = self._buffers.get(key)
        return float(buf[-1]) if buf else 0.0

    def summary(self) -> Dict[str, float]:
        return {k: self.mean(k) for k in self._buffers}

    def reset(self) -> None:
        self._buffers.clear()

    def __repr__(self) -> str:
        return f"MetricsTracker(keys={list(self._buffers.keys())})"


# ---------------------------------------------------------------------------
# Exploration scheduler
# ---------------------------------------------------------------------------

class EpsilonScheduler:
    """
    Decays epsilon from start to end over a given number of steps.

    Supports linear and exponential schedules.
    """

    def __init__(
        self,
        start: float = 1.0,
        end:   float = 0.05,
        steps: int   = 50_000,
        schedule: str = "linear",   # "linear" | "exponential"
    ):
        self.start    = start
        self.end      = end
        self.steps    = steps
        self.schedule = schedule
        self._step    = 0

    def value(self) -> float:
        frac = min(self._step / max(self.steps, 1), 1.0)
        if self.schedule == "linear":
            return self.start + frac * (self.end - self.start)
        elif self.schedule == "exponential":
            return self.end + (self.start - self.end) * math.exp(-5.0 * frac)
        return self.end

    def step(self) -> float:
        v = self.value()
        self._step += 1
        return v

    def reset(self) -> None:
        self._step = 0


# ---------------------------------------------------------------------------
# Learning-rate scheduler (cosine annealing with linear warm-up)
# ---------------------------------------------------------------------------

class LRScheduler:
    """
    Cosine annealing with optional linear warm-up.

    Compatible with any optimiser; call scheduler.step() each update.
    """

    def __init__(
        self,
        base_lr:    float = 3e-4,
        min_lr:     float = 1e-6,
        total_steps: int  = 100_000,
        warmup_steps: int = 1_000,
    ):
        self.base_lr      = base_lr
        self.min_lr       = min_lr
        self.total_steps  = total_steps
        self.warmup_steps = warmup_steps
        self._step        = 0

    def value(self) -> float:
        t = self._step
        if t < self.warmup_steps:
            return self.base_lr * (t + 1) / max(self.warmup_steps, 1)
        progress = (t - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self) -> float:
        v = self.value()
        self._step += 1
        return v

    def apply(self, optimizer) -> None:
        """Apply current LR to a PyTorch optimizer."""
        lr = self.step()
        for pg in optimizer.param_groups:
            pg["lr"] = lr


# ---------------------------------------------------------------------------
# PPO diagnostic
# ---------------------------------------------------------------------------

def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Fraction of variance in y_true explained by y_pred.
    1.0 = perfect; 0.0 = constant predictor; negative = worse than constant.
    """
    var_y = np.var(y_true)
    if var_y < 1e-10:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / var_y)


# ---------------------------------------------------------------------------
# Action-mask utilities
# ---------------------------------------------------------------------------

def softmax_with_mask(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Numerically stable softmax respecting a boolean action mask."""
    logits = np.where(mask, logits, -1e9)
    logits -= logits.max()
    probs   = np.exp(logits)
    probs   = np.where(mask, probs, 0.0)
    total   = probs.sum()
    return probs / (total + 1e-8)


def sample_masked(logits: np.ndarray, mask: np.ndarray) -> int:
    """Sample an action from masked softmax distribution."""
    probs = softmax_with_mask(logits, mask)
    return int(np.random.choice(len(probs), p=probs))
