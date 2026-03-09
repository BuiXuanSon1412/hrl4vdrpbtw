"""
src/envs/wrappers.py
─────────────────────────────────────────────────────────────────────────────
Gymnasium-style wrappers that add functionality to VRPBTWEnv without
modifying its core logic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from .vrpbtw_env import VRPBTWEnv


class NormalizeRewardWrapper(gym.Wrapper):
    """
    Clips and rescales rewards to a configurable range.
    Useful when mixing different reward functions or comparing agents.
    """

    def __init__(self, env: VRPBTWEnv, clip_low: float = -1.0, clip_high: float = 1.0):
        super().__init__(env)
        self.clip_low = clip_low
        self.clip_high = clip_high

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(np.clip(reward, self.clip_low, self.clip_high))
        return obs, reward, terminated, truncated, info


class CurriculumWrapper(gym.Wrapper):
    """
    Gradually increases the number of customers as the agent improves.

    The wrapper tracks a rolling service-rate window and bumps the problem
    size when the agent consistently performs above `threshold`.

    Usage
    ─────
        env = CurriculumWrapper(
            base_env,
            start_customers=20,
            max_customers=100,
            increment=10,
            threshold=0.85,        # 85% service rate triggers progression
            window=50,             # rolling average over last 50 episodes
        )
        env.report_episode(service_rate)   # call after every episode
    """

    def __init__(
        self,
        env: VRPBTWEnv,
        start_customers: int = 20,
        max_customers: int = 100,
        increment: int = 10,
        threshold: float = 0.85,
        window: int = 50,
    ):
        super().__init__(env)
        self.max_customers = max_customers
        self.increment = increment
        self.threshold = threshold
        self._window = window
        self._history: list = []
        self._current_n = start_customers
        env.num_customers = start_customers

    def report_episode(self, service_rate: float):
        """Call at the end of every training episode with the achieved service rate."""
        self._history.append(service_rate)
        if len(self._history) >= self._window:
            avg = float(np.mean(self._history[-self._window :]))
            if avg >= self.threshold and self._current_n < self.max_customers:
                self._current_n = min(
                    self._current_n + self.increment, self.max_customers
                )
                self.env.num_customers = self._current_n
                self._history.clear()

    @property
    def current_num_customers(self) -> int:
        return self._current_n


class TimeLimitWrapper(gym.Wrapper):
    """Hard episode time-limit, overriding the env's own max_steps."""

    def __init__(self, env: VRPBTWEnv, max_steps: int):
        super().__init__(env)
        self._max_steps = max_steps
        self._steps = 0

    def reset(self, **kwargs):
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        if self._steps >= self._max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info
