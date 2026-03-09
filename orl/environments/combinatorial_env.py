"""
environments/combinatorial_env.py
----------------------------------
Gym-style MDP environment that wraps any CombinatorialProblem.

The environment handles:
  - Episode reset / step lifecycle
  - Action masking enforcement
  - Optional reward shaping (baseline subtraction, normalisation)
  - Step-limit truncation
  - Statistics accumulation across episodes
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core.problem import CombinatorialProblem, ActionMask


class CombinatorialEnv:
    """
    MDP environment wrapper for CombinatorialProblem.

    Parameters
    ----------
    problem : CombinatorialProblem
        The problem instance (already configured with encode_instance).
    max_steps : int
        Hard truncation limit per episode.  Set to None for no limit.
    reward_scale : float
        Multiply all rewards by this factor (helps stabilise training).
    subtract_baseline : bool
        If True and the problem provides heuristic_solution(), subtract
        the heuristic value from the terminal reward so the agent learns
        to *improve* over the heuristic.
    dense_shaping : bool
        If True, use incremental rewards at every step (dense).
        If False, give reward=0 until termination (sparse terminal).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        problem: CombinatorialProblem,
        max_steps: Optional[int] = None,
        reward_scale: float = 1.0,
        subtract_baseline: bool = False,
        dense_shaping: bool = True,
        seed: Optional[int] = None,
    ):
        self.problem = problem
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.subtract_baseline = subtract_baseline
        self.dense_shaping = dense_shaping
        self._rng = np.random.default_rng(seed)

        # Episode state
        self._state: Any = None
        self._current_action_mask: Optional[ActionMask] = None
        self._step_count: int = 0
        self._episode_reward: float = 0.0
        self._decision_sequence: List[int] = []

        # Running statistics
        self._total_episodes: int = 0
        self._total_steps: int = 0
        self._episode_rewards: List[float] = []
        self._episode_objectives: List[float] = []

        # Baseline (for reward shaping)
        self._baseline: Optional[float] = None

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def reset(
        self, raw_instance: Any, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Start a new episode on the given problem instance.

        Args:
            raw_instance: Raw input data passed to problem.encode_instance().
            seed:         Optional per-episode seed.

        Returns:
            obs:  Initial observation array.
            info: Dict with action_mask, instance metadata.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset problem & obtain initial state
        self._state = self.problem.reset(raw_instance)
        self._current_action_mask = self.problem.get_action_mask(self._state)
        self._step_count = 0
        self._episode_reward = 0.0
        self._decision_sequence = []

        # Compute baseline for reward shaping
        if self.subtract_baseline:
            self._baseline = self.problem.heuristic_solution()
        else:
            self._baseline = None

        obs = self.problem.state_to_obs(self._state)
        info = {
            "action_mask": self._current_action_mask.mask,
            "feasible_actions": self._current_action_mask.action_indices,
            "step": 0,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one action in the environment.

        Args:
            action: Integer action index; must be feasible.

        Returns:
            obs:        Next observation.
            reward:     Shaped scalar reward.
            terminated: True when the problem is naturally complete.
            truncated:  True when max_steps is reached.
            info:       Auxiliary information.

        Raises:
            RuntimeError: If called before reset().
            ValueError:   If action is infeasible.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        # Validate action
        if not self._current_action_mask.mask[action]:
            raise ValueError(
                f"Action {action} is infeasible. "
                f"Feasible: {self._current_action_mask.action_indices.tolist()}"
            )

        # Apply action
        result = self.problem.apply_action(self._state, action)
        self._state = result.next_state
        self._current_action_mask = result.action_mask
        self._decision_sequence.append(action)
        self._step_count += 1
        self._total_steps += 1

        # ---- Reward shaping ----------------------------------------
        reward = self._shape_reward(result.reward, result.terminated)
        self._episode_reward += reward

        # ---- Truncation --------------------------------------------
        truncated = (
            self.max_steps is not None and self._step_count >= self.max_steps
        ) or result.truncated
        terminated = result.terminated

        # ---- Episode end bookkeeping --------------------------------
        if terminated or truncated:
            self._total_episodes += 1
            self._episode_rewards.append(self._episode_reward)
            if terminated:
                obj = self.problem.evaluate(self._state)
                self._episode_objectives.append(obj)
                result.info["episode_objective"] = obj
            result.info["episode_reward"] = self._episode_reward
            result.info["episode_steps"] = self._step_count

        next_obs = self.problem.state_to_obs(self._state)
        info = {
            **result.info,
            "action_mask": self._current_action_mask.mask,
            "feasible_actions": self._current_action_mask.action_indices,
            "step": self._step_count,
            "decision_sequence": list(self._decision_sequence),
        }
        return next_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _shape_reward(self, raw_reward: float, terminated: bool) -> float:
        """Apply scaling, sparsification, and baseline subtraction."""
        if not self.dense_shaping and not terminated:
            reward = 0.0
        else:
            reward = raw_reward

        # Subtract heuristic baseline at terminal step only
        if terminated and self._baseline is not None:
            objective = self.problem.evaluate(self._state)
            reward = (objective - self._baseline) * self.reward_scale
        else:
            reward = reward * self.reward_scale

        return reward

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def current_obs(self) -> np.ndarray:
        """Return observation for the current state without stepping."""
        return self.problem.state_to_obs(self._state)

    def current_action_mask(self) -> np.ndarray:
        """Return the boolean action mask for the current state."""
        return self._current_action_mask.mask

    def current_feasible_actions(self) -> np.ndarray:
        """Return indices of feasible actions for the current state."""
        return self._current_action_mask.action_indices

    def decode_current_solution(self):
        """Decode current state as a Solution (may be partial)."""
        return self.problem.decode_solution(self._state)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        rewards = self._episode_rewards
        objs = self._episode_objectives
        return {
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
            "mean_episode_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_episode_reward": float(np.std(rewards)) if rewards else 0.0,
            "mean_objective": float(np.mean(objs)) if objs else 0.0,
            "best_objective": float(np.max(objs)) if objs else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"CombinatorialEnv(problem={self.problem.name!r}, "
            f"max_steps={self.max_steps}, "
            f"episodes={self._total_episodes})"
        )
