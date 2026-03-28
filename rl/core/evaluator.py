"""
core/evaluator.py
-----------------
Evaluation module — runs the agent on fresh instances, collects
solution quality metrics, and supports multiple decoding strategies.

Decoding strategies
-------------------
  greedy   : argmax at every step  (fast, deterministic)
  sampling : stochastic, run N times, keep best
  beam     : beam search over partial solutions

No circular dependency
----------------------
Evaluator imports BaseAgent from core.agent.
Trainer accepts Evaluator as a constructor argument (Any type),
so trainer.py never imports evaluator.py.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core.agent import BaseAgent
from core.environment import Environment
from core.problem import Solution, SolutionPool


class Evaluator:
    """
    Evaluate a trained agent on fresh problem instances.

    Parameters
    ----------
    agent         : Any BaseAgent subclass.
    env           : Environment wrapping the problem.
    n_episodes    : Number of fresh instances per evaluation call.
    deterministic : True → greedy; False → stochastic.
    n_samples     : Rollouts per instance for sampling decoding.
    beam_width    : Beam width (1 = greedy).
    """

    def __init__(
        self,
        agent: BaseAgent,
        env: Environment,
        n_episodes: int = 20,
        deterministic: bool = True,
        n_samples: int = 1,
        beam_width: int = 1,
    ):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.deterministic = deterministic
        self.n_samples = n_samples
        self.beam_width = beam_width

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        instance_generator: Callable[..., Any],
        size: Optional[int] = None,
    ) -> Dict[str, float]:
        objectives: List[float] = []
        rewards: List[float] = []
        times: List[float] = []

        gen = (lambda: instance_generator(size=size)) if size else instance_generator

        for _ in range(self.n_episodes):
            raw = gen()
            t0 = time.time()

            if self.beam_width > 1:
                sol = self._beam_search(raw)
            elif self.n_samples > 1:
                sol = self._sampling_decode(raw, self.n_samples)
            else:
                sol = self._greedy_decode(raw)

            times.append(time.time() - t0)
            objectives.append(sol.objective)
            rewards.append(sol.metadata.get("episode_reward", sol.objective))

        stats: Dict[str, float] = {
            "mean_objective": float(np.mean(objectives)),
            "std_objective": float(np.std(objectives)),
            "best_objective": float(np.max(objectives)),
            "worst_objective": float(np.min(objectives)),
            "mean_reward": float(np.mean(rewards)),
            "mean_time_s": float(np.mean(times)),
            "n_episodes": float(self.n_episodes),
        }

        heuristic = self.env.problem.heuristic_solution()
        if heuristic is not None and heuristic != 0:
            gap = (heuristic - stats["mean_objective"]) / abs(heuristic) * 100
            stats["optimality_gap_pct"] = gap

        return stats

    # ------------------------------------------------------------------
    # Decoding strategies
    # ------------------------------------------------------------------

    def _greedy_decode(self, raw_instance: Any) -> Solution:
        obs, info = self.env.reset(raw_instance)
        mask = info["action_mask"]
        ep_reward = 0.0
        actions: List[int] = []

        done = False
        while not done:
            action, _, _ = self.agent.select_action(
                obs, mask, training=not self.deterministic
            )
            obs, reward, terminated, truncated, info = self.env.step(action)
            mask = info["action_mask"]
            ep_reward += reward
            actions.append(action)
            done = terminated or truncated

        sol = self.env.decode_current_solution()
        sol.decision_sequence = actions
        sol.metadata["episode_reward"] = ep_reward
        return sol

    def _sampling_decode(self, raw_instance: Any, n: int) -> Solution:
        pool = SolutionPool(capacity=1)
        for _ in range(n):
            pool.add(self._greedy_decode(raw_instance))
        assert pool.best is not None
        return pool.best

    def _beam_search(self, raw_instance: Any) -> Solution:
        """Beam search over partial solution states."""
        problem = self.env.problem
        problem.reset(raw_instance)

        # (neg_log_prob, state, action_sequence)
        beam: List[Tuple[float, Any, List[int]]] = [(0.0, problem.initial_state(), [])]
        completed: List[Tuple[float, Any, List[int]]] = []

        while beam:
            candidates: List[Tuple[float, Any, List[int]]] = []

            for score, state, seq in beam:
                if problem.is_complete(state):
                    completed.append((score, state, seq))
                    continue

                action_mask = problem.get_action_mask(state)
                obs = problem.state_to_obs(state)

                log_probs = self._get_log_probs(obs, action_mask.mask)

                feasible = action_mask.action_indices
                top_k = min(self.beam_width, len(feasible))
                top_acts = feasible[np.argsort(log_probs[feasible])[-top_k:][::-1]]

                for action in top_acts:
                    result = problem.apply_action(state, int(action))
                    new_score = float(score) - float(log_probs[int(action)])
                    candidates.append((new_score, result.next_state, seq + [action]))
                    if result.terminated:
                        completed.append((new_score, result.next_state, seq + [action]))

            candidates.sort(key=lambda x: x[0])
            beam = [c for c in candidates if not problem.is_complete(c[1])][
                : self.beam_width
            ]

        if not completed:
            return self._greedy_decode(raw_instance)

        best = max(
            completed,
            key=lambda x: problem.scalar_objective(x[1]),
        )
        sol = problem.decode_solution(best[1])
        sol.decision_sequence = best[2]
        return sol

    def _get_log_probs(self, obs: Any, mask: np.ndarray) -> np.ndarray:
        """Extract per-action log-probabilities from the policy network."""

        network = self.agent.network  # assign once to a local variable
        if network is None:
            # Fallback: uniform over feasible actions
            lp = np.full(len(mask), -1e9, dtype=np.float32)
            lp[mask] = 0.0
            return lp

        obs_t: Any
        if isinstance(obs, dict):
            obs_t = {
                k: torch.FloatTensor(v).unsqueeze(0)
                for k, v in obs.items()
                if isinstance(v, np.ndarray)
            }
        else:
            obs_t = torch.FloatTensor(obs[np.newaxis])

        mask_t = torch.tensor(mask[np.newaxis], dtype=torch.bool)

        with torch.no_grad():
            logits, _ = network.forward(obs_t, mask_t)

        lp = F.log_softmax(logits.squeeze(0), dim=-1).numpy()
        return lp

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def evaluate_solutions(self, solutions: List[Solution]) -> Dict[str, float]:
        objs = [s.objective for s in solutions]
        return {
            "mean_objective": float(np.mean(objs)),
            "std_objective": float(np.std(objs)),
            "best_objective": float(np.max(objs)),
            "worst_objective": float(np.min(objs)),
            "n_solutions": float(len(solutions)),
        }
