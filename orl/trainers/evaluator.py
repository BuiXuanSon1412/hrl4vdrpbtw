"""
training/evaluator.py
---------------------
Evaluation module: runs the agent greedily on held-out instances,
collects solution quality metrics, and supports beam search decoding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent
from core.solution import Solution, SolutionPool
from environments.combinatorial_env import Env

Agent = Union[PPOAgent, DQNAgent]


class Evaluator:
    """
    Evaluate a trained agent on fresh problem instances.

    Supports three decoding strategies:
      - greedy:  argmax at every step  (fast, deterministic)
      - sampling: stochastic, run N times, keep best  (diverse)
      - beam:    beam search over partial solutions  (best quality)

    Parameters
    ----------
    agent         : Trained PPOAgent or DQNAgent.
    env           : Env wrapping the problem.
    n_episodes    : Number of fresh instances to evaluate on.
    deterministic : True → greedy decoding; False → stochastic.
    n_samples     : For sampling decoding: rollouts per instance.
    beam_width    : Width for beam search (1 = greedy).
    """

    def __init__(
        self,
        agent: Agent,
        env: Env,
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
    # Main evaluate method
    # ------------------------------------------------------------------

    def evaluate(
        self,
        instance_generator: Callable[..., Any],
        size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the agent on ``n_episodes`` fresh instances.

        Args:
            instance_generator: Callable returning a raw instance.
            size: Optional size keyword forwarded to the generator.

        Returns:
            Dictionary of aggregated metrics.
        """
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

        stats = {
            "mean_objective": float(np.mean(objectives)),
            "std_objective": float(np.std(objectives)),
            "best_objective": float(np.max(objectives)),
            "worst_objective": float(np.min(objectives)),
            "mean_reward": float(np.mean(rewards)),
            "mean_time_s": float(np.mean(times)),
            "n_episodes": self.n_episodes,
        }

        # Optimality gap vs heuristic (if available)
        heuristic = self.env.problem.heuristic_solution()
        if heuristic is not None and heuristic != 0:
            gap = (heuristic - stats["mean_objective"]) / abs(heuristic) * 100
            stats["optimality_gap_pct"] = gap

        return stats

    # ------------------------------------------------------------------
    # Greedy decoding
    # ------------------------------------------------------------------

    def _greedy_decode(self, raw_instance: Any) -> Solution:
        """Roll out the agent greedily on one instance."""
        obs, info = self.env.reset(raw_instance)
        mask = info["action_mask"]
        ep_reward = 0.0
        actions = []

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

    # ------------------------------------------------------------------
    # Sampling decoding  (run N times, return best)
    # ------------------------------------------------------------------

    def _sampling_decode(self, raw_instance: Any, n: int) -> Solution:
        """Run stochastic decoding N times and return the best solution."""
        pool = SolutionPool(capacity=1)
        for _ in range(n):
            sol = self._greedy_decode(raw_instance)  # training=True → stochastic
            pool.add(sol)
        return pool.best

    # ------------------------------------------------------------------
    # Beam search decoding
    # ------------------------------------------------------------------

    def _beam_search(self, raw_instance: Any) -> Solution:
        """
        Beam search over partial solution states.

        Maintains ``beam_width`` candidate states in parallel,
        expanding the top-scoring feasible actions at each step.

        Returns the best complete solution found.
        """
        problem = self.env.problem
        problem.reset(raw_instance)

        # Each beam entry: (neg_log_prob, state, action_sequence)
        BeamEntry = Tuple[float, Any, List[int]]
        beam: List[BeamEntry] = [(0.0, problem.initial_state(), [])]
        completed: List[BeamEntry] = []

        while beam:
            candidates: List[BeamEntry] = []

            for score, state, seq in beam:
                if problem.is_complete(state):
                    completed.append((score, state, seq))
                    continue

                mask = problem.get_action_mask(state)
                obs = problem.state_to_obs(state)

                # Get log-probs from policy
                if hasattr(self.agent, "network"):
                    try:
                        import torch

                        obs_t = torch.FloatTensor(obs).unsqueeze(0)
                        mask_t = torch.BoolTensor(mask.mask).unsqueeze(0)
                        with torch.no_grad():
                            logits, _ = self.agent.network.forward(obs_t, mask_t)
                        import torch.nn.functional as F

                        log_probs = F.log_softmax(logits, dim=-1).squeeze(0).numpy()
                    except Exception:
                        log_probs = self._numpy_log_probs(obs, mask.mask)
                else:
                    log_probs = self._numpy_log_probs(obs, mask.mask)

                # Expand top-k feasible actions
                feasible = mask.action_indices
                top_k = min(self.beam_width, len(feasible))
                top_acts = feasible[np.argsort(log_probs[feasible])[-top_k:][::-1]]

                for action in top_acts:
                    result = problem.apply_action(state, int(action))
                    new_score = score - log_probs[action]  # accumulate neg-log
                    candidates.append((new_score, result.next_state, seq + [action]))

                    if result.terminated:
                        completed.append((new_score, result.next_state, seq + [action]))

            # Keep best beam_width non-complete states
            candidates.sort(key=lambda x: x[0])
            beam = [c for c in candidates if not problem.is_complete(c[1])][
                : self.beam_width
            ]

        if not completed:
            # Fallback to greedy
            return self._greedy_decode(raw_instance)

        # Pick best completed solution (highest objective)
        best_state = max(completed, key=lambda x: problem.evaluate(x[1]))
        sol = problem.decode_solution(best_state[1])
        sol.decision_sequence = best_state[2]
        return sol

    def _numpy_log_probs(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        logits, _ = self.agent.network.forward(obs[np.newaxis], mask[np.newaxis])
        logits = logits[0]
        logits = np.where(mask, logits, -1e9)
        logits -= logits.max()
        log_probs = logits - np.log(np.exp(logits).sum())
        return log_probs

    # ------------------------------------------------------------------
    # Batch evaluation utility
    # ------------------------------------------------------------------

    def evaluate_solutions(self, solutions: List[Solution]) -> Dict[str, float]:
        """Aggregate statistics for a pre-computed list of solutions."""
        objs = [s.objective for s in solutions]
        return {
            "mean_objective": float(np.mean(objs)),
            "std_objective": float(np.std(objs)),
            "best_objective": float(np.max(objs)),
            "worst_objective": float(np.min(objs)),
            "n_solutions": len(solutions),
        }
