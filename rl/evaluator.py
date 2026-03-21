"""
evaluator.py
---------------------
Evaluation module: runs the agent greedily on held-out instances,
collects solution quality metrics, and supports beam search decoding.
"""

from typing import Union, Callable, Optional, Any, List, Dict, Tuple

import numpy as np
import time
from agent import PPOAgent
from core import Environment
from problem import Solution, SolutionPool

Agent = Union[PPOAgent]


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
            sol = self._greedy_decode(raw_instance)
            pool.add(sol)
        assert pool.best is not None, "SolutionPool is empty after sampling"
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

        # beam entry: (neg_log_prob, state, action_sequence)
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

                # normalise obs to array for log-prob computation
                obs_array = obs["node_features"] if isinstance(obs, dict) else obs

                # Get log-probs from policy
                if hasattr(self.agent, "network"):
                    try:
                        import torch
                        import torch.nn.functional as F

                        if isinstance(obs, dict):
                            obs_t = {
                                k: torch.FloatTensor(v).unsqueeze(0)
                                for k, v in obs.items()
                                if isinstance(v, np.ndarray)
                            }
                        else:
                            obs_t = torch.FloatTensor(obs).unsqueeze(0)
                        mask_t: torch.Tensor = torch.tensor(
                            action_mask.mask, dtype=torch.bool
                        ).unsqueeze(0)
                        with torch.no_grad():
                            logits, _ = self.agent.network.forward(obs_t, mask_t)
                        log_probs = F.log_softmax(logits, dim=-1).squeeze(0).numpy()
                    except Exception:
                        log_probs = self._numpy_log_probs(obs_array, action_mask.mask)
                else:
                    log_probs = self._numpy_log_probs(obs_array, action_mask.mask)

                # Expand top-k feasible actions
                feasible = action_mask.action_indices
                top_k = min(self.beam_width, len(feasible))
                top_acts = feasible[np.argsort(log_probs[feasible])[-top_k:][::-1]]

                for action in top_acts:
                    result = problem.apply_action(state, int(action))
                    new_score = float(score) - float(log_probs[int(action)])
                    candidates.append((new_score, result.next_state, seq + [action]))
                    if result.terminated:
                        completed.append((new_score, result.next_state, seq + [action]))

            # Keep best beam_width non-complete states
            candidates.sort(key=lambda x: x[0])
            beam = [c for c in candidates if not problem.is_complete(c[1])][
                : self.beam_width
            ]

        if not completed:
            return self._greedy_decode(raw_instance)

        # Pick best completed solution (highest scalar objective)
        best_state = max(
            completed,
            key=lambda x: (
                problem.scalar_objective(x[1])
                if hasattr(problem, "scalar_objective")
                else float(problem.evaluate(x[1]))  # type: ignore[arg-type]
            ),
        )
        sol = problem.decode_solution(best_state[1])
        sol.decision_sequence = best_state[2]
        return sol

    def _numpy_log_probs(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch

        obs_t = torch.FloatTensor(obs[np.newaxis])
        mask_t: torch.Tensor = torch.tensor(mask[np.newaxis], dtype=torch.bool)
        logits, _ = self.agent.network.forward(  # type: ignore[union-attr]
            obs_t, mask_t
        )
        logits_np: np.ndarray = logits[0].detach().numpy()
        logits_np = np.where(mask, logits_np, -1e9)
        logits_np -= logits_np.max()
        log_probs = logits_np - np.log(np.exp(logits_np).sum())
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
