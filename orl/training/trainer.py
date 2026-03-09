"""
training/trainer.py
--------------------
Master training loop orchestrating the full RL procedure:

  1. Instance generation  (curriculum scheduling)
  2. Rollout / step collection
  3. Agent update
  4. Logging (console + optional file)
  5. Evaluation checkpoints
  6. Early stopping
  7. Best-model checkpointing

Works with both PPOAgent (on-policy) and DQNAgent (off-policy)
through a unified interface.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent
from environments.combinatorial_env import CombinatorialEnv
from .evaluator import Evaluator

Agent = Union[PPOAgent, DQNAgent]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    # Duration
    total_timesteps: int = 1_000_000
    # Logging
    log_interval: int = 10  # log every N iterations
    eval_interval: int = 50  # evaluate every N iterations
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    experiment_name: str = "rl_combinatorial"
    # Early stopping
    patience: int = 100  # iterations without improvement
    min_delta: float = 1e-4
    # Curriculum
    curriculum: bool = False  # gradually increase problem size
    curriculum_start: int = 10  # starting problem size
    curriculum_end: int = 100
    curriculum_steps: int = 500_000  # timesteps to reach full size
    # Evaluation
    n_eval_episodes: int = 20
    eval_deterministic: bool = True


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Orchestrates the complete RL training pipeline.

    Parameters
    ----------
    agent             : PPOAgent or DQNAgent
    env               : CombinatorialEnv
    instance_generator: Callable[..., Any]
        A function that returns a fresh raw problem instance.
        Receives keyword argument ``size`` if curriculum=True.
    cfg               : TrainerConfig
    evaluator         : Evaluator (optional; auto-created if None)

    Usage
    -----
    trainer = Trainer(agent, env, gen_fn, cfg)
    trainer.train()
    """

    def __init__(
        self,
        agent: Agent,
        env: CombinatorialEnv,
        instance_generator: Callable[..., Any],
        cfg: Optional[TrainerConfig] = None,
        evaluator: Optional[Evaluator] = None,
    ):
        self.agent = agent
        self.env = env
        self.instance_generator = instance_generator
        self.cfg = cfg or TrainerConfig()
        self.evaluator = evaluator or Evaluator(
            agent=agent,
            env=env,
            n_episodes=self.cfg.n_eval_episodes,
            deterministic=self.cfg.eval_deterministic,
        )

        self._timestep = 0
        self._iteration = 0
        self._best_objective = float("-inf")
        self._patience_counter = 0
        self._training_log: List[Dict] = []

        # Directories
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)

        self._log_file = Path(self.cfg.log_dir) / f"{self.cfg.experiment_name}.jsonl"

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop until total_timesteps is reached
        or early stopping fires.

        Returns:
            Summary dict with best objective, total steps, training time.
        """
        print(self._header())
        start_time = time.time()
        stop_reason = "timestep_limit"

        while self._timestep < self.cfg.total_timesteps:
            iter_start = time.time()
            self._iteration += 1

            # ---- Curriculum problem size ----------------------------
            size = self._curriculum_size()

            # ---- Collect experience / rollout -----------------------
            rollout_stats = self._collect(size)
            self._timestep += rollout_stats.get(
                "rollout/steps_collected", self._steps_per_iter()
            )

            # ---- Agent update ---------------------------------------
            update_metrics = self._update()

            # ---- Logging -------------------------------------------
            iter_time = time.time() - iter_start
            log_entry = {
                "iteration": self._iteration,
                "timestep": self._timestep,
                "iter_time": round(iter_time, 3),
                **rollout_stats,
                **(update_metrics or {}),
            }

            if self._iteration % self.cfg.log_interval == 0:
                self._print_log(log_entry)

            # ---- Evaluation ----------------------------------------
            if self._iteration % self.cfg.eval_interval == 0:
                eval_stats = self.evaluator.evaluate(self.instance_generator)
                log_entry.update({f"eval/{k}": v for k, v in eval_stats.items()})

                mean_obj = eval_stats.get("mean_objective", float("-inf"))
                improved = mean_obj > self._best_objective + self.cfg.min_delta
                if improved:
                    self._best_objective = mean_obj
                    self._patience_counter = 0
                    self._save_checkpoint("best")
                else:
                    self._patience_counter += 1

                print(
                    f"  [Eval] mean_obj={mean_obj:.4f}  "
                    f"best={self._best_objective:.4f}  "
                    f"patience={self._patience_counter}/{self.cfg.patience}"
                )

                if self._patience_counter >= self.cfg.patience:
                    stop_reason = "early_stopping"
                    print(f"\n⚑ Early stopping at iteration {self._iteration}.")
                    break

            # ---- Periodic checkpoint --------------------------------
            if self._iteration % (self.cfg.eval_interval * 5) == 0:
                self._save_checkpoint(f"iter_{self._iteration}")

            self._training_log.append(log_entry)
            self._write_log(log_entry)

        total_time = time.time() - start_time
        summary = {
            "stop_reason": stop_reason,
            "total_iterations": self._iteration,
            "total_timesteps": self._timestep,
            "best_objective": self._best_objective,
            "training_time_s": round(total_time, 1),
        }
        self._save_checkpoint("final")
        self._save_summary(summary)
        print(self._footer(summary))
        return summary

    # ------------------------------------------------------------------
    # Internal dispatch (PPO vs DQN)
    # ------------------------------------------------------------------

    def _collect(self, size: Optional[int]) -> Dict[str, float]:
        """Dispatch rollout collection to the correct agent type."""
        gen = self._make_generator(size)
        if isinstance(self.agent, PPOAgent):
            return self.agent.collect_rollout(self.env, gen)
        else:
            # DQN: run one full episode of steps
            return self._dqn_episode(gen)

    def _update(self) -> Optional[Dict[str, float]]:
        """Dispatch gradient update to the correct agent type."""
        if isinstance(self.agent, PPOAgent):
            return self.agent.update()
        else:
            return self.agent.update()

    def _dqn_episode(self, gen: Callable) -> Dict[str, float]:
        """Run one episode of environment interaction for DQN."""
        from core.buffers import Transition

        raw = gen()
        obs, info = self.env.reset(raw)
        mask = info["action_mask"]
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = self.agent.select_action(obs, mask, training=True)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_mask = info["action_mask"]
            done = terminated or truncated

            self.agent.store_transition(
                Transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                    action_mask=next_mask,
                )
            )
            self.agent.update()

            total_reward += reward
            steps += 1
            obs = next_obs
            mask = next_mask

        return {
            "rollout/mean_reward": total_reward,
            "rollout/mean_ep_length": steps,
            "rollout/steps_collected": steps,
        }

    def _steps_per_iter(self) -> int:
        if isinstance(self.agent, PPOAgent):
            return self.agent.cfg.rollout_len
        return 1

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _curriculum_size(self) -> Optional[int]:
        if not self.cfg.curriculum:
            return None
        frac = min(self._timestep / max(self.cfg.curriculum_steps, 1), 1.0)
        return int(
            self.cfg.curriculum_start
            + frac * (self.cfg.curriculum_end - self.cfg.curriculum_start)
        )

    def _make_generator(self, size: Optional[int]) -> Callable:
        if size is not None:
            return lambda: self.instance_generator(size=size)
        return self.instance_generator

    # ------------------------------------------------------------------
    # Checkpointing & logging
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str) -> None:
        path = Path(self.cfg.checkpoint_dir) / f"{self.cfg.experiment_name}_{tag}.pt"
        self.agent.save(str(path))

    def _write_log(self, entry: Dict) -> None:
        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _save_summary(self, summary: Dict) -> None:
        path = Path(self.cfg.log_dir) / f"{self.cfg.experiment_name}_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def _header(self) -> str:
        a_type = type(self.agent).__name__
        return (
            f"\n{'=' * 60}\n"
            f"  Experiment : {self.cfg.experiment_name}\n"
            f"  Agent      : {a_type}\n"
            f"  Problem    : {self.env.problem.name}\n"
            f"  Budget     : {self.cfg.total_timesteps:,} steps\n"
            f"{'=' * 60}"
        )

    def _footer(self, summary: Dict) -> str:
        return (
            f"\n{'=' * 60}\n"
            f"  Training complete ({summary['stop_reason']})\n"
            f"  Iterations  : {summary['total_iterations']:,}\n"
            f"  Timesteps   : {summary['total_timesteps']:,}\n"
            f"  Best Obj    : {summary['best_objective']:.4f}\n"
            f"  Time        : {summary['training_time_s']:.1f}s\n"
            f"{'=' * 60}\n"
        )

    def _print_log(self, entry: Dict) -> None:
        it = entry.get("iteration", "?")
        ts = entry.get("timestep", 0)
        rr = entry.get("rollout/mean_reward", 0)
        pl = entry.get("train/policy_loss", entry.get("train/td_loss", 0))
        eps = entry.get("train/epsilon", "")
        eps_str = f"  ε={eps:.3f}" if eps != "" else ""
        print(f"  iter={it:5d}  ts={ts:9,}  r̄={rr:+.3f}  loss={pl:.4f}{eps_str}")
