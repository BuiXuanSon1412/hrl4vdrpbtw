"""
core/trainer.py
---------------
Training loop abstractions.

Classes
-------
  BaseTrainer      — minimal interface: train() → summary dict
  OnPolicyTrainer  — full PPO-style collect → update → log → eval loop

Design notes on trainer / evaluator / MAML
-------------------------------------------
The Trainer orchestrates the *loop*; the Agent owns the *update math*.
This separation means:

  Standard PPO:  OnPolicyTrainer  + PPOAgent   (ppo_update inside agent)
  MAML:          MetaTrainer      + MAMLAgent   (meta_update inside agent)
  Eval-only:     run evaluate.py directly       (no trainer needed)

To add MAML:
  1. Add MAMLAgent(BaseAgent) to core/agent.py with its meta_update().
  2. Add MetaTrainer(BaseTrainer) here that calls
     agent.meta_update(support_batch, query_batch).
  Evaluator stays completely unchanged.

Circular dependency fix
-----------------------
The old code did `from evaluator import Agent` inside __init__ to avoid
a circular import.  Here the Evaluator is accepted as a plain Any-typed
argument to BaseTrainer.__init__, so there is no import of evaluator.py
from trainer.py at all.  The concrete type is only instantiated in
train.py / evaluate.py where both modules are already imported.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from core.agent import BaseAgent
from core.environment import Environment


# ---------------------------------------------------------------------------
# BaseTrainer
# ---------------------------------------------------------------------------


class BaseTrainer(ABC):
    """
    Minimal training-loop interface.

    Subclasses implement train() and any curriculum / meta-learning logic.
    """

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Run the full training loop and return a summary dict."""
        ...


# ---------------------------------------------------------------------------
# OnPolicyTrainer  (PPO / A2C style)
# ---------------------------------------------------------------------------


class OnPolicyTrainer(BaseTrainer):
    """
    Algorithm-agnostic on-policy training loop.

    Responsibilities
    ----------------
    - collect → update → log → eval → checkpoint
    - Curriculum scheduling
    - Early stopping
    - Saving best and periodic checkpoints

    NOT responsible for
    -------------------
    - Algorithm math        (inside agent.update())
    - Network architecture  (inside agent.network)
    - Problem definition    (inside env.problem)
    - Any isinstance dispatch on agent or problem type
    """

    def __init__(
        self,
        agent: BaseAgent,
        env: Environment,
        instance_generator: Callable,
        cfg: Any,  # ExperimentConfig
        evaluator: Any,  # Evaluator — passed in, not imported
        logger: Any,  # Logger
    ):
        self.agent = agent
        self.env = env
        self.instance_generator = instance_generator
        self.cfg = cfg
        self.tcfg = cfg.train
        self.evaluator = evaluator
        self.logger = logger

        self._timestep = 0
        self._iteration = 0
        self._best_objective = float("-inf")
        self._patience_counter = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        self._print_header()
        start_time = time.time()
        stop_reason = "timestep_limit"

        while self._timestep < self.tcfg.total_timesteps:
            iter_start = time.time()
            self._iteration += 1

            size = self._curriculum_size()
            gen = (
                (lambda s=size: self.instance_generator(size=s))
                if size is not None
                else self.instance_generator
            )

            # ── Collect + update ────────────────────────────────────
            collect_stats = self.agent.collect(self.env, gen)
            self._timestep += int(collect_stats.get("rollout/steps", 1))

            update_metrics = self.agent.update() or {}

            # ── Logging ─────────────────────────────────────────────
            iter_time = time.time() - iter_start
            all_metrics = {**collect_stats, **update_metrics, "iter_time_s": iter_time}

            if self._iteration % self.tcfg.log_interval == 0:
                self.logger.log_metrics(
                    all_metrics,
                    step=self._timestep,
                    print_keys=[
                        "rollout/mean_reward",
                        "train/policy_loss",
                        "train/explained_var",
                        "train/grad_norm",
                    ],
                )

            # ── Evaluation ──────────────────────────────────────────
            if self._iteration % self.tcfg.eval_interval == 0:
                eval_stats = self.evaluator.evaluate(self.instance_generator)
                self.logger.log_metrics(eval_stats, step=self._timestep, prefix="eval")

                mean_obj = eval_stats.get("mean_objective", float("-inf"))
                if mean_obj > self._best_objective + self.tcfg.min_delta:
                    self._best_objective = mean_obj
                    self._patience_counter = 0
                    self._save_checkpoint("best")
                    self.logger.log_event(
                        "best_checkpoint", self._timestep, objective=f"{mean_obj:.4f}"
                    )
                else:
                    self._patience_counter += 1

                if self._patience_counter >= self.tcfg.patience:
                    stop_reason = "early_stopping"
                    self.logger.log_event(
                        "early_stop", self._timestep, patience=self.tcfg.patience
                    )
                    break

            # ── Periodic checkpoint ──────────────────────────────────
            if self._iteration % self.tcfg.checkpoint_interval == 0:
                self._save_checkpoint(f"iter_{self._iteration}")

        summary = {
            "stop_reason": stop_reason,
            "total_iterations": self._iteration,
            "total_timesteps": self._timestep,
            "best_objective": self._best_objective,
            "training_time_s": round(time.time() - start_time, 1),
        }
        self._save_checkpoint("final")
        self.logger.log_event("training_complete", self._timestep, **summary)
        self.logger.close()
        self._print_footer(summary)
        return summary

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _curriculum_size(self) -> Optional[int]:
        if not self.tcfg.curriculum:
            return None
        frac = min(self._timestep / max(self.tcfg.curriculum_steps, 1), 1.0)
        return int(
            self.tcfg.curriculum_start
            + frac * (self.tcfg.curriculum_end - self.tcfg.curriculum_start)
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str) -> None:
        path = f"{self.tcfg.checkpoint_dir}/{self.cfg.name}_{tag}.pt"
        self.agent.save(path)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _print_header(self) -> None:
        p = self.env.problem
        print(
            f"\n{'=' * 64}\n"
            f"  Experiment : {self.cfg.name}\n"
            f"  Algorithm  : {self.cfg.algorithm.upper()}\n"
            f"  Network    : {self.cfg.network.network_type}\n"
            f"  Problem    : {p.name}\n"
            f"  Obs shape  : {p.observation_shape}\n"
            f"  Actions    : {p.action_space_size}\n"
            f"  Budget     : {self.tcfg.total_timesteps:,} steps\n"
            f"  Device     : {self.cfg.device}\n"
            f"{'=' * 64}"
        )

    def _print_footer(self, summary: Dict) -> None:
        print(
            f"\n{'=' * 64}\n"
            f"  Done ({summary['stop_reason']})\n"
            f"  Iterations : {summary['total_iterations']:,}\n"
            f"  Timesteps  : {summary['total_timesteps']:,}\n"
            f"  Best Obj   : {summary['best_objective']:.4f}\n"
            f"  Time       : {summary['training_time_s']:.1f}s\n"
            f"{'=' * 64}\n"
        )
