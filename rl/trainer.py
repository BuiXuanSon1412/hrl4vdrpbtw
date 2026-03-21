from __future__ import annotations

import time
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from pathlib import Path


from agent import BaseAgent
from core import Environment
from config import ExperimentConfig, TrainConfig, save_config
from logger import Logger
from evaluator import Evaluator

"""
trainers/trainer.py
--------------------
Master training loop.

Responsibilities
----------------
- Orchestrate collect → update → log → eval → checkpoint
- Curriculum scheduling
- Early stopping
- Saving best and periodic checkpoints

NOT responsible for
-------------------
- Algorithm math (in algorithms/)
- Network architecture (in networks/)
- Problem definition (in core/)
- Any isinstance dispatch on agent type — uses BaseAgent interface only

Key fix vs original
--------------------
No isinstance(agent, PPOAgent) / isinstance(agent, DQNAgent) anywhere.
Both agent types expose collect() and update() — the trainer is truly
algorithm-agnostic.
"""


class Trainer:
    """
    Algorithm-agnostic training loop.

    Parameters
    ----------
    agent              : Any BaseAgent (PPO, DQN, SAC, …).
    env                : Env.
    instance_generator : Callable[..., Any] — returns a raw problem instance.
                         Receives ``size=`` kwarg when curriculum is active.
    cfg                : ExperimentConfig.
    """

    def __init__(
        self,
        agent: BaseAgent,
        env: Environment,
        instance_generator: Callable,
        cfg: ExperimentConfig,
    ):
        self.agent = agent
        self.env = env
        self.instance_generator = instance_generator
        self.cfg = cfg
        self.tcfg: TrainConfig = cfg.train

        self.logger = Logger(
            log_dir=self.tcfg.log_dir,
            experiment_name=cfg.name,
            verbose=True,
        )
        from evaluator import Agent

        self.evaluator = Evaluator(
            agent=cast(Agent, agent),
            env=env,
            n_episodes=self.tcfg.n_eval_episodes,
            deterministic=self.tcfg.eval_deterministic,
            beam_width=self.tcfg.eval_beam_width,
        )

        self._timestep = 0
        self._iteration = 0
        self._best_objective = float("-inf")
        self._patience_counter = 0

        # Persist config next to checkpoints for full reproducibility
        Path(self.tcfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        save_config(cfg, f"{self.tcfg.checkpoint_dir}/{cfg.name}_config.json")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """
        Run the training loop to completion.

        Returns a summary dict.
        """
        self._print_header()
        start_time = time.time()
        stop_reason = "timestep_limit"

        while self._timestep < self.tcfg.total_timesteps:
            iter_start = time.time()
            self._iteration += 1

            size = self._curriculum_size()
            gen = self._make_generator(size)

            # ── Collect + update (algorithm-agnostic) ──────────────────
            collect_stats = self.agent.collect(self.env, gen)
            self._timestep += int(collect_stats.get("rollout/steps", 1))

            update_metrics = self.agent.update() or {}

            # ── Logging ────────────────────────────────────────────────
            iter_time = time.time() - iter_start
            all_metrics = {**collect_stats, **update_metrics, "iter_time_s": iter_time}

            if self._iteration % self.tcfg.log_interval == 0:
                self.logger.log_metrics(
                    all_metrics,
                    step=self._timestep,
                    print_keys=[
                        "rollout/mean_reward",
                        "train/policy_loss",
                        "train/td_loss",
                        "train/explained_var",
                        "train/epsilon",
                        "train/grad_norm",
                    ],
                )

            # ── Evaluation ─────────────────────────────────────────────
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

            # ── Periodic checkpoint ────────────────────────────────────
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

    def _make_generator(self, size: Optional[int]) -> Callable:
        if size is not None:
            return lambda: self.instance_generator(size=size)
        return self.instance_generator

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str) -> None:
        path = f"{self.tcfg.checkpoint_dir}/{self.cfg.name}_{tag}.pt"
        self.agent.save(path)

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def _print_header(self) -> None:
        print(
            f"\n{'=' * 64}\n"
            f"  Experiment : {self.cfg.name}\n"
            f"  Algorithm  : {self.cfg.algorithm.upper()}\n"
            f"  Network    : {self.cfg.network.network_type}\n"
            f"  Problem    : {self.env.problem.name}\n"
            f"  Obs shape  : {self.env.problem.observation_shape}\n"
            f"  Actions    : {self.env.problem.action_space_size}\n"
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
