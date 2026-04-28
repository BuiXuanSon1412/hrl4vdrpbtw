"""
core/trainer.py
---------------
Training-loop implementations.

MetaTrainer — multi-task MAML with curriculum expansion
POMOTrainer — Policy Optimization with Multiple Optima

Design principle:
  - Agent holds the policy network; each trainer computes loss via its own method
  - MetaTrainer coordinates multi-task learning with inner-loop adaptation and outer meta-updates
  - Curriculum expansion monitored via task entropy, integrated into train() method
  - POMOTrainer optimizes over multiple starting points per instance
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from globals import DEVICE
from core.agent import BaseAgent
from core.buffer import RolloutBuffer
from core.policy import BasePolicy
from core.utils import obs_to_tensor


# ---------------------------------------------------------------------------
# BaseTrainer (abstract interface)
# ---------------------------------------------------------------------------


class BaseTrainer(ABC):
    """Abstract base class for training strategies."""

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Run training loop and return summary."""
        ...

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        trainer_cfg: Dict[str, Any],
        agents: Dict[str, BaseAgent],
        env: Any,
        evaluator: Any,
        logger: Any,
    ) -> "BaseTrainer":
        """Factory method: instantiate trainer from config.

        Args:
            trainer_cfg: trainer config dict (cfg.trainer)
            agents: dict of agent instances (keyed by name)
            env: environment instance (has tasks list and reset interface)
            evaluator: evaluator instance
            logger: logger instance
        """
        ...


# ---------------------------------------------------------------------------
# MetaTrainer: Full MAML with curriculum
# ---------------------------------------------------------------------------


class MetaTrainer(BaseTrainer):
    """
    MAML trainer (second-order) with curriculum learning.

    Coordinates:
      1. Task sampling from TaskManager (respects curriculum)
      2. Support/query data collection for each task
      3. Inner-loop adaptation per task
      4. Outer-loop meta-gradient accumulation and update
      5. Curriculum expansion monitoring
    """

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        env: Any,
        trainer_cfg: Dict[str, Any],
        evaluator: Any,
        logger: Any,
    ):
        self.agents = agents
        self.agent = agents["meta_agent"]  # Primary trainable agent
        self.env = env
        self.active_tasks = {env.tasks[0]}  # Start with first (easiest) task
        self.evaluator = evaluator
        self.logger = logger

        # Extract config from trainer structure
        phase_cfg = trainer_cfg.get("phase", {})

        # Get meta_learning phase
        meta_phase = phase_cfg.get("meta_learning", {})
        meta_agent_cfg = meta_phase.get("agents", {}).get("meta_agent", {})
        sub_agent_cfg = meta_phase.get("agents", {}).get("sub_agent", {})
        curriculum_cfg = meta_phase.get("curriculum", {})
        meta_control_cfg = meta_phase.get("control", {})
        meta_logging_cfg = meta_control_cfg.get("logging", {})
        meta_early_stop = meta_control_cfg.get("early_stopping", {})

        # Only load curriculum and rollout settings; learning rates are already in agents
        self.mcfg = {
            "support_rollout_len": int(meta_agent_cfg.get("rollout_length", 256)),
            "query_rollout_len": int(sub_agent_cfg.get("rollout_length", 256)),
            "entropy_threshold": float(curriculum_cfg.get("entropy_threshold", 0.5)),
            "curriculum_check_interval": int(curriculum_cfg.get("check_interval", 1)),
            "discount_factor": float(meta_phase.get("discount_factor", 0.99)),
            "total_timesteps": int(meta_control_cfg.get("total_timesteps", 2000000)),
            "log_interval": int(meta_logging_cfg.get("log_interval", 5)),
            "eval_interval": int(meta_logging_cfg.get("eval_interval", 20)),
            "checkpoint_interval": int(
                meta_logging_cfg.get("checkpoint_interval", 100)
            ),
            "patience": int(meta_early_stop.get("patience", 150)),
            "min_delta": float(meta_early_stop.get("min_delta", 0.0001)),
        }

        # Get fine_tuning phase config
        fine_tune_phase = phase_cfg.get("fine_tuning", {})
        fine_control_cfg = fine_tune_phase.get("control", {})
        fine_logging_cfg = fine_control_cfg.get("logging", {})
        fine_early_stop = fine_control_cfg.get("early_stopping", {})

        self.fcfg = {
            "discount_factor": float(fine_tune_phase.get("discount_factor", 0.99)),
            "total_timesteps": int(fine_control_cfg.get("total_timesteps", 2000000)),
            "log_interval": int(fine_logging_cfg.get("log_interval", 5)),
            "eval_interval": int(fine_logging_cfg.get("eval_interval", 20)),
            "checkpoint_interval": int(
                fine_logging_cfg.get("checkpoint_interval", 100)
            ),
            "patience": int(fine_early_stop.get("patience", 150)),
            "min_delta": float(fine_early_stop.get("min_delta", 0.0001)),
        }

        # Build training config dict (for compatibility, use meta_cfg as default)
        self.tcfg = self.mcfg.copy()

        # Training state
        self._timestep = 0
        self._iteration = 0
        self._best_objective = float("-inf")
        self._patience_counter = 0
        self._curriculum_check_counter = 0

    @classmethod
    def from_config(
        cls,
        trainer_cfg: Dict[str, Any],
        agents: Dict[str, BaseAgent],
        env: Any,
        evaluator: Any,
        logger: Any,
    ) -> "MetaTrainer":
        if not env.tasks:
            raise ValueError("MetaTrainer.from_config requires env.tasks")

        if "meta_agent" not in agents:
            raise ValueError("MetaTrainer requires 'meta_agent' in agents dict")

        return cls(
            agents=agents,
            env=env,
            trainer_cfg=trainer_cfg,
            evaluator=evaluator,
            logger=logger,
        )

    def train(self) -> Dict[str, Any]:
        """Run full MAML pipeline: meta-training + fine-tuning."""
        start_time = time.time()

        # Phase 1: Meta-training
        meta_summary = self.meta_train()

        # Phase 2: Fine-tuning (if agent is available in fine_tuning phase)
        if "agent" in self.agents:
            fine_tune_summary = self.fine_tune()
        else:
            fine_tune_summary = {}

        # Combine summaries
        summary = {
            "stop_reason": meta_summary.get("stop_reason", "completed"),
            "total_iterations": meta_summary.get("total_iterations", 0)
            + fine_tune_summary.get("total_iterations", 0),
            "total_timesteps": meta_summary.get("total_timesteps", 0)
            + fine_tune_summary.get("total_timesteps", 0),
            "best_objective": max(
                meta_summary.get("best_objective", float("-inf")),
                fine_tune_summary.get("best_objective", float("-inf")),
            ),
            "training_time_s": round(time.time() - start_time, 1),
        }
        self.logger.log_event(
            "training_complete", summary.get("total_timesteps", 0), **summary
        )
        self.logger.close()
        self._print_footer(summary)
        return summary

    def meta_train(self) -> Dict[str, Any]:
        """Run meta-training loop and return summary."""
        self._print_header()
        start_time = time.time()
        stop_reason = "timestep_limit"

        while self._timestep < self.tcfg["total_timesteps"]:
            iter_start = time.time()
            self._iteration += 1

            # Compute meta-loss across active tasks
            meta_loss, task_losses, task_metrics, total_steps = (
                self._compute_meta_loss()
            )

            # Update meta-policy
            self.agent.update(meta_loss)

            # Curriculum expansion: find task with highest entropy
            max_entropy = None
            max_entropy_task = None
            for task_id, metrics_dict in task_metrics.items():
                entropy = metrics_dict.get("entropy", 0)
                if max_entropy is None or entropy > max_entropy:
                    max_entropy = entropy
                    max_entropy_task = task_id

            if max_entropy is not None:
                self._curriculum_check_counter += 1
                if (
                    self._curriculum_check_counter
                    >= self.mcfg["curriculum_check_interval"]
                ):
                    self._curriculum_check_counter = 0
                    if max_entropy < self.mcfg["entropy_threshold"]:
                        # Add next task from sorted task list
                        if len(self.active_tasks) < len(self.env.tasks):
                            next_task = self.env.tasks[len(self.active_tasks)]
                            self.active_tasks.add(next_task)
                            print(
                                f"[MetaTrainer] Curriculum expanded to {len(self.active_tasks)} tasks"
                            )

            self._timestep += total_steps

            # Build metrics dict
            metrics = {
                "train/meta_loss": float(np.mean(task_losses)) if task_losses else 0.0,
                "train/update_count": float(self._iteration),
                "num_active_tasks": len(self.active_tasks),
            }
            for task_id, task_metric in task_metrics.items():
                for key, val in task_metric.items():
                    if (
                        key != "entropy"
                    ):  # entropy is just for curriculum, don't log separately
                        metrics[f"train/task_{task_id}_{key}"] = val

            metrics["iter_time_s"] = time.time() - iter_start

            if self._iteration % self.tcfg["log_interval"] == 0:
                self.logger.log_metrics(
                    metrics,
                    step=self._timestep,
                    print_keys=[
                        "train/meta_loss",
                        "train/update_count",
                        "num_active_tasks",
                    ],
                )

            if self._iteration % self.tcfg["eval_interval"] == 0:
                # Get median task for evaluation
                median_idx = len(self.env.tasks) // 2
                eval_task_id = self.env.tasks[median_idx]
                eval_stats = self.evaluator.evaluate(eval_task_id)
                self.logger.log_metrics(eval_stats, step=self._timestep, prefix="eval")

                mean_obj = eval_stats.get("mean_objective", float("-inf"))
                if mean_obj > self._best_objective + self.tcfg["min_delta"]:
                    self._best_objective = mean_obj
                    self._patience_counter = 0
                    self.logger.save_checkpoint(
                        "meta_best",
                        {
                            "network_state": self.agent.policy.state_dict(),
                            "iteration": self._iteration,
                        },
                    )
                    self.logger.log_event(
                        "best_checkpoint", self._timestep, objective=f"{mean_obj:.4f}"
                    )
                else:
                    self._patience_counter += 1

                if self._patience_counter >= self.tcfg["patience"]:
                    stop_reason = "early_stopping"
                    self.logger.log_event(
                        "early_stop", self._timestep, patience=self.tcfg["patience"]
                    )
                    break

        summary = {
            "stop_reason": stop_reason,
            "total_iterations": self._iteration,
            "total_timesteps": self._timestep,
            "best_objective": self._best_objective,
            "training_time_s": round(time.time() - start_time, 1),
            "final_num_active_tasks": len(self.active_tasks),
        }
        self.logger.save_checkpoint(
            "meta_final",
            {
                "network_state": self.agent.policy.state_dict(),
                "iteration": self._iteration,
            },
        )
        self.logger.log_event("meta_training_complete", self._timestep, **summary)
        return summary

    def fine_tune(self) -> Dict[str, Any]:
        """Fine-tune policy on each task independently after meta-training.

        Uses the agent from fine_tuning phase to adapt the meta-learned policy
        to each task independently.
        """
        self._print_header_tune()
        start_time = time.time()
        agent = self.agents["agent"]

        total_iterations = 0
        total_timesteps = 0
        best_objective = float("-inf")

        for task_id in self.env.tasks:
            self._print_header_task(task_id)
            task_timestep = 0
            task_iteration = 0
            task_best_objective = float("-inf")
            task_patience_counter = 0
            stop_reason = "timestep_limit"

            while task_timestep < self.tcfg["total_timesteps"]:
                iter_start = time.time()
                task_iteration += 1

                # Collect and update on this task
                self.env.reset(task_id)
                buf = agent.collect(self.env)
                loss = self._compute_ppo_loss(
                    agent.policy, buf, gamma=self.fcfg["discount_factor"]
                )
                agent.update(loss)
                task_timestep += buf._ptr

                metrics = {
                    "tune/loss": float(loss.item()),
                    "iter_time_s": time.time() - iter_start,
                }

                if task_iteration % self.tcfg["log_interval"] == 0:
                    self.logger.log_metrics(
                        metrics,
                        step=task_timestep,
                        print_keys=["tune/loss"],
                    )

                if task_iteration % self.tcfg["eval_interval"] == 0:
                    eval_stats = self.evaluator.evaluate(task_id)
                    self.logger.log_metrics(
                        eval_stats, step=task_timestep, prefix="tune_eval"
                    )

                    mean_obj = eval_stats.get("mean_objective", float("-inf"))
                    if mean_obj > task_best_objective + self.tcfg["min_delta"]:
                        task_best_objective = mean_obj
                        task_patience_counter = 0
                        self.logger.save_checkpoint(
                            f"tune_best_{task_id}",
                            {
                                "network_state": self.agent.policy.state_dict(),
                                "iteration": self._iteration,
                            },
                        )
                        self.logger.log_event(
                            "tune_best_checkpoint",
                            task_timestep,
                            task=task_id,
                            objective=f"{mean_obj:.4f}",
                        )
                    else:
                        task_patience_counter += 1

                    if task_patience_counter >= self.tcfg["patience"]:
                        stop_reason = "early_stopping"
                        self.logger.log_event(
                            "tune_early_stop",
                            task_timestep,
                            task=task_id,
                            patience=self.tcfg["patience"],
                        )
                        break

            total_timesteps += task_timestep
            total_iterations += task_iteration
            best_objective = max(best_objective, task_best_objective)
            self.logger.save_checkpoint(
                f"tune_final_{task_id}",
                {
                    "network_state": self.agent.policy.state_dict(),
                    "iteration": self._iteration,
                },
            )

        summary = {
            "stop_reason": "completed",
            "total_iterations": total_iterations,
            "total_timesteps": total_timesteps,
            "best_objective": best_objective,
            "training_time_s": round(time.time() - start_time, 1),
        }
        self.logger.log_event("fine_tuning_complete", total_timesteps, **summary)
        return summary

    def _print_header_tune(self) -> None:
        print(
            f"\n{'=' * 64}\n"
            f"  Fine-Tuning Phase\n"
            f"  Tasks: {len(self.env.tasks)}\n"
            f"  Budget/task: {self.tcfg['total_timesteps']:,} steps\n"
            f"{'=' * 64}"
        )

    def _compute_ppo_loss(
        self,
        policy: "BasePolicy",
        buffer: RolloutBuffer,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        """Compute PPO loss from buffer of transitions with discount factor."""
        n = buffer._ptr
        if n == 0:
            return torch.tensor(0.0, device=DEVICE, requires_grad=True)

        # Compute returns (discounted cumulative)
        returns = []
        cumsum = 0.0
        for t in reversed(range(n)):
            tr = buffer._data[t]
            cumsum = tr.reward + gamma * cumsum * (1.0 - float(tr.done))
            returns.insert(0, cumsum)
        returns = np.array(returns)

        # Compute advantages
        baseline = float(np.mean(returns))
        advantages = returns - baseline

        total_loss = None
        for i, tr in enumerate(buffer._data[:n]):
            obs_t = obs_to_tensor(tr.obs, DEVICE)
            act_t = torch.tensor([tr.action], dtype=torch.long, device=DEVICE)
            mask_t = torch.tensor(
                tr.action_mask, dtype=torch.bool, device=DEVICE
            ).unsqueeze(0)
            adv = torch.tensor(advantages[i], dtype=torch.float32, device=DEVICE)
            ret = torch.tensor(returns[i], dtype=torch.float32, device=DEVICE)
            # Handle both tensor and float log_prob
            if isinstance(tr.log_prob, torch.Tensor):
                old_log_prob = tr.log_prob.to(DEVICE)
            else:
                old_log_prob = torch.tensor(
                    tr.log_prob, dtype=torch.float32, device=DEVICE
                )

            # Evaluate current policy
            log_prob, value, entropy = policy.evaluate_actions(obs_t, act_t, mask_t)

            # PPO clipped objective
            ratio = torch.exp(log_prob - old_log_prob)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2)

            # Value loss
            value_loss = 0.5 * torch.nn.functional.mse_loss(value, ret)

            # Entropy bonus
            entropy_loss = -entropy

            loss_i = (
                policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            ) / n
            total_loss = loss_i if total_loss is None else total_loss + loss_i

        return (
            total_loss
            if total_loss is not None
            else torch.tensor(0.0, device=DEVICE, requires_grad=True)
        )

    def _compute_meta_loss(
        self,
    ) -> Tuple[torch.Tensor, List[float], Dict[Any, Dict[str, float]], int]:
        """Compute second-order meta-loss across all active tasks.

        For each active task:
          1. Collect support buffer with meta_agent
          2. Adapt sub_agent on support buffer
          3. Collect query buffer with adapted sub_agent
          4. Compute query loss and entropy on adapted policy

        Returns:
            (meta_loss, task_losses, task_metrics, total_steps)
        """
        total_steps = 0
        task_losses: List[float] = []
        task_metrics: Dict[Any, Dict[str, float]] = {}

        n_active_tasks = len(self.active_tasks)
        meta_loss = None

        # Process each active task
        for task_id in self.active_tasks:
            # Support: collect with meta_agent
            self.env.reset(task_id)
            buf = self.agent.collect(self.env)
            support_loss = self._compute_ppo_loss(
                self.agent.policy, buf, gamma=self.mcfg["discount_factor"]
            )
            total_steps += buf._ptr

            # Adapt: clone sub_agent with meta_agent's policy, then update on support buffer
            sub_agent = self.agents["sub_agent"]
            sub_agent.clone_policy(self.agent)
            sub_agent.update(support_loss)

            # Query: collect with adapted sub_agent
            self.env.reset(task_id)
            buf = sub_agent.collect(self.env)
            query_loss = self._compute_ppo_loss(
                sub_agent.policy, buf, gamma=self.mcfg["discount_factor"]
            )
            total_steps += buf._ptr

            task_losses.append(query_loss.item())
            task_metrics[task_id] = {
                "support_loss": support_loss.item(),
                "query_loss": query_loss.item(),
                "improvement": (support_loss.item() - query_loss.item()),
                "entropy": sub_agent.policy.compute_entropy(buf),
            }

            # Accumulate meta-loss
            if meta_loss is None:
                meta_loss = query_loss / n_active_tasks
            else:
                meta_loss = meta_loss + query_loss / n_active_tasks

        return (
            meta_loss
            if meta_loss is not None
            else torch.tensor(0.0, device=DEVICE, requires_grad=True),
            task_losses,
            task_metrics,
            total_steps,
        )

    def _print_header(self) -> None:
        active_task_ids = sorted(self.active_tasks)
        total_tasks = len(self.env.tasks)
        print(
            f"\n{'=' * 64}\n"
            f"  Algorithm  : MAML (Meta-Learning)\n"
            f"  Tasks      : {active_task_ids} (of {total_tasks} total)\n"
            f"  Budget     : {self.tcfg['total_timesteps']:,} steps\n"
            f"  Device     : {DEVICE}\n"
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

    def _print_header_task(self, task_id: str) -> None:
        print(f"\n{'-' * 64}\n  Task: {task_id}\n{'-' * 64}")


# ---------------------------------------------------------------------------
# POMOTrainer: Policy Optimization with Multiple Optima
# ---------------------------------------------------------------------------


class POMOTrainer(BaseTrainer):
    """
    POMO trainer: policy optimization with multiple optima.

    Collects complete episodes from multiple candidate starting points for each instance,
    then uses REINFORCEEstimator to compute policy gradients with a baseline.
    This encourages the policy to find good solutions regardless of starting point.

    Collection strategy (POMO):
      - For each instance: get candidate starting states via env.get_candidate_starts()
      - Roll out complete episode from each candidate
      - Accumulate all transitions into a RolloutBuffer

    Loss computation (delegated to REINFORCEEstimator):
      - Compute returns and advantage = return - baseline
      - Loss = -log_prob * advantage
    """

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        env: Any,
        trainer_cfg: Dict[str, Any],
        evaluator: Any,
        logger: Any,
    ):
        self.agents = agents
        self.env = env
        self.evaluator = evaluator
        self.logger = logger

        # Setup task iteration from env
        if not env.tasks:
            raise ValueError("POMOTrainer requires env.tasks")

        # Extract config from hierarchical structure
        phase_cfg = trainer_cfg.get("phase", {})
        training_phase = phase_cfg.get("training", {})
        control_cfg = training_phase.get("control", {})

        logging_cfg = control_cfg.get("logging", {})

        self.tcfg = {
            "epochs": int(control_cfg.get("epochs", 100)),
            "batches_per_epoch": int(control_cfg.get("batches_per_epoch", 10000)),
            "instances_per_batch": int(control_cfg.get("instances_per_batch", 64)),
            "eval_instances": int(control_cfg.get("eval_instances", 10000)),
            "log_interval": int(logging_cfg.get("log_interval", 50)),
            "checkpoint_interval": int(logging_cfg.get("checkpoint_interval", 10)),
        }

        self.pcfg = {
            "discount_factor": float(training_phase.get("discount_factor", 0.99)),
        }

        self._timestep = 0
        self._iteration = 0

    @classmethod
    def from_config(
        cls,
        trainer_cfg: Dict[str, Any],
        agents: Dict[str, BaseAgent],
        env: Any,
        evaluator: Any,
        logger: Any,
    ) -> "POMOTrainer":
        return cls(
            agents=agents,
            env=env,
            trainer_cfg=trainer_cfg,
            evaluator=evaluator,
            logger=logger,
        )

    def train(self) -> Dict[str, Any]:
        """Run POMO training loop with epoch-based batch training per task."""
        start_time = time.time()
        self._print_header()
        agent = self.agents["agent"]

        epochs = self.tcfg["epochs"]
        batches_per_epoch = self.tcfg["batches_per_epoch"]
        instances_per_batch = self.tcfg["instances_per_batch"]
        eval_instances = self.tcfg["eval_instances"]
        log_interval = self.tcfg["log_interval"]
        checkpoint_interval = self.tcfg["checkpoint_interval"]

        for task_id in self.env.tasks:
            self._print_header_task(task_id)
            self.env.retask(task_id)  # Set up environment for this task

            for epoch in range(epochs):
                epoch_start = time.time()
                epoch_losses = []
                epoch_returns = []
                grad_norm = -1.0

                for batch_idx in range(batches_per_epoch):
                    batch_loss = None
                    batch_returns = []
                    # Accumulate loss over instances in batch
                    for _ in range(instances_per_batch):
                        self.env.retask(task_id)

                        loss, episode_returns, _ = self._compute_pomo_loss(agent)
                        batch_returns.extend(episode_returns)

                        if batch_loss is None:
                            batch_loss = loss
                        else:
                            batch_loss = batch_loss + loss

                    # Update after each batch
                    if batch_loss is not None:
                        batch_avg_loss = batch_loss / instances_per_batch
                        grad_norm = agent.update(batch_avg_loss)
                    else:
                        batch_avg_loss = None

                    epoch_losses.append(
                        float(batch_avg_loss.item())
                        if batch_avg_loss is not None
                        else 0.0
                    )
                    epoch_returns.extend(batch_returns)

                    # Logging
                    if (batch_idx + 1) % log_interval == 0:
                        avg_loss = float(
                            np.mean(epoch_losses[-(log_interval):])
                            if epoch_losses
                            else 0.0
                        )
                        avg_return = (
                            float(np.mean(batch_returns)) if batch_returns else 0.0
                        )
                        metrics = {
                            "train/batch_avg_loss": avg_loss,
                            "train/batch_avg_return": avg_return,
                            "train/grad_norm": grad_norm,
                        }
                        global_step = (
                            self._iteration + epoch * batches_per_epoch + batch_idx + 1
                        )
                        self.logger.log_metrics(
                            metrics,
                            step=global_step,
                            print_keys=[
                                "train/batch_loss",
                                "train/batch_avg_return",
                                "train/grad_norm",
                            ],
                        )

                # Evaluate after each epoch
                eval_loss = None
                eval_returns = []
                for _ in range(eval_instances):
                    self.env.retask(task_id)
                    self.env.reset()

                    loss, episode_returns, _ = self._compute_pomo_loss(agent)
                    if eval_loss is None:
                        eval_loss = loss
                    else:
                        eval_loss = eval_loss + loss
                    eval_returns.extend(episode_returns)

                eval_metrics = {
                    "eval/epoch_loss": float(eval_loss.item())
                    if eval_loss is not None
                    else 0.0,
                    "eval/avg_return": float(np.mean(eval_returns))
                    if eval_returns
                    else 0.0,
                    "eval/std_return": float(np.std(eval_returns))
                    if eval_returns
                    else 0.0,
                }
                self.logger.log_metrics(
                    eval_metrics,
                    step=self._iteration + (epoch + 1) * batches_per_epoch,
                    print_keys=["eval/epoch_loss", "eval/avg_return"],
                )

                # Checkpoint
                if (epoch + 1) % checkpoint_interval == 0:
                    agent = self.agents["agent"]
                    self.logger.save_checkpoint(
                        f"{task_id}_epoch_{epoch + 1}",
                        {
                            "network_state": agent.policy.state_dict(),
                            "iteration": self._iteration,
                        },
                    )

                epoch_time = time.time() - epoch_start
                print(
                    f"[Task {task_id}] Epoch {epoch + 1}/{epochs} - Time: {epoch_time:.1f}s"
                )

            agent = self.agents["agent"]
            self.logger.save_checkpoint(
                f"{task_id}_final",
                {
                    "network_state": agent.policy.state_dict(),
                    "iteration": self._iteration,
                },
            )
            self._iteration += epochs * batches_per_epoch

        summary = {
            "stop_reason": "completed",
            "total_iterations": self._iteration,
            "total_timesteps": self._timestep,
            "training_time_s": round(time.time() - start_time, 1),
        }
        self.logger.log_event("training_complete", self._timestep, **summary)
        self.logger.close()
        self._print_footer(summary)
        return summary

    def _compute_pomo_loss(
        self, agent: BaseAgent
    ) -> Tuple[torch.Tensor, List[float], int]:
        """Compute POMO loss for a task.

        For each feasible starting action:
          - Collect complete episode, accumulating log probabilities
          - Get final solution objective value
          - Compute advantage = objective - baseline
          - Compute loss = -episode_log_prob * advantage

        Returns:
            (loss, episode_returns, n_episodes)
        """
        obs, info = self.env.reset()

        # Get log probabilities for all feasible starting actions
        action_log_probs = agent.action_to_log_prob(obs, info["action_mask"])

        if len(action_log_probs) == 0:
            # No feasible starting actions
            return torch.tensor(0.0, device=DEVICE, requires_grad=True), [], 0

        episode_returns = []
        episode_log_probs = []
        n_episodes = 0

        for starting_action in action_log_probs.keys():
            obs, info = self.env.reset()

            # First step with starting action
            episode_log_prob = action_log_probs[int(starting_action)]
            next_obs, _, terminated, truncated, next_info = self.env.step(
                int(starting_action)
            )

            if not terminated and not truncated and next_info["action_mask"].any():
                obs = next_obs
                mask = next_info["action_mask"]

                # Collect trajectory, accumulating log probabilities
                while True:
                    action, lp, _ = agent.select_action(obs, mask, training=True)
                    episode_log_prob = episode_log_prob + lp
                    next_obs, _, terminated, truncated, info = self.env.step(action)

                    if terminated or truncated or not info["action_mask"].any():
                        break

                    obs = next_obs
                    mask = info["action_mask"]

            # Get final solution quality (negated and normalized)
            n_episodes += 1
            episode_return = self.env.compute_return()

            episode_returns.append(episode_return)
            episode_log_probs.append(episode_log_prob)

        # Compute baseline and per-episode advantages
        baseline = float(np.mean(episode_returns)) if episode_returns else 0.0
        advantages = np.array(episode_returns) - baseline

        # Compute POMO loss: -sum(log_probs_in_episode) * advantage_episode
        episode_log_probs_tensor = torch.stack(episode_log_probs)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        total_loss = (
            -torch.sum(episode_log_probs_tensor * advantages_tensor) / n_episodes
        )

        return total_loss, episode_returns, n_episodes

    def _print_header(self) -> None:
        print(
            f"\n{'=' * 64}\n"
            f"  Algorithm  : POMO (Multiple Optima)\n"
            f"  Tasks      : {len(self.env.tasks)}\n"
            f"  Device     : {DEVICE}\n"
            f"{'=' * 64}"
        )

    def _print_header_task(self, task_id: str) -> None:
        print(f"\n{'-' * 64}\n  Task: {task_id}\n{'-' * 64}")

    def _print_footer(self, summary: Dict) -> None:
        print(
            f"\n{'=' * 64}\n"
            f"  Done ({summary['stop_reason']})\n"
            f"  Iterations : {summary['total_iterations']:,}\n"
            f"  Timesteps  : {summary['total_timesteps']:,}\n"
            f"  Time       : {summary['training_time_s']:.1f}s\n"
            f"{'=' * 64}\n"
        )
