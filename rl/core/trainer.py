"""
core/trainer.py
---------------
Training-loop implementations.

MetaTrainer — multi-task MAML with curriculum expansion
POMOTrainer — Policy Optimization with Multiple Optima

Design principle:
  - Agent holds the policy network and estimator
  - Estimator computes gradients (PPO, A2C, etc.) — algorithm lives here
  - MetaTrainer coordinates multi-task learning with inner-loop adaptation and outer meta-updates
  - POMOTrainer optimizes over multiple starting points per instance
  - TaskManager, CurriculumScheduler used by MetaTrainer only
  - FineTuner adapts trained meta-policy to individual tasks
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.agent import BaseAgent
from core.buffer import RolloutBuffer
from core.task import SimpleTask, TaskManager


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
        cfg: Dict[str, Any],
        agent: BaseAgent,
        env: Any,
        tasks: Dict[str, Tuple[Any, Callable]],
        evaluator: Any,
        logger: Any,
    ) -> "BaseTrainer":
        """Factory method: instantiate trainer from config.

        Args:
            cfg: config dict
            agent: agent instance
            env: environment template (used for step/reset interface)
            tasks: dict mapping task_id -> (problem, generator)
            evaluator: evaluator instance
            logger: logger instance
        """
        ...


# ---------------------------------------------------------------------------
# Inner-loop adaptation (inlined from InnerUpdater)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Curriculum scheduler
# ---------------------------------------------------------------------------


class CurriculumScheduler:
    """
    Monitor task entropy and expand curriculum when policy is ready.

    Curriculum expands when entropy on hardest task falls below threshold,
    indicating the policy has mastered the current difficulty level.
    """

    def __init__(
        self,
        entropy_threshold: float = 0.5,
        expand_interval: int = 1,
        check_metric: str = "entropy",
    ):
        self.entropy_threshold = entropy_threshold
        self.expand_interval = expand_interval
        self.check_metric = check_metric
        self._check_counter = 0
        self.recent_entropies: Dict[Any, list] = {}
        self.recent_losses: Dict[Any, list] = {}

    def should_expand(
        self,
        task_manager: Any,
        task_id: Any,
        metric_value: float,
    ) -> bool:
        """Check if curriculum should expand based on task metric."""
        if self.check_metric not in ["entropy", "loss"]:
            return False

        if task_id not in self.recent_entropies:
            self.recent_entropies[task_id] = []
        if task_id not in self.recent_losses:
            self.recent_losses[task_id] = []

        if self.check_metric == "entropy":
            self.recent_entropies[task_id].append(metric_value)
            if len(self.recent_entropies[task_id]) > 5:
                self.recent_entropies[task_id].pop(0)
            avg_entropy = np.mean(self.recent_entropies[task_id])
            return bool(avg_entropy < self.entropy_threshold)

        return False

    def update(
        self,
        task_manager: Any,
        task_entropies: Dict[Any, float],
    ) -> bool:
        """
        Update curriculum and return True if expansion occurred.

        Args:
            task_manager: TaskManager instance
            task_entropies: dict mapping task_id -> entropy value

        Returns:
            True if curriculum was expanded, False otherwise
        """
        self._check_counter += 1
        if self._check_counter < self.expand_interval:
            return False

        self._check_counter = 0

        active_tasks = task_manager.get_active_task_ids()
        if not active_tasks:
            return False

        hardest_task_id = active_tasks[-1]
        if hardest_task_id not in task_entropies:
            return False

        hardest_entropy = task_entropies[hardest_task_id]
        if self.should_expand(task_manager, hardest_task_id, hardest_entropy):
            task_manager.activate_next()
            return True

        return False


# ---------------------------------------------------------------------------
# Fine-tuner: adapt meta-policy to specific tasks
# ---------------------------------------------------------------------------


class FineTuner:
    """
    Fine-tune trained meta-policy on individual tasks.

    After meta-learning, produces task-specific sub-policies by adapting
    the meta-policy to each task.
    """

    def __init__(self, agent: BaseAgent, task_manager: Any, cfg: Dict[str, Any]):
        self.agent = agent
        self.task_manager = task_manager
        self.cfg = cfg
        self.task_agents: Dict[Any, Any] = {}

    def initialize(self) -> None:
        """Initialize fine-tuning by cloning agent for each active task."""
        for task_id in self.task_manager.get_active_task_ids():
            self.task_agents[task_id] = self.agent.clone()

    def finetune_task(
        self,
        task_id: Any,
        task_problem: Any,
        task_gen: Callable,
        optimizer: optim.Optimizer,
        num_steps: int,
    ) -> int:
        """
        Fine-tune the meta-policy on a single task.

        Args:
            task_id: task identifier
            task_problem: environment for the task
            task_gen: generator for task instances
            optimizer: optimizer for fine-tuning
            num_steps: number of gradient steps

        Returns:
            number of environment steps collected
        """
        task_agent = self.task_agents[task_id]
        total_steps = 0

        for _ in range(num_steps):
            buffer = self._collect_rollout(task_problem, task_agent, task_gen)
            total_steps += buffer._ptr

            if task_agent.estimator is None:
                continue

            loss = task_agent.estimator.compute_loss(task_agent.policy, buffer)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_steps

    def _collect_rollout(
        self, env: Any, agent: Any, generator: Callable
    ) -> RolloutBuffer:
        """Collect rollout for fine-tuning: generic strategy.

        Collects transitions until buffer is full (256 default).
        """
        buffer = RolloutBuffer(capacity=256)

        def _fresh_episode() -> Tuple[Any, np.ndarray]:
            for _ in range(100):
                raw = generator()
                obs, info = env.reset(raw)
                mask = info["action_mask"]
                if mask.any():
                    return obs, mask
            raise RuntimeError(
                "FineTuner._collect_rollout: 100 consecutive dead-start instances."
            )

        obs, action_mask = _fresh_episode()

        while buffer._ptr < buffer.capacity and not buffer.is_full:
            action, lp, val = agent.select_action(obs, action_mask, training=True)

            if not action_mask[action]:
                feasible = np.where(action_mask)[0]
                if len(feasible) > 0:
                    action = int(np.random.choice(feasible))
                    lp = 0.0
                else:
                    obs, action_mask = _fresh_episode()
                    continue

            next_obs, reward, terminated, truncated, info = env.step(action)

            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=(terminated or truncated),
                log_prob=lp,
                value=val,
                action_mask=action_mask,
            )

            obs = next_obs
            action_mask = info["action_mask"]

            if terminated or truncated or not action_mask.any():
                obs, action_mask = _fresh_episode()

        return buffer

    def get_task_agent(self, task_id: Any) -> Any:
        """Retrieve fine-tuned agent for a task."""
        return self.task_agents.get(task_id)

    def get_all_agents(self) -> Dict[Any, Any]:
        """Get all fine-tuned task agents."""
        return self.task_agents

    def save_task_policies(self, save_dir: str) -> None:
        """Save all task-specific policies."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for task_id, agent in self.task_agents.items():
            task_save_path = save_path / f"task_{task_id}_best.pt"
            torch.save(
                {"network_state": agent.policy.state_dict()},
                task_save_path,
            )

    def load_task_policies(self, load_dir: str) -> None:
        """Load task-specific policies from disk."""
        load_path = Path(load_dir)

        for task_id, agent in self.task_agents.items():
            task_load_path = load_path / f"task_{task_id}_best.pt"
            if task_load_path.exists():
                checkpoint = torch.load(task_load_path, map_location="cpu")
                agent.policy.load_state_dict(checkpoint["network_state"])


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
        agent: BaseAgent,
        task_manager: TaskManager,
        eval_problem: Any,
        eval_gen: Callable,
        cfg: Dict[str, Any],
        evaluator: Any,
        logger: Any,
    ):
        self.agent = agent
        self.task_manager = task_manager
        self.cfg = cfg
        self.evaluator = evaluator
        self.logger = logger

        # Handle experiment name (can be string or dict)
        experiment = cfg.get("experiment", {})
        if isinstance(experiment, str):
            self.experiment_name = cfg.get("name") or experiment or "experiment"
        else:
            self.experiment_name = cfg.get("name") or experiment.get("name", "experiment")
        self.device = cfg.get("device", "cpu")

        # Extract config from trainer structure
        trainer_cfg = cfg.get("trainer", {})
        meta_cfg = trainer_cfg.get("meta_learning", {})
        training_cfg = cfg.get("training", cfg.get("train", {}))

        # Build meta training config from trainer.meta_learning structure
        meta_agent_cfg = meta_cfg.get("meta_agent", {})
        sub_agent_cfg = meta_cfg.get("sub_agent", {})
        curriculum_cfg = meta_cfg.get("curriculum", {})

        self.mcfg = {
            "meta_lr": float(meta_agent_cfg.get("learning_rate", 0.0003)),
            "max_grad_norm": float(meta_cfg.get("max_grad_norm", 0.5)),
            "inner_lr": float(sub_agent_cfg.get("learning_rate", 0.01)),
            "support_rollout_len": int(meta_agent_cfg.get("rollout_length", 256)),
            "query_rollout_len": int(sub_agent_cfg.get("rollout_length", 256)),
            "entropy_threshold": float(curriculum_cfg.get("entropy_threshold", 0.5)),
            "curriculum_check_interval": int(curriculum_cfg.get("check_interval", 1)),
        }

        # Build training config dict from hierarchical structure
        train_logging = training_cfg.get("logging", {})
        early_stop = training_cfg.get("early_stopping", {})

        self.tcfg = {
            "total_timesteps": training_cfg.get(
                "total_timesteps", cfg.get("train", {}).get("total_timesteps", 2000000)
            ),
            "log_interval": train_logging.get(
                "log_interval", cfg.get("train", {}).get("log_interval", 5)
            ),
            "eval_interval": train_logging.get(
                "eval_interval", cfg.get("train", {}).get("eval_interval", 20)
            ),
            "checkpoint_interval": train_logging.get(
                "checkpoint_interval",
                cfg.get("train", {}).get("checkpoint_interval", 100),
            ),
            "checkpoint_dir": train_logging.get(
                "checkpoint_dir",
                cfg.get("train", {}).get("checkpoint_dir", "checkpoints"),
            ),
            "log_dir": train_logging.get(
                "log_dir", cfg.get("train", {}).get("log_dir", "logs")
            ),
            "patience": early_stop.get(
                "patience", cfg.get("train", {}).get("patience", 150)
            ),
            "min_delta": early_stop.get(
                "min_delta", cfg.get("train", {}).get("min_delta", 0.0001)
            ),
        }

        # Evaluation task
        self.eval_problem = eval_problem
        self.eval_gen = eval_gen

        # Meta-optimizer
        self._meta_optimizer = optim.Adam(
            agent.policy.parameters(), lr=self.mcfg["meta_lr"]
        )

        # Curriculum scheduler
        self.curriculum_scheduler = CurriculumScheduler(
            entropy_threshold=self.mcfg["entropy_threshold"],
            expand_interval=self.mcfg["curriculum_check_interval"],
            check_metric="entropy",
        )

        # Training state
        self._timestep = 0
        self._iteration = 0
        self._best_objective = float("-inf")
        self._patience_counter = 0

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        agent: BaseAgent,
        env: Any,
        tasks: Dict[str, Tuple[Any, Callable]],
        evaluator: Any,
        logger: Any,
    ) -> "MetaTrainer":
        if not tasks:
            raise ValueError("MetaTrainer.from_config requires tasks dict")

        task_list = []
        for task_id in list(tasks.keys()):
            problem, gen = tasks[task_id]
            task = SimpleTask(task_id=task_id, problem=problem, generator=gen)
            task_list.append(task)
        task_manager = TaskManager(task_list)

        # Use median task size as eval anchor
        eval_task_ids = list(tasks.keys())
        eval_task_id = eval_task_ids[len(eval_task_ids) // 2]
        eval_problem, eval_gen = tasks[eval_task_id]

        return cls(
            agent=agent,
            task_manager=task_manager,
            eval_problem=eval_problem,
            eval_gen=eval_gen,
            cfg=cfg,
            evaluator=evaluator,
            logger=logger,
        )

    def _collect(self, env: Any, agent: Any, generator: Callable) -> RolloutBuffer:
        """Collect complete episodes into buffer (same strategy as Trainer).

        Collects full episodes from environment up to buffer capacity.
        Uses support_rollout_len from mcfg as default capacity.

        Args:
            env: environment with reset/step interface
            agent: agent for select_action
            generator: callable that generates raw problem instances

        Returns:
            RolloutBuffer with collected transitions
        """
        capacity = self.mcfg.get("support_rollout_len", 256)
        buffer = RolloutBuffer(capacity=capacity)
        rollout_len = buffer.capacity

        def _fresh_episode() -> Tuple[Any, np.ndarray]:
            for _ in range(100):
                raw = generator()
                obs, info = env.reset(raw)
                mask = info["action_mask"]
                if mask.any():
                    return obs, mask
            raise RuntimeError(
                "MetaTrainer._collect: 100 consecutive dead-start instances."
            )

        obs, action_mask = _fresh_episode()

        while buffer._ptr < rollout_len and not buffer.is_full:
            action, lp, val = agent.select_action(obs, action_mask, training=True)

            if not action_mask[action]:
                feasible = np.where(action_mask)[0]
                if len(feasible) > 0:
                    action = int(np.random.choice(feasible))
                    lp = 0.0
                else:
                    obs, action_mask = _fresh_episode()
                    continue

            next_obs, reward, terminated, truncated, info = env.step(action)

            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=(terminated or truncated),
                log_prob=lp,
                value=val,
                action_mask=action_mask,
            )

            obs = next_obs
            action_mask = info["action_mask"]

            if terminated or truncated or not action_mask.any():
                obs, action_mask = _fresh_episode()

        return buffer

    def train(self) -> Dict[str, Any]:
        """Run meta-training loop and return summary."""
        self._print_header()
        start_time = time.time()
        stop_reason = "timestep_limit"

        while self._timestep < self.tcfg["total_timesteps"]:
            iter_start = time.time()
            self._iteration += 1
            print("iteration ", self._iteration)
            metrics = self._update_meta_policy()
            self._timestep += int(metrics.pop("_steps", 0))

            metrics["iter_time_s"] = time.time() - iter_start
            metrics["num_active_tasks"] = self.task_manager.num_active_tasks()

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
                eval_stats = self.evaluator.evaluate(self.eval_gen)
                self.logger.log_metrics(eval_stats, step=self._timestep, prefix="eval")

                mean_obj = eval_stats.get("mean_objective", float("-inf"))
                if mean_obj > self._best_objective + self.tcfg["min_delta"]:
                    self._best_objective = mean_obj
                    self._patience_counter = 0
                    self._save_checkpoint("best")
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
            "final_num_active_tasks": self.task_manager.num_active_tasks(),
        }
        self._save_checkpoint("final")
        self.logger.log_event("training_complete", self._timestep, **summary)
        self.logger.close()
        self._print_footer(summary)
        return summary

    def _update_sub_policy(
        self,
        network: nn.Module,
        loss_fn: Callable[[nn.Module, Any], torch.Tensor],
        support_data: Any,
        inner_optimizer: optim.Optimizer,
        n_steps: int,
    ) -> None:
        """Adapt network in-place using support data (second-order MAML).

        Computes support loss, backpropagates through adaptation, and performs
        optimizer step. Inner-loop gradients are tracked (create_graph=True) so
        that query-loss backprop can flow through the adaptation step.
        """
        for _ in range(n_steps):
            loss = loss_fn(network, support_data)
            inner_optimizer.zero_grad()
            loss.backward(create_graph=True)
            inner_optimizer.step()

    def _update_meta_policy(self) -> Dict[str, float]:
        """
        Execute one MAML meta-update with second-order meta-gradients.

        Algorithm:
        1. For each active task:
           a. Clone meta-policy → task-specific policy (fast-weights)
           b. Inner loop: ∇_support on support data (1st-order gradient)
           c. Outer loop: L_query on adapted policy
           d. Compute query loss (2nd-order meta-gradient flows through adaptation)
        2. Average second-order meta-gradients across tasks
        3. Meta-optimizer step: update meta-policy with meta_lr
        """
        active_task_ids = self.task_manager.get_active_task_ids()
        # n_tasks = min(self.mcfg["n_tasks_per_update"], len(active_task_ids))
        # sampled_task_ids = random.sample(active_task_ids, n_tasks)

        total_steps = 0
        task_losses: List[float] = []
        task_metrics: Dict[Any, Dict[str, float]] = {}

        # Collect support/query data for sampled tasks
        task_trajectories: Dict[Any, Tuple[Any, Any]] = {}
        for task_id in active_task_ids:
            task = self.task_manager.get_task(task_id)

            # Support data (inner-loop training) — temporarily override buffer capacity
            _orig_support_len = self.tcfg.get("rollout_len", 256)
            self.tcfg["rollout_len"] = self.mcfg["support_rollout_len"]
            sup_buf = self._collect(task.problem, self.agent, task.generator)
            self.tcfg["rollout_len"] = _orig_support_len
            total_steps += sup_buf._ptr

            # Query data (meta-gradient computation) — temporarily override buffer capacity
            self.tcfg["rollout_len"] = self.mcfg["query_rollout_len"]
            qry_buf = self._collect(task.problem, self.agent, task.generator)
            self.tcfg["rollout_len"] = _orig_support_len
            total_steps += qry_buf._ptr

            task_trajectories[task_id] = (sup_buf, qry_buf)

        # Compute second-order meta-gradients
        assert self.agent.estimator is not None, "Estimator required for training"
        estimator = self.agent.estimator

        # Zero meta-gradients before accumulation
        self._meta_optimizer.zero_grad()

        # Compute average of sub-policy losses on query rollouts
        n_active_tasks = len(active_task_ids)
        meta_loss = None

        # Track hardest task for entropy-based curriculum expansion
        hardest_task_id = max(
            active_task_ids, key=lambda tid: int(str(tid).split("_")[0])
        )
        hardest_task_entropy = None

        for task_id in active_task_ids:
            if task_id not in task_trajectories:
                continue

            sup_buf, qry_buf = task_trajectories[task_id]

            # Inner loop: task-specific adaptation on support data
            sub_agent = self.agent.clone()

            # Support loss (before adaptation)
            support_loss = estimator.compute_loss(self.agent.policy, sup_buf)

            # Create inner optimizer (SGD) for task-specific adaptation
            inner_optimizer = optim.SGD(
                sub_agent.policy.parameters(),
                lr=self.mcfg["inner_lr"],
            )

            # Adapt fast-weights via gradient descent on support loss
            self._update_sub_policy(
                network=sub_agent.policy,
                loss_fn=lambda net, data: estimator.compute_loss(net, data),
                support_data=sup_buf,
                inner_optimizer=inner_optimizer,
                n_steps=1,
            )

            # Outer loop: compute query loss with adapted task-specific policy
            query_loss = estimator.compute_loss(sub_agent.policy, qry_buf)

            task_losses.append(query_loss.item())
            task_metrics[task_id] = {
                "support_loss": support_loss.item(),
                "query_loss": query_loss.item(),
                "improvement": (support_loss.item() - query_loss.item()),
            }

            # Compute entropy of adapted policy on the hardest task
            if task_id == hardest_task_id:
                hardest_task_entropy = sub_agent.policy.compute_entropy(qry_buf)
                task_metrics[task_id]["entropy"] = hardest_task_entropy

            if meta_loss is None:
                meta_loss = query_loss / n_active_tasks
            else:
                meta_loss = meta_loss + query_loss / n_active_tasks

        # Single backward pass on average meta-loss
        if meta_loss is not None:
            meta_loss.backward()

        # Compute gradient norm before clipping (diagnostic)
        meta_grad_norm = 0.0
        for p in self.agent.policy.parameters():
            if p.grad is not None:
                meta_grad_norm += p.grad.data.norm(2.0) ** 2
        meta_grad_norm = float(meta_grad_norm**0.5)

        # Stabilize 2nd-order meta-updates via gradient clipping
        nn.utils.clip_grad_norm_(
            self.agent.policy.parameters(),
            self.mcfg["max_grad_norm"],
        )

        # Meta-optimizer step: update meta-policy using averaged 2nd-order meta-gradients
        # Uses meta_lr (0.0003) for careful meta-policy initialization updates
        self._meta_optimizer.step()

        # Curriculum expansion based on hardest task entropy
        if hardest_task_entropy is not None:
            expanded = self.curriculum_scheduler.should_expand(
                self.task_manager,
                hardest_task_id,
                hardest_task_entropy,
            )
            if expanded:
                self.task_manager.expand_curriculum()
                print(
                    f"[MetaTrainer] Curriculum expanded to {self.task_manager.num_active_tasks()} tasks"
                )

        metrics = {
            "train/meta_loss": float(np.mean(task_losses)) if task_losses else 0.0,
            "train/meta_grad_norm": meta_grad_norm,
            "train/update_count": float(self._iteration),
            "_steps": total_steps,
        }

        # Add per-task metrics
        for task_id, task_metric in task_metrics.items():
            for key, val in task_metric.items():
                metrics[f"train/task_{task_id}_{key}"] = val

        return metrics

    def _save_checkpoint(self, tag: str) -> None:
        path = f"{self.tcfg['checkpoint_dir']}/{self.experiment_name}_{tag}.pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network_state": self.agent.policy.state_dict(),
                "meta_optimizer_state": self._meta_optimizer.state_dict(),
                "iteration": self._iteration,
            },
            path,
        )

    def _print_header(self) -> None:
        task_ids = sorted(self.task_manager.get_active_task_ids())
        total_tasks = self.task_manager.num_total_tasks()
        print(
            f"\n{'=' * 64}\n"
            f"  Experiment : {self.experiment_name}\n"
            f"  Algorithm  : MAML (Meta-Learning)\n"
            f"  Tasks      : {task_ids} (of {total_tasks} total)\n"
            f"  Inner lr   : {self.mcfg['inner_lr']}   "
            f"Meta lr : {self.mcfg['meta_lr']}\n"
            f"  Budget     : {self.tcfg['total_timesteps']:,} steps\n"
            f"  Device     : {self.device}\n"
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
        agent: BaseAgent,
        env: Any,
        tasks: Dict[str, Tuple[Any, Callable]],
        cfg: Dict[str, Any],
        evaluator: Any,
        logger: Any,
    ):
        self.agent = agent
        self.env = env
        self.tasks = tasks
        self.cfg = cfg
        self.evaluator = evaluator
        self.logger = logger

        # Handle experiment name (can be string or dict)
        experiment = cfg.get("experiment", {})
        if isinstance(experiment, str):
            self.experiment_name = cfg.get("name") or experiment or "pomo_experiment"
        else:
            self.experiment_name = cfg.get("name") or experiment.get("name", "pomo_experiment")
        self.device = cfg.get("device", "cpu")

        # Setup task iteration
        if not tasks:
            raise ValueError("POMOTrainer requires at least one task")
        self.task_ids = list(tasks.keys())

        # Extract config from hierarchical structure
        trainer_cfg = cfg.get("trainer", {})
        agent_cfg = trainer_cfg.get("agent", {})
        hparams_cfg = trainer_cfg.get("hparams", {})
        training_cfg = trainer_cfg.get("training", cfg.get("training", cfg.get("train", {})))

        self.pcfg = {
            "learning_rate": float(agent_cfg.get("learning_rate", 0.001)),
            "max_grad_norm": float(hparams_cfg.get("max_grad_norm", 0.5)),
        }

        train_logging = training_cfg.get("logging", {})
        early_stop = training_cfg.get("early_stopping", {})

        self.tcfg = {
            "total_timesteps": training_cfg.get(
                "total_timesteps", cfg.get("train", {}).get("total_timesteps", 1000000)
            ),
            "log_interval": train_logging.get(
                "log_interval", cfg.get("train", {}).get("log_interval", 5)
            ),
            "eval_interval": train_logging.get(
                "eval_interval", cfg.get("train", {}).get("eval_interval", 20)
            ),
            "checkpoint_interval": train_logging.get(
                "checkpoint_interval",
                cfg.get("train", {}).get("checkpoint_interval", 100),
            ),
            "checkpoint_dir": train_logging.get(
                "checkpoint_dir",
                cfg.get("train", {}).get("checkpoint_dir", "checkpoints"),
            ),
            "log_dir": train_logging.get(
                "log_dir", cfg.get("train", {}).get("log_dir", "logs")
            ),
            "patience": early_stop.get(
                "patience", cfg.get("train", {}).get("patience", 150)
            ),
            "min_delta": early_stop.get(
                "min_delta", cfg.get("train", {}).get("min_delta", 0.0001)
            ),
        }

        if agent._opt_policy is None:
            self._optimizer: optim.Optimizer = optim.Adam(
                agent.policy.parameters(), lr=self.pcfg["learning_rate"]
            )
            agent._opt_policy = self._optimizer
        else:
            opt = agent.opt_policy
            if opt is None:
                raise ValueError(
                    "POMOTrainer requires an optimizer. Agent must have opt_policy or allow creation."
                )
            self._optimizer = opt

        self._timestep = 0
        self._iteration = 0
        self._best_objective = float("-inf")
        self._patience_counter = 0

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        agent: BaseAgent,
        env: Any,
        tasks: Dict[str, Tuple[Any, Callable]],
        evaluator: Any,
        logger: Any,
    ) -> "POMOTrainer":
        return cls(
            agent=agent,
            env=env,
            tasks=tasks,
            cfg=cfg,
            evaluator=evaluator,
            logger=logger,
        )

    def train(self) -> Dict[str, Any]:
        """Run POMO training loop, training one model per task."""
        start_time = time.time()

        for task_id in self.task_ids:
            _, generator = self.tasks[task_id]
            self._print_header_task(task_id)
            task_start_time = time.time()
            stop_reason = "timestep_limit"
            task_timestep = 0
            task_iteration = 0
            task_best_objective = float("-inf")
            task_patience_counter = 0

            while task_timestep < self.tcfg["total_timesteps"]:
                iter_start = time.time()
                task_iteration += 1

                metrics = self._update_policy(generator)
                task_timestep += int(metrics.pop("_steps", 0))

                metrics["iter_time_s"] = time.time() - iter_start

                if task_iteration % self.tcfg["log_interval"] == 0:
                    self.logger.log_metrics(
                        metrics,
                        step=task_timestep,
                        print_keys=["train/pomo_loss", "train/avg_episode_return"],
                    )

                if task_iteration % self.tcfg["eval_interval"] == 0:
                    eval_stats = self.evaluator.evaluate(generator)
                    self.logger.log_metrics(eval_stats, step=task_timestep, prefix="eval")

                    mean_obj = eval_stats.get("mean_objective", float("-inf"))
                    if mean_obj > task_best_objective + self.tcfg["min_delta"]:
                        task_best_objective = mean_obj
                        task_patience_counter = 0
                        self._save_checkpoint(f"best_{task_id}")
                        self.logger.log_event(
                            "best_checkpoint", task_timestep, objective=f"{mean_obj:.4f}"
                        )
                    else:
                        task_patience_counter += 1

                    if task_patience_counter >= self.tcfg["patience"]:
                        stop_reason = "early_stopping"
                        self.logger.log_event(
                            "early_stop", task_timestep, patience=self.tcfg["patience"]
                        )
                        break

            self._timestep += task_timestep
            self._iteration += task_iteration
            self._best_objective = max(self._best_objective, task_best_objective)

        summary = {
            "stop_reason": "completed",
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

    def _collect(self, generator: Callable) -> Tuple[RolloutBuffer, List[float]]:
        """Collect rollouts using POMO multi-start strategy.

        For each problem instance, collect complete episodes from multiple
        candidate starting points.

        Args:
            generator: callable that generates raw problem instances

        Returns:
            (buffer, episode_returns): collected transitions and episode metrics
        """
        buffer = RolloutBuffer(capacity=self.tcfg.get("rollout_len", 2048))
        episode_returns = []

        for attempt in range(100):
            raw = generator()
            obs, info = self.env.reset(raw)
            if info["action_mask"].any():
                break
        else:
            return buffer, episode_returns

        try:
            candidates = self.env.get_candidate_starts()
        except NotImplementedError:
            candidates = [(self.env._current_state, obs, info)]

        if not candidates:
            return buffer, episode_returns

        for state, start_obs, start_info in candidates:
            if buffer.is_full:
                break

            self.env._current_state = state
            traj_obs = start_obs
            traj_mask = start_info["action_mask"]
            episode_return = 0.0

            while not buffer.is_full:
                action, lp, val = self.agent.select_action(
                    traj_obs, traj_mask, training=True
                )

                if not traj_mask[action]:
                    feasible = np.where(traj_mask)[0]
                    if len(feasible) > 0:
                        action = int(np.random.choice(feasible))
                        lp = 0.0
                    else:
                        break

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                episode_return += reward

                buffer.add(
                    obs=traj_obs,
                    action=action,
                    reward=reward,
                    done=(terminated or truncated),
                    log_prob=lp,
                    value=val,
                    action_mask=traj_mask,
                )

                if terminated or truncated or not info["action_mask"].any():
                    episode_returns.append(episode_return)
                    break

                traj_obs = next_obs
                traj_mask = info["action_mask"]

        return buffer, episode_returns

    def _update_policy(self, generator: Callable) -> Dict[str, float]:
        """
        Perform one POMO update using REINFORCEEstimator.

        Steps:
          1. Collect rollouts from multiple candidate starting points
          2. Compute loss via estimator.compute_loss()
          3. Backprop and step optimizer
          4. Return metrics
        """
        buffer, episode_returns = self._collect(generator)

        if buffer._ptr == 0:
            return {
                "train/pomo_loss": 0.0,
                "train/avg_episode_return": 0.0,
                "_steps": 0,
            }

        loss = self.agent.estimator.compute_loss(self.agent.policy, buffer)

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.agent.policy.parameters(), self.pcfg["max_grad_norm"]
        )
        self._optimizer.step()

        avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0

        return {
            "train/pomo_loss": float(loss.item()),
            "train/avg_episode_return": avg_return,
            "_steps": buffer._ptr,
        }

    def _save_checkpoint(self, tag: str) -> None:
        path = f"{self.tcfg['checkpoint_dir']}/{self.experiment_name}_{tag}.pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network_state": self.agent.policy.state_dict(),
                "iteration": self._iteration,
            },
            path,
        )

    def _print_header(self) -> None:
        print(
            f"\n{'=' * 64}\n"
            f"  Experiment : {self.experiment_name}\n"
            f"  Algorithm  : POMO (Multiple Optima)\n"
            f"  Tasks      : {len(self.task_ids)}\n"
            f"  Budget/task: {self.tcfg['total_timesteps']:,} steps\n"
            f"  Device     : {self.device}\n"
            f"{'=' * 64}"
        )

    def _print_header_task(self, task_id: str) -> None:
        print(
            f"\n{'-' * 64}\n"
            f"  Task: {task_id}\n"
            f"{'-' * 64}"
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
