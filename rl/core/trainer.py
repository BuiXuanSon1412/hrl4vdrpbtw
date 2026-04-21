"""
core/trainer.py
---------------
Training-loop implementations.

Trainer — single-environment trainer (algorithm-agnostic via estimator)
MetaTrainer — multi-task MAML with curriculum expansion

Design principle:
  - Agent holds the policy network and estimator
  - Estimator computes gradients (PPO, A2C, etc.) — algorithm lives here
  - Trainer is algorithm-agnostic: collect → compute_loss → backward → step
  - MetaTrainer extends this with inner-loop adaptation and outer meta-updates
  - TaskManager, CurriculumScheduler used by MetaTrainer only
  - FineTuner adapts trained meta-policy to individual tasks
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.agent import Agent
from core.buffer import RolloutBuffer
from core.task import TaskManager


# ---------------------------------------------------------------------------
# Standalone collect function
# ---------------------------------------------------------------------------


def collect(
    env: Any,
    policy: Agent,
    buffer: RolloutBuffer,
    instance_gen: Callable,
) -> int:
    """
    Collect rollout data into buffer.

    Args:
        env: environment with reset/step interface
        policy: agent for select_action
        buffer: rollout buffer to fill
        instance_gen: callable that generates raw problem instances

    Returns:
        number of steps collected
    """
    rollout_len = buffer.capacity

    def _fresh_episode() -> Tuple[Any, np.ndarray]:
        for _ in range(100):
            raw = instance_gen()
            obs, info = env.reset(raw)
            mask = info["action_mask"]
            if mask.any():
                return obs, mask
        raise RuntimeError("collect: 100 consecutive dead-start instances.")

    obs, action_mask = _fresh_episode()
    steps_collected = 0

    while buffer._ptr < rollout_len and not buffer.is_full:
        action, lp, val = policy.select_action(obs, action_mask, training=True)

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
        steps_collected += 1

        obs = next_obs
        action_mask = info["action_mask"]

        if terminated or truncated or not action_mask.any():
            obs, action_mask = _fresh_episode()

    return steps_collected


# ---------------------------------------------------------------------------
# Trainer: single-environment training (no meta-learning)
# ---------------------------------------------------------------------------


class Trainer:
    """
    Simple single-environment trainer.

    Trains a single policy on one environment without meta-learning.
    Simpler than MetaTrainer: collect → update → repeat.

    The estimator (injected via agent) determines the gradient algorithm
    (PPO, A2C, etc.) — this trainer is algorithm-agnostic.
    """

    def __init__(
        self,
        agent: Agent,
        env: Any,
        generator: Callable,
        cfg: Dict[str, Any],
        evaluator: Any,
        logger: Any,
    ):
        self.agent = agent
        self.env = env
        self.generator = generator
        self.cfg = cfg
        self.evaluator = evaluator
        self.logger = logger

        # Extract experiment name and device from hierarchical structure
        self.experiment_name = cfg.get("name") or cfg.get("experiment", {}).get(
            "name", "experiment"
        )
        self.device = cfg.get("device", "cpu")

        # Extract training config (algorithm is determined by the estimator)
        training_cfg = cfg.get("training", cfg.get("train", {}))

        # Build training config dict
        train_logging = training_cfg.get("logging", {})
        early_stop = training_cfg.get("early_stopping", {})

        self.tcfg = {
            "total_timesteps": training_cfg.get(
                "total_timesteps", cfg.get("train", {}).get("total_timesteps", 1000000)
            ),
            "rollout_len": training_cfg.get(
                "rollout_length", cfg.get("train", {}).get("rollout_len", 256)
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

        # Training state
        self._timestep = 0
        self._iteration = 0
        self._best_objective = float("-inf")
        self._patience_counter = 0

        # Create optimizer if agent doesn't have one
        if agent._opt_policy is None:
            learning_rate = cfg.get("learning_rate", 0.001)
            self._optimizer = optim.Adam(agent.policy.parameters(), lr=learning_rate)
            agent._opt_policy = self._optimizer
        else:
            assert agent.opt_policy is not None
            self._optimizer = agent.opt_policy

    def train(self) -> Dict[str, Any]:
        """Run training loop and return summary."""
        self._print_header()
        start_time = time.time()
        stop_reason = "timestep_limit"

        while self._timestep < self.tcfg["total_timesteps"]:
            iter_start = time.time()
            self._iteration += 1

            metrics = self._update()
            self._timestep += int(metrics.pop("_steps", 0))

            metrics["iter_time_s"] = time.time() - iter_start

            if self._iteration % self.tcfg["log_interval"] == 0:
                self.logger.log_metrics(
                    metrics,
                    step=self._timestep,
                    print_keys=["train/loss", "train/value"],
                )

            if self._iteration % self.tcfg["eval_interval"] == 0:
                eval_stats = self.evaluator.evaluate(self.generator)
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
        }
        self._save_checkpoint("final")
        self.logger.log_event("training_complete", self._timestep, **summary)
        self.logger.close()
        self._print_footer(summary)
        return summary

    def _update(self) -> Dict[str, float]:
        """Collect rollout and perform gradient update."""
        buffer = RolloutBuffer(capacity=self.tcfg["rollout_len"])
        steps_collected = collect(self.env, self.agent, buffer, self.generator)

        assert self.agent.estimator is not None, "Estimator required for training"
        loss = self.agent.estimator.compute_loss(self.agent.policy, buffer)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        metrics = {
            "train/loss": float(loss.item()),
            "_steps": steps_collected,
        }

        return metrics

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
            f"  Algorithm  : Single-Environment Training\n"
            f"  Budget     : {self.tcfg['total_timesteps']:,} steps\n"
            f"  Rollout    : {self.tcfg['rollout_len']} steps\n"
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
# Inner-loop adaptation (inlined from InnerUpdater)
# ---------------------------------------------------------------------------


def _adapt(
    network: nn.Module,
    loss_fn: Callable[[nn.Module, Any], torch.Tensor],
    support_data: Any,
    inner_optimizer: optim.Optimizer,
    n_steps: int,
) -> None:
    """
    Adapt network in-place using support data (second-order MAML).

    Computes support loss, backpropagates through adaptation, and performs
    optimizer step. Inner-loop gradients are tracked (create_graph=True) so
    that query-loss backprop can flow through the adaptation step.

    Args:
        network: policy network to adapt (modified in-place)
        loss_fn: callable (network, support_data) -> scalar loss
        support_data: support rollout buffer
        inner_optimizer: SGD optimizer for inner-loop adaptation
        n_steps: number of gradient steps
    """
    for _ in range(n_steps):
        loss = loss_fn(network, support_data)

        inner_optimizer.zero_grad()
        loss.backward(create_graph=True)
        inner_optimizer.step()


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

    def __init__(self, agent: Agent, task_manager: Any, cfg: Dict[str, Any]):
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
            buffer = RolloutBuffer(capacity=256)
            steps = collect(task_problem, task_agent, buffer, task_gen)
            total_steps += steps

            if task_agent.estimator is None:
                continue

            loss = task_agent.estimator.compute_loss(task_agent.policy, buffer)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_steps

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


class MetaTrainer:
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
        agent: Agent,
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

        # Extract experiment name and device from hierarchical structure
        self.experiment_name = cfg.get("name") or cfg.get("experiment", {}).get(
            "name", "experiment"
        )
        self.device = cfg.get("device", "cpu")

        # Extract config from hierarchical structure
        algo_cfg = cfg.get("algorithm", cfg.get("maml", {}))
        if isinstance(algo_cfg, str):
            algo_cfg = cfg.get("maml", {})
        training_cfg = cfg.get("training", cfg.get("train", {}))

        # Build maml config dict from hierarchical structure
        outer_loop = algo_cfg.get("outer_loop", {})
        inner_loop = algo_cfg.get("inner_loop", {})
        data_coll = algo_cfg.get("data_collection", {})
        curriculum = algo_cfg.get("curriculum", {})

        self.mcfg = {
            "meta_lr": outer_loop.get(
                "meta_learning_rate", cfg.get("maml", {}).get("meta_lr", 0.0003)
            ),
            "n_tasks_per_update": outer_loop.get(
                "n_tasks_per_update", cfg.get("maml", {}).get("n_tasks_per_update", 4)
            ),
            "max_grad_norm": outer_loop.get(
                "max_gradient_norm", cfg.get("maml", {}).get("max_grad_norm", 0.5)
            ),
            "inner_lr": inner_loop.get(
                "adaptation_learning_rate", cfg.get("maml", {}).get("inner_lr", 0.01)
            ),
            "n_inner_steps": inner_loop.get(
                "n_adaptation_steps", cfg.get("maml", {}).get("n_inner_steps", 1)
            ),
            "support_rollout_len": data_coll.get(
                "support_rollout_length",
                cfg.get("maml", {}).get("support_rollout_len", 256),
            ),
            "query_rollout_len": data_coll.get(
                "query_rollout_length",
                cfg.get("maml", {}).get("query_rollout_len", 256),
            ),
            "entropy_threshold": curriculum.get(
                "entropy_threshold", cfg.get("maml", {}).get("entropy_threshold", 0.5)
            ),
            "curriculum_check_interval": curriculum.get(
                "check_interval",
                cfg.get("maml", {}).get("curriculum_check_interval", 1),
            ),
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

    def train(self) -> Dict[str, Any]:
        """Run meta-training loop and return summary."""
        self._print_header()
        start_time = time.time()
        stop_reason = "timestep_limit"

        while self._timestep < self.tcfg["total_timesteps"]:
            iter_start = time.time()
            self._iteration += 1
            print("iteration ", self._iteration)
            metrics = self._meta_update()
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

    def _meta_update(self) -> Dict[str, float]:
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

            # Support data (inner-loop training)
            sup_buf = RolloutBuffer(capacity=self.mcfg["support_rollout_len"])
            print("start collect: ...")
            collect(task.problem, self.agent, sup_buf, task.generator)
            print("finish collect !")
            total_steps += sup_buf._ptr

            # Query data (meta-gradient computation)
            qry_buf = RolloutBuffer(capacity=self.mcfg["query_rollout_len"])
            collect(task.problem, self.agent, qry_buf, task.generator)
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
            _adapt(
                network=sub_agent.policy,
                loss_fn=lambda net, data: estimator.compute_loss(net, data),
                support_data=sup_buf,
                inner_optimizer=inner_optimizer,
                n_steps=self.mcfg["n_inner_steps"],
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
            f"  Algorithm  : MAML (Phase 1: Meta-Learning)\n"
            f"  Tasks      : {task_ids} (of {total_tasks} total)\n"
            f"  Inner lr   : {self.mcfg['inner_lr']}   "
            f"Meta lr : {self.mcfg['meta_lr']}\n"
            f"  Inner steps: {self.mcfg['n_inner_steps']}   "
            f"Tasks/update: {self.mcfg['n_tasks_per_update']}\n"
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


def _collect_episode_from_state(
    env: Any,
    policy: Agent,
    state: Any,
    obs: Any,
    action_mask: np.ndarray,
) -> Tuple[float, List[Tuple[Any, int, np.ndarray]]]:
    """
    Collect a single episode from a given starting state.

    Args:
        env: environment
        policy: policy agent
        state: starting state (internal)
        obs: observation at this state
        action_mask: feasible actions at this state

    Returns:
        (episode_return, list of (obs, action, mask) tuples for gradient recomputation)
    """
    trajectory = []
    cumulative_reward = 0.0

    while True:
        action, _, _ = policy.select_action(obs, action_mask, training=True)

        if not action_mask[action]:
            feasible = np.where(action_mask)[0]
            if len(feasible) > 0:
                action = int(np.random.choice(feasible))
            else:
                break

        next_obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        trajectory.append((obs, action, action_mask.copy()))

        if terminated or truncated or not info["action_mask"].any():
            break

        obs = next_obs
        action_mask = info["action_mask"]

    return cumulative_reward, trajectory


class POMOTrainer:
    """
    POMO trainer: policy optimization with multiple optima.

    For each training instance:
      1. Get candidate starting states from env.get_candidate_starts()
      2. Roll out N independent episodes using the same policy
      3. Compute REINFORCE loss with baseline = mean reward across all starts
      4. Advantage for each trajectory: A_i = R_i - baseline
      5. Loss = -1/N * sum_i(A_i * sum(log_prob_ij))
      6. Update policy

    This encourages the policy to find good solutions regardless of starting point.
    """

    def __init__(
        self,
        agent: Agent,
        env: Any,
        generator: Callable,
        cfg: Dict[str, Any],
        evaluator: Any,
        logger: Any,
    ):
        self.agent = agent
        self.env = env
        self.generator = generator
        self.cfg = cfg
        self.evaluator = evaluator
        self.logger = logger

        self.experiment_name = cfg.get("name") or cfg.get("experiment", {}).get(
            "name", "pomo_experiment"
        )
        self.device = cfg.get("device", "cpu")

        pomo_cfg = cfg.get("pomo", {})
        training_cfg = cfg.get("training", cfg.get("train", {}))

        self.pcfg = {
            "n_instances_per_update": pomo_cfg.get(
                "n_instances_per_update",
                cfg.get("train", {}).get("n_instances_per_update", 4),
            ),
            "entropy_coef": pomo_cfg.get("entropy_coef", 0.01),
            "learning_rate": cfg.get("learning_rate", 0.001),
            "max_grad_norm": pomo_cfg.get("max_grad_norm", 0.5),
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
                raise ValueError("POMOTrainer requires an optimizer. Agent must have opt_policy or allow creation.")
            self._optimizer = opt

        self._timestep = 0
        self._iteration = 0
        self._best_objective = float("-inf")
        self._patience_counter = 0

    def train(self) -> Dict[str, Any]:
        """Run POMO training loop and return summary."""
        self._print_header()
        start_time = time.time()
        stop_reason = "timestep_limit"

        while self._timestep < self.tcfg["total_timesteps"]:
            iter_start = time.time()
            self._iteration += 1

            metrics = self._pomo_update()
            self._timestep += int(metrics.pop("_steps", 0))

            metrics["iter_time_s"] = time.time() - iter_start

            if self._iteration % self.tcfg["log_interval"] == 0:
                self.logger.log_metrics(
                    metrics,
                    step=self._timestep,
                    print_keys=["train/pomo_loss", "train/avg_episode_return"],
                )

            if self._iteration % self.tcfg["eval_interval"] == 0:
                eval_stats = self.evaluator.evaluate(self.generator)
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
        }
        self._save_checkpoint("final")
        self.logger.log_event("training_complete", self._timestep, **summary)
        self.logger.close()
        self._print_footer(summary)
        return summary

    def _pomo_update(self) -> Dict[str, float]:
        """
        Perform one POMO update.

        For each of n_instances_per_update problem instances:
          1. Encode instance
          2. Get candidate starts via env.get_candidate_starts()
          3. Roll out N episodes from each start
          4. Compute REINFORCE loss with advantage = R_i - mean(R_j)
        """
        n_instances = self.pcfg["n_instances_per_update"]
        total_steps = 0
        batch_loss = None
        all_returns = []

        for _ in range(n_instances):
            for attempt in range(100):
                raw = self.generator()
                obs, info = self.env.reset(raw)
                if info["action_mask"].any():
                    break
            else:
                continue

            try:
                candidates = self.env.get_candidate_starts()
            except NotImplementedError:
                candidates = [(self.env._current_state, obs, info)]

            if not candidates:
                continue

            episode_returns = []
            episode_trajectories = []

            for state, start_obs, start_info in candidates:
                self.env._current_state = state
                ret, trajectory = _collect_episode_from_state(
                    self.env,
                    self.agent,
                    state,
                    start_obs,
                    start_info["action_mask"],
                )
                episode_returns.append(ret)
                episode_trajectories.append(trajectory)
                total_steps += len(trajectory)

            baseline = float(np.mean(episode_returns))
            loss = self._compute_pomo_loss(
                episode_returns, episode_trajectories, baseline
            )

            if batch_loss is None:
                batch_loss = loss / n_instances
            else:
                batch_loss = batch_loss + loss / n_instances

            all_returns.extend(episode_returns)

        if total_steps == 0 or batch_loss is None:
            return {
                "train/pomo_loss": 0.0,
                "train/avg_episode_return": 0.0,
                "_steps": 0,
            }

        self._optimizer.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_norm_(
            self.agent.policy.parameters(), self.pcfg["max_grad_norm"]
        )
        self._optimizer.step()

        return {
            "train/pomo_loss": float(batch_loss.item()),
            "train/avg_episode_return": float(np.mean(all_returns))
            if all_returns
            else 0.0,
            "_steps": total_steps,
        }

    def _compute_pomo_loss(
        self,
        episode_returns: List[float],
        episode_trajectories: List[List[Tuple]],
        baseline: float,
    ) -> torch.Tensor:
        """
        Compute POMO loss: -1/N * sum_i(A_i * sum(log_prob_ij))

        Recomputes log probs via forward pass to maintain gradient flow.

        Args:
            episode_returns: R_i for each trajectory i
            episode_trajectories: list of trajectory lists. Each trajectory is list of (obs, action, mask)
            baseline: mean return across all trajectories

        Returns:
            scalar loss tensor for backpropagation
        """
        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        n_episodes = len(episode_returns)

        for ret, trajectory in zip(episode_returns, episode_trajectories):
            if not trajectory:
                continue

            advantage = float(ret - baseline)
            trajectory_log_prob = torch.tensor(
                0.0, dtype=torch.float32, device=self.device
            )

            for obs, action, mask in trajectory:
                obs_t = self.agent.prepare_obs(obs)
                mask_t = torch.tensor(
                    mask, dtype=torch.bool, device=self.device
                ).unsqueeze(0)
                action_t = torch.tensor(
                    action, dtype=torch.long, device=self.device
                ).unsqueeze(0)

                _, lp_t, _ = self.agent.policy.get_action_and_log_prob(
                    obs_t, mask_t, deterministic=False
                )

                trajectory_log_prob = trajectory_log_prob + lp_t.squeeze(0)

            loss = loss + advantage * trajectory_log_prob / n_episodes

        return -loss

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
            f"  Budget     : {self.tcfg['total_timesteps']:,} steps\n"
            f"  Instances  : {self.pcfg['n_instances_per_update']} per update\n"
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
