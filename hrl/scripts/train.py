"""
scripts/train.py
─────────────────────────────────────────────────────────────────────────────
Main training entry-point.

Usage
─────
    # Single training run with the base config
    python scripts/train.py

    # Override individual keys on the command line (Hydra syntax)
    python scripts/train.py experiment.name=my_run training.num_episodes=2000

    # Use a named experiment config
    python scripts/train.py --config-name experiments/large_scale

    # Install and use as CLI tool (after pip install -e .)
    vrpbtw-train experiment.name=prod_run
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from src.agents import build_agent
from src.envs import VRPBTWEnv, NormalizeRewardWrapper, CurriculumWrapper
from src.rewards.default import DefaultRewardFn
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import Logger
from src.utils.metrics import EpisodeResult, EvalStats

console = Console()


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ── Reproducibility ───────────────────────────────────────────────────
    seed = cfg.experiment.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    console.rule(f"[bold blue]VRPBTW-RL Training — {cfg.experiment.name}")

    # ── Environment ───────────────────────────────────────────────────────
    reward_fn = DefaultRewardFn(
        cost_weight=cfg.env.cost_weight,
        tardiness_weight=cfg.env.tardiness_weight,
    )
    env = VRPBTWEnv(
        num_customers=cfg.problem.num_customers,
        num_vehicles=cfg.problem.num_vehicles,
        num_drones=cfg.problem.num_drones,
        vehicle_capacity=cfg.problem.vehicle_capacity,
        drone_capacity=cfg.problem.drone_capacity,
        drone_battery=cfg.problem.drone_battery,
        map_size=cfg.problem.map_size,
        time_horizon=cfg.problem.time_horizon,
        linehaul_ratio=cfg.problem.linehaul_ratio,
        reward_fn=reward_fn,
    )

    if cfg.training.curriculum.enabled:
        env = CurriculumWrapper(
            env,
            start_customers=cfg.training.curriculum.start_customers,
            max_customers=cfg.problem.num_customers,
            increment=cfg.training.curriculum.increment_by,
        )

    # ── Agent ─────────────────────────────────────────────────────────────
    agent = build_agent(cfg, env)
    console.print(f"Agent parameters: [cyan]{agent.num_parameters:,}[/]")

    # ── Infrastructure ────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.checkpoint.dir) / cfg.experiment.name
    mgr = CheckpointManager(ckpt_dir, keep_last_n=cfg.checkpoint.keep_last_n)
    logger = Logger.from_cfg(cfg)

    # ── Training loop ─────────────────────────────────────────────────────
    best_service_rate = 0.0
    train_history = []

    console.print(
        f"Starting training for [bold]{cfg.training.num_episodes}[/] episodes\n"
    )

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training...", total=cfg.training.num_episodes)

        for ep in range(1, cfg.training.num_episodes + 1):
            # ── Collect experience ────────────────────────────────────────
            result = agent.rollout(training=True)
            train_history.append(result)

            # ── Update exploration ────────────────────────────────────────
            agent.update_exploration(ep)

            # ── Gradient update ───────────────────────────────────────────
            losses = {}
            if ep % cfg.training.train_every_n_episodes == 0:
                losses = agent.train_step()

            # ── Periodic logging ──────────────────────────────────────────
            if ep % cfg.logging.log_every_n_episodes == 0:
                log_data = {
                    "train/reward": result["total_reward"],
                    "train/cost": result["total_cost"],
                    "train/tardiness": result["total_tardiness"],
                    "train/service_rate": result["service_rate"],
                    "explore/epsilon": agent.epsilon,
                    "explore/temperature": agent.temperature,
                    **{f"loss/{k}": v for k, v in losses.items()},
                }
                logger.log(log_data, step=ep)

            # ── Periodic evaluation ───────────────────────────────────────
            if ep % cfg.evaluation.eval_every_n_episodes == 0:
                eval_results = [
                    EpisodeResult(**agent.rollout(training=False))
                    for _ in range(cfg.evaluation.num_eval_episodes)
                ]
                stats = EvalStats.from_results(eval_results)

                console.print(f"\n[bold]Ep {ep:>5}[/] | {stats}")
                logger.log(
                    {f"eval/{k}": v for k, v in stats.to_dict().items()},
                    step=ep,
                )

                # Save checkpoint
                mgr.save(
                    agent,
                    ep,
                    metric=stats.mean_service_rate,
                    higher_is_better=True,
                    extra_meta={"eval_stats": stats.to_dict()},
                )

                if stats.mean_service_rate < 0.9:
                    console.print(
                        f"  [yellow]⚠  Low service rate "
                        f"({stats.mean_service_rate * 100:.1f}%)[/]"
                    )

            # ── Periodic buffer flush ─────────────────────────────────────
            if ep % 100 == 0:
                agent.clear_buffers()

            # ── Curriculum progression ────────────────────────────────────
            if cfg.training.curriculum.enabled and hasattr(env, "report_episode"):
                env.report_episode(result["service_rate"])

            progress.advance(task)

    # ── Final save ────────────────────────────────────────────────────────
    final_path = ckpt_dir / "final.pt"
    agent.save(final_path)
    console.print(f"\n[green]Training complete.[/] Final checkpoint: {final_path}")

    logger.close()


if __name__ == "__main__":
    main()
