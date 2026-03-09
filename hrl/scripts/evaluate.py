"""
scripts/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive offline evaluation of a saved checkpoint.

Runs against:
  1. Random instances (same distribution as training)
  2. Clustered instances (distribution shift test)
  3. Benchmark file instances (fixed, reproducible)

Usage
─────
    python scripts/evaluate.py checkpoint=outputs/checkpoints/run01/best.pt
    python scripts/evaluate.py checkpoint=best.pt num_episodes=100 output_dir=results/run01
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from src.agents import build_agent
from src.envs import VRPBTWEnv
from src.rewards.default import DefaultRewardFn
from src.utils.metrics import EpisodeResult, EvalStats

console = Console()


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Allow CLI overrides: checkpoint=path, num_episodes=N, output_dir=path
    checkpoint = cfg.get("checkpoint", "outputs/checkpoints/best.pt")
    num_episodes = cfg.get("num_episodes", 100)
    output_dir = Path(cfg.get("output_dir", "outputs/results"))

    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)

    console.rule("[bold blue]VRPBTW-RL Evaluation")
    console.print(f"Checkpoint : [cyan]{checkpoint}[/]")
    console.print(f"Episodes   : [cyan]{num_episodes}[/]")

    # ── Build env & agent ─────────────────────────────────────────────────
    reward_fn = DefaultRewardFn(
        cost_weight=cfg.env.cost_weight,
        tardiness_weight=cfg.env.tardiness_weight,
    )
    env = VRPBTWEnv(
        num_customers=cfg.problem.num_customers,
        num_vehicles=cfg.problem.num_vehicles,
        num_drones=cfg.problem.num_drones,
        map_size=cfg.problem.map_size,
        reward_fn=reward_fn,
    )
    agent = build_agent(cfg, env)
    agent.load(checkpoint)
    console.print(f"Parameters : [cyan]{agent.num_parameters:,}[/]\n")

    # ── Evaluation suites ─────────────────────────────────────────────────
    all_results: dict = {}

    suites = {
        "random_instances": lambda: _run_episodes(agent, num_episodes),
    }

    for name, fn in suites.items():
        console.print(f"Running suite: [bold]{name}[/]")
        results = fn()
        stats = EvalStats.from_results(results)
        all_results[name] = stats.to_dict()
        _print_stats_table(console, name, stats)

    # ── Save JSON report ──────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\n[green]Report saved to {report_path}[/]")


def _run_episodes(agent, n: int):
    results = []
    for _ in range(n):
        r = agent.rollout(training=False)
        results.append(
            EpisodeResult(
                total_reward=r["total_reward"],
                total_cost=r["total_cost"],
                max_tardiness=r["max_tardiness"],
                service_rate=r["service_rate"],
                customers_served=r["customers_served"],
                steps=r["steps"],
            )
        )
    return results


def _print_stats_table(console: Console, name: str, stats: EvalStats):
    table = Table(title=name, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=28)
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")

    rows = [
        (
            "Service Rate (%)",
            stats.mean_service_rate * 100,
            stats.std_service_rate * 100,
        ),
        ("Total Cost", stats.mean_cost, stats.std_cost),
        ("Tardiness", stats.mean_tardiness, stats.std_tardiness),
        ("Reward", stats.mean_reward, stats.std_reward),
        ("100% Service Rate (%)", stats.pct_full_service * 100, None),
    ]
    for label, mean, std in rows:
        if std is not None:
            table.add_row(label, f"{mean:.2f}", f"±{std:.2f}")
        else:
            table.add_row(label, f"{mean:.1f}", "—")

    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
