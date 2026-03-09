"""
scripts/generate_data.py
─────────────────────────────────────────────────────────────────────────────
Generate and cache VRPBTW benchmark instances in CSV format.

Supports:
  • Random uniform instances
  • Clustered instances (Solomon-style)
  • Multiple problem sizes

Usage
─────
    python scripts/generate_data.py
    python scripts/generate_data.py sizes=[20,50,100,150,200] output_dir=data/generated
    vrpbtw-data sizes=[20,50] seed=0
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

console = Console()


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    sizes = cfg.get("sizes", cfg.data.benchmark_sizes)
    output_dir = Path(cfg.get("output_dir", cfg.data.generated_dir))
    seed = cfg.get("seed", cfg.experiment.seed)
    t_max = cfg.problem.time_horizon / 20.0  # normalise to reasonable TW range

    output_dir.mkdir(parents=True, exist_ok=True)
    console.rule("[bold blue]VRPBTW Data Generation")

    summary = Table(title="Generated Instances", show_header=True)
    summary.add_column("File", style="cyan")
    summary.add_column("N Customers", justify="right")
    summary.add_column("N Linehaul", justify="right")
    summary.add_column("N Backhaul", justify="right")

    for n in sizes:
        for dist in ["random", "clustered"]:
            df = (
                generate_random(n, seed=seed + n, t_max=t_max)
                if dist == "random"
                else generate_clustered(
                    n, n_clusters=max(3, n // 20), seed=seed + n, t_max=t_max
                )
            )
            fname = f"VRPBTW_N{n}_{dist}.csv"
            df.to_csv(output_dir / fname, index=False)

            n_lh = (df["DEMAND"] > 0).sum()
            n_bh = (df["DEMAND"] < 0).sum()
            summary.add_row(fname, str(n), str(n_lh), str(n_bh))

    console.print(summary)
    console.print(f"\n[green]All instances saved to {output_dir}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Generators
# ─────────────────────────────────────────────────────────────────────────────


def generate_random(
    n: int,
    seed: int = 42,
    t_max: float = 10.0,
    speed_factor: float = 1.0,
) -> pd.DataFrame:
    """Uniform-random Solomon-style VRPBTW instance."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    depot = {
        "ID": 0,
        "X_COORD": 0.0,
        "Y_COORD": 0.0,
        "DEMAND": 0,
        "SERVICE_TIME": 0.0,
        "READY_TIME": 0.0,
        "DUE_TIME": t_max,
        "CUSTOMER_TYPE": "Depot",
    }

    n_lh = n // 2
    all_ids = list(range(1, n + 1))
    random.shuffle(all_ids)
    linehaul_set = set(all_ids[:n_lh])

    coords = rng.uniform(0, 1, size=(n, 2))
    rows = [depot]

    for i in range(1, n + 1):
        x, y = coords[i - 1]
        is_lh = i in linehaul_set
        demand = random.randint(1, 10) if is_lh else random.randint(-10, -1)
        svc = round(random.uniform(0.01, 0.1), 2)
        dist_depot = math.hypot(x, y) * speed_factor
        tau_a = random.uniform(0.1, 0.9)
        tau_b = random.uniform(0.1, 0.9)
        ready = max(0.0, dist_depot - tau_a * dist_depot)
        due = dist_depot + svc + tau_b * (t_max - dist_depot)
        if ready >= due:
            ready = max(0.0, dist_depot - 0.5)
            due = min(t_max, ready + 1.0)
        rows.append(
            {
                "ID": i,
                "X_COORD": round(x, 4),
                "Y_COORD": round(y, 4),
                "DEMAND": demand,
                "SERVICE_TIME": svc,
                "READY_TIME": round(ready, 2),
                "DUE_TIME": round(due, 2),
                "CUSTOMER_TYPE": "Linehaul" if is_lh else "Backhaul",
            }
        )

    return pd.DataFrame(rows)


def generate_clustered(
    n: int,
    n_clusters: int = 5,
    seed: int = 42,
    t_max: float = 10.0,
    cluster_std: float = 0.1,
) -> pd.DataFrame:
    """Clustered instance: customers grouped around random cluster centres."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    centres = rng.uniform(0.1, 0.9, size=(n_clusters, 2))
    depot = {
        "ID": 0,
        "X_COORD": 0.0,
        "Y_COORD": 0.0,
        "DEMAND": 0,
        "SERVICE_TIME": 0.0,
        "READY_TIME": 0.0,
        "DUE_TIME": t_max,
        "CUSTOMER_TYPE": "Depot",
    }

    n_lh = n // 2
    all_ids = list(range(1, n + 1))
    random.shuffle(all_ids)
    linehaul_set = set(all_ids[:n_lh])

    rows = [depot]
    cluster_assigns = rng.integers(0, n_clusters, size=n)

    for idx in range(n):
        i = idx + 1
        cx, cy = centres[cluster_assigns[idx]]
        x = float(np.clip(rng.normal(cx, cluster_std), 0, 1))
        y = float(np.clip(rng.normal(cy, cluster_std), 0, 1))
        is_lh = i in linehaul_set
        demand = random.randint(1, 10) if is_lh else random.randint(-10, -1)
        svc = round(random.uniform(0.01, 0.1), 2)
        dist_depot = math.hypot(x, y)
        ready = max(0.0, dist_depot * 0.8)
        due = min(t_max, dist_depot + svc + random.uniform(0.5, 2.0))
        if ready >= due:
            due = min(t_max, ready + 0.5)
        rows.append(
            {
                "ID": i,
                "X_COORD": round(x, 4),
                "Y_COORD": round(y, 4),
                "DEMAND": demand,
                "SERVICE_TIME": svc,
                "READY_TIME": round(ready, 2),
                "DUE_TIME": round(due, 2),
                "CUSTOMER_TYPE": "Linehaul" if is_lh else "Backhaul",
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
