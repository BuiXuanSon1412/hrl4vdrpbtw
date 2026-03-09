"""
scripts/infer.py
─────────────────────────────────────────────────────────────────────────────
Inference / deployment script.

Given a trained checkpoint and a problem instance (either a CSV file or
randomly generated), produce and save the optimised vehicle routes.

Usage
─────
    # Random instance
    python scripts/infer.py checkpoint=best.pt num_customers=50

    # From a generated CSV
    python scripts/infer.py checkpoint=best.pt instance=data/generated/VRPBTW_N100.csv

    # Batch inference over all CSVs in a folder
    python scripts/infer.py checkpoint=best.pt instance_dir=data/generated/ output_dir=outputs/results/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from rich.console import Console

from src.agents import build_agent
from src.envs import VRPBTWEnv
from src.rewards.default import DefaultRewardFn

console = Console()


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint", "outputs/checkpoints/best.pt")
    instance = cfg.get("instance", None)
    instance_dir = cfg.get("instance_dir", None)
    output_dir = Path(cfg.get("output_dir", "outputs/results/inference"))

    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]VRPBTW-RL Inference")

    # ── Build agent ───────────────────────────────────────────────────────
    reward_fn = DefaultRewardFn()
    env = VRPBTWEnv(
        num_customers=cfg.problem.num_customers,
        num_vehicles=cfg.problem.num_vehicles,
        num_drones=cfg.problem.num_drones,
        map_size=cfg.problem.map_size,
        reward_fn=reward_fn,
    )
    agent = build_agent(cfg, env)
    agent.load(checkpoint)
    console.print(f"[green]✓[/] Loaded checkpoint: {checkpoint}\n")

    # ── Determine instances to solve ──────────────────────────────────────
    if instance_dir:
        files = sorted(Path(instance_dir).glob("*.csv"))
        console.print(f"Batch inference over {len(files)} instances")
        for f in files:
            _solve_and_save(agent, env, str(f), output_dir)
    elif instance:
        _solve_and_save(agent, env, instance, output_dir)
    else:
        # Random instance
        console.print("No instance file given — solving random instance")
        _solve_and_save(agent, env, None, output_dir)


def _solve_and_save(agent, env, instance_path, output_dir: Path):
    """Solve one instance and write the solution JSON."""
    options = {}
    if instance_path:
        options["instance"] = _load_csv_instance(instance_path, env)
        name = Path(instance_path).stem
    else:
        name = f"random_{int(time.time())}"

    t0 = time.perf_counter()
    result = agent.rollout(training=False)
    elapsed = time.perf_counter() - t0

    solution = {
        "instance": name,
        "solve_time_sec": round(elapsed, 4),
        "service_rate": result["service_rate"],
        "customers_served": result["customers_served"],
        "total_cost": result["total_cost"],
        "max_tardiness": result["max_tardiness"],
        "total_reward": result["total_reward"],
        "routes": [
            {
                "vehicle_id": v.id,
                "route": v.route,
                "load": v.current_load,
                "time": v.current_time,
            }
            for v in env.vehicles
        ],
    }

    out_path = output_dir / f"{name}_solution.json"
    with open(out_path, "w") as f:
        json.dump(solution, f, indent=2)

    console.print(
        f"[green]✓[/] {name} | "
        f"ServiceRate={result['service_rate'] * 100:.1f}% | "
        f"Cost={result['total_cost']:.1f} | "
        f"Time={elapsed:.3f}s | "
        f"Saved → {out_path}"
    )


def _load_csv_instance(path: str, env: VRPBTWEnv) -> dict:
    """
    Parse a CSV file produced by scripts/generate_data.py and convert it
    into a dict of Customer / Vehicle / Drone objects the env can load.
    """
    import pandas as pd
    from src.envs.vrpbtw_env import Customer, CustomerType, Vehicle, Drone

    df = pd.read_csv(path)
    depot_row = df[df["ID"] == 0].iloc[0]
    cust_rows = df[df["ID"] != 0]

    customers = []
    for _, row in cust_rows.iterrows():
        ctype = CustomerType.LINEHAUL if row["DEMAND"] > 0 else CustomerType.BACKHAUL
        customers.append(
            Customer(
                id=int(row["ID"]) - 1,
                x=float(row["X_COORD"]) * env.map_size,
                y=float(row["Y_COORD"]) * env.map_size,
                demand=abs(float(row["DEMAND"])),
                time_window_start=float(row["READY_TIME"]) * (env.time_horizon / 10.0),
                time_window_end=float(row["DUE_TIME"]) * (env.time_horizon / 10.0),
                service_time=float(row["SERVICE_TIME"]) * (env.time_horizon / 10.0),
                customer_type=ctype,
            )
        )

    vehicles = [
        Vehicle(
            id=i,
            capacity=env.vehicle_capacity,
            current_load=env.vehicle_capacity,
            x=depot_row["X_COORD"] * env.map_size,
            y=depot_row["Y_COORD"] * env.map_size,
            current_time=0.0,
        )
        for i in range(env.num_vehicles)
    ]
    drones = [
        Drone(
            id=i,
            capacity=env.drone_capacity,
            battery_capacity=env.drone_battery,
            current_battery=env.drone_battery,
            speed=2.0,
            is_available=True,
            x=depot_row["X_COORD"] * env.map_size,
            y=depot_row["Y_COORD"] * env.map_size,
        )
        for i in range(env.num_drones)
    ]

    return {"customers": customers, "vehicles": vehicles, "drones": drones}


if __name__ == "__main__":
    main()
