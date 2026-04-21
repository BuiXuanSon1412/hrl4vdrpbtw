"""
registry.py
-----------
Factory functions that wire concrete implementations to abstract interfaces.

train.py and evaluate.py are the only callers.

Rules
-----
- No hyperparameter values live here; all come from ExperimentConfig.
- n_fleets is read from the built Problem (problem.n_fleets) and passed
  forward explicitly — never re-read from problem_kwargs with a string key.
- build_network does not receive action_space_size (it does not use it).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List
import numpy as np

# Problems
from impl.environment import VRPBTWEnv

# Networks
from impl.policy import VRPBTWPolicy

# Core
from core.agent import Agent, Agent
from core.policy import BasePolicy
from core.estimator import PPOEstimator

_DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_GENERATED_ROOT = _DEFAULT_DATA_ROOT / "generated"

# Add data/generated to sys.path for direct import
sys.path.insert(0, str(_GENERATED_ROOT))
from generate import create_instance  # type: ignore[import]


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


def build_problem(cfg: Dict[str, Any]) -> VRPBTWEnv:
    # Support hierarchical config structure
    env_cfg = cfg.get("environment", cfg)
    problem_cfg = env_cfg.get("problem", {})
    name = problem_cfg.get("name", env_cfg.get("problem_name", "vrpbtw"))
    problem_kwargs = problem_cfg.get("kwargs", env_cfg.get("problem_kwargs", {}))
    kwargs = dict(problem_kwargs)

    if name == "vrpbtw":
        return VRPBTWEnv(
            n_customers=kwargs.get("n_customers", 10),
            n_fleets=kwargs.get("n_fleets", 2),
        )

    raise ValueError(
        f"Unknown problem {name!r}.  Register it in registry.py.  Known: ['vrpbtw']"
    )


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------


def _normalize_generated_instance(
    data: Dict[str, Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    general = data["Config"]["General"]
    vehicles = data["Config"]["Vehicles"]
    depot = data["Config"]["Depot"]

    return {
        "depot": list(depot["coord"]),
        "customers": [
            [
                float(node["coord"][0]),
                float(node["coord"][1]),
                float(node["tw_h"][0]),
                float(node["tw_h"][1]),
                float(node["demand"]),
            ]
            for node in data["Nodes"]
        ],
        "n_fleets": int(vehicles["NUM_TRUCKS"]),
        "truck_capacity": float(vehicles["CAPACITY_TRUCK"]),
        "drone_capacity": float(vehicles["CAPACITY_DRONE"]),
        "system_duration": float(general["T_MAX_SYSTEM_H"]),
        "trip_duration": float(vehicles["DRONE_DURATION_H"]),
        "truck_speed": float(vehicles["V_TRUCK_KM_H"]),
        "drone_speed": float(vehicles["V_DRONE_KM_H"]),
        "truck_cost": float(kwargs.get("truck_cost", 1.0)),
        "drone_cost": float(kwargs.get("drone_cost", 0.5)),
        "launch_time": float(vehicles["DRONE_TAKEOFF_MIN"]) / 60.0,
        "land_time": float(vehicles["DRONE_LANDING_MIN"]) / 60.0,
        "service_time": float(vehicles["SERVICE_TIME_MIN"]) / 60.0,
        "lambda_weight": float(kwargs.get("lambda_weight", 0.5)),
        "max_coord": float(general["MAX_COORD_KM"]),
    }


def _load_generated_config() -> Dict[str, Any]:
    with (_GENERATED_ROOT / "config.json").open() as fh:
        return json.load(fh)


def get_generator(cfg: Dict[str, Any]) -> Callable[..., Any]:
    # Support hierarchical config structure
    env_cfg = cfg.get(
        "environment", cfg
    )  # Fallback to cfg if environment key not found
    problem_cfg = env_cfg.get("problem", {})
    name = problem_cfg.get("name", env_cfg.get("problem_name", "vrpbtw"))

    if name == "vrpbtw":
        problem_kwargs = problem_cfg.get("kwargs", {})
        kwargs = dict(problem_kwargs)
        generated_cfg = _load_generated_config()

        def _generator(
            size: Optional[int] = None,
            rng=None,
            **extra_kwargs: Any,
        ) -> Dict[str, Any]:
            n_customers = int(
                size if size is not None else kwargs.get("n_customers", 10)
            )
            dist = str(extra_kwargs.get("dist", kwargs.get("coord_distribution", "RC")))
            ratio = float(extra_kwargs.get("ratio", kwargs.get("linehaul_ratio", 0.5)))
            seed_offset = int(
                extra_kwargs.get(
                    "generator_seed_offset",
                    kwargs.get("generator_seed_offset", 0),
                )
            )
            seed = (
                int(rng.integers(0, 2**31 - 1)) + seed_offset
                if rng is not None
                else int(extra_kwargs.get("seed", kwargs.get("seed", 42))) + seed_offset
            )
            data = create_instance(generated_cfg, n_customers, dist, ratio, seed)
            raw = _normalize_generated_instance(data, kwargs)

            # Allow RL-specific overrides while keeping the data/generated pattern.
            for key in (
                "n_fleets",
                "truck_capacity",
                "drone_capacity",
                "truck_speed",
                "drone_speed",
                "truck_cost",
                "drone_cost",
                "launch_time",
                "land_time",
                "service_time",
                "trip_duration",
                "lambda_weight",
            ):
                if key in kwargs:
                    raw[key] = kwargs[key]
                if key in extra_kwargs:
                    raw[key] = extra_kwargs[key]

            return raw

        return _generator

    raise ValueError(f"No generator for problem {name!r}.  Known: ['vrpbtw']")


# ---------------------------------------------------------------------------
# MAML task pool
# ---------------------------------------------------------------------------


def build_task_pool(cfg: Dict[str, Any]) -> Dict[str, Tuple[Any, Callable]]:
    """
    Build a MAML task pool with multiple coordinate distributions.

    Task pool structure: {task_id: (VRPBTWEnv, generator_fn)}
    Task ID format: "{size}_{distribution}" (e.g., "10_R", "20_C", "50_RC")

    Fleet sizes are read from data/generated/config.json FLEET_SIZES.
    Each task-distribution combination gets its own seeded RNG for reproducibility.

    The returned dict is compatible with MetaTrainer, which internally converts
    the task pool into Task objects and a TaskManager for curriculum-based
    meta-learning.
    """
    # Support hierarchical config structure
    algo_cfg = cfg.get("algorithm", {})
    task_pool_cfg = algo_cfg.get("task_pool", {})
    task_sizes = task_pool_cfg.get(
        "sizes", cfg.get("maml", {}).get("task_sizes", [10, 20, 50, 100])
    )
    task_distributions = task_pool_cfg.get(
        "distributions", cfg.get("maml", {}).get("task_distributions", ["RC"])
    )

    data_cfg = _load_generated_config()
    fleet_map: Dict[int, int] = {
        int(k): int(v) for k, v in data_cfg["FLEET_SIZES"].items()
    }

    base_gen = get_generator(cfg)
    pool: Dict[str, Tuple[Any, Callable]] = {}

    for size in task_sizes:
        n_fleets = fleet_map.get(size, 2)

        for dist in task_distributions:
            task_id = f"{size}_{dist}"

            # Create problem instance for this size/distribution combo
            problem = VRPBTWEnv(n_customers=size, n_fleets=n_fleets)

            # Each task has a dedicated RNG: hash of (size, dist) ensures determinism
            # while keeping distinct streams for different tasks
            reproducibility_cfg = cfg.get("reproducibility", {})
            seed_cfg = reproducibility_cfg.get("seed", cfg.get("seed", {}))
            global_seed = seed_cfg.get("global_seed", 42)
            _rng = np.random.default_rng(global_seed + hash((size, dist)) % (2**31 - 1))

            # Closure captures size, distribution, fleet count, and RNG
            gen = _make_task_generator(size, dist, n_fleets, base_gen, _rng)
            pool[task_id] = (problem, gen)

    return pool


def _make_task_generator(
    size: int,
    dist: str,
    n_fleets: int,
    base_gen: Callable,
    rng: np.random.Generator,
) -> Callable:
    """Factory for task-specific generators with captured parameters."""

    def _gen() -> Dict[str, Any]:
        return base_gen(size=size, dist=dist, n_fleets=n_fleets, rng=rng)

    return _gen


def _parse_task_id(task_id: str) -> Tuple[int, str]:
    """Parse task ID string (format: "{size}_{dist}") into (size, dist) tuple.

    Used for proper sorting of task IDs by size (numeric) then distribution (alphabetic).
    """
    parts = task_id.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid task_id format: {task_id!r}. Expected '{{size}}_{{dist}}'"
        )
    try:
        size = int(parts[0])
    except ValueError:
        raise ValueError(f"Invalid size in task_id {task_id!r}: {parts[0]!r}")
    return size, parts[1]


def sort_task_ids(task_ids: List[str]) -> List[str]:
    """Sort task IDs by size (numeric) then distribution (alphabetic).

    Examples:
        ["10_R", "100_C", "10_C"] → ["10_C", "10_R", "100_C"]
    """
    return sorted(task_ids, key=_parse_task_id)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def build_network(
    cfg: Dict[str, Any],
) -> BasePolicy:
    net_cfg = cfg["network"]
    net_type = net_cfg.get("type") or net_cfg.get("network_type", "hgnn")

    if net_type == "hgnn":
        return VRPBTWPolicy(cfg=net_cfg)

    raise ValueError(
        f"Unknown network type {net_type!r}.  "
        f"Register it in registry.py.  Known: ['hgnn']"
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def build_agent(cfg: Dict[str, Any]) -> Agent:
    network = build_network(cfg)
    device = cfg.get("device", "cpu")
    network = network.to_device(device)

    # Build estimator (PPO for all algorithms)
    algo_cfg = cfg.get("algorithm", {})
    if isinstance(algo_cfg, dict):
        algo_name = algo_cfg.get("name", "").lower()
    else:
        algo_name = str(algo_cfg).lower()

    # Extract RL objective config (works for both MAML and single-env)
    rl_obj_cfg = algo_cfg.get("rl_objective", {}) if isinstance(algo_cfg, dict) else {}
    if not rl_obj_cfg:
        rl_obj_cfg = cfg.get("rl_objective", {})

    estimator = PPOEstimator(
        device=device,
        gamma=rl_obj_cfg.get("gamma", 0.99),
        gae_lambda=rl_obj_cfg.get("gae_lambda", 0.95),
        value_coef=rl_obj_cfg.get("value_coefficient", 0.5),
        entropy_coef=rl_obj_cfg.get("entropy_coefficient", 0.02),
        clip_ratio=rl_obj_cfg.get("clip_eps", 0.2),
    )

    return Agent(policy=network, estimator=estimator, device=device)
