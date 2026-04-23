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
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List
import numpy as np
import torch.optim as optim

# Problems
from impl.environment import VRPBTWEnv

# Networks
from impl.policy import HGNNPolicy

# Core
from core.agent import PolicyAgent
from core.policy import BasePolicy
from core.estimator import BaseEstimator, PPOEstimator, REINFORCEEstimator
from core.trainer import BaseTrainer, MetaTrainer, POMOTrainer

_DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_GENERATED_ROOT = _DEFAULT_DATA_ROOT / "generated"

# Add data/generated to sys.path for direct import

# ---------------------------------------------------------------------------
# Registry: factory method dispatch tables
# ---------------------------------------------------------------------------

_POLICY_REGISTRY: Dict[str, type] = {
    "hgnn": HGNNPolicy,
}

_ESTIMATOR_REGISTRY: Dict[str, type] = {
    "ppo": PPOEstimator,
    "reinforce": REINFORCEEstimator,
}

_AGENT_REGISTRY: Dict[str, type] = {
    "policy": PolicyAgent,
}

_TRAINER_REGISTRY: Dict[str, type] = {
    "meta": MetaTrainer,
    "pomo": POMOTrainer,
}

_ENVIRONMENT_REGISTRY: Dict[str, type] = {
    "vrpbtw": VRPBTWEnv,
}

_OPTIMIZER_REGISTRY: Dict[str, type] = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
}


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------


def _generate_coords(num_customers: int, max_coord: float, dist_type: str, rng: np.random.Generator) -> List[List[float]]:
    """Generate customer coordinates based on distribution type (R, C, RC)."""
    if dist_type == "R":
        return rng.uniform(0, max_coord, size=(num_customers, 2)).tolist()

    elif dist_type == "C":
        coords = []
        remaining_nodes = num_customers

        if num_customers <= 200:
            std_dev = max_coord / 25
        elif num_customers <= 400:
            std_dev = max_coord / 32
        else:
            std_dev = max_coord / 40
        min_dist = std_dev * 4

        centers = []
        while remaining_nodes > 0:
            current_cluster_size = min(remaining_nodes, int(rng.uniform(10, 16)))

            proposal = rng.uniform(5, max_coord - 5, size=(2,))
            valid_center = False
            attempts = 0
            while not valid_center and attempts < 1000:
                proposal = rng.uniform(5, max_coord - 5, size=(2,))
                if not centers:
                    valid_center = True
                else:
                    dists = [np.linalg.norm(proposal - np.array(c)) for c in centers]
                    if min(dists) >= min_dist:
                        valid_center = True
                attempts += 1

            centers.append(proposal.tolist())

            for _ in range(current_cluster_size):
                point = rng.normal(proposal, std_dev)
                coords.append(np.clip(point, 0, max_coord).tolist())

            remaining_nodes -= current_cluster_size

        return coords

    elif dist_type == "RC":
        n_c = num_customers // 2
        rng1 = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        rng2 = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        return _generate_coords(n_c, max_coord, "C", rng1) + _generate_coords(
            num_customers - n_c, max_coord, "R", rng2
        )

    raise ValueError(f"Unknown distribution type: {dist_type}")


def get_vrpbtw_generator(cfg: Dict[str, Any]) -> Callable[..., Any]:
    """Get a VRPBTW instance generator from config.

    Returns a generator function that accepts (size, dist, n_fleets, rng, **kwargs)
    and creates normalized VRPBTW instances.
    """
    # Load config.json for demand ranges and other constants
    with (_GENERATED_ROOT / "config.json").open() as fh:
        data_cfg = json.load(fh)

    env_cfg = cfg.get("environment", cfg)
    props = env_cfg.get("properties", env_cfg)  # Support both old and new structure

    def _generator(
        size: int,
        dist: str,
        n_fleets: int,
        rng=None,
        **extra_kwargs: Any,
    ) -> Dict[str, Any]:
        if rng is None:
            rng = np.random.default_rng(42)

        max_coord = float(props.get("max_coord", 100.0))
        t_max_system = float(props.get("t_max_system_h", 24.0))
        ratio = float(extra_kwargs.get("ratio", 0.5))

        # Generate coordinates
        coords = _generate_coords(size, max_coord, dist, rng)
        depot_coord = [max_coord / 2.0, max_coord / 2.0]

        # Generate customers with time windows and demands
        customers = []
        linehaul_count = int(size * ratio)
        types = ["LINEHAUL"] * linehaul_count + ["BACKHAUL"] * (size - linehaul_count)
        rng.shuffle(types)

        for i in range(size):
            node_type = types[i]
            demand_range = (
                data_cfg["DEMAND_RANGE_LINEHAUL"]
                if node_type == "LINEHAUL"
                else data_cfg["DEMAND_RANGE_BACKHAUL"]
            )

            dist_km = np.linalg.norm(np.array(coords[i]) - np.array(depot_coord))
            min_reach_time = dist_km / float(env_cfg.get("v_truck_km_h", 40.0))
            ready_h = float(rng.uniform(min_reach_time * 1.1, t_max_system * 0.7))
            width_h = float(
                rng.uniform(
                    1.0,
                    1.0 + (data_cfg["TIME_WINDOW_SCALING_FACTOR"] * (dist_km / max_coord)),
                )
            )

            customers.append([
                float(coords[i][0]),
                float(coords[i][1]),
                float(round(ready_h, 4)),
                float(round(min(ready_h + width_h, t_max_system), 4)),
                float(int(rng.integers(demand_range[0], demand_range[1] + 1))),
            ])

        raw = {
            "depot": depot_coord,
            "customers": customers,
            "n_fleets": n_fleets,
            "truck_capacity": float(props.get("capacity_truck", 200)),
            "drone_capacity": float(props.get("capacity_drone", 20)),
            "system_duration": t_max_system,
            "trip_duration": float(props.get("drone_duration_h", 1.0)),
            "truck_speed": float(props.get("v_truck_km_h", 40.0)),
            "drone_speed": float(props.get("v_drone_km_h", 60.0)),
            "truck_cost": float(props.get("truck_cost_unit", 1.0)),
            "drone_cost": float(props.get("drone_cost_unit", 0.5)),
            "launch_time": float(props.get("drone_takeoff_min", 1.0)) / 60.0,
            "land_time": float(props.get("drone_landing_min", 1.0)) / 60.0,
            "service_time": float(props.get("service_time_min", 5.0)) / 60.0,
            "max_coord": max_coord,
        }

        return raw

    return _generator


def _parse_task_id(task_id: str) -> Tuple[int, int, str]:
    """Parse task ID format: {difficulty}_N{customers}_F{fleets}_{distribution}.

    Returns: (n_customers, n_fleets, distribution)
    Example: "001_N10_F2_RC" -> (10, 2, "RC")
    """
    parts = task_id.split("_")
    if len(parts) < 4:
        raise ValueError(f"Invalid task_id format: {task_id!r}")

    try:
        n_customers = int(parts[1][1:])  # Skip 'N'
        n_fleets = int(parts[2][1:])  # Skip 'F'
        dist = parts[3]
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse task_id {task_id!r}: {e}")

    return n_customers, n_fleets, dist


def build_tasks(cfg: Dict[str, Any]) -> Dict[str, Tuple[Any, Callable]]:
    """Build a MAML task pool from environment config.

    Task pool structure: {task_id: (VRPBTWEnv, generator_fn)}
    Task IDs come from environment.properties.tasks in config (format: {difficulty}_N{customers}_F{fleets}_{distribution})
    Example: "001_N10_F2_RC"

    Each task gets its own seeded RNG for reproducibility.
    The returned dict is compatible with MetaTrainer for curriculum-based meta-learning.
    """
    env_cfg = cfg.get("environment", cfg)
    props_cfg = env_cfg.get("properties", env_cfg)
    task_ids = props_cfg.get("tasks", [])

    if not task_ids:
        raise ValueError("No tasks configured in environment.tasks")

    base_gen = get_vrpbtw_generator(cfg)
    pool: Dict[str, Tuple[Any, Callable]] = {}

    reproducibility_cfg = cfg.get("reproducibility", {})
    seed_cfg = reproducibility_cfg.get("seed", cfg.get("seed", {}))
    global_seed = seed_cfg.get("global_seed", 42)

    for task_id in task_ids:
        n_customers, n_fleets, dist = _parse_task_id(task_id)

        # Create generic environment instance (instance-specific params set in reset)
        env = build_environment(cfg)

        # Deterministic RNG per task
        task_seed = global_seed + hash((n_customers, n_fleets, dist)) % (2**31 - 1)
        _rng = np.random.default_rng(task_seed)

        # Closure captures task parameters and RNG
        def _gen(n=n_customers, d=dist, nf=n_fleets, rng=_rng) -> Dict[str, Any]:
            return base_gen(size=n, dist=d, n_fleets=nf, rng=rng)

        pool[task_id] = (env, _gen)

    return pool


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def build_trainer(
    cfg: Dict[str, Any],
    agent: Any,
    env: Any,
    generator: Callable,
    evaluator: Any,
    logger: Any,
) -> BaseTrainer:
    """Build trainer from config.

    Dispatches to registry based on trainer type, then calls from_config.
    Builds task pool (all trainers train on multiple tasks sequentially).
    """
    trainer_cfg = cfg.get("trainer", {})
    trainer_type = trainer_cfg.get("name", "meta")

    if trainer_type not in _TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown trainer type {trainer_type!r}. "
            f"Register it in registry.py. Known: {list(_TRAINER_REGISTRY.keys())}"
        )

    cls = _TRAINER_REGISTRY[trainer_type]

    # Build task pool: all trainers use this (train multiple tasks sequentially)
    tasks = build_tasks(cfg)

    return cls.from_config(
        cfg=cfg,
        agent=agent,
        env=env,
        tasks=tasks,
        evaluator=evaluator,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Network / Policy
# ---------------------------------------------------------------------------


def build_policy(cfg: Dict[str, Any]) -> BasePolicy:
    """Build policy network from config.

    Dispatches to registry based on policy name, then calls from_config.
    """
    policy_cfg = cfg.get("policy", cfg.get("network", {}))  # Support both old and new
    net_type = policy_cfg.get("name", policy_cfg.get("type", "hgnn"))

    if net_type not in _POLICY_REGISTRY:
        raise ValueError(
            f"Unknown network type {net_type!r}. "
            f"Register it in registry.py. Known: {list(_POLICY_REGISTRY.keys())}"
        )

    cls = _POLICY_REGISTRY[net_type]
    return cls.from_config(policy_cfg)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


def build_estimator(cfg: Dict[str, Any], estimator_name: Optional[str] = None) -> BaseEstimator:
    """Build estimator from config.

    Dispatches to registry based on estimator type, then calls from_config.
    """
    # Get estimator name from parameter or config
    if estimator_name is None:
        estimator_name = cfg.get("estimator", {}).get("name", "ppo")

    if estimator_name not in _ESTIMATOR_REGISTRY:
        raise ValueError(
            f"Unknown estimator type {estimator_name!r}. "
            f"Register it in registry.py. Known: {list(_ESTIMATOR_REGISTRY.keys())}"
        )

    cls = _ESTIMATOR_REGISTRY[estimator_name]
    return cls.from_config(cfg)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def build_agent(cfg: Dict[str, Any]) -> PolicyAgent:
    """Build agent from config.

    Reads agent configuration from trainer config, then composes policy,
    estimator, optimizer, and agent via their from_config methods.
    """
    device = cfg.get("device", "cpu")

    # Get agent config from trainer (supports both meta and pomo)
    trainer_cfg = cfg.get("trainer", {})
    agent_cfg = trainer_cfg.get("agent", trainer_cfg.get("meta_learning", {}).get("meta_agent", {}))

    # Build policy
    policy = build_policy(cfg)
    policy = policy.to_device(device)

    # Build estimator using the estimator name from agent config
    estimator_name = agent_cfg.get("estimator", "ppo")
    estimator = build_estimator(cfg, estimator_name=estimator_name)

    # Build optimizer
    opt_type = agent_cfg.get("optimizer", "adam")
    opt_lr = agent_cfg.get("learning_rate", 0.001)

    if opt_type not in _OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer type {opt_type!r}. "
            f"Register it in registry.py. Known: {list(_OPTIMIZER_REGISTRY.keys())}"
        )

    opt_class = _OPTIMIZER_REGISTRY[opt_type]
    optimizer = opt_class(policy.parameters(), lr=opt_lr)

    # Build agent
    agent_type = agent_cfg.get("name", "policy")
    if agent_type not in _AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent type {agent_type!r}. "
            f"Register it in registry.py. Known: {list(_AGENT_REGISTRY.keys())}"
        )

    cls = _AGENT_REGISTRY[agent_type]
    return cls.from_config(cfg, policy, estimator, optimizer)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def build_environment(cfg: Dict[str, Any]) -> VRPBTWEnv:
    """Build environment from config.

    Dispatches to registry based on environment name, then calls from_config.
    Instance-specific parameters are determined dynamically in reset() via raw_instance.
    """
    # Support hierarchical config structure
    env_cfg = cfg.get("environment", cfg)
    env_name = env_cfg.get("name", "vrpbtw")

    if env_name not in _ENVIRONMENT_REGISTRY:
        raise ValueError(
            f"Unknown environment {env_name!r}. "
            f"Register it in registry.py. Known: {list(_ENVIRONMENT_REGISTRY.keys())}"
        )

    cls = _ENVIRONMENT_REGISTRY[env_name]
    return cls.from_config(env_cfg)
