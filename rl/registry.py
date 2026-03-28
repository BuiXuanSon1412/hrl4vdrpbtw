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

from typing import Any, Callable, Tuple

from config import EnvironmentConfig, ExperimentConfig

# Problems
from problems.vrpbtw import VRPBTWProblem, generate_vrpbtw

# Networks
from networks.hacn import PolicyNetwork

# Core
from core.agent import BaseAgent, PPOAgent
from core.module import BaseNetwork


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


def build_problem(env_cfg: EnvironmentConfig) -> VRPBTWProblem:
    """Instantiate the correct Problem subclass from EnvConfig."""
    name = env_cfg.problem_name
    kwargs = dict(env_cfg.problem_kwargs)

    if name == "vrpbtw":
        return VRPBTWProblem(
            n_customers=kwargs.get("n_customers", 10),
            n_fleets=kwargs.get("n_fleets", 2),
        )

    raise ValueError(
        f"Unknown problem {name!r}.  Register it in registry.py.  Known: ['vrpbtw']"
    )


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------


def get_generator(env_cfg: EnvironmentConfig) -> Callable[..., Any]:
    """Return the raw-instance generator for the configured problem."""
    name = env_cfg.problem_name

    if name == "vrpbtw":
        return generate_vrpbtw

    raise ValueError(f"No generator for problem {name!r}.  Known: ['vrpbtw']")


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def build_network(
    obs_shape: Tuple[int, ...],
    cfg: ExperimentConfig,
    n_fleets: int,
) -> BaseNetwork:
    """
    Instantiate the policy network.

    n_fleets is supplied by the caller (who already has the built Problem)
    so we never re-parse problem_kwargs here.
    """
    net_type = cfg.network.network_type

    if net_type == "hacn":
        return PolicyNetwork(
            obs_shape=obs_shape,
            cfg=cfg.network,
            n_fleets=n_fleets,
        )

    raise ValueError(
        f"Unknown network type {net_type!r}.  "
        f"Register it in registry.py.  Known: ['hacn']"
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def build_agent(
    obs_shape: Tuple[int, ...],
    action_space_size: int,
    cfg: ExperimentConfig,
    n_fleets: int,
) -> BaseAgent:
    """
    Build the network, move it to the target device, wrap in an agent.
    """
    network = build_network(obs_shape, cfg, n_fleets)
    network = network.to_device(cfg.device)

    if cfg.algorithm == "ppo":
        return PPOAgent(
            network=network,
            obs_shape=obs_shape,
            action_space_size=action_space_size,
            cfg=cfg.ppo,
            device=cfg.device,
        )

    raise ValueError(
        f"Unknown algorithm {cfg.algorithm!r}.  "
        f"Register it in registry.py.  Known: ['ppo']"
    )
