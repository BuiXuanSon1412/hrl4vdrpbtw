"""
agents/registry.py
------------------
Factory: build the right agent from config.

The factory receives shapes from the problem (source of truth),
builds the network first, then injects it into the agent.

Adding a new algorithm
----------------------
1. Implement the algorithm in algorithms/
2. Create a thin agent shell in agents/
3. Register it here.
"""

from __future__ import annotations

from configs import ExperimentConfig
from networks.registry import build_network
from .ppo_agent import PPOAgent
from .dqn_agent import DQNAgent
from .base_agent import BaseAgent


def build_agent(
    obs_shape,
    action_space_size: int,
    cfg: ExperimentConfig,
) -> BaseAgent:
    """
    Build a fully-configured agent.

    Parameters
    ----------
    obs_shape        : problem.observation_shape
    action_space_size: problem.action_space_size
    cfg              : Full ExperimentConfig.

    Returns
    -------
    Configured agent, with network on the correct device.
    """
    network = build_network(obs_shape, action_space_size, cfg.network)
    network = network.to_device(cfg.device)

    algo = cfg.algorithm.lower()

    if algo == "ppo":
        return PPOAgent(
            network=network,
            obs_shape=obs_shape,
            action_space_size=action_space_size,
            cfg=cfg.ppo,
            device=cfg.device,
        )

    if algo == "dqn":
        return DQNAgent(
            network=network,
            obs_shape=obs_shape,
            action_space_size=action_space_size,
            cfg=cfg.dqn,
            device=cfg.device,
        )

    raise ValueError(f"Unknown algorithm={algo!r}. Available: 'ppo', 'dqn'.")
