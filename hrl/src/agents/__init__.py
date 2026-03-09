"""
src/agents/__init__.py
─────────────────────────────────────────────────────────────────────────────
Agent registry — maps config string → agent class.
Add a new agent class here to make it accessible from the CLI.
"""

from .base import BaseAgent
from .hierarchical import HierarchicalAgent

AGENT_REGISTRY: dict = {
    "hierarchical": HierarchicalAgent,
}


def build_agent(cfg, env) -> BaseAgent:
    """Instantiate an agent from a config dict/object."""
    agent_type = cfg.agent.type
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent type '{agent_type}'. "
            f"Available: {list(AGENT_REGISTRY.keys())}"
        )
    return AGENT_REGISTRY[agent_type](env=env, cfg=cfg)
