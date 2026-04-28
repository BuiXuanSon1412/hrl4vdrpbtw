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

from pathlib import Path
from typing import Any, Dict
import torch.optim as optim

# Problems
from impl.environment import VRPBTWEnv

# Networks
from impl.policy import HGNNPolicy

# Core
from core.agent import PolicyAgent
from core.policy import BasePolicy
from core.trainer import BaseTrainer, MetaTrainer, POMOTrainer
from core.evaluator import Evaluator
from core.logger import Logger

# ---------------------------------------------------------------------------
# Registry: factory method dispatch tables
# ---------------------------------------------------------------------------

_POLICY_REGISTRY: Dict[str, type] = {
    "hgnn": HGNNPolicy,
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

_OPTIMIZER_REGISTRY: Dict[str, type | None] = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
    "unspecified": None,
}


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


def build_logger(cfg: Dict[str, Any]) -> Logger:
    """Build logger from config.

    Reads logger settings from logger section:
    - base_dir: base directory for experiments (e.g., experiment/train)
    - verbose: print to console
    - tensorboard: enable TensorBoard
    """
    exp_cfg = cfg.get("experiment", {})
    logger_cfg = cfg.get("logger", {})

    exp_name = exp_cfg.get("name", "experiment")
    base_dir = logger_cfg.get("base_dir", "experiment/train")
    exp_dir = str(Path(base_dir) / exp_name)

    return Logger(
        dir=exp_dir,
        verbose=logger_cfg.get("verbose", True),
        tensorboard=logger_cfg.get("tensorboard", False),
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def build_evaluator(cfg: Dict[str, Any], agent: Any, env: Any) -> Evaluator:
    """Build evaluator from config.

    Reads evaluation settings from trainer.control.evaluation.
    """
    trainer_cfg = cfg.get("trainer", {})
    control_cfg = trainer_cfg.get("control", {})
    eval_cfg = control_cfg.get("evaluation", {})

    return Evaluator(
        agent=agent,
        env=env,
        n_episodes=eval_cfg.get("n_episodes", 20),
        deterministic=eval_cfg.get("deterministic", True),
        beam_width=eval_cfg.get("decoding", {}).get("beam_width", 1),
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def build_trainer(
    cfg: Dict[str, Any],
    agents: Dict[str, Any],
    env: Any,
    evaluator: Any,
    logger: Any,
) -> BaseTrainer:
    """Build trainer from config.

    Dispatches to registry based on trainer type, then calls from_config.
    """
    trainer_cfg = cfg.get("trainer", {})
    trainer_type = trainer_cfg.get("name", "meta")

    if trainer_type not in _TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown trainer type {trainer_type!r}. "
            f"Register it in registry.py. Known: {list(_TRAINER_REGISTRY.keys())}"
        )

    cls = _TRAINER_REGISTRY[trainer_type]

    return cls.from_config(
        trainer_cfg=trainer_cfg,
        agents=agents,
        env=env,
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
    net_type = policy_cfg.get("name", "hgnn")

    if net_type not in _POLICY_REGISTRY:
        raise ValueError(
            f"Unknown network type {net_type!r}. "
            f"Register it in registry.py. Known: {list(_POLICY_REGISTRY.keys())}"
        )

    cls = _POLICY_REGISTRY[net_type]
    return cls.from_config(policy_cfg)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def build_agent(cfg: Dict[str, Any]) -> Dict[str, "PolicyAgent"]:
    """Build agent(s) from config.

    Iterates through all phases in trainer.phase and builds agents from each,
    merging into single dict. Agent keys must be unique across phases.

    Returns dict of agents keyed by agent name.
    """
    trainer_cfg = cfg.get("trainer", {})
    agents = {}

    # Iterate through all phases
    phase_cfg = trainer_cfg.get("phase", {})
    for phase_name, phase_data in phase_cfg.items():
        agents_cfg = phase_data.get("agents", {})

        for agent_name, agent_cfg in agents_cfg.items():
            # Build policy
            policy = build_policy(cfg)

            # Build optimizer with agent-specific learning rate
            opt_type = agent_cfg.get("optimizer", "adam")
            opt_lr = agent_cfg.get("learning_rate", 0.001)

            if opt_type not in _OPTIMIZER_REGISTRY:
                raise ValueError(
                    f"Unknown optimizer type {opt_type!r}. "
                    f"Register it in registry.py. Known: {list(_OPTIMIZER_REGISTRY.keys())}"
                )

            opt_class = _OPTIMIZER_REGISTRY[opt_type]
            optimizer = (
                None if opt_class is None else opt_class(policy.parameters(), lr=opt_lr)
            )

            # Build agent
            agent_type = agent_cfg.get("name", "policy")
            if agent_type not in _AGENT_REGISTRY:
                raise ValueError(
                    f"Unknown agent type {agent_type!r}. "
                    f"Register it in registry.py. Known: {list(_AGENT_REGISTRY.keys())}"
                )

            cls = _AGENT_REGISTRY[agent_type]
            agents[agent_name] = cls.from_config(agent_cfg, policy, optimizer)

    return agents


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def build_environment(cfg: Dict[str, Any]) -> VRPBTWEnv:
    """Build environment from config.

    Dispatches to registry based on environment name, then calls from_config.
    Instance-specific parameters are determined dynamically in reset() via raw_instance.
    """
    # Support hierarchical config structure
    env_cfg = cfg.get("environment", {})
    env_name = env_cfg.get("name", "vrpbtw")

    if env_name not in _ENVIRONMENT_REGISTRY:
        raise ValueError(
            f"Unknown environment {env_name!r}. "
            f"Register it in registry.py. Known: {list(_ENVIRONMENT_REGISTRY.keys())}"
        )

    cls = _ENVIRONMENT_REGISTRY[env_name]
    return cls.from_config(env_cfg)
