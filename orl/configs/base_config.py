"""
configs/base_config.py
----------------------
Centralised, fully-serialisable experiment configuration.

Design principles
-----------------
- Every hyperparameter lives in exactly ONE dataclass.
- Configs are plain Python dataclasses so they serialise to/from JSON/YAML
  with zero external dependencies.
- NetworkConfig owns architecture; AlgorithmConfig owns learning;
  TrainConfig owns loop logistics.  They are never mixed.
- The top-level ExperimentConfig is the single object passed to Trainer.
  Passing it to `agent.save()` lets checkpoints be self-contained.

Usage
-----
from configs.base_config import ExperimentConfig, load_config, save_config

cfg = ExperimentConfig()            # sensible defaults
cfg = load_config("cfg.json")       # from file
save_config(cfg, "cfg.json")        # to file
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Seed / reproducibility
# ---------------------------------------------------------------------------


@dataclass
class SeedConfig:
    """Reproducibility settings."""

    global_seed: int = 42
    env_seed: Optional[int] = None  # None → derive from global_seed
    data_seed: Optional[int] = None  # None → derive from global_seed


# ---------------------------------------------------------------------------
# Problem / environment
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    """Environment and problem settings."""

    problem_name: str = "knapsack"  # registered name in ENV_REGISTRY
    problem_kwargs: Dict[str, Any] = field(default_factory=dict)
    max_steps: Optional[int] = None  # hard episode truncation
    reward_scale: float = 1.0
    subtract_baseline: bool = False
    dense_shaping: bool = True


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------


@dataclass
class NetworkConfig:
    """
    Neural network architecture hyperparameters.

    These describe the MODEL.  No algorithm-specific fields belong here.
    """

    network_type: str = "attention"  # "attention" | "pointer" | "mlp"
    embed_dim: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    dropout: float = 0.0
    clip_logits: float = 10.0
    ortho_init: bool = True
    use_instance_norm: bool = True  # True=InstanceNorm (paper), False=LayerNorm
    # MLP-only
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])


# ---------------------------------------------------------------------------
# Algorithm hyperparameters — one per algorithm, no mixing
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """
    PPO-specific hyperparameters only.

    No network fields.  No environment fields.
    """

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    mini_batch_size: int = 256
    rollout_len: int = 2048
    target_kl: Optional[float] = 0.015  # None → disable KL early-stop
    normalize_advantages: bool = True
    normalize_rewards: bool = True
    reward_norm_eps: float = 1e-8


@dataclass
class DQNConfig:
    """DQN-specific hyperparameters only."""

    lr: float = 1e-4
    gamma: float = 0.99
    buffer_capacity: int = 100_000
    batch_size: int = 64
    target_update_freq: int = 500
    tau: float = 1.0  # 1.0 = hard copy; <1.0 = soft
    train_freq: int = 4
    learning_starts: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 100_000


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """
    Training loop logistics — completely independent of the algorithm.
    """

    total_timesteps: int = 1_000_000
    log_interval: int = 10  # log every N iterations
    eval_interval: int = 50  # evaluate every N iterations
    checkpoint_interval: int = 250  # save periodic snapshot every N iters
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    # Early stopping
    patience: int = 100  # evals without improvement
    min_delta: float = 1e-4
    # Evaluation
    n_eval_episodes: int = 20
    eval_deterministic: bool = True
    eval_beam_width: int = 1
    # Curriculum
    curriculum: bool = False
    curriculum_start: int = 5
    curriculum_end: int = 50
    curriculum_steps: int = 500_000


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """
    Single object capturing a complete, reproducible experiment.

    Pass this to Trainer.  Save it alongside checkpoints.
    """

    name: str = "experiment"
    algorithm: str = "ppo"  # "ppo" | "dqn"
    seed: SeedConfig = field(default_factory=SeedConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def config_to_dict(cfg: Any) -> Dict:
    """Recursively convert dataclass to plain dict."""
    return dataclasses.asdict(cfg)


def config_from_dict(d: Dict, cls) -> Any:
    """Recursively reconstruct dataclass from plain dict."""
    import typing

    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        val = d.get(
            f.name,
            f.default if f.default is not dataclasses.MISSING else f.default_factory(),
        )
        actual_type = hints.get(f.name, None)
        if (
            actual_type
            and dataclasses.is_dataclass(actual_type)
            and isinstance(val, dict)
        ):
            val = config_from_dict(val, actual_type)
        kwargs[f.name] = val
    return cls(**kwargs)


def save_config(cfg: ExperimentConfig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(config_to_dict(cfg), fh, indent=2)


def load_config(path: str) -> ExperimentConfig:
    with open(path) as fh:
        d = json.load(fh)
    return config_from_dict(d, ExperimentConfig)
