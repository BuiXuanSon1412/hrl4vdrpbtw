"""
config.py
---------
Experiment configuration: dataclasses + YAML/JSON serialisation.

Structure
---------
Each component has its own dataclass.  ExperimentConfig composes them.
YAML is the preferred hand-edited format; JSON is used for machine-written
checkpoint configs.  Both directions are supported transparently via the
file extension.

Ablation / multi-file workflow
------------------------------
For ablation studies, keep one base config and small override files:

    python train.py --config configs/base.yaml --override configs/ablations/no_curriculum.yaml

merge_configs(base_path, override_path) loads and deep-merges two files.
Only keys present in the override file are changed; everything else keeps
the base value.  This is cleaner than duplicating full config files.

Dependencies
------------
  pyyaml   pip install pyyaml
  json     stdlib
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Component dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SeedConfig:
    global_seed: int = 42
    env_seed: Optional[int] = None  # None → derived from global_seed
    data_seed: Optional[int] = None  # None → derived from global_seed


@dataclass
class EnvironmentConfig:
    problem_name: str = "vrpbtw"
    problem_kwargs: Dict[str, Any] = field(default_factory=dict)
    max_steps: Optional[int] = None
    reward_scale: float = 1.0
    subtract_baseline: bool = False
    dense_shaping: bool = True


@dataclass
class NetworkConfig:
    network_type: str = "hacn"
    embed_dim: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    dropout: float = 0.0
    clip_logits: float = 10.0
    ortho_init: bool = True
    use_instance_norm: bool = True
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])


@dataclass
class PPOConfig:
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
    target_kl: Optional[float] = 0.015
    normalize_advantages: bool = True
    normalize_rewards: bool = True
    reward_norm_eps: float = 1e-8


@dataclass
class DQNConfig:
    lr: float = 1e-4
    gamma: float = 0.99
    buffer_capacity: int = 100_000
    batch_size: int = 64
    target_update_freq: int = 500
    tau: float = 1.0
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


@dataclass
class TrainConfig:
    total_timesteps: int = 1_000_000
    log_interval: int = 10
    eval_interval: int = 50
    checkpoint_interval: int = 250
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    patience: int = 100
    min_delta: float = 1e-4
    n_eval_episodes: int = 20
    eval_deterministic: bool = True
    eval_beam_width: int = 1
    curriculum: bool = False
    curriculum_start: int = 5
    curriculum_end: int = 50
    curriculum_steps: int = 500_000


@dataclass
class ExperimentConfig:
    """Single object capturing a complete, reproducible experiment."""

    name: str = "experiment"
    algorithm: str = "ppo"
    device: str = "cpu"
    seed: SeedConfig = field(default_factory=SeedConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def config_to_dict(cfg: Any) -> Dict:
    return dataclasses.asdict(cfg)


def _from_dict(d: Dict, cls: type) -> Any:
    """
    Reconstruct a dataclass from a plain dict.
    Unknown keys are silently ignored → forward-compatible with new fields.
    """
    import typing

    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: Dict[str, Any] = {}

    for name, f in fields.items():
        if name in d:
            val = d[name]
        elif f.default is not dataclasses.MISSING:
            val = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            val = f.default_factory()  # type: ignore[misc]
        else:
            raise ValueError(
                f"Required field {name!r} missing in {cls.__name__} and has no default."
            )
        # Recurse into nested dataclasses
        actual = hints.get(name)
        if (
            actual is not None
            and isinstance(actual, type)
            and dataclasses.is_dataclass(actual)
            and isinstance(val, dict)
        ):
            val = _from_dict(val, actual)

        kwargs[name] = val

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _infer_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".json":
        return "json"
    raise ValueError(
        f"Cannot infer format from extension {suffix!r}.  Use .yaml, .yml, or .json."
    )


def _load_yaml(path: str) -> Dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required.  Install with:  pip install pyyaml"
        ) from exc
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _dump_yaml(data: Dict, path: str) -> None:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required.  Install with:  pip install pyyaml"
        ) from exc
    with open(path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False, indent=2)


def load_config(path: str) -> ExperimentConfig:
    """
    Load ExperimentConfig from a YAML or JSON file.
    Missing fields fall back to dataclass defaults (forward-compatible).
    """
    fmt = _infer_format(path)
    data = _load_yaml(path) if fmt == "yaml" else json.loads(Path(path).read_text())
    return _from_dict(data, ExperimentConfig)


def save_config(cfg: ExperimentConfig, path: str) -> None:
    """
    Save config to YAML or JSON.
    YAML output omits the unused algorithm block (dqn when ppo, etc.)
    to keep hand-edited files clean.  JSON always saves the full schema.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fmt = _infer_format(path)
    data = config_to_dict(cfg)

    if fmt == "yaml":
        if cfg.algorithm == "ppo":
            data.pop("dqn", None)
        elif cfg.algorithm == "dqn":
            data.pop("ppo", None)
        _dump_yaml(data, path)
    else:
        Path(path).write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Ablation helper: deep-merge two config dicts
# ---------------------------------------------------------------------------


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge override into base.
    Only keys present in override are modified; base keys not in override
    are preserved unchanged.
    """
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def merge_configs(
    base_path: str,
    override_path: str,
) -> ExperimentConfig:
    """
    Load a base config and apply an override file on top of it.

    Useful for ablation studies:
        merge_configs("configs/base.yaml", "configs/ablations/no_curriculum.yaml")

    The override file only needs to contain the keys you want to change.
    """
    fmt_b = _infer_format(base_path)
    fmt_o = _infer_format(override_path)

    base_data = (
        _load_yaml(base_path)
        if fmt_b == "yaml"
        else json.loads(Path(base_path).read_text())
    )
    override_data = (
        _load_yaml(override_path)
        if fmt_o == "yaml"
        else json.loads(Path(override_path).read_text())
    )

    merged = _deep_merge(base_data, override_data)
    return _from_dict(merged, ExperimentConfig)
