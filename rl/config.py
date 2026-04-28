"""
config.py
---------
Simple configuration loading from YAML/JSON.

Usage
-----
  cfg = load_config("configs/base.yaml")
  cfg = merge_configs("configs/base.yaml", "configs/ablations/override.yaml")

  Access via dict syntax:
    cfg["algorithm"]
    cfg["seed"]["global_seed"]
    cfg["maml"]["task_sizes"]
"""

import json
from pathlib import Path
from typing import Any, Dict


def _infer_format(path: str) -> str:
    """Infer file format from extension."""
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".json":
        return "json"
    raise ValueError(
        f"Cannot infer format from extension {suffix!r}.  Use .yaml, .yml, or .json."
    )


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML file."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("pyyaml is required.  Install with: pip install pyyaml") from exc
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    return json.loads(Path(path).read_text())


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Supports _include key to load and merge component configs:
        _include:
          - trainer: configs/trainer/pomo.yaml
          - policy: configs/policy/hgnn.yaml
          - environment: configs/environment/vrpbtw.yaml

    Included configs are merged in order, then main config merged on top.

    Args:
        path: path to config file (.yaml or .json)

    Returns:
        dict with all configuration keys
    """
    fmt = _infer_format(path)
    cfg = _load_yaml(path) if fmt == "yaml" else _load_json(path)

    # Process _include directives
    if "_include" in cfg:
        includes = cfg.pop("_include")
        if not isinstance(includes, list):
            includes = [includes]

        result = {}
        for inc in includes:
            if isinstance(inc, dict):
                for component_name, component_path in inc.items():
                    component_path = Path(path).parent / component_path
                    component_cfg = load_config(str(component_path))
                    result = _deep_merge(result, component_cfg)
            elif isinstance(inc, str):
                component_path = Path(path).parent / inc
                component_cfg = load_config(str(component_path))
                result = _deep_merge(result, component_cfg)

        cfg = _deep_merge(result, cfg)

    return cfg


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def merge_configs(base_path: str, override_path: str) -> Dict[str, Any]:
    """
    Load base config and deep-merge override on top.

    Useful for ablation studies: only keys in override are changed.

    Args:
        base_path: path to base config
        override_path: path to override config

    Returns:
        merged dict
    """
    base = load_config(base_path)
    override = load_config(override_path)
    return _deep_merge(base, override)


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """Save config dict to YAML or JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fmt = _infer_format(path)

    if fmt == "yaml":
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("pyyaml is required.  Install with: pip install pyyaml") from exc
        with open(path, "w") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False, indent=2)
    else:
        Path(path).write_text(json.dumps(cfg, indent=2))
