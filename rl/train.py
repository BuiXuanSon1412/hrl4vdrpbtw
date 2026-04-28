"""
train.py
--------
Training entry point.

All hyperparameters live in configs/train.yaml (edit this file for ablations).
Outputs are saved to: experiment/train/{experiment.name}/

CLI flags  (runtime only — not hyperparameters)
-----------------------------------------------
  --config   PATH    Config file               [default: configs/train.yaml]
  --override PATH    Config override file      [optional]
  --device   DEVICE  Override reproducibility.device
  --name     NAME    Override experiment.name
  --beam     WIDTH   Override evaluation.decoding.beam_width

Usage
-----
  # Standard run (reads configs/train.yaml, outputs to experiment/train/{name}/)
  python train.py

  # Custom config file
  python train.py --config configs/custom_train.yaml

  # Config override
  python train.py --override custom_override.yaml

  # GPU override without editing the file
  python train.py --device cuda --name my_experiment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import globals
from config import load_config, merge_configs, save_config
from core import SeedManager
from core.registry import (
    build_agent,
    build_environment,
    build_evaluator,
    build_logger,
    build_trainer,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RL training — all hyperparameters in --config YAML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="configs/train.yaml",
        metavar="PATH",
        help="Training config file (default: configs/train.yaml).",
    )
    p.add_argument(
        "--override",
        default=None,
        metavar="PATH",
        help="Config override file (merged on top of --config).",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="PyTorch device.  Overrides reproducibility.device.",
    )
    p.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="Experiment name.  Overrides experiment.name.",
    )
    p.add_argument(
        "--beam",
        default=None,
        type=int,
        metavar="WIDTH",
        help="Beam width.  Overrides evaluation.decoding.beam_width.",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _build_parser().parse_args()

    # ── 1. Config ───────────────────────────────────────────────────────
    cfg = (
        merge_configs(args.config, args.override)
        if args.override
        else load_config(args.config)
    )
    # Apply CLI overrides to hierarchical config
    if args.device:
        cfg["device"] = args.device

    # Set global device for all rl components
    globals.DEVICE = cfg.get("device", "cpu")
    if args.name:
        cfg["name"] = args.name
        if "experiment" not in cfg:
            cfg["experiment"] = {}
        cfg["experiment"]["name"] = args.name
    if args.beam:
        if "evaluation" not in cfg:
            cfg["evaluation"] = {}
        if "decoding" not in cfg["evaluation"]:
            cfg["evaluation"]["decoding"] = {}
        cfg["evaluation"]["decoding"]["beam_width"] = args.beam

    print(
        f"\n  Config     : {args.config}"
        + (f"  +  {args.override}" if args.override else "")
    )
    exp_name = cfg.get("name") or cfg.get("experiment", {}).get("name", "experiment")
    algo_name = cfg.get("algorithm", "")
    if isinstance(algo_name, dict):
        algo_name = algo_name.get("name", "").upper()
    else:
        algo_name = algo_name.upper()
    net_type = cfg.get("network", {}).get("type", "hgnn")
    device = cfg.get("device", "cpu")
    print(f"  Experiment : {exp_name}")
    print(f"  Algorithm  : {algo_name}  |  Network: {net_type}  |  Device: {device}")

    # ── 2. Reproducibility ──────────────────────────────────────────────
    reproducibility_cfg = cfg.get("reproducibility", {})
    seed_cfg = reproducibility_cfg.get("seed", cfg.get("seed", {}))
    seed_mgr = SeedManager(
        random_seed=seed_cfg.get("random_seed", 42),
        numpy_seed=seed_cfg.get("numpy_seed", 42),
        torch_seed=seed_cfg.get("torch_seed", 42),
    )
    seed_mgr.seed_everything()
    print(f"  {seed_mgr}")

    # ── 3. Initialize logger and save config ────────────────────────────
    logger = build_logger(cfg)

    # Save merged config
    logger.save_config(cfg)
    logger_base_dir = Path(logger.config_path).parent

    print(f"  Experiment dir: {logger_base_dir}")
    print(f"  Logs:           {logger.log_dir.relative_to(logger_base_dir)}")
    print(f"  Checkpoints:    {logger.checkpoint_dir.relative_to(logger_base_dir)}")
    print(f"  Config:         {logger.config_path.name}")
    print(f"  Artifacts:      {logger.artifacts_dir.relative_to(logger_base_dir)}")

    # ── 4. Build components using registry pattern ──────────────────────
    # Build environment
    env = build_environment(cfg)
    print(f"  Environment: {type(env).__name__}")

    # Build agents from config (returns dict keyed by agent name)
    agents = build_agent(cfg=cfg)
    print(f"  Agents     : {list(agents.keys())}\n")

    # Build evaluator using first available agent
    agent = agents["agent"]
    evaluator = build_evaluator(cfg, agent, env)

    # Build trainer using factory pattern (dispatched by cfg.trainer)
    trainer = build_trainer(
        cfg=cfg,
        agents=agents,
        env=env,
        evaluator=evaluator,
        logger=logger,
    )
    print(f"  Trainer    : {type(trainer).__name__}\n")

    # ── 5. Training ─────────────────────────────────────────────────────
    trainer.train()


# ---------------------------------------------------------------------------
# Shared display helper (also used by evaluate.py)
# ---------------------------------------------------------------------------


def _print_eval(stats: dict, label: str = "Evaluation") -> None:
    print(f"\n{label}:")
    for k, v in stats.items():
        print(f"  {k:<28}: {v:.4f}" if isinstance(v, float) else f"  {k:<28}: {v}")


if __name__ == "__main__":
    main()
