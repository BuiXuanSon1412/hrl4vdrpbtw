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

from config import load_config, merge_configs, save_config
from core import (
    Evaluator,
    Logger,
    SeedManager,
)
from registry import (
    build_agent,
    build_environment,
    build_trainer,
    get_vrpbtw_generator,
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
        global_seed=seed_cfg.get("global_seed", 42),
        env_seed=seed_cfg.get("env_seed"),
        data_seed=seed_cfg.get("data_seed"),
    )
    seed_mgr.seed_everything()
    data_rng = seed_mgr.make_data_rng()
    print(f"  {seed_mgr}")

    # ── 3. Logger ───────────────────────────────────────────────────────
    training_cfg = cfg.get("training", cfg.get("train", {}))
    train_logging = training_cfg.get("logging", {})

    # Create experiment-specific output directories
    base_checkpoint_dir = train_logging.get("checkpoint_dir", "experiment/train")
    checkpoint_dir = Path(base_checkpoint_dir) / exp_name
    log_dir = checkpoint_dir / "logs"
    tensorboard_dir = checkpoint_dir / "tensorboard"

    # Ensure directories exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(
        log_dir=str(log_dir),
        experiment_name=exp_name,
        verbose=True,
    )

    # ── 3b. Save config snapshot ────────────────────────────────────────
    config_path = checkpoint_dir / f"{exp_name}_config.yaml"
    save_config(cfg, str(config_path))
    print(f"  Config saved: {config_path}")

    # ── 4. Build components using registry pattern ──────────────────────
    # Build environment
    env = build_environment(cfg)
    print(f"  Environment: {type(env).__name__}")

    # Build agent (composes policy + estimator)
    agent = build_agent(cfg=cfg)
    print(f"  Agent      : {agent}\n")

    # Create instance generator for training/evaluation
    generator = get_vrpbtw_generator(cfg)

    # Build evaluator
    eval_cfg = training_cfg.get("evaluation", cfg.get("evaluation", {}))
    eval_decoding = eval_cfg.get("decoding", {})
    n_eval_episodes = eval_cfg.get("n_eval_episodes", eval_cfg.get("n_episodes", 20))
    deterministic = eval_cfg.get("deterministic", True)
    beam_width = eval_decoding.get("beam_width", 1)

    evaluator = Evaluator(
        agent=agent,
        env=env,
        n_episodes=n_eval_episodes,
        deterministic=deterministic,
        beam_width=beam_width,
    )

    # Build trainer using factory pattern (dispatched by cfg.trainer)
    trainer = build_trainer(
        cfg=cfg,
        agent=agent,
        env=env,
        generator=generator,
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
