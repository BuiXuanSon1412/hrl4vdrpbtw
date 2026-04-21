"""
strain.py
---------
Single-environment training entry point.

Trains a single policy on one environment without meta-learning.
Hyperparameters live in configs/strain.yaml.
Outputs saved to: experiment/train/{experiment.name}/

CLI flags (runtime only)
------------------------
  --config   PATH    Config file               [default: configs/strain.yaml]
  --override PATH    Config override file      [optional]
  --device   DEVICE  Override reproducibility.device
  --name     NAME    Override experiment.name

Usage
-----
  python strain.py
  python strain.py --device cuda --name my_single_task
  python strain.py --override overrides/strain_100nodes.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, merge_configs, save_config
from core import (
    Evaluator,
    Logger,
    Trainer,
    SeedManager,
)
from registry import build_agent, build_problem, get_generator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-environment RL training — all hyperparameters in --config YAML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="configs/strain.yaml",
        metavar="PATH",
        help="Training config file (default: configs/strain.yaml).",
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
        help="PyTorch device. Overrides reproducibility.device.",
    )
    p.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="Experiment name. Overrides experiment.name.",
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
    # Apply CLI overrides
    if args.device:
        cfg["device"] = args.device
    if args.name:
        cfg["name"] = args.name
        if "experiment" not in cfg:
            cfg["experiment"] = {}
        cfg["experiment"]["name"] = args.name

    print(
        f"\n  Config     : {args.config}"
        + (f"  +  {args.override}" if args.override else "")
    )
    exp_name = cfg.get("name") or cfg.get("experiment", {}).get("name", "experiment")
    net_type = cfg.get("network", {}).get("type", "hgnn")
    device = cfg.get("device", "cpu")
    print(f"  Experiment : {exp_name}")
    print(f"  Network    : {net_type}  |  Device: {device}")

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

    base_checkpoint_dir = train_logging.get("checkpoint_dir", "experiment/train")
    checkpoint_dir = Path(base_checkpoint_dir) / exp_name
    log_dir = checkpoint_dir / "logs"

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

    # ── 4. Build environment + agent ─────────────────────────────────
    problem = build_problem(cfg)
    generator = get_generator(cfg)

    agent = build_agent(cfg=cfg)
    print(f"  Agent      : {agent}\n")

    # ── 5. Build evaluator ──────────────────────────────────────────────
    eval_cfg = training_cfg.get("evaluation", cfg.get("evaluation", {}))
    eval_decoding = eval_cfg.get("decoding", {})
    n_episodes = eval_cfg.get("n_episodes", 20)
    deterministic = eval_cfg.get("deterministic", True)
    beam_width = eval_decoding.get("beam_width", 1)

    evaluator = Evaluator(
        agent=agent,
        env=problem,
        n_episodes=n_episodes,
        deterministic=deterministic,
        beam_width=beam_width,
    )

    # ── 6. Create trainer and run ────────────────────────────────────
    trainer = Trainer(
        agent=agent,
        env=problem,
        generator=generator,
        cfg=cfg,
        evaluator=evaluator,
        logger=logger,
    )

    summary = trainer.train()

    # ── 7. Load best checkpoint and final eval ──────────────────────
    best_ckpt_path = checkpoint_dir / f"{exp_name}_best.pt"
    try:
        agent.load(str(best_ckpt_path))
        print(f"\nLoaded best checkpoint: {best_ckpt_path}")
    except Exception as exc:
        print(f"Could not load best checkpoint ({exc}); using final weights.")

    final_eval = evaluator.evaluate(generator)
    _print_eval(final_eval, label="Final Evaluation")


# ---------------------------------------------------------------------------
# Shared display helper
# ---------------------------------------------------------------------------


def _print_eval(stats: dict, label: str = "Evaluation") -> None:
    print(f"\n{label}:")
    for k, v in stats.items():
        print(f"  {k:<28}: {v:.4f}" if isinstance(v, float) else f"  {k:<28}: {v}")


if __name__ == "__main__":
    main()
