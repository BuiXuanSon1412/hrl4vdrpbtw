"""
train_debug.py - Debug version of train.py with verbose output to trace hangs.

Run this instead of train.py to identify where the process is hanging.
Each major phase prints a timestamp and checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional, Dict

import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, merge_configs, save_config
from core import (
    Evaluator,
    FineTuner,
    Logger,
    MetaTrainer,
    SeedManager,
)
from core.task import TaskManager, SimpleTask
from registry import (
    build_agent,
    build_problem,
    build_task_pool,
    get_generator,
    sort_task_ids,
)


def log_checkpoint(msg: str) -> None:
    """Print timestamped debug message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def main() -> None:
    log_checkpoint("=== TRAINING START ===")
    args = _build_parser().parse_args()

    # ── 1. Config ───────────────────────────────────────────────────────
    log_checkpoint(f"Loading config from {args.config}")
    cfg = (
        merge_configs(args.config, args.override)
        if args.override
        else load_config(args.config)
    )
    log_checkpoint(f"Config loaded. Keys: {list(cfg.keys())}")

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

    exp_name = cfg.get("name") or cfg.get("experiment", {}).get("name", "experiment")
    log_checkpoint(f"Experiment: {exp_name}")

    # ── 2. Reproducibility ──────────────────────────────────────────────
    log_checkpoint("Setting up reproducibility...")
    reproducibility_cfg = cfg.get("reproducibility", {})
    seed_cfg = reproducibility_cfg.get("seed", cfg.get("seed", {}))
    seed_mgr = SeedManager(
        global_seed=seed_cfg.get("global_seed", 42),
        env_seed=seed_cfg.get("env_seed"),
        data_seed=seed_cfg.get("data_seed"),
    )
    seed_mgr.seed_everything()
    data_rng = seed_mgr.make_data_rng()
    log_checkpoint(f"Reproducibility set: {seed_mgr}")

    # ── 3. Logger ───────────────────────────────────────────────────────
    log_checkpoint("Setting up logger...")
    training_cfg = cfg.get("training", cfg.get("train", {}))
    train_logging = training_cfg.get("logging", {})

    base_checkpoint_dir = train_logging.get("checkpoint_dir", "experiment/train")
    checkpoint_dir = Path(base_checkpoint_dir) / exp_name
    log_dir = checkpoint_dir / "logs"
    tensorboard_dir = checkpoint_dir / "tensorboard"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(
        log_dir=str(log_dir),
        experiment_name=exp_name,
        verbose=True,
    )
    log_checkpoint(f"Logger initialized. Log dir: {log_dir}")

    # ── 3b. Save config snapshot ────────────────────────────────────────
    config_path = checkpoint_dir / f"{exp_name}_config.yaml"
    save_config(cfg, str(config_path))
    log_checkpoint(f"Config saved to {config_path}")

    # ── 4. Build components ─────────────────────────────────────────────
    algo_name = cfg.get("algorithm", "")
    if isinstance(algo_name, dict):
        algo_name = algo_name.get("name", "").upper()
    else:
        algo_name = algo_name.upper()

    algo_name_lower = algo_name.lower()

    if algo_name_lower == "maml":
        log_checkpoint("Building task pool...")
        task_pool = build_task_pool(cfg)
        log_checkpoint(f"Task pool built: {list(task_pool.keys())}")

        log_checkpoint("Creating TaskManager...")
        tasks = []
        for task_id in sorted(task_pool.keys()):
            problem, gen = task_pool[task_id]
            task = SimpleTask(task_id=task_id, problem=problem, generator=gen)
            tasks.append(task)
        task_manager = TaskManager(tasks)
        log_checkpoint(f"TaskManager created with {len(tasks)} tasks")

        log_checkpoint("Selecting evaluation anchor task...")
        eval_task_ids = sort_task_ids(list(task_pool.keys()))
        eval_task_id = eval_task_ids[len(eval_task_ids) // 2]
        eval_problem, eval_gen = task_pool[eval_task_id]
        log_checkpoint(f"Eval anchor: {eval_task_id}")

        log_checkpoint("Building agent...")
        agent = build_agent(cfg=cfg)
        log_checkpoint(f"Agent built: {type(agent).__name__}")

        log_checkpoint("Building evaluator...")
        eval_cfg = training_cfg.get("evaluation", cfg.get("evaluation", {}))
        eval_decoding = eval_cfg.get("decoding", {})
        n_episodes = eval_cfg.get("n_episodes", 20)
        deterministic = eval_cfg.get("deterministic", True)
        beam_width = eval_decoding.get("beam_width", 1)

        evaluator = Evaluator(
            agent=agent,
            env=eval_problem,
            n_episodes=n_episodes,
            deterministic=deterministic,
            beam_width=beam_width,
        )
        log_checkpoint("Evaluator built")

        log_checkpoint("Building MetaTrainer...")
        trainer = MetaTrainer(
            agent=agent,
            task_manager=task_manager,
            eval_problem=eval_problem,
            eval_gen=eval_gen,
            cfg=cfg,
            evaluator=evaluator,
            logger=logger,
        )
        log_checkpoint("MetaTrainer built")

        # Phase 1: Meta-learning
        log_checkpoint("=" * 64)
        log_checkpoint("STARTING PHASE 1: META-LEARNING")
        log_checkpoint("=" * 64)
        phase1_summary = trainer.train()
        log_checkpoint("Phase 1 complete")

        best_ckpt_path = checkpoint_dir / f"{exp_name}_best.pt"
        try:
            agent.load(str(best_ckpt_path))
            log_checkpoint(f"Loaded best checkpoint: {best_ckpt_path}")
        except Exception as exc:
            log_checkpoint(f"Could not load best checkpoint ({exc}); using final weights.")

        log_checkpoint("=" * 64)
        log_checkpoint("TRAINING COMPLETE")
        log_checkpoint("=" * 64)


if __name__ == "__main__":
    main()
