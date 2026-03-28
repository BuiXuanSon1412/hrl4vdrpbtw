"""
train.py
--------
Training entry point.

All hyperparameters live in a YAML config file.
This script only wires services together.

CLI flags  (runtime only — not hyperparameters)
-----------------------------------------------
  --config   PATH    Base config file          [required]
  --override PATH    Ablation override file    [optional]
  --device   DEVICE  Override cfg.device
  --name     NAME    Override cfg.name
  --beam     WIDTH   Override cfg.train.eval_beam_width

Usage
-----
  # Standard run
  python train.py --config configs/base.yaml

  # Ablation
  python train.py --config configs/base.yaml \\
                  --override configs/ablations/no_curriculum.yaml

  # GPU override without editing the file
  python train.py --config configs/base.yaml --device cuda --name my_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, merge_configs, save_config, ExperimentConfig
from core import (
    Environment,
    Evaluator,
    Logger,
    OnPolicyTrainer,
    SeedManager,
)
from registry import build_agent, build_problem, get_generator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RL training — all hyperparameters in --config YAML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", required=True, metavar="PATH", help="Base ExperimentConfig YAML."
    )
    p.add_argument(
        "--override",
        default=None,
        metavar="PATH",
        help="Ablation override YAML (merged on top of --config).",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="PyTorch device.  Overrides cfg.device.",
    )
    p.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="Experiment name prefix.  Overrides cfg.name.",
    )
    p.add_argument(
        "--beam",
        default=None,
        type=int,
        metavar="WIDTH",
        help="Eval beam width.  Overrides cfg.train.eval_beam_width.",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _build_parser().parse_args()

    # ── 1. Config ───────────────────────────────────────────────────────
    cfg: ExperimentConfig = (
        merge_configs(args.config, args.override)
        if args.override
        else load_config(args.config)
    )
    if args.device:
        cfg.device = args.device
    if args.name:
        cfg.name = args.name
    if args.beam:
        cfg.train.eval_beam_width = args.beam

    print(
        f"\n  Config     : {args.config}"
        + (f"  +  {args.override}" if args.override else "")
    )
    print(f"  Experiment : {cfg.name}")
    print(
        f"  Algorithm  : {cfg.algorithm.upper()}  |  "
        f"Network: {cfg.network.network_type}  |  "
        f"Device: {cfg.device}"
    )

    # ── 2. Reproducibility ──────────────────────────────────────────────
    seed_mgr = SeedManager(
        global_seed=cfg.seed.global_seed,
        env_seed=cfg.seed.env_seed,
        data_seed=cfg.seed.data_seed,
    )
    seed_mgr.seed_everything()
    data_rng = seed_mgr.make_data_rng()
    print(f"  {seed_mgr}")

    # ── 3. Problem ──────────────────────────────────────────────────────
    problem = build_problem(cfg.env)
    base_gen = get_generator(cfg.env)

    dummy = base_gen(**{**cfg.env.problem_kwargs, "rng": seed_mgr.make_eval_rng()})
    problem.encode_instance(dummy)
    n_fleets: int = problem.n_fleets

    print(
        f"  Problem    : {problem}  |  "
        f"Obs: {problem.observation_shape}  |  "
        f"Actions: {problem.action_space_size}  |  "
        f"Fleets: {n_fleets}"
    )

    # ── 4. Environment ──────────────────────────────────────────────────
    env = Environment(problem, cfg.env)

    # ── 5. Instance generator ───────────────────────────────────────────
    def instance_generator(size: Optional[int] = None, **_: Any) -> Dict[str, Any]:
        kw = dict(cfg.env.problem_kwargs)
        if size is not None and cfg.env.problem_name == "vrpbtw":
            kw["n_customers"] = size
        kw["rng"] = data_rng

        raw: Dict[str, Any] = base_gen(**kw)  # safe default
        for _attempt in range(50):  # _attempt, not overwriting raw
            candidate: Dict[str, Any] = base_gen(**kw)
            problem.encode_instance(candidate)
            state = problem.initial_state()
            mask = problem.get_action_mask(state)
            if not mask.is_empty():
                raw = candidate
                break
        return raw

    # ── 6. Agent ────────────────────────────────────────────────────────
    agent = build_agent(
        obs_shape=problem.observation_shape,
        action_space_size=problem.action_space_size,
        cfg=cfg,
        n_fleets=n_fleets,
    )
    print(f"  Agent      : {agent}\n")

    # ── 7. Logger + Evaluator ───────────────────────────────────────────
    logger = Logger(
        log_dir=cfg.train.log_dir,
        experiment_name=cfg.name,
        verbose=True,
    )
    evaluator = Evaluator(
        agent=agent,
        env=env,
        n_episodes=cfg.train.n_eval_episodes,
        deterministic=cfg.train.eval_deterministic,
        beam_width=cfg.train.eval_beam_width,
    )

    # ── 8. Save config snapshot alongside checkpoints ───────────────────
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, str(ckpt_dir / f"{cfg.name}_config.yaml"))

    # ── 9. Train ────────────────────────────────────────────────────────
    trainer = OnPolicyTrainer(
        agent=agent,
        env=env,
        instance_generator=instance_generator,
        cfg=cfg,
        evaluator=evaluator,
        logger=logger,
    )
    trainer.train()

    # ── 10. Load best checkpoint → final evaluation ──────────────────────
    best_ckpt = ckpt_dir / f"{cfg.name}_best.pt"
    try:
        agent.load(str(best_ckpt))
        print(f"\nLoaded best checkpoint: {best_ckpt}")
    except Exception as exc:
        print(f"Could not load best checkpoint ({exc}); using final weights.")

    _print_eval(evaluator.evaluate(instance_generator), label="Final evaluation")


# ---------------------------------------------------------------------------
# Shared display helper (also used by evaluate.py)
# ---------------------------------------------------------------------------


def _print_eval(stats: dict, label: str = "Evaluation") -> None:
    print(f"\n{label}:")
    for k, v in stats.items():
        print(f"  {k:<28}: {v:.4f}" if isinstance(v, float) else f"  {k:<28}: {v}")


if __name__ == "__main__":
    main()
