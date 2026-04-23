"""
evaluate.py
-----------
Standalone evaluation entry point.

Loads a trained checkpoint and evaluates it on fresh instances.
No training is performed.

CLI flags
---------
  --config     PATH   Config used during training  [required]
  --checkpoint PATH   Model checkpoint .pt file    [required]
  --override   PATH   Optional config override
  --device     DEVICE Override cfg.device
  --beam       WIDTH  Beam width (1=greedy, >1=beam search)
  --episodes   N      Number of evaluation episodes
  --samples    N      Rollouts per instance for sampling decode (>1 = best-of-N)

Usage
-----
  # Greedy evaluation (loads from experiment/train automatically)
  python evaluate.py --config vrpbtw_maml_base_config.yaml \\
                     --checkpoint vrpbtw_maml_base_best.pt

  # Beam search
  python evaluate.py --config vrpbtw_maml_base_config.yaml \\
                     --checkpoint vrpbtw_maml_base_best.pt \\
                     --beam 5

  # Best-of-N sampling
  python evaluate.py --config vrpbtw_maml_base_config.yaml \\
                     --checkpoint vrpbtw_maml_base_best.pt \\
                     --samples 8

  # Or specify full paths
  python evaluate.py --config experiment/train/vrpbtw_maml_base_config.yaml \\
                     --checkpoint experiment/train/vrpbtw_maml_base_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, merge_configs
from core import Evaluator, SeedManager
from registry import build_agent, build_environment, get_generator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained RL checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="ExperimentConfig YAML (saved alongside the checkpoint).",
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Checkpoint .pt file to evaluate.",
    )
    p.add_argument(
        "--override",
        default=None,
        metavar="PATH",
        help="Optional ablation override YAML.",
    )
    p.add_argument(
        "--device", default=None, metavar="DEVICE", help="Override cfg.device."
    )
    p.add_argument(
        "--beam",
        default=None,
        type=int,
        metavar="WIDTH",
        help="Beam width (1 = greedy).  Overrides cfg.train.eval_beam_width.",
    )
    p.add_argument(
        "--episodes",
        default=None,
        type=int,
        metavar="N",
        help="Number of evaluation episodes.  Overrides cfg.train.n_eval_episodes.",
    )
    p.add_argument(
        "--samples",
        default=1,
        type=int,
        metavar="N",
        help="Best-of-N sampling rollouts per instance (default: 1).",
    )
    p.add_argument(
        "--customers",
        default=None,
        type=int,
        metavar="N",
        help="Override n_customers for evaluation instances.  "
        "Allows evaluating a checkpoint on larger or smaller problems "
        "than the training size.",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _build_parser().parse_args()

    # ── 1. Config ───────────────────────────────────────────────────────
    # Support both direct config path and config within experiment/train folder
    config_path = args.config
    if not Path(config_path).exists():
        # Try to find config in experiment/train folder
        exp_config = Path("experiment/train") / args.config
        if exp_config.exists():
            config_path = str(exp_config)

    cfg = (
        merge_configs(config_path, args.override)
        if args.override
        else load_config(config_path)
    )
    if args.device:
        cfg["device"] = args.device

    # Extract config values with hierarchical support
    device = cfg.get("device", "cpu")
    exp_name = cfg.get("name") or cfg.get("experiment", {}).get("name", "experiment")

    # Handle training config override for beam/episodes
    training_cfg = cfg.get("training", cfg.get("train", {}))
    eval_cfg = training_cfg.get("evaluation", cfg.get("evaluation", {}))
    eval_decoding = eval_cfg.get("decoding", {})

    beam_width = args.beam if args.beam else eval_decoding.get("beam_width", 1)
    n_episodes = args.episodes if args.episodes else eval_cfg.get("n_episodes", 20)
    deterministic = eval_cfg.get("deterministic", True)

    print(f"\n  Config     : {args.config}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Experiment : {exp_name}")
    print(f"  Device     : {device}")

    # ── 2. Reproducibility ──────────────────────────────────────────────
    reproducibility_cfg = cfg.get("reproducibility", {})
    seed_cfg = reproducibility_cfg.get("seed", cfg.get("seed", {}))
    seed_mgr = SeedManager(
        global_seed=seed_cfg.get("global_seed", 42),
        env_seed=seed_cfg.get("env_seed"),
        data_seed=seed_cfg.get("data_seed"),
    )
    seed_mgr.seed_everything()
    eval_rng = seed_mgr.make_eval_rng()  # fixed RNG → reproducible eval

    # ── 3. Environment ──────────────────────────────────────────────────
    env = build_environment(cfg)
    base_gen = get_generator(cfg)

    print(f"  Environment: {env}")

    # ── 4. Instance generator (fixed eval RNG for reproducibility) ──────
    def instance_generator(size: Optional[int] = None, **_: Any) -> Any:
        env_cfg = cfg.get("environment", {})
        problem_cfg = env_cfg.get("problem", {})
        problem_kwargs = problem_cfg.get("kwargs", {})
        kw = dict(problem_kwargs)
        if args.customers is not None:
            kw["n_customers"] = args.customers
        elif size is not None and problem_cfg.get("name", "vrpbtw") == "vrpbtw":
            kw["n_customers"] = size
        kw["rng"] = eval_rng
        return base_gen(**kw)

    # ── 6. Agent ────────────────────────────────────────────────────────
    agent = build_agent(cfg=cfg)

    # Support both direct checkpoint path and checkpoint within experiment/train folder
    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).exists():
        # Try to find checkpoint in experiment/train folder
        exp_checkpoint = Path("experiment/train") / args.checkpoint
        if exp_checkpoint.exists():
            checkpoint_path = str(exp_checkpoint)

    agent.load(checkpoint_path)
    print(f"  Agent      : {agent}\n")

    # ── 7. Evaluate ─────────────────────────────────────────────────────
    evaluator = Evaluator(
        agent=agent,
        env=env,
        n_episodes=n_episodes,
        deterministic=deterministic,
        n_samples=args.samples,
        beam_width=beam_width,
    )
    stats = evaluator.evaluate(instance_generator)

    print("Evaluation results:")
    for k, v in stats.items():
        print(f"  {k:<28}: {v:.4f}" if isinstance(v, float) else f"  {k:<28}: {v}")


if __name__ == "__main__":
    main()
