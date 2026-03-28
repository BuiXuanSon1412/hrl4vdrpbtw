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
  --override   PATH   Optional ablation override
  --device     DEVICE Override cfg.device
  --beam       WIDTH  Beam width (1=greedy, >1=beam search)
  --episodes   N      Number of evaluation episodes
  --samples    N      Rollouts per instance for sampling decode (>1 = best-of-N)

Usage
-----
  # Greedy evaluation
  python evaluate.py --config checkpoints/vrpbtw_ppo_base_config.yaml \\
                     --checkpoint checkpoints/vrpbtw_ppo_base_best.pt

  # Beam search
  python evaluate.py --config checkpoints/vrpbtw_ppo_base_config.yaml \\
                     --checkpoint checkpoints/vrpbtw_ppo_base_best.pt \\
                     --beam 5

  # Best-of-N sampling
  python evaluate.py --config checkpoints/vrpbtw_ppo_base_config.yaml \\
                     --checkpoint checkpoints/vrpbtw_ppo_base_best.pt \\
                     --samples 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, merge_configs, ExperimentConfig
from core import Environment, Evaluator, SeedManager
from registry import build_agent, build_problem, get_generator


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
    if args.beam:
        cfg.train.eval_beam_width = args.beam
    if args.episodes:
        cfg.train.n_eval_episodes = args.episodes

    print(f"\n  Config     : {args.config}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Experiment : {cfg.name}")
    print(f"  Device     : {cfg.device}")

    # ── 2. Reproducibility ──────────────────────────────────────────────
    seed_mgr = SeedManager(
        global_seed=cfg.seed.global_seed,
        env_seed=cfg.seed.env_seed,
        data_seed=cfg.seed.data_seed,
    )
    seed_mgr.seed_everything()
    eval_rng = seed_mgr.make_eval_rng()  # fixed RNG → reproducible eval

    # ── 3. Problem ──────────────────────────────────────────────────────
    problem = build_problem(cfg.env)
    base_gen = get_generator(cfg.env)

    dummy = base_gen(**{**cfg.env.problem_kwargs, "rng": eval_rng})
    problem.encode_instance(dummy)
    n_fleets: int = problem.n_fleets

    print(
        f"  Problem    : {problem}  |  "
        f"Obs: {problem.observation_shape}  |  "
        f"Actions: {problem.action_space_size}"
    )

    # ── 4. Environment ──────────────────────────────────────────────────
    env = Environment(problem, cfg.env)

    # ── 5. Instance generator (fixed eval RNG for reproducibility) ──────
    def instance_generator(size: Optional[int] = None, **_: Any) -> Any:
        kw = dict(cfg.env.problem_kwargs)
        if size is not None and cfg.env.problem_name == "vrpbtw":
            kw["n_customers"] = size
        kw["rng"] = eval_rng
        return base_gen(**kw)

    # ── 6. Agent ────────────────────────────────────────────────────────
    agent = build_agent(
        obs_shape=problem.observation_shape,
        action_space_size=problem.action_space_size,
        cfg=cfg,
        n_fleets=n_fleets,
    )
    agent.load(args.checkpoint)
    print(f"  Agent      : {agent}\n")

    # ── 7. Evaluate ─────────────────────────────────────────────────────
    evaluator = Evaluator(
        agent=agent,
        env=env,
        n_episodes=cfg.train.n_eval_episodes,
        deterministic=cfg.train.eval_deterministic,
        n_samples=args.samples,
        beam_width=cfg.train.eval_beam_width,
    )
    stats = evaluator.evaluate(instance_generator)

    print("Evaluation results:")
    for k, v in stats.items():
        print(f"  {k:<28}: {v:.4f}" if isinstance(v, float) else f"  {k:<28}: {v}")


if __name__ == "__main__":
    main()
