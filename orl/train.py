"""
train.py
--------
Single entry point for all experiments.

This file is the only place that wires everything together:
  1. Load / build ExperimentConfig
  2. Seed everything
  3. Build problem → env
  4. Build network → agent  (shapes flow from problem)
  5. Run Trainer.train()
  6. Evaluate best checkpoint

Usage
-----
# PPO on Knapsack (defaults)
python train.py

# DQN on Knapsack
python train.py --algorithm dqn

# PPO with pointer network on VRPBTW
python train.py --problem vrpbtw --network pointer --steps 200000

# Reproduce an existing experiment exactly
python train.py --config checkpoints/my_exp_config.json

# Evaluate only (no training)
python train.py --eval-only checkpoints/my_exp_best.pt --config checkpoints/my_exp_config.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make orl importable when run from the project root
sys.path.insert(0, str(Path(__file__).parent))

from configs import ExperimentConfig, EnvConfig, NetworkConfig, PPOConfig, DQNConfig
from configs import TrainConfig, SeedConfig, load_config, save_config
from utils.seed import SeedManager
from problems.registry import build_problem, get_generator
from environments.combinatorial_env import Env
from agents.registry import build_agent
from trainers.trainer import Trainer
from trainers.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Config builder from CLI args
# ---------------------------------------------------------------------------


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build or load ExperimentConfig from CLI args."""
    if args.config:
        cfg = load_config(args.config)
        # CLI overrides (explicit args take priority over file)
        if args.algorithm:
            cfg.algorithm = args.algorithm
        if args.problem:
            cfg.env.problem_name = args.problem
        if args.network:
            cfg.network.network_type = args.network
        if args.steps:
            cfg.train.total_timesteps = args.steps
        if args.device:
            cfg.device = args.device
        return cfg

    n_items = args.n_items or 20
    n_cust = args.customers or 10
    n_fleets = args.fleets or 2

    problem_name = args.problem or "knapsack"
    problem_kwargs = {}
    if problem_name == "knapsack":
        problem_kwargs = {"n_items": n_items}
    elif problem_name == "vrpbtw":
        problem_kwargs = {"n_customers": n_cust, "n_fleets": n_fleets}

    algorithm = args.algorithm or "ppo"
    network = args.network or ("hacn" if problem_name == "vrpbtw" else "attention")

    cfg = ExperimentConfig(
        name=args.name or f"{problem_name}_{algorithm}",
        algorithm=algorithm,
        device=args.device or "cpu",
        seed=SeedConfig(global_seed=args.seed or 42),
        env=EnvConfig(
            problem_name=problem_name,
            problem_kwargs=problem_kwargs,
            max_steps=args.max_steps,
            subtract_baseline=True,
            dense_shaping=True,
        ),
        network=NetworkConfig(
            network_type=network,
            embed_dim=args.embed_dim or 128,
            n_heads=8,
            n_encoder_layers=3,
        ),
        ppo=PPOConfig(
            lr=args.lr or 3e-4,
            rollout_len=1024,
            n_epochs=4,
            entropy_coef=0.02,
            normalize_rewards=True,
            normalize_advantages=True,
        ),
        dqn=DQNConfig(
            lr=args.lr or 1e-4,
            eps_decay_steps=50_000,
        ),
        train=TrainConfig(
            total_timesteps=args.steps or 100_000,
            log_interval=5,
            eval_interval=20,
            checkpoint_interval=100,
            checkpoint_dir=args.checkpoint_dir or "checkpoints",
            log_dir=args.log_dir or "logs",
            patience=60,
            n_eval_episodes=10,
            curriculum=not args.no_curriculum,
            curriculum_start=3,
            curriculum_end=n_cust if problem_name == "vrpbtw" else n_items,
            curriculum_steps=(args.steps or 100_000) // 2,
        ),
    )
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ORL — Combinatorial RL framework")

    # Config file (overrides everything)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to ExperimentConfig JSON. CLI args override fields.",
    )

    # Problem
    parser.add_argument(
        "--problem", type=str, default=None, choices=["knapsack", "vrpbtw"]
    )
    parser.add_argument("--n-items", type=int, default=None)
    parser.add_argument("--customers", type=int, default=None)
    parser.add_argument("--fleets", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)

    # Algorithm & network
    parser.add_argument("--algorithm", type=str, default=None, choices=["ppo", "dqn"])
    parser.add_argument(
        "--network",
        type=str,
        default=None,
        choices=["attention", "pointer", "mlp", "hacn"],
    )
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    # Training
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)

    # Evaluation
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument(
        "--eval-only",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Skip training; load checkpoint and evaluate.",
    )

    args = parser.parse_args()
    cfg = build_config(args)

    # ── 1. Reproducibility ──────────────────────────────────────────────
    seed_mgr = SeedManager(
        global_seed=cfg.seed.global_seed,
        env_seed=cfg.seed.env_seed,
        data_seed=cfg.seed.data_seed,
    )
    seed_mgr.seed_everything()
    data_rng = seed_mgr.make_data_rng()
    print(f"  {seed_mgr}")

    # ── 2. Problem + environment ─────────────────────────────────────────
    problem = build_problem(cfg.env)
    env = Env(problem, cfg.env)

    # Instance generator uses the dedicated data RNG for reproducibility
    _base_gen = get_generator(cfg.env)

    def instance_generator(size: int = None, **kwargs):
        kw = dict(cfg.env.problem_kwargs)
        if size is not None:
            # Curriculum: update the size-governing kwarg
            if cfg.env.problem_name == "knapsack":
                kw["n_items"] = size
            elif cfg.env.problem_name == "vrpbtw":
                kw["n_customers"] = size
        kw["rng"] = data_rng
        return _base_gen(**kw)

    # Encode a dummy instance so problem knows obs/action shapes
    dummy = instance_generator()
    problem.encode_instance(dummy)

    print(f"  Problem    : {problem}")
    print(f"  Obs shape  : {problem.observation_shape}")
    print(f"  Actions    : {problem.action_space_size}")

    # ── 3. Agent (network injected via factory) ──────────────────────────
    agent = build_agent(
        obs_shape=problem.observation_shape,
        action_space_size=problem.action_space_size,
        cfg=cfg,
    )
    print(f"  Agent      : {agent}")

    # ── 4. Eval-only mode ─────────────────────────────────────────────────
    if args.eval_only:
        print(f"\nLoading checkpoint: {args.eval_only}")
        agent.load(args.eval_only)
        evaluator = Evaluator(
            agent=agent,
            env=env,
            n_episodes=20,
            deterministic=True,
            beam_width=args.beam,
        )
        stats = evaluator.evaluate(instance_generator)
        print("\nEvaluation results:")
        for k, v in stats.items():
            print(f"  {k:<28}: {v:.4f}" if isinstance(v, float) else f"  {k:<28}: {v}")
        return

    # ── 5. Train ─────────────────────────────────────────────────────────
    trainer = Trainer(
        agent=agent,
        env=env,
        instance_generator=instance_generator,
        cfg=cfg,
    )
    summary = trainer.train()

    # ── 6. Load best and evaluate ─────────────────────────────────────────
    best_ckpt = f"{cfg.train.checkpoint_dir}/{cfg.name}_best.pt"
    try:
        agent.load(best_ckpt)
        print(f"\nLoaded best checkpoint: {best_ckpt}")
    except Exception as e:
        print(f"Could not load best checkpoint ({e}); using final weights.")

    evaluator = Evaluator(
        agent=agent,
        env=env,
        n_episodes=20,
        deterministic=True,
        beam_width=args.beam,
    )
    eval_stats = evaluator.evaluate(instance_generator)
    print("\nFinal evaluation:")
    for k, v in eval_stats.items():
        print(f"  {k:<28}: {v:.4f}" if isinstance(v, float) else f"  {k:<28}: {v}")


if __name__ == "__main__":
    main()
