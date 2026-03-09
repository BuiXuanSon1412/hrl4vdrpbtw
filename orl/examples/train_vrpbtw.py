"""
examples/train_vrpbtw.py
------------------------
Full end-to-end training script for VRPBTW using the RL framework.

Stages covered
--------------
  1. Problem instantiation     (VRPBTWProblem)
  2. Environment configuration (CombinatorialEnv)
  3. Agent creation            (PPOAgent with attention network)
  4. Training loop             (Trainer with curriculum)
  5. Evaluation & decoding     (Evaluator with greedy + beam search)
  6. Checkpoint reload & final inference

Run
---
  python examples/train_vrpbtw.py                    # default PPO
  python examples/train_vrpbtw.py --customers 15     # larger instances
  python examples/train_vrpbtw.py --fleets 3         # more fleets
  python examples/train_vrpbtw.py --beam 3           # beam search eval
  python examples/train_vrpbtw.py --steps 200000     # longer training
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# ── Framework imports ──────────────────────────────────────────────────────
from vrpbtw_problem import VRPBTWProblem, generate_vrpbtw
from environments.combinatorial_env import CombinatorialEnv
from agents.ppo_agent import PPOAgent, PPOConfig
from training.trainer import Trainer, TrainerConfig
from training.evaluator import Evaluator


# ---------------------------------------------------------------------------
# 1.  Environment factory
# ---------------------------------------------------------------------------


def make_env(
    n_customers: int = 10,
    n_fleets: int = 2,
    dense: bool = True,
) -> CombinatorialEnv:
    """
    Wrap VRPBTWProblem in a CombinatorialEnv.

    subtract_baseline=True  → reward = objective − heuristic_solution()
                               so the agent is rewarded for *improving*
                               over a nearest-neighbour truck-only baseline.
    """
    problem = VRPBTWProblem(n_customers=n_customers, n_fleets=n_fleets)
    return CombinatorialEnv(
        problem=problem,
        max_steps=n_customers * n_fleets * 4,  # generous step budget
        reward_scale=1.0 / max(n_customers, 1),  # normalise by problem size
        subtract_baseline=True,
        dense_shaping=dense,
    )


# ---------------------------------------------------------------------------
# 2.  Instance generator
# ---------------------------------------------------------------------------


def make_instance_generator(n_fleets: int = 2):
    """
    Returns a callable that generates fresh VRPBTW instances.
    The `size` kwarg is used by the Trainer's curriculum scheduler.
    """

    def generator(size: int = 10) -> dict:
        return generate_vrpbtw(
            n_customers=size,
            n_fleets=n_fleets,
            grid_size=100.0,
            seed=None,  # fresh random each episode
        )

    return generator


# ---------------------------------------------------------------------------
# 3.  Agent factory
# ---------------------------------------------------------------------------


def make_ppo_agent(env: CombinatorialEnv) -> PPOAgent:
    """
    PPO with Transformer encoder-decoder.

    Why use_attention=True?
    -----------------------
    The observation is a (N+1, 10) node-feature matrix. The Transformer
    encoder processes all nodes with self-attention, naturally capturing
    spatial relationships between the depot, customers, and current vehicle
    positions — essential for routing problems.

    Hyper-parameter notes
    ---------------------
    - embed_dim=128, n_heads=8: standard for N≤50 customers
    - entropy_coef=0.02: higher than default to encourage exploration
      early in training (action space is large: K*2*(N+1))
    - rollout_len=1024: collect ~2 full episodes before each update
      (adjust if episodes are longer)
    - n_epochs=4, mini_batch_size=128: standard PPO schedule
    """
    cfg = PPOConfig(
        # Network architecture
        embed_dim=128,
        n_heads=8,
        n_encoder_layers=3,
        dropout=0.1,
        use_attention=True,  # Transformer for graph-structured obs
        clip_logits=10.0,
        # PPO hyper-parameters
        lr=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.02,  # encourage exploration of large action space
        max_grad_norm=0.5,
        n_epochs=4,
        mini_batch_size=128,
        rollout_len=1024,
        target_kl=0.02,
        normalize_rewards=True,
        device="cpu",  # change to "cuda" if GPU available
    )

    return PPOAgent(
        obs_shape=env.problem.observation_shape,
        action_space_size=env.problem.action_space_size,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# 4.  Training
# ---------------------------------------------------------------------------


def train(
    agent: PPOAgent,
    env: CombinatorialEnv,
    instance_gen,
    n_customers: int = 10,
    n_fleets: int = 2,
    timesteps: int = 100_000,
    experiment: str = "vrpbtw_ppo",
) -> str:
    """
    Run the full training loop with curriculum scheduling.

    Curriculum strategy
    -------------------
    Start with small instances (3 customers) and gradually increase to
    n_customers over the first half of training. This helps the agent
    learn basic feasibility and routing before facing the full problem size.
    """
    cfg = TrainerConfig(
        total_timesteps=timesteps,
        log_interval=5,
        eval_interval=20,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        experiment_name=experiment,
        patience=60,  # stop if no improvement for 60 evals
        min_delta=1e-3,
        n_eval_episodes=10,
        eval_deterministic=True,
        # Curriculum: grow from 3 customers to n_customers
        curriculum=True,
        curriculum_start=min(3, n_customers),
        curriculum_end=n_customers,
        curriculum_steps=timesteps // 2,  # reach full size halfway through
    )

    trainer = Trainer(
        agent=agent,
        env=env,
        instance_generator=instance_gen,
        cfg=cfg,
    )

    summary = trainer.train()

    print("\n── Training Summary ────────────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:<25}: {v}")

    best_ckpt = f"checkpoints/{experiment}_best.pt"
    return best_ckpt


# ---------------------------------------------------------------------------
# 5.  Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    agent: PPOAgent,
    env: CombinatorialEnv,
    instance_gen,
    beam_width: int = 1,
    n_episodes: int = 20,
) -> None:
    """
    Evaluate the trained agent on fresh instances.

    Three decoding modes are demonstrated:
      1. Greedy (deterministic argmax at every step)
      2. Sampling (stochastic; run 5 times per instance, keep best)
      3. Beam search (if beam_width > 1)
    """
    print("\n── Greedy Evaluation ───────────────────────────────────")
    greedy_eval = Evaluator(
        agent=agent,
        env=env,
        n_episodes=n_episodes,
        deterministic=True,
        beam_width=1,
    )
    greedy_stats = greedy_eval.evaluate(instance_gen)
    _print_eval_stats(greedy_stats, "Greedy")

    print("\n── Sampling Evaluation (5 samples/instance) ────────────")
    sampling_eval = Evaluator(
        agent=agent,
        env=env,
        n_episodes=n_episodes,
        deterministic=False,
        n_samples=5,
    )
    sampling_stats = sampling_eval.evaluate(instance_gen)
    _print_eval_stats(sampling_stats, "Sampling×5")

    if beam_width > 1:
        print(f"\n── Beam Search Evaluation (width={beam_width}) ─────────────")
        beam_eval = Evaluator(
            agent=agent,
            env=env,
            n_episodes=n_episodes,
            deterministic=True,
            beam_width=beam_width,
        )
        beam_stats = beam_eval.evaluate(instance_gen)
        _print_eval_stats(beam_stats, f"Beam-{beam_width}")

    # ── Single instance deep dive ──────────────────────────────────
    print("\n── Single Instance Deep Dive ───────────────────────────")
    raw = instance_gen()
    obs, info = env.reset(raw)
    mask = info["action_mask"]
    done = False
    steps = 0

    while not done:
        action, _, _ = agent.select_action(obs, mask, training=False)
        obs, r, terminated, truncated, info = env.step(action)
        mask = info["action_mask"]
        done = terminated or truncated
        steps += 1

    sol = env.decode_current_solution()
    print(f"\n{sol.summary()}")
    print(f"\n  Steps taken     : {steps}")
    print(
        f"  Customers served: {sol.metadata['served_count']} / {env.problem.n_customers}"
    )
    print(f"  Unserved        : {sol.metadata['unserved']}")
    print(f"\n  Truck routes    :")
    for k, route in enumerate(sol.metadata["truck_routes"]):
        dist = sol.metadata["truck_dist"][k]
        print(f"    Fleet {k}: {route}  (dist={dist:.2f})")
    print(f"\n  Drone routes    :")
    for k, route in enumerate(sol.metadata["drone_routes"]):
        dist = sol.metadata["drone_dist"][k]
        print(f"    Fleet {k}: {route}  (dist={dist:.2f})")

    heuristic = env.problem.heuristic_solution()
    if heuristic is not None:
        gap = (heuristic - sol.objective) / (abs(heuristic) + 1e-6) * 100
        print(f"\n  Heuristic (NN truck-only): {heuristic:.3f}")
        print(f"  Agent objective          : {sol.objective:.3f}")
        print(f"  Gap vs heuristic         : {gap:.1f}%")


def _print_eval_stats(stats: dict, label: str) -> None:
    print(f"  [{label}]")
    for k, v in stats.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k:<28}: {val}")


# ---------------------------------------------------------------------------
# 6.  Main entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train PPO on VRPBTW")
    parser.add_argument(
        "--customers", type=int, default=10, help="Number of customers per instance"
    )
    parser.add_argument(
        "--fleets", type=int, default=2, help="Number of truck-drone fleets"
    )
    parser.add_argument(
        "--steps", type=int, default=100_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--beam", type=int, default=1, help="Beam width for evaluation decoding"
    )
    parser.add_argument(
        "--no-curriculum", action="store_true", help="Disable curriculum scheduling"
    )
    parser.add_argument(
        "--eval-only",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Skip training; load checkpoint and evaluate",
    )
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────
    env = make_env(n_customers=args.customers, n_fleets=args.fleets)
    agent = make_ppo_agent(env)
    instance_gen = make_instance_generator(n_fleets=args.fleets)
    experiment = f"vrpbtw_c{args.customers}_f{args.fleets}"

    print(f"\n{'=' * 60}")
    print(f"  VRPBTW RL Solver")
    print(f"  Customers    : {args.customers}")
    print(f"  Fleets       : {args.fleets}  (truck + drone each)")
    print(f"  Action space : {env.problem.action_space_size}")
    print(f"  Obs shape    : {env.problem.observation_shape}")
    print(f"  launch_time  : {env.problem.launch_time}  (separate from land_time)")
    print(f"  land_time    : {env.problem.land_time}")
    print(f"  Agent        : {agent}")
    print(f"{'=' * 60}")

    # ── Eval-only mode ────────────────────────────────────────────────
    if args.eval_only:
        print(f"\nLoading checkpoint: {args.eval_only}")
        agent.load(args.eval_only)
        evaluate(agent, env, instance_gen, beam_width=args.beam, n_episodes=20)
        return

    # ── Train ─────────────────────────────────────────────────────────
    best_ckpt = train(
        agent=agent,
        env=env,
        instance_gen=instance_gen,
        n_customers=args.customers,
        n_fleets=args.fleets,
        timesteps=args.steps,
        experiment=experiment,
    )

    # ── Load best and evaluate ────────────────────────────────────────
    print(f"\nLoading best checkpoint: {best_ckpt}")
    try:
        agent.load(best_ckpt)
    except Exception as e:
        print(f"  (Could not load checkpoint: {e}; using final weights)")

    evaluate(agent, env, instance_gen, beam_width=args.beam, n_episodes=20)


if __name__ == "__main__":
    main()
