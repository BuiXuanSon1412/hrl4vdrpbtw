"""
examples/train_knapsack.py
---------------------------
Full end-to-end training script demonstrating every stage of the RL pipeline:

  Stage 1 — Problem definition     (KnapsackProblem)
  Stage 2 — Environment setup      (CombinatorialEnv)
  Stage 3 — Agent instantiation    (PPOAgent or DQNAgent)
  Stage 4 — Training loop          (Trainer)
  Stage 5 — Evaluation & decoding  (Evaluator, beam search)
  Stage 6 — Checkpoint reload & inference

Run:
    python examples/train_knapsack.py            # PPO (default)
    python examples/train_knapsack.py --dqn      # DQN
    python examples/train_knapsack.py --beam 3   # beam search decoding
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# ── Framework imports ──────────────────────────────────────────────────────
from examples.knapsack_problem import KnapsackProblem, generate_knapsack
from environments.combinatorial_env import CombinatorialEnv
from agents.ppo_agent import PPOAgent, PPOConfig
from agents.dqn_agent import DQNAgent, DQNConfig
from training.trainer import Trainer, TrainerConfig
from training.evaluator import Evaluator


# ---------------------------------------------------------------------------
# 1.  Problem & Environment
# ---------------------------------------------------------------------------

N_ITEMS = 20


def make_env(dense: bool = True) -> CombinatorialEnv:
    problem = KnapsackProblem(n_items=N_ITEMS)
    return CombinatorialEnv(
        problem=problem,
        max_steps=N_ITEMS + 5,
        reward_scale=1.0 / N_ITEMS,  # normalise rewards
        subtract_baseline=True,  # reward = obj - heuristic
        dense_shaping=dense,
    )


def instance_gen(size: int = N_ITEMS) -> dict:
    return generate_knapsack(n_items=size)


# ---------------------------------------------------------------------------
# 2.  Agent factory
# ---------------------------------------------------------------------------


def make_ppo_agent(env: CombinatorialEnv) -> PPOAgent:
    cfg = PPOConfig(
        embed_dim=64,
        n_heads=4,
        n_encoder_layers=2,
        use_attention=False,  # flat 5-dim obs → MLP is sufficient
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.02,
        n_epochs=4,
        mini_batch_size=128,
        rollout_len=512,
        normalize_rewards=True,
        device="cpu",
    )
    return PPOAgent(
        obs_shape=env.problem.observation_shape,
        action_space_size=env.problem.action_space_size,
        cfg=cfg,
    )


def make_dqn_agent(env: CombinatorialEnv) -> DQNAgent:
    cfg = DQNConfig(
        embed_dim=64,
        use_attention=False,
        lr=1e-3,
        gamma=0.99,
        buffer_capacity=20_000,
        batch_size=64,
        target_update_freq=200,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=10_000,
        use_per=False,
        device="cpu",
    )
    return DQNAgent(
        obs_shape=env.problem.observation_shape,
        action_space_size=env.problem.action_space_size,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# 3.  Training
# ---------------------------------------------------------------------------


def train(agent, env: CombinatorialEnv, timesteps: int = 50_000) -> str:
    trainer_cfg = TrainerConfig(
        total_timesteps=timesteps,
        log_interval=10,
        eval_interval=25,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        experiment_name=f"knapsack_{type(agent).__name__.lower()}",
        patience=40,
        n_eval_episodes=10,
        eval_deterministic=True,
        curriculum=True,
        curriculum_start=5,
        curriculum_end=N_ITEMS,
        curriculum_steps=timesteps // 2,
    )
    trainer = Trainer(
        agent=agent,
        env=env,
        instance_generator=instance_gen,
        cfg=trainer_cfg,
    )
    summary = trainer.train()
    print("\nTraining summary:", summary)
    best_ckpt = f"checkpoints/knapsack_{type(agent).__name__.lower()}_best.pt"
    return best_ckpt


# ---------------------------------------------------------------------------
# 4.  Evaluation
# ---------------------------------------------------------------------------


def evaluate(agent, env: CombinatorialEnv, beam_width: int = 1):
    print("\n── Evaluation ──────────────────────────────────────────")

    evaluator = Evaluator(
        agent=agent,
        env=env,
        n_episodes=20,
        deterministic=True,
        beam_width=beam_width,
    )
    stats = evaluator.evaluate(instance_gen)
    for k, v in stats.items():
        print(f"  {k:<25}: {v:.4f}" if isinstance(v, float) else f"  {k:<25}: {v}")

    # Single-instance deep dive
    print("\n── Single Instance Deep Dive ───────────────────────────")
    raw = generate_knapsack(n_items=N_ITEMS, seed=99)
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
    print(f"  {sol.summary()}")
    print(
        f"  Weight used : {sol.metadata.get('total_weight', '?'):.2f} "
        f"/ {sol.metadata.get('capacity', '?'):.2f}"
    )
    print(f"  Steps taken : {steps}")
    heur = env.problem.heuristic_solution()
    if heur:
        gap = (heur - sol.objective) / abs(heur) * 100
        print(f"  Heuristic UB: {heur:.2f}  |  Gap: {gap:.1f}%")


# ---------------------------------------------------------------------------
# 5.  Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn", action="store_true", help="Use DQN instead of PPO")
    parser.add_argument("--beam", type=int, default=1, help="Beam width for decoding")
    parser.add_argument("--steps", type=int, default=30_000, help="Training timesteps")
    args = parser.parse_args()

    env = make_env(dense=True)
    agent = make_dqn_agent(env) if args.dqn else make_ppo_agent(env)

    print(f"\nAgent  : {agent}")
    print(f"Env    : {env}")
    print(f"Problem: {env.problem}")

    # ── Train ────────────────────────────────────────────────────────────
    best_ckpt = train(agent, env, timesteps=args.steps)

    # ── Load best checkpoint & evaluate ─────────────────────────────────
    try:
        agent.load(best_ckpt)
        print(f"\nLoaded best checkpoint from {best_ckpt}")
    except Exception as e:
        print(f"Could not load checkpoint ({e}); using final weights.")

    evaluate(agent, env, beam_width=args.beam)


if __name__ == "__main__":
    main()
