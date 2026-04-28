"""
evaluate.py
-----------
Standalone evaluation entry point.

Loads a trained checkpoint and evaluates it on fresh instances.
No training is performed.

CLI flags
---------
  --config     PATH   Config file [default: configs/evaluate.yaml]
  --checkpoint PATH   Model checkpoint .pt file [default: all checkpoints in training_experiment]
  --override   PATH   Optional config override
  --device     DEVICE Override cfg.device
  --beam       WIDTH  Beam width (1=greedy, >1=beam search)
  --episodes   N      Number of evaluation episodes
  --samples    N      Rollouts per instance for sampling decode (>1 = best-of-N)

Usage
-----
  # Default: evaluate all checkpoints from default training experiment
  python evaluate.py

  # Evaluate specific checkpoint with default config
  python evaluate.py --checkpoint vrpbtw_maml_base_best.pt

  # Beam search with specific config
  python evaluate.py --config configs/custom_evaluate.yaml --checkpoint model_best.pt --beam 5

  # Best-of-N sampling
  python evaluate.py --checkpoint model_best.pt --samples 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, merge_configs
from core import Evaluator, SeedManager
from core.registry import build_agent, build_environment
import torch

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
        default="configs/evaluate.yaml",
        metavar="PATH",
        help="Evaluation config [default: configs/evaluate.yaml]",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Checkpoint .pt file to evaluate. If not provided, evaluates all checkpoints in training_experiment dir.",
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


def _find_checkpoints(training_experiment: str) -> list:
    """Find all .pt checkpoint files in training experiment directory."""
    checkpoint_dir = Path("experiment/train") / training_experiment / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    return sorted(checkpoint_dir.glob("*.pt"))


def _evaluate_checkpoint(
    checkpoint_path: Path,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """Evaluate a single checkpoint."""
    import csv

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

    # Parse task_id from checkpoint filename (format: {exp_name}_{tag}.pt)
    checkpoint_stem = checkpoint_path.stem
    parts = checkpoint_stem.rsplit("_", 1)
    task_id = parts[-1] if len(parts) > 1 else None

    print(f"\n{'=' * 70}")
    print(f"  Checkpoint : {checkpoint_path.name}")
    print(f"  Task ID    : {task_id}")
    print(f"  Experiment : {exp_name}")
    print(f"  Device     : {device}")

    # ── Reproducibility ─────────────────────────────────────────────────
    reproducibility_cfg = cfg.get("reproducibility", {})
    seed_cfg = reproducibility_cfg.get("seed", cfg.get("seed", {}))
    seed_mgr = SeedManager(
        random_seed=seed_cfg.get("random_seed", 42),
        numpy_seed=seed_cfg.get("numpy_seed", 42),
        torch_seed=seed_cfg.get("torch_seed", 42),
    )
    seed_mgr.seed_everything()

    # ── Environment ─────────────────────────────────────────────────────
    env = build_environment(cfg)
    print(f"  Environment: {env}")

    # ── Agent ───────────────────────────────────────────────────────────

    agents = build_agent(cfg=cfg)
    agent = next(iter(agents.values()))  # Use first agent for evaluation

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load checkpoint into agent's policy
    if isinstance(checkpoint, dict) and "agent" in checkpoint:
        agent.policy.load_state_dict(checkpoint["agent"])
    elif isinstance(checkpoint, dict):
        agent.policy.load_state_dict(checkpoint)

    print(f"  Agents     : {list(agents.keys())}")
    print(f"  Agent      : {agent}")

    # ── Evaluate ────────────────────────────────────────────────────────
    evaluator = Evaluator(
        agent=agent,
        env=env,
        n_episodes=n_episodes,
        deterministic=deterministic,
        n_samples=args.samples,
        beam_width=beam_width,
    )
    stats = evaluator.evaluate(task_id)

    print("\n  Results:")
    for k, v in stats.items():
        print(f"    {k:<26}: {v:.4f}" if isinstance(v, float) else f"    {k:<26}: {v}")

    # ── Save results to CSV ──────────────────────────────────────────────
    checkpoint_name = checkpoint_path.stem
    csv_path = artifacts_dir / f"{checkpoint_name}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in stats.items():
            writer.writerow([k, v])

    print(f"  CSV saved: {csv_path}")

    return stats


def main() -> None:
    args = _build_parser().parse_args()

    # ── 1. Load evaluate config ──────────────────────────────────────────
    config_path = args.config
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    eval_cfg = (
        merge_configs(config_path, args.override)
        if args.override
        else load_config(config_path)
    )

    print(f"\n  Evaluate Config: {config_path}")

    # ── 2. Get training and evaluation experiment names ──────────────────
    eval_exp_name = eval_cfg.get("experiment", {}).get("name")
    if not eval_exp_name:
        raise ValueError("experiment.name must be specified in evaluate config")

    dirs_cfg = eval_cfg.get("directories", {})
    training_exp_name = dirs_cfg.get("source")
    if not training_exp_name:
        raise ValueError("directories.source must be specified in evaluate config")

    training_config_path = (
        Path("experiment/train") / training_exp_name / "config" / "config.yaml"
    )
    if not training_config_path.exists():
        raise FileNotFoundError(
            f"Training config not found: {training_config_path}\n"
            f"Make sure training experiment '{training_exp_name}' exists in experiment/train/"
        )

    # Load training config and merge evaluate config on top
    cfg = load_config(str(training_config_path))

    # Merge evaluate settings (device, evaluation, directories, reproducibility)
    if "device" in eval_cfg:
        cfg["device"] = eval_cfg["device"]
    if "evaluation" in eval_cfg:
        cfg["evaluation"] = eval_cfg["evaluation"]
    if "directories" in eval_cfg:
        cfg["directories"] = eval_cfg["directories"]
    if "reproducibility" in eval_cfg:
        cfg["reproducibility"] = eval_cfg["reproducibility"]

    if args.device:
        cfg["device"] = args.device

    print(f"  Training Experiment: {training_exp_name}")
    print(f"  Evaluation Experiment: {eval_exp_name}")
    print(f"  Training Config: {training_config_path}")

    # ── 3. Determine checkpoints to evaluate ─────────────────────────────
    if args.checkpoint:
        # Single checkpoint specified
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            # Try to find in training experiment folder
            exp_checkpoint = (
                Path("experiment/train")
                / training_exp_name
                / "checkpoints"
                / args.checkpoint
            )
            if exp_checkpoint.exists():
                checkpoint_path = exp_checkpoint
            else:
                raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        checkpoints = [checkpoint_path]
    else:
        # Find all checkpoints in training experiment directory
        checkpoints = _find_checkpoints(training_exp_name)
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found in experiment/train/{training_exp_name}/checkpoints/"
            )
        print(f"  Found {len(checkpoints)} checkpoint(s)")

    # ── 4. Setup output directory ───────────────────────────────────────
    base_dir = cfg.get("directories", {}).get("base_dir", "experiment/evaluate")
    output_dir = Path(base_dir) / training_exp_name / eval_exp_name
    artifacts_dir = output_dir / cfg.get("directories", {}).get(
        "artifacts", "artifacts"
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluate config
    eval_config_path = output_dir / "evaluate.yaml"
    from config import save_config

    save_config(eval_cfg, str(eval_config_path))

    print(f"  Output Directory: {output_dir}")
    print(f"  Artifacts Directory: {artifacts_dir}")

    # ── 5. Evaluate each checkpoint ──────────────────────────────────────
    results = {}
    for checkpoint_path in checkpoints:
        results[checkpoint_path.name] = _evaluate_checkpoint(
            checkpoint_path, cfg, args, artifacts_dir
        )

    # ── 6. Summary and aggregate results ────────────────────────────────
    if len(checkpoints) > 1:
        print(f"\n{'=' * 70}")
        print("\nSummary of all checkpoints:")

        # Create summary CSV
        import csv

        summary_csv = artifacts_dir / "summary.csv"

        # Get all metrics from first checkpoint
        first_stats = next(iter(results.values()))
        metrics = list(first_stats.keys())

        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["checkpoint"] + metrics)
            for ckpt_name, stats in results.items():
                row = [ckpt_name] + [stats.get(m, "") for m in metrics]
                writer.writerow(row)

        print(f"  Summary CSV saved: {summary_csv}\n")

        for ckpt_name, stats in results.items():
            print(f"\n  {ckpt_name}:")
            for k, v in stats.items():
                print(
                    f"    {k:<26}: {v:.4f}"
                    if isinstance(v, float)
                    else f"    {k:<26}: {v}"
                )

    print(f"\n  Evaluation results saved to: {output_dir}")


if __name__ == "__main__":
    main()
