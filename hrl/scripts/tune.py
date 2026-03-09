"""
scripts/tune.py
─────────────────────────────────────────────────────────────────────────────
Hyperparameter optimisation using Optuna.

Reads the search space from configs/tuning/optuna_search.yaml and runs N
trials, each training for a short budget. The best hyperparameters are
saved as configs/tuning/best_params.yaml for use in a full training run.

Usage
─────
    python scripts/tune.py
    python scripts/tune.py tuning.n_trials=200 tuning.n_jobs=8

    # After tuning, train with best params:
    python scripts/train.py --config-name tuning/best_params
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import optuna
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from src.agents import build_agent
from src.envs import VRPBTWEnv
from src.rewards.default import DefaultRewardFn
from src.utils.metrics import EpisodeResult, EvalStats

console = Console()


@hydra.main(
    config_path="../configs",
    config_name="tuning/optuna_search",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    console.rule("[bold blue]VRPBTW-RL Hyperparameter Tuning (Optuna)")

    study = optuna.create_study(
        study_name=cfg.tuning.study_name,
        direction=cfg.tuning.direction,
        sampler=_build_sampler(cfg.tuning.sampler),
        pruner=_build_pruner(cfg.tuning.pruner),
    )

    def objective(trial: optuna.Trial) -> float:
        # ── Sample hyperparameters ────────────────────────────────────────
        trial_cfg = _sample_cfg(trial, cfg)

        # ── Build env & agent ─────────────────────────────────────────────
        reward_fn = DefaultRewardFn(
            cost_weight=float(trial_cfg["env"]["cost_weight"]),
            tardiness_weight=1.0 - float(trial_cfg["env"]["cost_weight"]),
        )
        env = VRPBTWEnv(
            num_customers=cfg.problem.num_customers,
            num_vehicles=cfg.problem.num_vehicles,
            num_drones=cfg.problem.num_drones,
            map_size=cfg.problem.map_size,
            reward_fn=reward_fn,
        )

        # Patch the cfg with sampled values so build_agent picks them up
        trial_omegacfg = OmegaConf.create(trial_cfg)
        agent = build_agent(trial_omegacfg, env)

        # ── Short training run ────────────────────────────────────────────
        budget = cfg.tuning.trial_budget.num_episodes
        eval_every = cfg.tuning.trial_budget.eval_every_n_episodes
        n_eval = cfg.tuning.trial_budget.num_eval_episodes

        best_metric = 0.0
        for ep in range(1, budget + 1):
            agent.rollout(training=True)
            agent.update_exploration(ep)

            if ep % cfg.training.train_every_n_episodes == 0:
                agent.train_step()

            if ep % eval_every == 0:
                results = [
                    EpisodeResult(**agent.rollout(training=False))
                    for _ in range(n_eval)
                ]
                stats = EvalStats.from_results(results)
                metric = stats.mean_service_rate
                best_metric = max(best_metric, metric)

                # Report to Optuna pruner
                trial.report(metric, step=ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return best_metric

    study.optimize(
        objective,
        n_trials=cfg.tuning.n_trials,
        n_jobs=cfg.tuning.n_jobs,
        show_progress_bar=True,
    )

    # ── Report & save best params ─────────────────────────────────────────
    console.print("\n[bold green]Best trial:[/]")
    best = study.best_trial
    for k, v in best.params.items():
        console.print(f"  {k}: {v}")
    console.print(f"  Metric: {best.value:.4f}")

    # Merge best params into base config and save
    best_cfg = OmegaConf.to_container(cfg, resolve=True)
    for key, value in best.params.items():
        _set_nested(best_cfg, key.split("."), value)

    out_path = Path("configs/tuning/best_params.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(best_cfg, f, default_flow_style=False)
    console.print(f"\n[green]Best params saved → {out_path}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_sampler(name: str):
    name = name.lower()
    if name == "tpe":
        return optuna.samplers.TPESampler()
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler()
    return optuna.samplers.RandomSampler()


def _build_pruner(name: str):
    name = name.lower()
    if name == "median":
        return optuna.pruners.MedianPruner()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    return optuna.pruners.NopPruner()


def _sample_cfg(trial: optuna.Trial, cfg: DictConfig) -> dict:
    """Build a mutable config dict with Optuna-sampled values."""
    base = OmegaConf.to_container(cfg, resolve=True)
    for key, spec in cfg.tuning.search_space.items():
        spec = dict(spec)
        t = spec.pop("type")
        if t == "float":
            val = trial.suggest_float(key, **spec)
        elif t == "int":
            val = trial.suggest_int(key, **spec)
        elif t == "categorical":
            val = trial.suggest_categorical(key, spec["choices"])
        else:
            raise ValueError(f"Unknown param type: {t}")
        _set_nested(base, key.split("."), val)
    return base


def _set_nested(d: dict, keys: list, value):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


if __name__ == "__main__":
    main()
