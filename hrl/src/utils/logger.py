"""
src/utils/logger.py
─────────────────────────────────────────────────────────────────────────────
Thin logging abstraction that writes to TensorBoard, W&B, or both,
controlled by config.  All scripts import this; never import tensorboard
or wandb directly in business logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class Logger:
    """
    Unified logger for training metrics.

    Usage
    ─────
        logger = Logger.from_cfg(cfg)
        logger.log({"train/reward": 0.42, "train/loss": 0.01}, step=100)
        logger.close()
    """

    def __init__(
        self,
        backend: str = "tensorboard",  # tensorboard | wandb | both | none
        log_dir: str | Path = "outputs/logs",
        experiment_name: str = "vrpbtw",
        wandb_project: Optional[str] = None,
        config_dict: Optional[Dict] = None,
    ):
        self.backend = backend
        self._tb_writer = None
        self._wandb = None

        if backend in ("tensorboard", "both"):
            from torch.utils.tensorboard import SummaryWriter

            self._tb_writer = SummaryWriter(
                log_dir=str(Path(log_dir) / experiment_name)
            )

        if backend in ("wandb", "both"):
            import wandb

            self._wandb = wandb
            wandb.init(
                project=wandb_project or "vrpbtw-rl",
                name=experiment_name,
                config=config_dict or {},
            )

    @classmethod
    def from_cfg(cls, cfg) -> "Logger":
        from omegaconf import OmegaConf

        return cls(
            backend=cfg.logging.backend,
            log_dir=cfg.logging.log_dir,
            experiment_name=cfg.experiment.name,
            wandb_project=cfg.logging.wandb_project,
            config_dict=OmegaConf.to_container(cfg, resolve=True),
        )

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self._tb_writer:
            for k, v in metrics.items():
                try:
                    self._tb_writer.add_scalar(k, float(v), global_step=step)
                except (TypeError, ValueError):
                    pass

        if self._wandb:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb:
            self._wandb.finish()
