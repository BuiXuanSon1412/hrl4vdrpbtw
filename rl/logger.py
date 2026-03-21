"""
utils/logger.py
---------------
Structured experiment logger with console + JSONL file output.

Design
------
- One Logger per experiment, created by the Trainer.
- All metrics arrive as plain dicts with a step tag.
- Console output is human-readable; file output is machine-readable JSONL.
- Supports optional TensorBoard / W&B backends without coupling to them.
- Tracks running statistics (window-mean) for smooth console display.

Usage
-----
from utils.logger import Logger

logger = Logger(log_dir="logs", experiment_name="ppo_knapsack")
logger.log_metrics({"loss": 0.5, "reward": 12.3}, step=1000, prefix="train")
logger.log_metrics({"objective": 87.2}, step=1000, prefix="eval")
logger.close()
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunningMean:
    """Tracks a windowed running mean for display purposes."""

    def __init__(self, window: int = 100):
        self._buf: deque = deque(maxlen=window)

    def update(self, value: float) -> None:
        self._buf.append(value)

    @property
    def mean(self) -> float:
        return float(sum(self._buf) / len(self._buf)) if self._buf else 0.0

    @property
    def latest(self) -> float:
        return float(self._buf[-1]) if self._buf else 0.0


class Logger:
    """
    Unified experiment logger.

    Backends
    --------
    - Console  : always active, controlled by ``verbose``
    - JSONL    : always written to ``log_dir/experiment_name.jsonl``
    - TensorBoard: enabled if ``tensorboard=True`` and tensorboard installed
    - W&B      : enabled if ``wandb_project`` is set and wandb installed
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        verbose: bool = True,
        tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.verbose = verbose
        self._start_time = time.time()
        self._running: Dict[str, RunningMean] = defaultdict(RunningMean)

        # JSONL file
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self.log_dir / f"{experiment_name}.jsonl"
        self._jsonl_fh = open(self._jsonl_path, "w")

        # Optional TensorBoard
        self._tb_writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = self.log_dir / "tensorboard" / experiment_name
                self._tb_writer = SummaryWriter(str(tb_dir))
            except ImportError:
                print("[Logger] TensorBoard not installed; skipping.", file=sys.stderr)

        # Optional W&B
        self._wandb = None
        if wandb_project:
            try:
                import wandb

                wandb.init(project=wandb_project, name=experiment_name, config=config)
                self._wandb = wandb
            except ImportError:
                print("[Logger] wandb not installed; skipping.", file=sys.stderr)

        if config:
            self._write_jsonl({"event": "config", "config": config, "step": 0})

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = "",
        print_keys: Optional[List[str]] = None,
    ) -> None:
        """
        Log a dict of scalar metrics.

        Parameters
        ----------
        metrics    : Dict of name → scalar value.
        step       : Global training step (for x-axis alignment across plots).
        prefix     : Prepended to all keys in file/TB (e.g. "train", "eval").
        print_keys : Subset of keys to display on console (None = all).
        """
        prefixed = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}

        # Update running means
        for k, v in prefixed.items():
            if isinstance(v, (int, float)):
                self._running[k].update(float(v))

        # JSONL
        self._write_jsonl({"step": step, **prefixed})

        # TensorBoard
        if self._tb_writer:
            for k, v in prefixed.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, global_step=step)

        # W&B
        if self._wandb:
            self._wandb.log({**prefixed, "global_step": step})

        # Console
        if self.verbose:
            keys_to_show = set(print_keys) if print_keys else set(prefixed.keys())
            display = {k: v for k, v in prefixed.items() if k in keys_to_show}
            if display:
                elapsed = time.time() - self._start_time
                row = "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in display.items()
                )
                print(f"[{elapsed:7.1f}s | step={step:>9,}]  {row}", flush=True)

    def log_event(self, event: str, step: int, **kwargs) -> None:
        """Log a named event (checkpoint saved, early stop, etc.)."""
        entry = {"event": event, "step": step, **kwargs}
        self._write_jsonl(entry)
        if self.verbose:
            print(
                f"  ▸ {event}  step={step:,}  "
                + "  ".join(f"{k}={v}" for k, v in kwargs.items()),
                flush=True,
            )

    def running_mean(self, key: str) -> float:
        """Return the windowed running mean for a metric key."""
        return self._running[key].mean

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_jsonl(self, entry: Dict) -> None:
        self._jsonl_fh.write(json.dumps(entry, default=str) + "\n")
        self._jsonl_fh.flush()

    def close(self) -> None:
        self._jsonl_fh.close()
        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb:
            self._wandb.finish()

    def __repr__(self) -> str:
        return f"Logger(experiment={self.experiment_name!r}, log_dir={self.log_dir})"
