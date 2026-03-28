"""
core/logger.py
--------------
Structured experiment logger: console + JSONL file + optional backends.

Kept as a separate file (not merged into utils) because Logger has
meaningful state (file handles, TB writer, W&B session) and a non-trivial
lifecycle (open → log → close).

Backends
--------
  Console    : always active
  JSONL      : always written to log_dir/experiment_name.jsonl
  TensorBoard: opt-in (tensorboard=True)
  W&B        : opt-in (wandb_project="my-project")
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


class _RunningMean:
    def __init__(self, window: int = 100):
        self._buf: Deque[float] = deque(maxlen=window)

    def update(self, value: float) -> None:
        self._buf.append(value)

    @property
    def mean(self) -> float:
        return float(sum(self._buf) / len(self._buf)) if self._buf else 0.0


class Logger:
    """
    Unified experiment logger.

    Parameters
    ----------
    log_dir         : Directory where the JSONL file is written.
    experiment_name : Used as the JSONL filename and W&B run name.
    verbose         : Print to console.
    tensorboard     : Enable TensorBoard SummaryWriter.
    wandb_project   : W&B project name (None → disabled).
    config          : Optional config dict logged as the first entry.
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
        self._running: Dict[str, _RunningMean] = defaultdict(_RunningMean)

        # JSONL
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self.log_dir / f"{experiment_name}.jsonl"
        self._jsonl_fh = open(self._jsonl_path, "w")

        # TensorBoard
        self._tb_writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = self.log_dir / "tensorboard" / experiment_name
                self._tb_writer = SummaryWriter(str(tb_dir))
            except ImportError:
                print("[Logger] TensorBoard not installed; skipping.", file=sys.stderr)

        # W&B
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
    # Logging
    # ------------------------------------------------------------------

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = "",
        print_keys: Optional[List[str]] = None,
    ) -> None:
        prefixed = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}

        for k, v in prefixed.items():
            if isinstance(v, (int, float)):
                self._running[k].update(float(v))

        self._write_jsonl({"step": step, **prefixed})

        if self._tb_writer:
            for k, v in prefixed.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, global_step=step)

        if self._wandb:
            self._wandb.log({**prefixed, "global_step": step})

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

    def log_event(self, event: str, step: int, **kwargs: Any) -> None:
        entry = {"event": event, "step": step, **kwargs}
        self._write_jsonl(entry)
        if self.verbose:
            kv = "  ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"  ▸ {event}  step={step:,}  {kv}", flush=True)

    def running_mean(self, key: str) -> float:
        return self._running[key].mean

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._jsonl_fh.close()
        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb:
            self._wandb.finish()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_jsonl(self, entry: Dict) -> None:
        self._jsonl_fh.write(json.dumps(entry, default=str) + "\n")
        self._jsonl_fh.flush()

    def __repr__(self) -> str:
        return f"Logger(experiment={self.experiment_name!r}, dir={self.log_dir})"
