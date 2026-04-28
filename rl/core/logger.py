"""
core/logger.py
--------------
Structured experiment logger: console + JSONL file + optional backends.

Kept as a separate file (not merged into utils) because Logger has
meaningful state (file handles, TB writer) and a non-trivial
lifecycle (open → log → close).

Responsibilities
--------
  Metrics/events logging to JSONL, console, TensorBoard
  Checkpoint saving
  Config file saving
  Summary file saving

Backends
--------
  Console    : always active
  JSONL      : always written to log_dir/metrics.jsonl
  TensorBoard: opt-in (tensorboard=True)
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import torch


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
    Unified experiment logger with checkpoint and config persistence.

    Parameters
    ----------
    dir         : Experiment directory (base_dir/{experiment.name}).
    verbose     : Print to console.
    tensorboard : Enable TensorBoard SummaryWriter.

    Subdirectories created under dir:
    - logs/        : metrics.jsonl, tensorboard/
    - checkpoints/ : model checkpoints
    - artifacts/   : evaluation results, plots, etc.
    - config.json  : merged configuration
    """

    def __init__(
        self,
        dir: str,
        verbose: bool = True,
        tensorboard: bool = False,
    ):
        exp_dir = Path(dir)
        self.experiment_name = exp_dir.name
        self.log_dir = exp_dir / "logs"
        self.checkpoint_dir = exp_dir / "checkpoints"
        self.artifacts_dir = exp_dir / "artifacts"
        self.config_path = exp_dir / "config.yaml"
        self.verbose = verbose
        self._start_time = time.time()
        self._running: Dict[str, _RunningMean] = defaultdict(_RunningMean)

        # JSONL
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_fh = open(self.log_dir / "metrics.jsonl", "w")

        # TensorBoard
        self._tb_writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = self.log_dir / "tensorboard"
                self._tb_writer = SummaryWriter(str(tb_dir))
            except ImportError:
                print("[Logger] TensorBoard not installed; skipping.", file=sys.stderr)

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

    def save_checkpoint(self, tag: str, checkpoint_dict: Dict[str, Any]) -> None:
        """Save model checkpoint with tag."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{self.experiment_name}_{tag}.pt"
        torch.save(checkpoint_dict, path)
        if self.verbose:
            print(f"  Checkpoint saved: {path}", flush=True)

    def save_config(self, config_dict: Dict[str, Any]) -> None:
        """Save merged config to config.yaml."""
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("pyyaml is required. Install with: pip install pyyaml") from exc

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)
        if self.verbose:
            print(f"  Config saved: {self.config_path}", flush=True)

    def save_summary(self, summary: Dict[str, Any]) -> None:
        """Save final training summary to summary.json."""
        summary_path = self.config_path.parent / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        if self.verbose:
            print(f"  Summary saved: {summary_path}", flush=True)

    def close(self) -> None:
        self._jsonl_fh.close()
        if self._tb_writer:
            self._tb_writer.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_jsonl(self, entry: Dict) -> None:
        self._jsonl_fh.write(json.dumps(entry, default=str) + "\n")
        self._jsonl_fh.flush()

    def __repr__(self) -> str:
        return f"Logger(experiment={self.experiment_name!r}, dir={self.log_dir})"
