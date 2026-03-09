"""
src/utils/checkpoint.py
─────────────────────────────────────────────────────────────────────────────
Manages saving, loading, and rotating agent checkpoints.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional


class CheckpointManager:
    """
    Handles:
    • Saving periodic checkpoints       (checkpoint_episode_{N}.pt)
    • Tracking and saving the best model (best.pt)
    • Rotating old checkpoints          (keep only last N)
    • Writing a JSON metadata sidecar   (metadata.json)

    Usage
    ─────
        mgr = CheckpointManager(dir="outputs/checkpoints/run_01", keep_last_n=5)
        mgr.save(agent, episode=100, metric=0.92, higher_is_better=True)
        mgr.load_best(agent)
    """

    BEST_NAME = "best.pt"
    META_NAME = "metadata.json"

    def __init__(
        self,
        directory: str | Path,
        keep_last_n: int = 5,
    ):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self._saved: List[Path] = []
        self._best_metric: Optional[float] = None
        self._meta: dict = {}

    def save(
        self,
        agent,
        episode: int,
        metric: float,
        higher_is_better: bool = True,
        extra_meta: Optional[dict] = None,
    ) -> Path:
        """Save a periodic checkpoint and update best if metric improved."""
        path = self.dir / f"checkpoint_ep{episode:06d}.pt"
        agent.save(path)
        self._saved.append(path)

        # Update best
        is_best = (
            self._best_metric is None
            or (higher_is_better and metric > self._best_metric)
            or (not higher_is_better and metric < self._best_metric)
        )
        if is_best:
            self._best_metric = metric
            shutil.copy2(path, self.dir / self.BEST_NAME)

        # Rotate old checkpoints
        self._rotate()

        # Write metadata
        self._meta[episode] = {
            "path": str(path),
            "metric": metric,
            "is_best": is_best,
            **(extra_meta or {}),
        }
        self._write_meta()

        return path

    def load_best(self, agent) -> None:
        best = self.dir / self.BEST_NAME
        if not best.exists():
            raise FileNotFoundError(f"No best checkpoint found at {best}")
        agent.load(best)

    def load_latest(self, agent) -> None:
        if not self._saved:
            raise RuntimeError("No checkpoints have been saved in this session.")
        agent.load(self._saved[-1])

    def load_episode(self, agent, episode: int) -> None:
        path = self.dir / f"checkpoint_ep{episode:06d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        agent.load(path)

    # ─────────────────────────────────────────────────────────────────────
    def _rotate(self):
        while len(self._saved) > self.keep_last_n:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink()

    def _write_meta(self):
        meta_path = self.dir / self.META_NAME
        with open(meta_path, "w") as f:
            json.dump(self._meta, f, indent=2)
