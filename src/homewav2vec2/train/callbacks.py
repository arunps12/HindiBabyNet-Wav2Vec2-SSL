"""Training callbacks — logging, checkpointing helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("homewav2vec2.callbacks")


class MetricsLogger:
    """Simple JSON-lines logger for training metrics."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "train_metrics.jsonl"
        self._fh = open(self.log_path, "a")

    def log(self, step: int, metrics: dict) -> None:
        record = {"step": step, **metrics}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
