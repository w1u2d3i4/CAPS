#!/usr/bin/env python3
"""
Metric writer helper — call from training scripts to write metrics
in the format that supervisor.py can read.

Usage in training code:
    from autoresearch.metric_writer import MetricWriter

    writer = MetricWriter("results/train_metrics.json")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(...)
        val_metrics = evaluate(...)

        writer.log(epoch, {
            "loss": train_loss,
            "val_loss": val_metrics["loss"],
            "avg_missing_top1": val_metrics.get("avg_missing_top1", 0),
            "s1_top1": val_metrics.get("s1_top1", 0),
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class MetricWriter:
    """Write metrics in the format expected by supervisor.py."""

    def __init__(self, output_path: str, max_history: int = 200):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.history = []
        self.best = {}
        self.best_epoch = {}

        # Load existing if resuming
        if self.output_path.exists():
            try:
                with open(self.output_path) as f:
                    data = json.load(f)
                    self.history = data.get("history", [])
                    self.best = data.get("best", {})
                    self.best_epoch = data.get("best_epoch", {})
            except (json.JSONDecodeError, IOError):
                pass

    def log(self, epoch: int, metrics: dict, direction_map: Optional[dict] = None):
        """
        Log metrics for an epoch.

        Args:
            epoch: Current epoch number
            metrics: Dict of metric_name -> value
            direction_map: Dict of metric_name -> "higher" or "lower" (default: "lower" for all)
                          Used to track best values.
        """
        if direction_map is None:
            direction_map = {}

        entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        self.history.append(entry)

        # Trim history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # Update best
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            direction = direction_map.get(k, "lower")
            if k not in self.best:
                self.best[k] = v
                self.best_epoch[k] = epoch
            elif direction == "lower" and v < self.best[k]:
                self.best[k] = v
                self.best_epoch[k] = epoch
            elif direction == "higher" and v > self.best[k]:
                self.best[k] = v
                self.best_epoch[k] = epoch

        # Write JSON atomically
        data = {
            "current": metrics,
            "current_epoch": epoch,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "updated_at": datetime.now().isoformat(),
        }

        tmp_path = self.output_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.output_path)

    def log_final(self, summary: dict):
        """Log final summary when training completes."""
        data = {
            "current": summary,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "final": True,
            "completed_at": datetime.now().isoformat(),
        }
        tmp_path = self.output_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.output_path)
