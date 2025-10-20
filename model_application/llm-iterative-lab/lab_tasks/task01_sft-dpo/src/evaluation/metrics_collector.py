"""Metrics collector for training/evaluation."""

from typing import Dict, Any, List
import json
from pathlib import Path


class MetricsCollector:
    """Collects and saves training/evaluation metrics."""

    def __init__(self, output_dir: str):
        """Initialize metrics collector.

        Args:
            output_dir: Directory to save metrics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []

    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add metrics to history.

        Args:
            metrics: Metrics dictionary
        """
        self.metrics_history.append(metrics)

    def save(self, filename: str = "metrics.json") -> None:
        """Save metrics to file.

        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def get_latest(self) -> Dict[str, Any]:
        """Get latest metrics.

        Returns:
            Latest metrics dictionary
        """
        return self.metrics_history[-1] if self.metrics_history else {}
