"""Logging callback for tracking training progress."""

from typing import Any, Dict, Optional
from pathlib import Path
import json

from src.callbacks.base_callback import Callback
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LoggingCallback(Callback):
    """Callback for logging training metrics.

    Args:
        log_file: Path to save logs (JSON format)
        verbose: Whether to print to console

    Example:
        >>> callback = LoggingCallback(log_file="logs/training.json")
    """

    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        self.log_file = Path(log_file) if log_file else None
        self.verbose = verbose
        self.history = []

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log training start."""
        if self.verbose:
            logger.info("🚀 Training started")

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log training end and save history."""
        if self.verbose:
            logger.info("✅ Training completed")

        if self.log_file:
            with open(self.log_file, 'w') as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"📝 Training history saved to: {self.log_file}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log epoch results.

        Args:
            epoch: Current epoch number
            logs: Dictionary with epoch results
        """
        if logs is None:
            return

        # Add to history
        epoch_log = {"epoch": epoch + 1, **logs}
        self.history.append(epoch_log)

        # Print to console
        if self.verbose:
            metrics_str = " - ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                     for k, v in logs.items()])
            logger.info(f"Epoch {epoch + 1}: {metrics_str}")
