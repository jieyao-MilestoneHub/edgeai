"""Checkpoint callback for saving model during training."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.callbacks.base_callback import Callback
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointCallback(Callback):
    """Callback for saving model checkpoints.

    Args:
        checkpoint_dir: Directory to save checkpoints
        save_frequency: Save every N epochs (0 = only save best)
        monitor: Metric to monitor for best model
        mode: 'min' or 'max' for the monitored metric
        verbose: Whether to print messages

    Example:
        >>> callback = CheckpointCallback(
        ...     checkpoint_dir="checkpoints",
        ...     save_frequency=1,
        ...     monitor="eval_loss",
        ...     mode="min"
        ... )
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_frequency: int = 1,
        monitor: str = "eval_loss",
        mode: str = "min",
        verbose: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_frequency = save_frequency
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save checkpoint if conditions are met.

        Args:
            epoch: Current epoch number
            logs: Dictionary with epoch results
        """
        if logs is None:
            return

        # Save periodically
        if self.save_frequency > 0 and (epoch + 1) % self.save_frequency == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}"
            self._save_checkpoint(checkpoint_path, epoch, logs)

        # Save best model
        if self.monitor in logs:
            current_value = logs[self.monitor]

            is_better = (
                (self.mode == 'min' and current_value < self.best_value) or
                (self.mode == 'max' and current_value > self.best_value)
            )

            if is_better:
                self.best_value = current_value
                self.best_epoch = epoch

                checkpoint_path = self.checkpoint_dir / "best_model"
                self._save_checkpoint(checkpoint_path, epoch, logs)

                if self.verbose:
                    logger.info(
                        f"💾 New best model saved! "
                        f"{self.monitor}={current_value:.4f} (epoch {epoch + 1})"
                    )

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        epoch: int,
        logs: Dict[str, Any]
    ) -> None:
        """Save a checkpoint.

        Note: Actual model saving is handled by the trainer.
        This method can be extended to save additional metadata.

        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch
            logs: Training logs
        """
        # Create metadata file
        metadata = {
            "epoch": epoch + 1,
            "metrics": logs,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch + 1,
        }

        metadata_path = checkpoint_path.parent / f"{checkpoint_path.name}_metadata.txt"
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")

        if self.verbose:
            logger.info(f"📝 Checkpoint metadata saved: {metadata_path}")
