"""
Base callback class for training hooks.

Follows the Open/Closed Principle: Extensible through inheritance.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Callback(ABC):
    """Abstract base class for training callbacks.

    Callbacks provide hooks into the training process, allowing custom
    behavior at various stages without modifying the trainer code.

    Example:
        >>> class CustomCallback(Callback):
        ...     def on_epoch_end(self, epoch, logs):
        ...         print(f"Epoch {epoch} finished with loss: {logs['loss']}")
    """

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training.

        Args:
            logs: Dictionary with training information
        """
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training.

        Args:
            logs: Dictionary with training results
        """
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary with epoch information
        """
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary with epoch results
        """
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each batch.

        Args:
            batch: Current batch number
            logs: Dictionary with batch information
        """
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each batch.

        Args:
            batch: Current batch number
            logs: Dictionary with batch results
        """
        pass

    def on_evaluation_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of evaluation.

        Args:
            logs: Dictionary with evaluation information
        """
        pass

    def on_evaluation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of evaluation.

        Args:
            logs: Dictionary with evaluation results
        """
        pass


class CallbackList:
    """Container for managing multiple callbacks.

    Example:
        >>> callbacks = CallbackList([callback1, callback2])
        >>> callbacks.on_epoch_end(epoch=1, logs={'loss': 0.5})
    """

    def __init__(self, callbacks: list[Callback]):
        """Initialize callback list.

        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_evaluation_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_evaluation_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_evaluation_begin(logs)

    def on_evaluation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_evaluation_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_evaluation_end(logs)
