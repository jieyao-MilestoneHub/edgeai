"""Evaluation callback for running evaluation during training."""

from typing import Any, Dict, Optional, Callable

from src.callbacks.base_callback import Callback
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationCallback(Callback):
    """Callback for running evaluation at specified intervals.

    Args:
        eval_function: Function to call for evaluation
        eval_frequency: Evaluate every N epochs
        verbose: Whether to print messages

    Example:
        >>> def my_eval():
        ...     return {"accuracy": 0.85}
        >>> callback = EvaluationCallback(eval_function=my_eval, eval_frequency=1)
    """

    def __init__(
        self,
        eval_function: Callable[[], Dict[str, Any]],
        eval_frequency: int = 1,
        verbose: bool = True
    ):
        self.eval_function = eval_function
        self.eval_frequency = eval_frequency
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Run evaluation if needed.

        Args:
            epoch: Current epoch number
            logs: Dictionary with epoch results
        """
        if (epoch + 1) % self.eval_frequency == 0:
            if self.verbose:
                logger.info(f"📊 Running evaluation at epoch {epoch + 1}...")

            eval_results = self.eval_function()

            if logs is not None:
                logs.update(eval_results)

            if self.verbose:
                metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in eval_results.items()])
                logger.info(f"Evaluation results: {metrics_str}")
