"""Base evaluator interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class Evaluator(ABC):
    """Abstract base class for evaluators."""

    @abstractmethod
    def evaluate(self, model, dataset) -> Dict[str, Any]:
        """Evaluate model on dataset.

        Args:
            model: Model to evaluate
            dataset: Dataset for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        pass
