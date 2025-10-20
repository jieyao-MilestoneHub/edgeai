"""Base data processor interface."""

from abc import ABC, abstractmethod
from typing import Any
from datasets import Dataset


class DataProcessor(ABC):
    """Abstract base class for data processors.

    Follows Interface Segregation Principle - minimal interface.
    """

    @abstractmethod
    def load_data(self) -> Dataset:
        """Load raw data.

        Returns:
            Raw dataset
        """
        pass

    @abstractmethod
    def process(self, dataset: Dataset) -> Dataset:
        """Process dataset.

        Args:
            dataset: Raw dataset

        Returns:
            Processed dataset
        """
        pass

    @abstractmethod
    def format_example(self, example: dict) -> dict:
        """Format a single example.

        Args:
            example: Raw example

        Returns:
            Formatted example
        """
        pass
