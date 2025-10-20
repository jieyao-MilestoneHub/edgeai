"""Preference data processor for DPO training."""

from typing import Dict
from datasets import Dataset, load_from_disk
from pathlib import Path

from src.data.base_processor import DataProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PreferenceDataProcessor(DataProcessor):
    """Processes preference pairs for DPO training.

    Expected format:
        {
            "prompt": str,
            "chosen": str,
            "rejected": str
        }
    """

    def __init__(self, data_path: str):
        """Initialize processor.

        Args:
            data_path: Path to preference dataset
        """
        self.data_path = Path(data_path)

    def load_data(self) -> Dataset:
        """Load preference dataset.

        Returns:
            Loaded dataset
        """
        logger.info(f"Loading preference data from: {self.data_path}")
        dataset = load_from_disk(str(self.data_path))
        logger.info(f"✅ Loaded {len(dataset.get('train', dataset))} preference pairs")
        return dataset

    def process(self, dataset: Dataset) -> Dataset:
        """Process preference dataset.

        Args:
            dataset: Raw dataset

        Returns:
            Processed dataset with required columns
        """
        required_columns = ['prompt', 'chosen', 'rejected']

        # Validate columns
        if isinstance(dataset, dict):
            for split in dataset:
                cols = dataset[split].column_names
                missing = set(required_columns) - set(cols)
                if missing:
                    raise ValueError(f"Missing columns in {split}: {missing}")
        else:
            cols = dataset.column_names
            missing = set(required_columns) - set(cols)
            if missing:
                raise ValueError(f"Missing columns: {missing}")

        return dataset

    def format_example(self, example: Dict) -> Dict:
        """Format preference pair for DPO.

        Args:
            example: Raw example with prompt, chosen, rejected

        Returns:
            Formatted example
        """
        # DPO expects this exact format
        return {
            "prompt": example['prompt'],
            "chosen": example['chosen'],
            "rejected": example['rejected']
        }
