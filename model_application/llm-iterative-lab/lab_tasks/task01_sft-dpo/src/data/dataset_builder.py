"""Dataset builder factory for creating different dataset types."""

from typing import Union
from datasets import Dataset

from src.data.base_processor import DataProcessor
from src.data.instruction_processor import InstructionDataProcessor
from src.data.preference_processor import PreferenceDataProcessor
from src.config.schemas import TaskType


class DatasetBuilder:
    """Factory for building datasets based on task type.

    Follows the Factory Pattern for object creation.
    """

    @staticmethod
    def create_processor(
        task_type: TaskType,
        data_path: str
    ) -> DataProcessor:
        """Create appropriate data processor based on task type.

        Args:
            task_type: Type of training task
            data_path: Path to dataset

        Returns:
            Configured DataProcessor instance

        Raises:
            ValueError: If task type is not supported
        """
        if task_type == TaskType.SFT:
            return InstructionDataProcessor(data_path)
        elif task_type in [TaskType.DPO, TaskType.ITERATIVE]:
            return PreferenceDataProcessor(data_path)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    @staticmethod
    def build_dataset(
        task_type: TaskType,
        data_path: str
    ) -> Dataset:
        """Build complete dataset for given task.

        Args:
            task_type: Type of training task
            data_path: Path to dataset

        Returns:
            Processed dataset ready for training
        """
        processor = DatasetBuilder.create_processor(task_type, data_path)
        raw_dataset = processor.load_data()
        processed_dataset = processor.process(raw_dataset)
        return processed_dataset
