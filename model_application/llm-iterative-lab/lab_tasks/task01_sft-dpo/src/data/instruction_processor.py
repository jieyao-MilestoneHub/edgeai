"""Instruction data processor for SFT training."""

from typing import Optional
from datasets import Dataset, load_from_disk
from pathlib import Path

from src.data.base_processor import DataProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InstructionDataProcessor(DataProcessor):
    """Processes instruction-following data for SFT.

    Expected format:
        {
            "instruction": str,
            "input": str (optional),
            "output": str
        }
    """

    def __init__(self, data_path: str):
        """Initialize processor.

        Args:
            data_path: Path to dataset
        """
        self.data_path = Path(data_path)

    def load_data(self) -> Dataset:
        """Load instruction dataset.

        Returns:
            Loaded dataset
        """
        logger.info(f"Loading instruction data from: {self.data_path}")
        dataset = load_from_disk(str(self.data_path))
        logger.info(f"✅ Loaded {len(dataset['train'])} training samples")
        return dataset

    def process(self, dataset: Dataset) -> Dataset:
        """Process instruction dataset.

        Args:
            dataset: Raw dataset

        Returns:
            Processed dataset
        """
        # Filter required columns
        required_columns = ['instruction', 'input', 'output']
        if 'train' in dataset:
            dataset['train'] = dataset['train'].select_columns(required_columns)
        if 'test' in dataset:
            dataset['test'] = dataset['test'].select_columns(required_columns)

        return dataset

    def format_example(self, example: dict) -> str:
        """Format example for Qwen chat template.

        Args:
            example: Raw example

        Returns:
            Formatted prompt string
        """
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')

        # Construct Qwen chat format
        prompt = "<|im_start|>system\nYou are a helpful AI assistant specialized in engineering and technical domains.<|im_end|>\n"
        prompt += f"<|im_start|>user\n{instruction}"

        if input_text:
            prompt += f"\n\n{input_text}"

        prompt += "<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{output}<|im_end|>"

        return prompt
