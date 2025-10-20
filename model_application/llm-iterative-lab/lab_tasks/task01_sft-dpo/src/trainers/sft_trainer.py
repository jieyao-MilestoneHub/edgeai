"""SFT (Supervised Fine-Tuning) Trainer implementation."""

from typing import Dict, Any, Optional, List
from pathlib import Path

from trl import SFTTrainer as HF_SFTTrainer, SFTConfig
from datasets import Dataset

from src.trainers.base_trainer import QwenTrainerBase
from src.config.schemas import ExperimentConfig
from src.callbacks.base_callback import Callback
from src.data.instruction_processor import InstructionDataProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QwenSFTWrapper(QwenTrainerBase):
    """Supervised Fine-Tuning Wrapper using TRL's SFTTrainer.

    Named QwenSFTWrapper to avoid conflict with trl.SFTTrainer.
    This wrapper handles instruction fine-tuning for language models.

    Example:
        >>> config = ConfigManager.load_config("config.yaml")
        >>> trainer = QwenSFTWrapper(config)
        >>> results = trainer.train()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        callbacks: Optional[List[Callback]] = None
    ):
        """Initialize SFT trainer.

        Args:
            config: Experiment configuration
            callbacks: Optional list of callbacks
        """
        super().__init__(config, callbacks)
        self.data_processor = None
        self.hf_trainer = None

    def prepare_data(self) -> Dataset:
        """Prepare instruction-following dataset.

        Returns:
            Processed dataset
        """
        # Determine data path
        if self.config.data.train_file:
            data_path = self.config.data.train_file
        else:
            # Default path
            data_path = "data_preparation/final_dataset"

        # Create processor and load data
        self.data_processor = InstructionDataProcessor(data_path)
        dataset = self.data_processor.load_data()
        dataset = self.data_processor.process(dataset)

        logger.info(f"✅ Data prepared: {len(dataset['train'])} train, {len(dataset.get('test', []))} test samples")

        return dataset

    def _train_impl(self) -> Dict[str, Any]:
        """Execute SFT training.

        Returns:
            Training results dictionary
        """
        # Prepare dataset
        dataset = self.prepare_data()

        # Create SFT config
        training_args = SFTConfig(
            # Output
            output_dir=self.config.training.output_dir,
            overwrite_output_dir=True,

            # Training
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,

            # Learning rate
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type=self.config.training.lr_scheduler_type,

            # Precision
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,

            # Optimizer
            optim=self.config.training.optim,
            max_grad_norm=self.config.training.max_grad_norm,

            # Logging & Saving
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps,
            save_total_limit=self.config.training.save_total_limit,
            eval_strategy="steps" if 'test' in dataset else "no",

            # SFT specific
            max_seq_length=self.config.data.max_length,
            packing=False,  # Don't pack multiple samples

            # Other
            report_to="tensorboard",
            seed=self.config.training.seed,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
        )

        # Formatting function
        def formatting_func(example):
            return self.data_processor.format_example(example)

        # Create HF SFTTrainer
        logger.info("Creating SFTTrainer...")
        self.hf_trainer = HF_SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset.get('test'),
            processing_class=self.tokenizer,
            formatting_func=formatting_func,
        )

        # Train
        logger.info("Starting training...")
        train_result = self.hf_trainer.train()

        # Evaluate
        eval_results = {}
        if 'test' in dataset:
            logger.info("Running final evaluation...")
            eval_results = self.hf_trainer.evaluate()

        # Compile results
        results = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get('train_runtime', 0),
            "train_samples_per_second": train_result.metrics.get('train_samples_per_second', 0),
        }

        if eval_results:
            results["eval_loss"] = eval_results.get('eval_loss', 0)

        logger.info(f"Training completed! Final train loss: {results['train_loss']:.4f}")

        return results
