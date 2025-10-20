"""DPO (Direct Preference Optimization) Trainer implementation."""

from typing import Dict, Any, Optional, List

from trl import DPOTrainer as HF_DPOTrainer, DPOConfig as HF_DPOConfig
from datasets import Dataset

from src.trainers.base_trainer import QwenTrainerBase
from src.config.schemas import ExperimentConfig
from src.callbacks.base_callback import Callback
from src.data.preference_processor import PreferenceDataProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QwenDPOWrapper(QwenTrainerBase):
    """Direct Preference Optimization Wrapper using TRL's DPOTrainer.

    Named QwenDPOWrapper to avoid conflict with trl.DPOTrainer.
    This wrapper aligns model outputs with human preferences using
    preference pairs (chosen vs rejected responses).

    Example:
        >>> config = ConfigManager.load_config("config.yaml")
        >>> trainer = QwenDPOWrapper(config)
        >>> results = trainer.train()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        callbacks: Optional[List[Callback]] = None,
        reference_model=None
    ):
        """Initialize DPO trainer.

        Args:
            config: Experiment configuration
            callbacks: Optional list of callbacks
            reference_model: Optional reference model (if None, will copy from main model)
        """
        super().__init__(config, callbacks)
        self.data_processor = None
        self.hf_trainer = None
        self.reference_model = reference_model

    def prepare_data(self) -> Dataset:
        """Prepare preference pair dataset.

        Returns:
            Processed dataset with prompt, chosen, rejected fields
        """
        # Determine data path
        if self.config.data.train_file:
            data_path = self.config.data.train_file
        else:
            # Default path for preference data
            data_path = "data/preference"

        # Create processor and load data
        self.data_processor = PreferenceDataProcessor(data_path)
        dataset = self.data_processor.load_data()
        dataset = self.data_processor.process(dataset)

        logger.info(f"✅ Preference data prepared: {len(dataset.get('train', dataset))} pairs")

        return dataset

    def _train_impl(self) -> Dict[str, Any]:
        """Execute DPO training.

        Returns:
            Training results dictionary
        """
        # Prepare dataset
        dataset = self.prepare_data()

        # Create DPO config
        training_args = HF_DPOConfig(
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
            eval_strategy="steps" if isinstance(dataset, dict) and 'test' in dataset else "no",

            # DPO specific
            beta=self.config.dpo.beta,
            label_smoothing=self.config.dpo.label_smoothing,
            loss_type=self.config.dpo.loss_type,
            max_length=self.config.dpo.max_length,
            max_prompt_length=self.config.dpo.max_prompt_length,

            # Other
            report_to="tensorboard",
            seed=self.config.training.seed,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
        )

        # Handle dataset format
        train_dataset = dataset['train'] if isinstance(dataset, dict) else dataset
        eval_dataset = dataset.get('test') if isinstance(dataset, dict) else None

        # Create HF DPOTrainer
        logger.info("Creating DPOTrainer...")
        self.hf_trainer = HF_DPOTrainer(
            model=self.model,
            ref_model=self.reference_model,  # Can be None - trainer will handle it
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        # Train
        logger.info("Starting DPO training...")
        train_result = self.hf_trainer.train()

        # Evaluate
        eval_results = {}
        if eval_dataset is not None:
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
            results["eval_accuracy"] = eval_results.get('eval_accuracy', 0)

        logger.info(f"DPO training completed! Final train loss: {results['train_loss']:.4f}")

        return results
