"""
Base Trainer class following the Template Method pattern.

Follows SOLID principles:
- Single Responsibility: Only handles training orchestration
- Open/Closed: Open for extension through inheritance
- Liskov Substitution: Subclasses can replace base class
- Interface Segregation: Minimal interface for subclasses
- Dependency Inversion: Depends on abstractions (callbacks, config)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from src.config.schemas import ExperimentConfig
from src.callbacks.base_callback import Callback, CallbackList
from src.utils.logger import get_logger
from src.utils.device import setup_device, print_memory_usage
from src.utils.reproducibility import set_seed

logger = get_logger(__name__)


class QwenTrainerBase(ABC):
    """Abstract base class for all Qwen trainers.

    This class implements the Template Method pattern, defining the skeleton
    of the training algorithm while allowing subclasses to override specific steps.

    Named QwenTrainerBase to avoid conflicts with transformers.Trainer and trl trainers.

    Attributes:
        config: Experiment configuration
        model: The language model (with or without LoRA)
        tokenizer: The tokenizer
        device: torch device
        callbacks: List of callbacks

    Example:
        >>> class MyWrapper(QwenTrainerBase):
        ...     def _train_impl(self):
        ...         # Custom training logic
        ...         pass
    """

    def __init__(
        self,
        config: ExperimentConfig,
        callbacks: Optional[List[Callback]] = None
    ):
        """Initialize trainer.

        Args:
            config: Experiment configuration
            callbacks: Optional list of callbacks
        """
        self.config = config
        self.callbacks = CallbackList(callbacks or [])

        # Setup device and reproducibility
        self.device = setup_device()
        set_seed(config.training.seed)

        # Initialize model and tokenizer (lazy loading)
        self.model = None
        self.tokenizer = None

        logger.info(f"✅ {self.__class__.__name__} initialized")

    def load_model_and_tokenizer(self) -> None:
        """Load model and tokenizer with quantization.

        This method loads the model with 4-bit quantization and applies LoRA.
        """
        logger.info("="*60)
        logger.info("🤖 Loading Model and Tokenizer...")
        logger.info("="*60)

        model_name = self.config.model.name
        logger.info(f"   Model: {model_name}")

        # Create quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.quantization.load_in_4bit,
            bnb_4bit_quant_type=self.config.quantization.quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.config.quantization.compute_dtype),
            bnb_4bit_use_double_quant=self.config.quantization.use_double_quant,
        )

        # Load tokenizer
        logger.info("   Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config.model.trust_remote_code
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with quantization
        logger.info("   Loading model with 4-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.config.model.trust_remote_code,
        )

        logger.info("✅ Model and tokenizer loaded successfully!")
        print_memory_usage("   ")

    def apply_lora(self) -> None:
        """Apply LoRA to the model.

        This method prepares the model for k-bit training and applies LoRA adapters.
        """
        logger.info("="*60)
        logger.info("🔧 Applying LoRA...")
        logger.info("="*60)

        # Prepare for k-bit training
        logger.info("   Preparing for k-bit training...")
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.training.gradient_checkpointing
        )

        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            bias=self.config.lora.bias,
            task_type="CAUSAL_LM",
            target_modules=self.config.lora.target_modules,
        )

        # Apply LoRA
        logger.info(f"   Applying LoRA (rank={self.config.lora.rank})...")
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        logger.info("✅ LoRA applied successfully!")
        print_memory_usage("   ")

    def save_model(self, output_dir: str) -> None:
        """Save model and tokenizer.

        Args:
            output_dir: Directory to save model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"💾 Saving model to: {output_path}")

        # Save LoRA adapter
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        logger.info("✅ Model saved successfully!")

    @abstractmethod
    def prepare_data(self) -> Any:
        """Prepare training data.

        This method must be implemented by subclasses to load and prepare
        their specific datasets (SFT, DPO, etc.).

        Returns:
            Prepared dataset(s)
        """
        pass

    @abstractmethod
    def _train_impl(self) -> Dict[str, Any]:
        """Internal training implementation.

        This method must be implemented by subclasses to define their
        specific training logic.

        Returns:
            Dictionary with training results
        """
        pass

    def train(self) -> Dict[str, Any]:
        """Main training method (Template Method).

        This method defines the training skeleton:
        1. Trigger callbacks
        2. Load model
        3. Apply LoRA
        4. Prepare data
        5. Execute training
        6. Save results

        Returns:
            Dictionary with training results
        """
        logger.info("\n" + "="*60)
        logger.info(f"🚀 Starting {self.__class__.__name__} Training")
        logger.info("="*60 + "\n")

        # Callbacks: training begin
        self.callbacks.on_train_begin()

        try:
            # Load model and tokenizer
            if self.model is None:
                self.load_model_and_tokenizer()

            # Apply LoRA
            if not hasattr(self.model, 'peft_config'):
                self.apply_lora()

            # Prepare data
            logger.info("\n" + "="*60)
            logger.info("📚 Preparing Data...")
            logger.info("="*60)
            dataset = self.prepare_data()

            # Execute training (implemented by subclass)
            logger.info("\n" + "="*60)
            logger.info("🏃 Executing Training...")
            logger.info("="*60)
            results = self._train_impl()

            # Save model
            output_dir = self.config.training.output_dir
            self.save_model(output_dir)

            # Callbacks: training end
            self.callbacks.on_train_end(logs=results)

            logger.info("\n" + "="*60)
            logger.info("✅ Training Completed Successfully!")
            logger.info("="*60)

            return results

        except KeyboardInterrupt:
            logger.warning("\n⚠️  Training interrupted by user")
            raise

        except Exception as e:
            logger.error(f"\n❌ Training failed with error: {e}")
            raise

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model.

        Returns:
            Dictionary with evaluation results
        """
        logger.info("📊 Running evaluation...")
        # Default implementation - can be overridden
        return {}
