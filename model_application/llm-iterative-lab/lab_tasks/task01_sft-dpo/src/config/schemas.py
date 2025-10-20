"""
Configuration schemas using dataclasses for type safety.

Follows the Single Responsibility Principle (SRP):
- Each config class handles one aspect of configuration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class TaskType(str, Enum):
    """Training task types."""
    SFT = "sft"
    DPO = "dpo"
    ITERATIVE = "iterative"
    HYBRID = "hybrid"


class OptimType(str, Enum):
    """Optimizer types."""
    ADAMW_8BIT = "paged_adamw_8bit"
    ADAMW = "adamw_torch"
    SGD = "sgd"


class SchedulerType(str, Enum):
    """Learning rate scheduler types."""
    LINEAR = "linear"
    COSINE = "cosine"
    CONSTANT = "constant"


@dataclass
class ModelConfig:
    """Model configuration.

    Attributes:
        name: Model identifier (e.g., "Qwen/Qwen2.5-3B")
        trust_remote_code: Whether to trust remote code
        use_flash_attention: Whether to use Flash Attention 2
    """
    name: str = "Qwen/Qwen2.5-3B"
    trust_remote_code: bool = True
    use_flash_attention: bool = False

    def __post_init__(self):
        """Validate model config."""
        if not self.name:
            raise ValueError("Model name cannot be empty")


@dataclass
class QuantizationConfig:
    """Quantization configuration for QLoRA.

    Attributes:
        load_in_4bit: Use 4-bit quantization
        quant_type: Quantization type (nf4, fp4)
        compute_dtype: Compute dtype (bfloat16, float16)
        use_double_quant: Use double quantization
    """
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration.

    Attributes:
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: LoRA dropout
        bias: Bias type (none, all, lora_only)
        target_modules: Modules to apply LoRA
    """
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    def __post_init__(self):
        """Validate LoRA config."""
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}")


@dataclass
class DataConfig:
    """Data configuration.

    Attributes:
        max_length: Maximum sequence length
        train_file: Training data file path
        eval_file: Evaluation data file path
        test_size: Test set ratio
        preprocessing_num_proc: Number of preprocessing processes
    """
    max_length: int = 512
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    test_size: float = 0.1
    preprocessing_num_proc: int = 4

    def __post_init__(self):
        """Validate data config."""
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")


@dataclass
class TrainingConfig:
    """Training configuration.

    Attributes:
        output_dir: Output directory
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Warmup ratio
        lr_scheduler_type: Learning rate scheduler type
        fp16: Use FP16 mixed precision
        bf16: Use BF16 mixed precision
        optim: Optimizer type
        max_grad_norm: Maximum gradient norm for clipping
        gradient_checkpointing: Use gradient checkpointing
        logging_steps: Logging frequency
        save_steps: Save checkpoint frequency
        eval_steps: Evaluation frequency
        save_total_limit: Maximum checkpoints to keep
        seed: Random seed
    """
    output_dir: str = "experiments/default"
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    seed: int = 42

    def __post_init__(self):
        """Validate training config."""
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both FP16 and BF16")


@dataclass
class DPOConfig:
    """DPO-specific configuration.

    Attributes:
        beta: DPO beta parameter (temperature)
        label_smoothing: Label smoothing for DPO loss
        loss_type: DPO loss type (sigmoid, hinge, ipo)
        reference_free: Whether to use reference-free DPO
        max_length: Maximum sequence length for chosen/rejected
        max_prompt_length: Maximum prompt length
    """
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    reference_free: bool = False
    max_length: int = 512
    max_prompt_length: int = 256

    def __post_init__(self):
        """Validate DPO config."""
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.loss_type not in ["sigmoid", "hinge", "ipo"]:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")


@dataclass
class IterativeConfig:
    """Iterative training configuration.

    Attributes:
        num_iterations: Number of SFT-DPO iterations
        sft_epochs_per_iter: SFT epochs per iteration
        dpo_epochs_per_iter: DPO epochs per iteration
        annotation_samples_per_iter: Number of samples to annotate per iteration
        evaluation_threshold: Evaluation score threshold to continue
    """
    num_iterations: int = 3
    sft_epochs_per_iter: int = 2
    dpo_epochs_per_iter: int = 1
    annotation_samples_per_iter: int = 100
    evaluation_threshold: float = 0.7

    def __post_init__(self):
        """Validate iterative config."""
        if self.num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {self.num_iterations}")


@dataclass
class HybridConfig:
    """Hybrid training configuration (SFT-dominant with micro-DPO triggers).

    This implements a closed-loop training paradigm:
    數據收集 → SFT → DPO(條件觸發) → 評估 → 數據收集 → ...

    Attributes:
        num_iterations: Maximum number of closed-loop iterations
        sft_steps_per_iteration: SFT training steps per iteration
        preference_batch_size: Number of preference pairs to trigger micro-DPO
        micro_dpo_steps: Training steps for each micro-DPO session
        micro_dpo_lr: Learning rate for micro-DPO (typically lower than SFT)
        quality_threshold: Minimum quality score to accept preference pair
        preference_data_file: Path to preference pairs JSONL file
        evaluation_interval: Evaluate every N iterations (0 = no evaluation)
        evaluation_threshold: Stop if evaluation score >= threshold
    """
    num_iterations: int = 10
    sft_steps_per_iteration: int = 100
    preference_batch_size: int = 5
    micro_dpo_steps: int = 50
    micro_dpo_lr: float = 5e-5
    quality_threshold: float = 0.7
    preference_data_file: Optional[str] = None
    evaluation_interval: int = 1
    evaluation_threshold: float = 0.95

    def __post_init__(self):
        """Validate hybrid config."""
        if self.num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {self.num_iterations}")
        if self.sft_steps_per_iteration <= 0:
            raise ValueError(f"sft_steps_per_iteration must be positive, got {self.sft_steps_per_iteration}")
        if self.preference_batch_size <= 0:
            raise ValueError(f"preference_batch_size must be positive, got {self.preference_batch_size}")
        if self.micro_dpo_steps <= 0:
            raise ValueError(f"micro_dpo_steps must be positive, got {self.micro_dpo_steps}")
        if self.micro_dpo_lr <= 0:
            raise ValueError(f"micro_dpo_lr must be positive, got {self.micro_dpo_lr}")
        if not 0 <= self.quality_threshold <= 1:
            raise ValueError(f"quality_threshold must be in [0, 1], got {self.quality_threshold}")
        if not 0 <= self.evaluation_threshold <= 1:
            raise ValueError(f"evaluation_threshold must be in [0, 1], got {self.evaluation_threshold}")


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.

    This is the main config class that aggregates all sub-configs.
    """
    task_type: TaskType = TaskType.SFT
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dpo: Optional[DPOConfig] = None
    iterative: Optional[IterativeConfig] = None
    hybrid: Optional[HybridConfig] = None

    def __post_init__(self):
        """Validate experiment config."""
        # Ensure DPO config exists for DPO tasks
        if self.task_type in [TaskType.DPO, TaskType.ITERATIVE, TaskType.HYBRID]:
            if self.dpo is None:
                self.dpo = DPOConfig()

        # Ensure iterative config exists for iterative tasks
        if self.task_type == TaskType.ITERATIVE:
            if self.iterative is None:
                self.iterative = IterativeConfig()

        # Ensure hybrid config exists for hybrid tasks
        if self.task_type == TaskType.HYBRID:
            if self.hybrid is None:
                self.hybrid = HybridConfig()
