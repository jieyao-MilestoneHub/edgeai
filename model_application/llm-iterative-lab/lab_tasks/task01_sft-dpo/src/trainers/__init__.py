"""Trainer wrapper modules for SFT, DPO, Iterative, and Hybrid training.

All trainer classes are named with 'Qwen' prefix and 'Wrapper' suffix
to avoid conflicts with transformers.Trainer and trl trainers.
"""

from src.trainers.base_trainer import QwenTrainerBase
from src.trainers.sft_trainer import QwenSFTWrapper
from src.trainers.dpo_trainer import QwenDPOWrapper
from src.trainers.iterative_trainer import QwenIterativeWrapper
from src.trainers.hybrid_trainer import QwenHybridWrapper

__all__ = [
    "QwenTrainerBase",
    "QwenSFTWrapper",
    "QwenDPOWrapper",
    "QwenIterativeWrapper",
    "QwenHybridWrapper"
]
