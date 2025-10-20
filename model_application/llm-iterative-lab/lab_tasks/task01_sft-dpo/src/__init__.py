"""
Enterprise-grade Qwen2.5-3B SFT + DPO Joint Optimization Framework

This package provides a modular, SOLID-compliant framework for:
- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- Iterative closed-loop training with continuous improvement

Author: Joel
License: MIT
"""

__version__ = "2.0.0"
__author__ = "Joel"

from src.trainers import BaseTrainer, SFTTrainer, DPOTrainer, IterativeTrainer
from src.config import ConfigManager, TrainingConfig, ModelConfig, DataConfig
from src.evaluation import Evaluator, InstructionEvaluator, PreferenceEvaluator

__all__ = [
    "BaseTrainer",
    "SFTTrainer",
    "DPOTrainer",
    "IterativeTrainer",
    "ConfigManager",
    "TrainingConfig",
    "ModelConfig",
    "DataConfig",
    "Evaluator",
    "InstructionEvaluator",
    "PreferenceEvaluator",
]
