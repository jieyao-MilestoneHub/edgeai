"""
Configuration Manager with YAML loading and validation.

Follows:
- Single Responsibility Principle: Only handles configuration
- Open/Closed Principle: Extensible through config classes
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from src.config.schemas import (
    ExperimentConfig,
    ModelConfig,
    QuantizationConfig,
    LoRAConfig,
    DataConfig,
    TrainingConfig,
    DPOConfig,
    IterativeConfig,
    TaskType,
)


class ConfigManager:
    """Manages configuration loading, validation, and merging.

    This class provides a centralized way to handle all configuration needs,
    ensuring type safety and validation through dataclass schemas.

    Example:
        >>> manager = ConfigManager()
        >>> config = manager.load_config("config.yaml")
        >>> config.training.learning_rate
        0.0002
    """

    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """Load YAML configuration file.

        Args:
            file_path: Path to YAML file

        Returns:
            Dictionary containing configuration

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            raise ValueError(f"Empty config file: {file_path}")

        return config_dict

    @staticmethod
    def dict_to_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Validated ExperimentConfig instance
        """
        # Extract task type
        task_type_str = config_dict.get('task_type', 'sft')
        task_type = TaskType(task_type_str)

        # Parse model config
        model_dict = config_dict.get('model', {})
        model = ModelConfig(**model_dict)

        # Parse quantization config
        quant_dict = config_dict.get('quantization', {})
        quantization = QuantizationConfig(**quant_dict)

        # Parse LoRA config
        lora_dict = config_dict.get('lora', {})
        lora = LoRAConfig(**lora_dict)

        # Parse data config
        data_dict = config_dict.get('data', {})
        data = DataConfig(**data_dict)

        # Parse training config
        training_dict = config_dict.get('training', {})
        training = TrainingConfig(**training_dict)

        # Parse DPO config (optional)
        dpo = None
        if 'dpo' in config_dict:
            dpo_dict = config_dict['dpo']
            dpo = DPOConfig(**dpo_dict)

        # Parse iterative config (optional)
        iterative = None
        if 'iterative' in config_dict:
            iter_dict = config_dict['iterative']
            iterative = IterativeConfig(**iter_dict)

        # Create experiment config
        return ExperimentConfig(
            task_type=task_type,
            model=model,
            quantization=quantization,
            lora=lora,
            data=data,
            training=training,
            dpo=dpo,
            iterative=iterative,
        )

    @staticmethod
    def merge_configs(
        base_config: ExperimentConfig,
        override_dict: Dict[str, Any]
    ) -> ExperimentConfig:
        """Merge override configuration into base configuration.

        Args:
            base_config: Base configuration
            override_dict: Override values

        Returns:
            Merged configuration

        Example:
            >>> base = manager.load_config("base.yaml")
            >>> override = {"training": {"learning_rate": 1e-3}}
            >>> merged = manager.merge_configs(base, override)
        """
        # Convert base config to dict
        base_dict = asdict(base_config)

        # Deep merge override into base
        def deep_merge(base: Dict, override: Dict) -> Dict:
            """Recursively merge dictionaries."""
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged_dict = deep_merge(base_dict, override_dict)

        # Convert back to ExperimentConfig
        return ConfigManager.dict_to_config(merged_dict)

    @staticmethod
    def load_config(
        config_path: str,
        override_path: Optional[str] = None,
        override_dict: Optional[Dict[str, Any]] = None
    ) -> ExperimentConfig:
        """Load and merge configurations.

        Args:
            config_path: Path to base configuration file
            override_path: Optional path to override configuration file
            override_dict: Optional dictionary with override values

        Returns:
            Validated and merged ExperimentConfig

        Example:
            >>> # Load base config only
            >>> config = ConfigManager.load_config("config.yaml")
            >>>
            >>> # Load with override file
            >>> config = ConfigManager.load_config(
            ...     "config.yaml",
            ...     override_path="experiments/exp1.yaml"
            ... )
            >>>
            >>> # Load with dictionary override
            >>> config = ConfigManager.load_config(
            ...     "config.yaml",
            ...     override_dict={"training": {"learning_rate": 1e-3}}
            ... )
        """
        # Load base config
        base_dict = ConfigManager.load_yaml(config_path)
        config = ConfigManager.dict_to_config(base_dict)

        # Apply file override if provided
        if override_path:
            override_file_dict = ConfigManager.load_yaml(override_path)
            config = ConfigManager.merge_configs(config, override_file_dict)

        # Apply dictionary override if provided
        if override_dict:
            config = ConfigManager.merge_configs(config, override_dict)

        return config

    @staticmethod
    def save_config(config: ExperimentConfig, output_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            config: Configuration to save
            output_path: Path to output file
        """
        config_dict = asdict(config)

        # Convert enums to strings
        if 'task_type' in config_dict:
            config_dict['task_type'] = config_dict['task_type'].value

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )

    @staticmethod
    def validate_config(config: ExperimentConfig) -> bool:
        """Validate configuration consistency.

        Args:
            config: Configuration to validate

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Check task-specific requirements
        if config.task_type in [TaskType.DPO, TaskType.ITERATIVE]:
            if config.dpo is None:
                raise ValueError(f"DPO config required for task type: {config.task_type}")

        if config.task_type == TaskType.ITERATIVE:
            if config.iterative is None:
                raise ValueError(f"Iterative config required for task type: {config.task_type}")

        # Check data consistency
        if config.data.train_file:
            train_path = Path(config.data.train_file)
            if not train_path.exists():
                raise ValueError(f"Training data file not found: {config.data.train_file}")

        return True

    @staticmethod
    def print_config(config: ExperimentConfig) -> None:
        """Pretty print configuration.

        Args:
            config: Configuration to print
        """
        config_dict = asdict(config)
        print("\n" + "="*60)
        print("📋 Experiment Configuration")
        print("="*60 + "\n")
        print(yaml.dump(config_dict, default_flow_style=False, allow_unicode=True, sort_keys=False))
