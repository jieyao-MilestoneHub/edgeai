"""
Run DPO (Direct Preference Optimization) training.

Usage:
    python run_dpo.py --config ../config.yaml --output experiments/dpo_r16
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ConfigManager
from src.trainers import QwenDPOWrapper
from src.callbacks import LoggingCallback, CheckpointCallback
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run DPO Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Path to override config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override preference data path"
    )

    args = parser.parse_args()

    # Load configuration
    logger.info("Loading configuration...")
    config = ConfigManager.load_config(args.config, override_path=args.override)

    # Apply command-line overrides
    if args.output:
        config.training.output_dir = args.output
    if args.data:
        config.data.train_file = args.data

    # Ensure task type is DPO and DPO config exists
    from src.config.schemas import TaskType, DPOConfig
    config.task_type = TaskType.DPO
    if config.dpo is None:
        config.dpo = DPOConfig()

    # Print configuration
    ConfigManager.print_config(config)

    # Create callbacks
    callbacks = [
        LoggingCallback(
            log_file=f"{config.training.output_dir}/training.json",
            verbose=True
        ),
        CheckpointCallback(
            checkpoint_dir=config.training.output_dir,
            save_frequency=0,  # Only save best
            monitor="eval_loss",
            mode="min"
        ),
    ]

    # Create trainer
    logger.info("Creating DPO Trainer...")
    trainer = QwenDPOWrapper(config=config, callbacks=callbacks)

    # Train
    results = trainer.train()

    # Print results
    logger.info("\n" + "="*60)
    logger.info("🎉 Training Results:")
    logger.info("="*60)
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.4f}")
        else:
            logger.info(f"   {key}: {value}")

    logger.info(f"\n✅ Model saved to: {config.training.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
