"""
Run iterative SFT+DPO training.

Usage:
    python run_iteration.py --config ../config.yaml --iterations 3
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ConfigManager
from src.trainers import QwenIterativeWrapper
from src.callbacks import LoggingCallback
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Iterative Training")
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
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations"
    )

    args = parser.parse_args()

    # Load configuration
    logger.info("Loading configuration...")
    config = ConfigManager.load_config(args.config, override_path=args.override)

    # Apply command-line overrides
    if args.output:
        config.training.output_dir = args.output
    if args.iterations:
        if config.iterative is None:
            from src.config.schemas import IterativeConfig
            config.iterative = IterativeConfig()
        config.iterative.num_iterations = args.iterations

    # Ensure task type is ITERATIVE and configs exist
    from src.config.schemas import TaskType, DPOConfig, IterativeConfig
    config.task_type = TaskType.ITERATIVE
    if config.dpo is None:
        config.dpo = DPOConfig()
    if config.iterative is None:
        config.iterative = IterativeConfig()

    # Print configuration
    ConfigManager.print_config(config)

    # Create callbacks
    callbacks = [
        LoggingCallback(
            log_file=f"{config.training.output_dir}/iterative_training.json",
            verbose=True
        ),
    ]

    # Create trainer
    logger.info("Creating Iterative Trainer...")
    trainer = QwenIterativeWrapper(config=config, callbacks=callbacks)

    # Train
    results = trainer.train()

    # Print results
    logger.info("\n" + "="*60)
    logger.info("🎉 Iterative Training Results:")
    logger.info("="*60)
    logger.info(f"   Total iterations: {results['total_iterations']}")
    logger.info(f"   Final SFT loss: {results['final_sft_loss']:.4f}")
    logger.info(f"   Final DPO loss: {results['final_dpo_loss']:.4f}")

    logger.info(f"\n✅ Final model saved to: {config.training.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
