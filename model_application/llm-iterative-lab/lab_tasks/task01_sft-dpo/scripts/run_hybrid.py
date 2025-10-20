"""
Run Hybrid Training (SFT + Micro-DPO).

This script runs the hybrid training mode where:
- Main training: SFT on instruction data
- Micro-DPO: Triggered every N preference pairs for quick alignment

Usage:
    python run_hybrid.py --config ../config.yaml --override ../configs/hybrid/default.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ConfigManager
from src.trainers import QwenHybridWrapper
from src.callbacks import LoggingCallback, CheckpointCallback
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Hybrid Training (SFT + Micro-DPO)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base config file"
    )
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Path to override config file (e.g., configs/hybrid/default.yaml)"
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
        help="Override SFT data path"
    )
    parser.add_argument(
        "--preference-data",
        type=str,
        default=None,
        help="Override preference data path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override preference batch size (trigger threshold)"
    )
    parser.add_argument(
        "--micro-dpo-steps",
        type=int,
        default=None,
        help="Override micro-DPO training steps"
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
    if args.preference_data:
        config.hybrid.preference_data_file = args.preference_data
    if args.batch_size:
        config.hybrid.preference_batch_size = args.batch_size
    if args.micro_dpo_steps:
        config.hybrid.micro_dpo_steps = args.micro_dpo_steps

    # Ensure task type is HYBRID
    from src.config.schemas import TaskType
    config.task_type = TaskType.HYBRID

    # Print configuration
    ConfigManager.print_config(config)

    # Validate hybrid config
    if config.hybrid is None:
        logger.error("❌ Hybrid configuration is missing!")
        logger.info("Please specify --override with a hybrid config file")
        sys.exit(1)

    # Print hybrid-specific info
    logger.info("\n" + "="*60)
    logger.info("🔄 Hybrid Training Configuration:")
    logger.info("="*60)
    logger.info(f"   Preference batch size: {config.hybrid.preference_batch_size}")
    logger.info(f"   Micro-DPO steps: {config.hybrid.micro_dpo_steps}")
    logger.info(f"   Micro-DPO learning rate: {config.hybrid.micro_dpo_lr}")
    logger.info(f"   Quality threshold: {config.hybrid.quality_threshold}")
    if config.hybrid.preference_data_file:
        logger.info(f"   Preference data: {config.hybrid.preference_data_file}")
    else:
        logger.warning("   ⚠️  No preference data file specified")
    logger.info("="*60 + "\n")

    # Create callbacks
    callbacks = [
        LoggingCallback(
            log_file=f"{config.training.output_dir}/training.json",
            verbose=True
        ),
        CheckpointCallback(
            checkpoint_dir=config.training.output_dir,
            save_frequency=0,  # Only save final model
            monitor="eval_loss",
            mode="min"
        ),
    ]

    # Create trainer
    logger.info("Creating Hybrid Trainer...")
    trainer = QwenHybridWrapper(config=config, callbacks=callbacks)

    # Train
    results = trainer.train()

    # Print results
    logger.info("\n" + "="*60)
    logger.info("🎉 Training Results:")
    logger.info("="*60)

    # SFT results
    if 'sft' in results:
        logger.info("\n📚 SFT Phase:")
        sft_results = results['sft']
        for key, value in sft_results.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.4f}")
            else:
                logger.info(f"   {key}: {value}")

    # Micro-DPO results
    logger.info(f"\n🎯 Micro-DPO Runs: {results.get('num_micro_dpo_runs', 0)}")
    if results.get('micro_dpo_runs'):
        for run in results['micro_dpo_runs']:
            logger.info(
                f"   Run #{run['run_id']}: "
                f"Loss={run['train_loss']:.4f}, "
                f"Steps={run['steps']}, "
                f"Pairs={run['num_pairs']}"
            )

    # Preference pool stats
    if 'preference_pool_stats' in results:
        logger.info("\n📊 Preference Pool Statistics:")
        stats = results['preference_pool_stats']
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.4f}")
            else:
                logger.info(f"   {key}: {value}")

    logger.info(f"\n✅ Model saved to: {config.training.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
