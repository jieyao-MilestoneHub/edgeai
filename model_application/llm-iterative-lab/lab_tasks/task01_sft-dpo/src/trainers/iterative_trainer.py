"""Iterative Trainer for closed-loop SFT+DPO optimization."""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from src.trainers.base_trainer import QwenTrainerBase
from src.trainers.sft_trainer import QwenSFTWrapper
from src.trainers.dpo_trainer import QwenDPOWrapper
from src.config.schemas import ExperimentConfig
from src.callbacks.base_callback import Callback
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QwenIterativeWrapper(QwenTrainerBase):
    """Iterative closed-loop wrapper orchestrating SFT → DPO → Evaluate → Collect cycle.

    Named QwenIterativeWrapper to avoid conflicts with TRL trainers.
    This wrapper implements the continuous improvement loop:
    1. SFT training on current instruction data
    2. DPO training on preference pairs
    3. Evaluation of model performance
    4. Data collection for next iteration

    Example:
        >>> config = ConfigManager.load_config("iterative_config.yaml")
        >>> trainer = QwenIterativeWrapper(config)
        >>> results = trainer.train()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        callbacks: Optional[List[Callback]] = None
    ):
        """Initialize iterative trainer.

        Args:
            config: Experiment configuration
            callbacks: Optional list of callbacks
        """
        super().__init__(config, callbacks)
        self.iteration_results = []

    def prepare_data(self) -> Any:
        """Prepare data (handled by sub-trainers)."""
        # Data preparation is handled by individual trainers
        return None

    def _run_sft_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run SFT phase of current iteration.

        Args:
            iteration: Current iteration number

        Returns:
            SFT training results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📚 Iteration {iteration + 1} - SFT Phase")
        logger.info(f"{'='*60}\n")

        # Update output directory for this iteration
        sft_output = Path(self.config.training.output_dir) / f"iteration_{iteration + 1}" / "sft"
        sft_config = self.config
        sft_config.training.output_dir = str(sft_output)
        sft_config.training.num_epochs = self.config.iterative.sft_epochs_per_iter

        # Create SFT trainer (reuse model if already loaded)
        sft_trainer = QwenSFTWrapper(sft_config)
        if self.model is not None:
            sft_trainer.model = self.model
            sft_trainer.tokenizer = self.tokenizer

        # Train
        sft_results = sft_trainer.train()

        # Update our model reference
        self.model = sft_trainer.model
        self.tokenizer = sft_trainer.tokenizer

        return sft_results

    def _run_dpo_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run DPO phase of current iteration.

        Args:
            iteration: Current iteration number

        Returns:
            DPO training results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Iteration {iteration + 1} - DPO Phase")
        logger.info(f"{'='*60}\n")

        # Update output directory for this iteration
        dpo_output = Path(self.config.training.output_dir) / f"iteration_{iteration + 1}" / "dpo"
        dpo_config = self.config
        dpo_config.training.output_dir = str(dpo_output)
        dpo_config.training.num_epochs = self.config.iterative.dpo_epochs_per_iter

        # Create DPO trainer (reuse model from SFT)
        dpo_trainer = QwenDPOWrapper(dpo_config)
        dpo_trainer.model = self.model
        dpo_trainer.tokenizer = self.tokenizer

        # Train
        dpo_results = dpo_trainer.train()

        # Update our model reference
        self.model = dpo_trainer.model

        return dpo_results

    def _evaluate_iteration(self, iteration: int) -> Dict[str, Any]:
        """Evaluate model after iteration.

        Args:
            iteration: Current iteration number

        Returns:
            Evaluation results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Iteration {iteration + 1} - Evaluation")
        logger.info(f"{'='*60}\n")

        # TODO: Implement actual evaluation
        # For now, return placeholder
        eval_results = {
            "iteration": iteration + 1,
            "evaluation_score": 0.0,  # Placeholder
            "pass_threshold": False
        }

        logger.info(f"Evaluation score: {eval_results['evaluation_score']:.4f}")

        return eval_results

    def _collect_new_data(self, iteration: int) -> bool:
        """Collect new training data for next iteration.

        Args:
            iteration: Current iteration number

        Returns:
            True if data collection successful
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📝 Iteration {iteration + 1} - Data Collection")
        logger.info(f"{'='*60}\n")

        # TODO: Implement data collection/annotation
        # This could involve:
        # 1. Generating responses with current model
        # 2. Human annotation of preferences
        # 3. Augmenting training dataset

        logger.info(f"Collecting {self.config.iterative.annotation_samples_per_iter} new samples...")
        logger.info("⚠️  Manual annotation required - see annotation tool")

        return True

    def _should_continue(self, iteration: int, eval_results: Dict[str, Any]) -> bool:
        """Determine if training should continue to next iteration.

        Args:
            iteration: Current iteration number
            eval_results: Evaluation results from current iteration

        Returns:
            True if should continue training
        """
        # Check if we've reached max iterations
        if iteration + 1 >= self.config.iterative.num_iterations:
            logger.info(f"✅ Reached maximum iterations ({self.config.iterative.num_iterations})")
            return False

        # Check evaluation threshold
        score = eval_results.get('evaluation_score', 0.0)
        threshold = self.config.iterative.evaluation_threshold

        if score >= threshold:
            logger.info(f"✅ Reached evaluation threshold ({score:.4f} >= {threshold})")
            return False

        logger.info(f"Continuing to next iteration...")
        return True

    def _train_impl(self) -> Dict[str, Any]:
        """Execute iterative SFT+DPO training loop.

        Returns:
            Complete training results across all iterations
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Starting Iterative Training")
        logger.info(f"   Iterations: {self.config.iterative.num_iterations}")
        logger.info(f"   SFT epochs/iter: {self.config.iterative.sft_epochs_per_iter}")
        logger.info(f"   DPO epochs/iter: {self.config.iterative.dpo_epochs_per_iter}")
        logger.info(f"{'='*60}\n")

        all_results = []

        for iteration in range(self.config.iterative.num_iterations):
            logger.info(f"\n{'#'*60}")
            logger.info(f"# Iteration {iteration + 1}/{self.config.iterative.num_iterations}")
            logger.info(f"{'#'*60}")

            iteration_log = {"iteration": iteration + 1}

            # Phase 1: SFT Training
            sft_results = self._run_sft_iteration(iteration)
            iteration_log["sft"] = sft_results

            # Phase 2: DPO Training
            dpo_results = self._run_dpo_iteration(iteration)
            iteration_log["dpo"] = dpo_results

            # Phase 3: Evaluation
            eval_results = self._evaluate_iteration(iteration)
            iteration_log["evaluation"] = eval_results

            # Phase 4: Data Collection (for next iteration)
            if iteration + 1 < self.config.iterative.num_iterations:
                data_collected = self._collect_new_data(iteration)
                iteration_log["data_collected"] = data_collected

            # Save iteration results
            all_results.append(iteration_log)
            self.iteration_results = all_results

            # Save iteration log
            self._save_iteration_log(iteration, iteration_log)

            # Check if should continue
            if not self._should_continue(iteration, eval_results):
                break

        # Final model saved by parent class
        return {
            "total_iterations": len(all_results),
            "iterations": all_results,
            "final_sft_loss": all_results[-1]["sft"]["train_loss"],
            "final_dpo_loss": all_results[-1]["dpo"]["train_loss"],
        }

    def _save_iteration_log(self, iteration: int, log: Dict[str, Any]) -> None:
        """Save iteration log to file.

        Args:
            iteration: Iteration number
            log: Log dictionary
        """
        log_dir = Path(self.config.training.output_dir) / f"iteration_{iteration + 1}"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "iteration_log.json"
        with open(log_file, 'w') as f:
            json.dump(log, f, indent=2)

        logger.info(f"📝 Iteration log saved: {log_file}")
