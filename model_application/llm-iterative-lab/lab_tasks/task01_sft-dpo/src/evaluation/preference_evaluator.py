"""Preference alignment evaluator for DPO models."""

from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
from datasets import Dataset
from tqdm import tqdm

from src.evaluation.base_evaluator import Evaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PreferenceEvaluator(Evaluator):
    """Evaluates preference alignment.

    Metrics computed:
    - Win Rate: Percentage of times model prefers chosen over rejected
    - Preference Accuracy: How well model rankings align with ground truth
    - Reward Margin: Average difference between chosen/rejected response scores
    - Mean Reward: Average log probability of chosen responses
    """

    def __init__(self, batch_size: int = 4, max_samples: Optional[int] = None):
        """Initialize evaluator.

        Args:
            batch_size: Batch size for evaluation
            max_samples: Maximum number of samples to evaluate (None = all)
        """
        self.batch_size = batch_size
        self.max_samples = max_samples

    def evaluate(self, model, dataset) -> Dict[str, Any]:
        """Evaluate preference alignment.

        Args:
            model: Model to evaluate (should have a tokenizer attribute)
            dataset: Preference pairs dataset (Dataset or dict with 'test'/'train' keys)

        Returns:
            Evaluation metrics dictionary
        """
        # Handle dataset format
        if isinstance(dataset, dict):
            eval_dataset = dataset.get('test') or dataset.get('train')
        else:
            eval_dataset = dataset

        if eval_dataset is None or len(eval_dataset) == 0:
            logger.warning("No evaluation dataset provided")
            return {
                "win_rate": 0.0,
                "preference_accuracy": 0.0,
                "reward_margin": 0.0,
                "mean_chosen_reward": 0.0,
                "mean_rejected_reward": 0.0,
                "num_samples": 0
            }

        # Validate required columns
        required_columns = ['prompt', 'chosen', 'rejected']
        if not all(col in eval_dataset.column_names for col in required_columns):
            logger.error(f"Dataset missing required columns. Found: {eval_dataset.column_names}")
            return {
                "win_rate": 0.0,
                "preference_accuracy": 0.0,
                "reward_margin": 0.0,
                "mean_chosen_reward": 0.0,
                "mean_rejected_reward": 0.0,
                "num_samples": 0
            }

        # Limit samples if specified
        num_samples = min(len(eval_dataset), self.max_samples) if self.max_samples else len(eval_dataset)
        eval_dataset = eval_dataset.select(range(num_samples))

        logger.info(f"Evaluating preference alignment on {num_samples} pairs...")

        # Set model to eval mode
        model.eval()
        device = next(model.parameters()).device

        # Get tokenizer from model or parent
        tokenizer = getattr(model, 'tokenizer', None)
        if tokenizer is None and hasattr(model, 'pretrained_model'):
            tokenizer = getattr(model.pretrained_model, 'tokenizer', None)

        if tokenizer is None:
            logger.error("No tokenizer found in model")
            return {
                "win_rate": 0.0,
                "preference_accuracy": 0.0,
                "reward_margin": 0.0,
                "mean_chosen_reward": 0.0,
                "mean_rejected_reward": 0.0,
                "num_samples": 0
            }

        total_wins = 0
        total_comparisons = 0
        chosen_rewards = []
        rejected_rewards = []
        reward_margins = []

        with torch.no_grad():
            for i in tqdm(range(0, len(eval_dataset), self.batch_size), desc="Evaluating preferences"):
                batch = eval_dataset[i:i + self.batch_size]

                # Evaluate batch
                batch_results = self._evaluate_batch(model, tokenizer, batch, device)

                total_wins += batch_results['wins']
                total_comparisons += batch_results['comparisons']
                chosen_rewards.extend(batch_results['chosen_rewards'])
                rejected_rewards.extend(batch_results['rejected_rewards'])
                reward_margins.extend(batch_results['margins'])

        # Calculate metrics
        win_rate = total_wins / total_comparisons if total_comparisons > 0 else 0.0
        preference_accuracy = win_rate  # Same as win rate for binary preferences
        mean_reward_margin = np.mean(reward_margins) if reward_margins else 0.0
        mean_chosen_reward = np.mean(chosen_rewards) if chosen_rewards else 0.0
        mean_rejected_reward = np.mean(rejected_rewards) if rejected_rewards else 0.0

        metrics = {
            "win_rate": float(win_rate),
            "preference_accuracy": float(preference_accuracy),
            "reward_margin": float(mean_reward_margin),
            "mean_chosen_reward": float(mean_chosen_reward),
            "mean_rejected_reward": float(mean_rejected_reward),
            "num_samples": num_samples
        }

        logger.info(
            f"Preference evaluation results: "
            f"Win Rate={win_rate:.4f}, "
            f"Reward Margin={mean_reward_margin:.4f}, "
            f"Chosen Reward={mean_chosen_reward:.4f}"
        )

        return metrics

    def _evaluate_batch(self, model, tokenizer, batch: Dict, device: torch.device) -> Dict[str, Any]:
        """Evaluate a batch of preference pairs.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            batch: Batch of preference pairs
            device: Device to run on

        Returns:
            Dictionary with batch evaluation results
        """
        batch_size = len(batch['prompt'])
        wins = 0
        chosen_rewards = []
        rejected_rewards = []
        margins = []

        for i in range(batch_size):
            prompt = batch['prompt'][i]
            chosen = batch['chosen'][i]
            rejected = batch['rejected'][i]

            # Calculate rewards (log probabilities) for chosen and rejected
            chosen_reward = self._calculate_reward(model, tokenizer, prompt, chosen, device)
            rejected_reward = self._calculate_reward(model, tokenizer, prompt, rejected, device)

            chosen_rewards.append(chosen_reward)
            rejected_rewards.append(rejected_reward)

            # Calculate margin and check if model prefers chosen
            margin = chosen_reward - rejected_reward
            margins.append(margin)

            if margin > 0:
                wins += 1

        return {
            'wins': wins,
            'comparisons': batch_size,
            'chosen_rewards': chosen_rewards,
            'rejected_rewards': rejected_rewards,
            'margins': margins
        }

    def _calculate_reward(
        self,
        model,
        tokenizer,
        prompt: str,
        response: str,
        device: torch.device
    ) -> float:
        """Calculate reward (log probability) for a response given a prompt.

        This follows the standard DPO evaluation approach where the reward
        is the average log probability of the response tokens.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            prompt: Input prompt
            response: Response to evaluate
            device: Device to run on

        Returns:
            Average log probability (reward) of the response
        """
        try:
            # Combine prompt and response
            full_text = f"{prompt}{response}"

            # Tokenize
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            full_ids = tokenizer.encode(full_text, add_special_tokens=True, truncation=True, max_length=2048)

            # Convert to tensors
            input_ids = torch.tensor([full_ids]).to(device)
            prompt_length = len(prompt_ids)

            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits

            # Calculate log probabilities for response tokens only
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, prompt_length-1:-1, :].contiguous()
            shift_labels = input_ids[:, prompt_length:].contiguous()

            # Calculate log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)

            # Gather log probs for actual tokens
            response_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Average log probability (reward)
            reward = response_log_probs.mean().item()

            return reward

        except Exception as e:
            logger.warning(f"Error calculating reward: {e}")
            return -100.0  # Large negative value for errors
