"""Instruction-following evaluator for SFT models."""

from typing import Dict, Any, Optional
import torch
import numpy as np
from datasets import Dataset
from rouge_score import rouge_scorer
from tqdm import tqdm

from src.evaluation.base_evaluator import Evaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InstructionEvaluator(Evaluator):
    """Evaluates instruction-following capability.

    Metrics computed:
    - Perplexity: Measures how well the model predicts the outputs
    - ROUGE-L: Measures overlap between generated and reference outputs
    - Token Accuracy: Exact match accuracy at token level
    """

    def __init__(self, batch_size: int = 8, max_samples: Optional[int] = None):
        """Initialize evaluator.

        Args:
            batch_size: Batch size for evaluation
            max_samples: Maximum number of samples to evaluate (None = all)
        """
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def evaluate(self, model, dataset) -> Dict[str, Any]:
        """Evaluate instruction-following performance.

        Args:
            model: Model to evaluate (should have a tokenizer attribute)
            dataset: Evaluation dataset (Dataset or dict with 'test'/'train' keys)

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
            return {"perplexity": 0.0, "rouge_l": 0.0, "token_accuracy": 0.0, "num_samples": 0}

        # Limit samples if specified
        num_samples = min(len(eval_dataset), self.max_samples) if self.max_samples else len(eval_dataset)
        eval_dataset = eval_dataset.select(range(num_samples))

        logger.info(f"Evaluating on {num_samples} samples...")

        # Set model to eval mode
        model.eval()
        device = next(model.parameters()).device

        # Get tokenizer from model or parent
        tokenizer = getattr(model, 'tokenizer', None)
        if tokenizer is None and hasattr(model, 'pretrained_model'):
            tokenizer = getattr(model.pretrained_model, 'tokenizer', None)

        if tokenizer is None:
            logger.error("No tokenizer found in model")
            return {"perplexity": 0.0, "rouge_l": 0.0, "token_accuracy": 0.0, "num_samples": 0}

        total_loss = 0.0
        total_tokens = 0
        total_correct_tokens = 0
        rouge_scores = []

        with torch.no_grad():
            for i in tqdm(range(0, len(eval_dataset), self.batch_size), desc="Evaluating"):
                batch = eval_dataset[i:i + self.batch_size]

                # Process batch
                batch_loss, batch_tokens, batch_correct = self._evaluate_batch(
                    model, tokenizer, batch, device
                )

                total_loss += batch_loss
                total_tokens += batch_tokens
                total_correct_tokens += batch_correct

                # Calculate ROUGE scores for this batch
                batch_rouge = self._calculate_rouge_batch(model, tokenizer, batch, device)
                rouge_scores.extend(batch_rouge)

        # Calculate metrics
        perplexity = np.exp(total_loss / total_tokens) if total_tokens > 0 else 0.0
        token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
        rouge_l = np.mean(rouge_scores) if rouge_scores else 0.0

        metrics = {
            "perplexity": float(perplexity),
            "rouge_l": float(rouge_l),
            "token_accuracy": float(token_accuracy),
            "num_samples": num_samples,
            "total_tokens": total_tokens
        }

        logger.info(f"Evaluation results: PPL={perplexity:.2f}, ROUGE-L={rouge_l:.4f}, Token Acc={token_accuracy:.4f}")

        return metrics

    def _evaluate_batch(self, model, tokenizer, batch: Dict, device: torch.device) -> tuple:
        """Evaluate a batch of samples.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            batch: Batch of samples
            device: Device to run on

        Returns:
            Tuple of (total_loss, total_tokens, correct_tokens)
        """
        # Format examples
        texts = []
        for i in range(len(batch['instruction'])):
            example = {
                'instruction': batch['instruction'][i],
                'input': batch.get('input', [''] * len(batch['instruction']))[i],
                'output': batch['output'][i]
            }
            # Format using Qwen chat template
            text = self._format_example(example)
            texts.append(text)

        # Tokenize
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(device)

        # Forward pass
        outputs = model(**encodings, labels=encodings['input_ids'])

        # Calculate loss and accuracy
        loss = outputs.loss.item() * encodings['input_ids'].numel()
        total_tokens = encodings['input_ids'].numel()

        # Calculate token accuracy
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Mask padding tokens
        mask = encodings['attention_mask'].bool()
        correct_tokens = ((predictions == encodings['input_ids']) & mask).sum().item()

        return loss, total_tokens, correct_tokens

    def _calculate_rouge_batch(self, model, tokenizer, batch: Dict, device: torch.device) -> list:
        """Calculate ROUGE scores for a batch by generating outputs.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            batch: Batch of samples
            device: Device to run on

        Returns:
            List of ROUGE-L scores
        """
        rouge_scores = []

        for i in range(len(batch['instruction'])):
            try:
                # Prepare prompt (without output)
                instruction = batch['instruction'][i]
                input_text = batch.get('input', [''] * len(batch['instruction']))[i]
                reference = batch['output'][i]

                prompt = f"<|im_start|>system\nYou are a helpful AI assistant specialized in engineering and technical domains.<|im_end|>\n"
                prompt += f"<|im_start|>user\n{instruction}"
                if input_text:
                    prompt += f"\n\n{input_text}"
                prompt += "<|im_end|>\n<|im_start|>assistant\n"

                # Generate
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                # Decode and extract generated text
                generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                # Calculate ROUGE
                scores = self.rouge_scorer.score(reference, generated)
                rouge_scores.append(scores['rougeL'].fmeasure)

            except Exception as e:
                logger.warning(f"Error generating for sample {i}: {e}")
                rouge_scores.append(0.0)

        return rouge_scores

    def _format_example(self, example: dict) -> str:
        """Format example for Qwen chat template.

        Args:
            example: Raw example with instruction, input, output

        Returns:
            Formatted prompt string
        """
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')

        prompt = "<|im_start|>system\nYou are a helpful AI assistant specialized in engineering and technical domains.<|im_end|>\n"
        prompt += f"<|im_start|>user\n{instruction}"

        if input_text:
            prompt += f"\n\n{input_text}"

        prompt += "<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{output}<|im_end|>"

        return prompt
