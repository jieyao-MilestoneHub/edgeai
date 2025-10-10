"""
Evaluation Script - Day 1-2
Evaluates model with comprehensive metrics: Accuracy, FLOPs, Params, Latency
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.loader import get_dataset
from tools.metrics import compute_flops, compute_params, normalize_metrics
from tools.latency import measure_latency


def evaluate(model, loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--baseline-metrics', type=str, help='Path to baseline metrics for normalization')
    parser.add_argument('--measure-latency', action='store_true', help='Measure inference latency')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device(args.device)
    print(f'Using device: {device}')

    # Load dataset
    _, val_loader, num_classes = get_dataset(
        config['dataset']['name'],
        config['dataset']['batch_size'],
        config['dataset']['num_workers']
    )

    # Load model
    from torchvision.models import resnet18
    model = resnet18(num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    print(f'\n✓ Loaded checkpoint from {args.checkpoint}')

    # Evaluate accuracy
    print('\n=== Evaluating Model ===')
    accuracy = evaluate(model, val_loader, device)
    print(f'Accuracy: {accuracy:.2f}%')

    # Compute computational metrics
    input_size = (1, 3, config['dataset']['input_size'], config['dataset']['input_size'])
    flops = compute_flops(model, input_size=input_size)
    params = compute_params(model)
    print(f'FLOPs: {flops/1e9:.2f} G')
    print(f'Params: {params/1e6:.2f} M')

    # Measure latency if requested
    latency_ms = None
    if args.measure_latency:
        print('\n=== Measuring Latency ===')
        latency_ms = measure_latency(model, input_size=input_size, device=device)
        print(f'Latency: {latency_ms:.2f} ms')

    # Normalize metrics
    baseline_acc = accuracy
    if args.baseline_metrics:
        with open(args.baseline_metrics, 'r') as f:
            baseline = yaml.safe_load(f)
            baseline_acc = baseline.get('accuracy', accuracy)

    metrics = normalize_metrics(
        flops=flops,
        params=params,
        accuracy=accuracy,
        baseline_acc=baseline_acc,
        latency_ms=latency_ms
    )

    # Print normalized metrics
    print('\n=== Normalized Metrics ===')
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f'{key}: {value:.4f}')
        else:
            print(f'{key}: {value}')

    # Save metrics
    output_dir = Path(args.checkpoint).parent
    with open(output_dir / 'eval_metrics.yaml', 'w') as f:
        yaml.dump(metrics, f)
    print(f'\n✓ Metrics saved to {output_dir / "eval_metrics.yaml"}')


if __name__ == '__main__':
    main()
