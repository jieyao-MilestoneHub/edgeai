"""
Unstructured Pruning Script - Day 3-7
Implements L1, Global Magnitude, Movement, and Taylor-Sensitivity pruning
Supports layerwise vs global sparsity allocation, one-shot vs iterative pruning
"""

import os
import sys
import yaml
import argparse
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.loader import get_dataset
from tools.metrics import compute_flops, compute_params, count_zero_weights
from tools.pruning_utils import (
    global_magnitude_pruning,
    layerwise_magnitude_pruning,
    taylor_importance_pruning,
    movement_pruning,
    remove_pruning_reparameterization
)


def fine_tune(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs):
    """Fine-tune pruned model"""
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Fine-tune Epoch {epoch}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f'{train_loss/total:.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch}: Val Acc = {val_acc:.2f}%')
        best_acc = max(best_acc, val_acc)

    return best_acc


def main():
    parser = argparse.ArgumentParser(description='Unstructured Pruning')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--method', type=str, default='l1',
                       choices=['l1', 'global', 'taylor', 'movement'],
                       help='Pruning method')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Target sparsity')
    parser.add_argument('--mode', type=str, default='layerwise',
                       choices=['layerwise', 'global'],
                       help='Sparsity allocation mode')
    parser.add_argument('--iterative', action='store_true', help='Use iterative pruning')
    parser.add_argument('--num-iterations', type=int, default=5, help='Number of iterative pruning steps')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='results/unstructured')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create save directory
    save_dir = Path(args.save_dir) / f'{args.method}_{args.mode}_sp{args.sparsity}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device(args.device)
    print(f'Using device: {device}')

    # Load dataset
    train_loader, val_loader, num_classes = get_dataset(
        config['dataset']['name'],
        config['dataset']['batch_size'],
        config['dataset']['num_workers']
    )

    # Load model
    from torchvision.models import resnet18
    model = resnet18(num_classes=num_classes)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        baseline_acc = checkpoint.get('best_acc', 0)
    else:
        model.load_state_dict(checkpoint)
        baseline_acc = 0
    model = model.to(device)

    print(f'\n=== Baseline Model ===')
    print(f'Accuracy: {baseline_acc:.2f}%')
    baseline_params = compute_params(model)
    print(f'Params: {baseline_params/1e6:.2f} M')

    # Pruning
    print(f'\n=== Pruning: {args.method.upper()} ({args.mode}) ===')
    print(f'Target Sparsity: {args.sparsity*100:.1f}%')

    if args.iterative:
        # Iterative pruning
        print(f'Mode: Iterative ({args.num_iterations} iterations)')
        sparsity_step = args.sparsity / args.num_iterations

        for i in range(args.num_iterations):
            current_sparsity = sparsity_step * (i + 1)
            print(f'\n--- Iteration {i+1}/{args.num_iterations} (sparsity={current_sparsity:.2f}) ---')

            # Apply pruning
            if args.method == 'global' or args.mode == 'global':
                global_magnitude_pruning(model, current_sparsity)
            else:
                layerwise_magnitude_pruning(model, current_sparsity)

            # Fine-tune
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=config['pruning']['finetune_lr'], momentum=0.9)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['pruning']['finetune_epochs'])

            acc = fine_tune(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                          config['pruning']['finetune_epochs'] // args.num_iterations)
            print(f'Accuracy after iteration {i+1}: {acc:.2f}%')

    else:
        # One-shot pruning
        print('Mode: One-shot')

        if args.method == 'global' or args.mode == 'global':
            global_magnitude_pruning(model, args.sparsity)
        elif args.method == 'taylor':
            taylor_importance_pruning(model, train_loader, device, args.sparsity)
        elif args.method == 'movement':
            movement_pruning(model, train_loader, device, args.sparsity, config['pruning']['finetune_epochs'])
        else:  # l1 layerwise
            layerwise_magnitude_pruning(model, args.sparsity)

        # Fine-tune
        print('\n=== Fine-tuning ===')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config['pruning']['finetune_lr'], momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['pruning']['finetune_epochs'])

        final_acc = fine_tune(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                            config['pruning']['finetune_epochs'])

    # Remove pruning reparameterization
    remove_pruning_reparameterization(model)

    # Final evaluation
    print('\n=== Pruned Model Metrics ===')
    zero_weights = count_zero_weights(model)
    total_weights = sum(p.numel() for p in model.parameters())
    actual_sparsity = zero_weights / total_weights
    print(f'Actual Sparsity: {actual_sparsity*100:.2f}%')
    print(f'Final Accuracy: {final_acc:.2f}%')
    print(f'Accuracy Drop: {baseline_acc - final_acc:.2f}%')

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': final_acc,
        'sparsity': actual_sparsity,
        'baseline_acc': baseline_acc,
        'method': args.method,
        'mode': args.mode
    }, save_dir / 'pruned_model.pth')

    # Save metrics
    metrics = {
        'method': args.method,
        'mode': args.mode,
        'target_sparsity': args.sparsity,
        'actual_sparsity': float(actual_sparsity),
        'baseline_accuracy': float(baseline_acc),
        'pruned_accuracy': float(final_acc),
        'accuracy_drop': float(baseline_acc - final_acc),
        'zero_weights': int(zero_weights),
        'total_weights': int(total_weights)
    }

    with open(save_dir / 'metrics.yaml', 'w') as f:
        yaml.dump(metrics, f)

    print(f'\n✓ Results saved to {save_dir}')


if __name__ == '__main__':
    main()
