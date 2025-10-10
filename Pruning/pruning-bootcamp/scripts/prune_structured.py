"""
Structured Pruning Script - Day 8-12
Implements Channel/Filter Pruning with L1 norm and BatchNorm Scaling (Network Slimming)
Uses torch-pruning for dependency graph handling
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch_pruning as tp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.loader import get_dataset
from tools.metrics import compute_flops, compute_params


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


def prune_with_l1(model, example_inputs, sparsity_ratio):
    """Prune using L1 magnitude-based importance"""
    imp = tp.importance.MagnitudeImportance(p=1)  # L1 norm

    # Ignored layers (typically first conv and last fc)
    ignored_layers = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and 'fc' in name:
            ignored_layers.append(m)

    # Initialize pruner
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=sparsity_ratio,
        ignored_layers=ignored_layers,
    )

    # Prune
    pruner.step()
    return model


def prune_with_bn_scaling(model, example_inputs, sparsity_ratio):
    """Prune using BatchNorm scaling factors (Network Slimming)"""
    imp = tp.importance.BNScaleImportance()

    # Ignored layers
    ignored_layers = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and 'fc' in name:
            ignored_layers.append(m)

    # Initialize pruner
    pruner = tp.pruner.BNScalePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=sparsity_ratio,
        ignored_layers=ignored_layers,
    )

    # Prune
    pruner.step()
    return model


def main():
    parser = argparse.ArgumentParser(description='Structured Pruning')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--method', type=str, default='l1',
                       choices=['l1', 'bn_scale'],
                       help='Pruning method: l1 (magnitude) or bn_scale (Network Slimming)')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Target channel pruning ratio')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='results/structured')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create save directory
    save_dir = Path(args.save_dir) / f'{args.method}_sp{args.sparsity}'
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

    # Baseline metrics
    print(f'\n=== Baseline Model ===')
    print(f'Accuracy: {baseline_acc:.2f}%')
    input_size = (1, 3, config['dataset']['input_size'], config['dataset']['input_size'])
    example_inputs = torch.randn(input_size).to(device)

    baseline_flops = compute_flops(model, input_size=input_size)
    baseline_params = compute_params(model)
    print(f'FLOPs: {baseline_flops/1e9:.2f} G')
    print(f'Params: {baseline_params/1e6:.2f} M')

    # Structured Pruning
    print(f'\n=== Structured Pruning: {args.method.upper()} ===')
    print(f'Target Sparsity: {args.sparsity*100:.1f}%')

    if args.method == 'l1':
        model = prune_with_l1(model, example_inputs, args.sparsity)
    elif args.method == 'bn_scale':
        model = prune_with_bn_scaling(model, example_inputs, args.sparsity)

    # Pruned model metrics
    pruned_flops = compute_flops(model, input_size=input_size)
    pruned_params = compute_params(model)
    flops_reduction = (1 - pruned_flops / baseline_flops) * 100
    params_reduction = (1 - pruned_params / baseline_params) * 100

    print(f'\n=== After Pruning ===')
    print(f'FLOPs: {pruned_flops/1e9:.2f} G ({flops_reduction:.1f}% reduction)')
    print(f'Params: {pruned_params/1e6:.2f} M ({params_reduction:.1f}% reduction)')

    # Fine-tune
    print('\n=== Fine-tuning ===')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['pruning']['finetune_lr'],
        momentum=0.9,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['pruning']['finetune_epochs']
    )

    final_acc = fine_tune(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, config['pruning']['finetune_epochs']
    )

    # Final results
    acc_drop = baseline_acc - final_acc
    print(f'\n=== Final Results ===')
    print(f'Baseline Accuracy: {baseline_acc:.2f}%')
    print(f'Pruned Accuracy: {final_acc:.2f}%')
    print(f'Accuracy Drop: {acc_drop:.2f}%')
    print(f'FLOPs Reduction: {flops_reduction:.1f}%')
    print(f'Params Reduction: {params_reduction:.1f}%')

    # Check if meets goal (≥40% FLOPs reduction, ≤1.5% accuracy drop)
    goal_met = flops_reduction >= 40.0 and acc_drop <= 1.5
    print(f'\n{"✓" if goal_met else "✗"} Goal {"MET" if goal_met else "NOT MET"} (≥40% FLOPs, ≤1.5% acc drop)')

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': final_acc,
        'baseline_acc': baseline_acc,
        'flops_reduction': flops_reduction,
        'params_reduction': params_reduction,
        'method': args.method
    }, save_dir / 'pruned_model.pth')

    # Save metrics
    metrics = {
        'method': args.method,
        'target_sparsity': args.sparsity,
        'baseline_accuracy': float(baseline_acc),
        'pruned_accuracy': float(final_acc),
        'accuracy_drop': float(acc_drop),
        'baseline_flops': float(baseline_flops),
        'pruned_flops': float(pruned_flops),
        'flops_reduction_pct': float(flops_reduction),
        'baseline_params': float(baseline_params),
        'pruned_params': float(pruned_params),
        'params_reduction_pct': float(params_reduction),
        'goal_met': goal_met
    }

    with open(save_dir / 'metrics.yaml', 'w') as f:
        yaml.dump(metrics, f)

    print(f'\n✓ Results saved to {save_dir}')


if __name__ == '__main__':
    main()
