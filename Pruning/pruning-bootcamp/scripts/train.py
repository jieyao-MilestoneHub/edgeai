"""
Baseline Training Script - Day 1-2
Trains ResNet-18 on CIFAR-10 or ImageNet-mini with metrics tracking
"""

import sys
import yaml
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.loader import get_dataset
from tools.metrics import compute_flops, compute_params, normalize_metrics


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} Train')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/total:.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Baseline Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='results/baseline')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Create save directory
    save_dir = Path(args.save_dir)
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
    model = model.to(device)

    # Compute baseline metrics
    print('\n=== Baseline Model Metrics ===')
    flops = compute_flops(model, input_size=(1, 3, config['dataset']['input_size'], config['dataset']['input_size']))
    params = compute_params(model)
    print(f'FLOPs: {flops/1e9:.2f} G')
    print(f'Params: {params/1e6:.2f} M')

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['lr'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['training']['milestones'],
        gamma=config['training']['gamma']
    )

    # Setup tensorboard
    writer = SummaryWriter(save_dir / 'logs')

    # Training loop
    best_acc = 0.0
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f'\nEpoch {epoch}: Train Loss={train_loss:.3f}, Train Acc={train_acc:.2f}%, '
              f'Val Loss={val_loss:.3f}, Val Acc={val_acc:.2f}%')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'flops': flops,
                'params': params
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f'✓ Saved best model with accuracy: {best_acc:.2f}%')

    print(f'\n=== Training Complete ===')
    print(f'Best Validation Accuracy: {best_acc:.2f}%')

    # Save final metrics
    metrics = normalize_metrics(flops, params, best_acc, baseline_acc=best_acc)
    with open(save_dir / 'metrics.yaml', 'w') as f:
        yaml.dump(metrics, f)

    writer.close()


if __name__ == '__main__':
    main()
