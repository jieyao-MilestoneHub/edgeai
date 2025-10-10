"""
Pruning Utilities - Day 3-7 & Day 15-16
Implements various pruning strategies:
- Global vs Layerwise Magnitude Pruning
- Taylor Importance
- Movement Pruning
- Iterative vs One-shot
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Dict
import copy


def get_prunable_parameters(model: nn.Module) -> List[tuple]:
    """
    Get list of prunable parameters (Conv2d and Linear weights)

    Args:
        model: PyTorch model

    Returns:
        List of (module, param_name) tuples
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    return parameters_to_prune


def layerwise_magnitude_pruning(model: nn.Module, sparsity: float):
    """
    Apply L1 magnitude-based pruning layerwise (same sparsity per layer)

    Args:
        model: PyTorch model
        sparsity: Target sparsity ratio [0, 1]
    """
    parameters_to_prune = get_prunable_parameters(model)

    for module, param_name in parameters_to_prune:
        prune.l1_unstructured(module, name=param_name, amount=sparsity)


def global_magnitude_pruning(model: nn.Module, sparsity: float):
    """
    Apply global magnitude-based pruning (considers all weights together)

    Args:
        model: PyTorch model
        sparsity: Target sparsity ratio [0, 1]
    """
    parameters_to_prune = get_prunable_parameters(model)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )


def taylor_importance_pruning(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    sparsity: float,
    num_batches: int = 10
):
    """
    Taylor expansion-based importance pruning
    Importance = |weight * gradient|

    Args:
        model: PyTorch model
        dataloader: Training data loader
        device: Device
        sparsity: Target sparsity
        num_batches: Number of batches to estimate importance
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Collect gradients
    importance_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            importance_scores[name] = torch.zeros_like(module.weight.data)

    # Accumulate importance over batches
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Accumulate |weight * gradient|
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                importance_scores[name] += torch.abs(module.weight.data * module.weight.grad.data)

    # Average importance
    for name in importance_scores:
        importance_scores[name] /= num_batches

    # Flatten all importance scores
    all_scores = torch.cat([scores.view(-1) for scores in importance_scores.values()])
    threshold = torch.quantile(all_scores, sparsity)

    # Apply pruning based on threshold
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            mask = importance_scores[name] > threshold
            prune.custom_from_mask(module, name='weight', mask=mask)


def movement_pruning(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    sparsity: float,
    num_epochs: int = 5,
    lr: float = 0.001
):
    """
    Movement Pruning: prunes weights that consistently move toward zero during training
    Reference: "Movement Pruning: Adaptive Sparsity by Fine-Tuning" (Sanh et al., 2020)

    Args:
        model: PyTorch model
        dataloader: Training data loader
        device: Device
        sparsity: Target sparsity
        num_epochs: Number of epochs to track movement
        lr: Learning rate
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Store initial weights
    initial_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            initial_weights[name] = module.weight.data.clone()

    # Train for a few epochs
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Compute movement scores: |w_final - w_initial| * sign(w_initial)
    # Negative movement = moving toward zero
    movement_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            movement = module.weight.data - initial_weights[name]
            # Score: weights moving toward zero have negative values
            movement_scores[name] = movement * torch.sign(initial_weights[name])

    # Prune weights with most negative movement (moving toward zero)
    all_scores = torch.cat([scores.view(-1) for scores in movement_scores.values()])
    threshold = torch.quantile(all_scores, sparsity)

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            mask = movement_scores[name] > threshold
            prune.custom_from_mask(module, name='weight', mask=mask)


def iterative_pruning(
    model: nn.Module,
    pruning_fn,
    target_sparsity: float,
    num_iterations: int,
    **pruning_kwargs
):
    """
    Iterative pruning: gradually increase sparsity over multiple iterations

    Args:
        model: PyTorch model
        pruning_fn: Pruning function to apply
        target_sparsity: Final target sparsity
        num_iterations: Number of pruning iterations
        **pruning_kwargs: Additional arguments for pruning function

    Returns:
        List of intermediate sparsity values
    """
    sparsity_schedule = []
    step_size = target_sparsity / num_iterations

    for i in range(num_iterations):
        current_sparsity = step_size * (i + 1)
        sparsity_schedule.append(current_sparsity)

        # Apply pruning
        pruning_fn(model, current_sparsity, **pruning_kwargs)

    return sparsity_schedule


def remove_pruning_reparameterization(model: nn.Module):
    """
    Remove pruning reparameterization and make pruning permanent

    Args:
        model: PyTorch model with pruning masks
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                # No pruning to remove
                pass


def compute_layerwise_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    Compute sparsity for each layer

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping layer names to sparsity ratios
    """
    layer_sparsity = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            zeros = torch.sum(weight == 0).item()
            total = weight.numel()
            layer_sparsity[name] = zeros / total if total > 0 else 0.0

    return layer_sparsity


def print_layerwise_sparsity(model: nn.Module):
    """
    Print sparsity statistics for each layer

    Args:
        model: PyTorch model
    """
    layer_sparsity = compute_layerwise_sparsity(model)

    print(f'\n{"="*70}')
    print(f'{"LAYER-WISE SPARSITY":^70}')
    print(f'{"="*70}')
    print(f'{"Layer Name":<40} {"Sparsity":>15} {"Zeros/Total":>15}')
    print(f'{"-"*70}')

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            zeros = torch.sum(weight == 0).item()
            total = weight.numel()
            sparsity = layer_sparsity[name]
            print(f'{name:<40} {sparsity*100:>14.2f}% {zeros:>7}/{total:<7}')

    # Overall sparsity
    total_zeros = sum(torch.sum(m.weight.data == 0).item()
                     for m in model.modules()
                     if isinstance(m, (nn.Conv2d, nn.Linear)))
    total_params = sum(m.weight.data.numel()
                      for m in model.modules()
                      if isinstance(m, (nn.Conv2d, nn.Linear)))
    overall_sparsity = total_zeros / total_params if total_params > 0 else 0.0

    print(f'{"-"*70}')
    print(f'{"OVERALL":<40} {overall_sparsity*100:>14.2f}% {total_zeros:>7}/{total_params:<7}')
    print(f'{"="*70}\n')


def adaptive_sparsity_allocation(
    model: nn.Module,
    target_sparsity: float,
    sensitivity_scores: Dict[str, float]
) -> Dict[str, float]:
    """
    Adaptively allocate sparsity based on layer sensitivity
    More sensitive layers get lower sparsity

    Args:
        model: PyTorch model
        target_sparsity: Overall target sparsity
        sensitivity_scores: Dictionary mapping layer names to sensitivity scores

    Returns:
        Dictionary mapping layer names to layer-specific sparsity ratios
    """
    # Normalize sensitivity scores
    total_sensitivity = sum(sensitivity_scores.values())
    normalized_sensitivity = {k: v/total_sensitivity for k, v in sensitivity_scores.items()}

    # Allocate sparsity: less sparsity for high-sensitivity layers
    layer_sparsity = {}
    for name, sensitivity in normalized_sensitivity.items():
        # Inverse relationship: high sensitivity -> low sparsity
        layer_sparsity[name] = target_sparsity * (1 - sensitivity)

    return layer_sparsity


def n_m_sparsity(model: nn.Module, n: int, m: int):
    """
    N:M structured sparsity (N zeros in every M consecutive elements)
    Hardware-friendly sparsity pattern (e.g., 2:4 sparsity supported by NVIDIA A100)

    Args:
        model: PyTorch model
        n: Number of zeros
        m: Block size
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            shape = weight.shape

            # Flatten to 2D for easier processing
            weight_2d = weight.view(-1, shape[-1])

            # Apply N:M sparsity pattern
            for i in range(0, weight_2d.shape[1], m):
                if i + m <= weight_2d.shape[1]:
                    block = weight_2d[:, i:i+m]
                    # Find n smallest magnitudes and zero them
                    _, indices = torch.topk(torch.abs(block), n, dim=1, largest=False)
                    block.scatter_(1, indices, 0)

            # Reshape back
            weight.copy_(weight_2d.view(shape))

    print(f'✓ Applied {n}:{m} sparsity pattern (hardware-friendly)')
