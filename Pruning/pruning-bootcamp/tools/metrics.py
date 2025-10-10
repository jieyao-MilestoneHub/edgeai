"""
Metrics Utilities
Provides comprehensive metrics computation: FLOPs, Params, Latency, Sparsity
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


def compute_flops(model: nn.Module, input_size: Tuple[int, int, int, int]) -> int:
    """
    Compute FLOPs (Floating Point Operations) for a model

    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)

    Returns:
        Total FLOPs
    """
    try:
        from thop import profile
        device = next(model.parameters()).device
        inputs = torch.randn(input_size).to(device)
        flops, _ = profile(model, inputs=(inputs,), verbose=False)
        return int(flops)
    except ImportError:
        # Fallback: rough estimation for Conv2d and Linear layers
        flops = 0
        input_h, input_w = input_size[2], input_size[3]

        def conv_flops(module, input, output):
            nonlocal flops
            batch_size = output.shape[0]
            output_h, output_w = output.shape[2], output.shape[3]
            kernel_h, kernel_w = module.kernel_size
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups

            # FLOPs = batch_size * output_spatial * kernel_ops * out_channels
            kernel_ops = kernel_h * kernel_w * (in_channels // groups)
            flops += batch_size * output_h * output_w * kernel_ops * out_channels

        def linear_flops(module, input, output):
            nonlocal flops
            batch_size = output.shape[0]
            flops += batch_size * module.in_features * module.out_features

        hooks = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(conv_flops))
            elif isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(linear_flops))

        device = next(model.parameters()).device
        inputs = torch.randn(input_size).to(device)
        model.eval()
        with torch.no_grad():
            model(inputs)

        for hook in hooks:
            hook.remove()

        return flops


def compute_params(model: nn.Module) -> int:
    """
    Compute total number of parameters

    Args:
        model: PyTorch model

    Returns:
        Total parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_zero_weights(model: nn.Module) -> int:
    """
    Count number of zero-valued weights (for sparsity measurement)

    Args:
        model: PyTorch model

    Returns:
        Number of zero weights
    """
    zero_count = 0
    for param in model.parameters():
        zero_count += torch.sum(param == 0).item()
    return zero_count


def compute_sparsity(model: nn.Module) -> float:
    """
    Compute model sparsity (percentage of zero weights)

    Args:
        model: PyTorch model

    Returns:
        Sparsity ratio [0, 1]
    """
    zero_weights = count_zero_weights(model)
    total_weights = sum(p.numel() for p in model.parameters())
    return zero_weights / total_weights if total_weights > 0 else 0.0


def normalize_metrics(
    flops: int,
    params: int,
    accuracy: float,
    baseline_acc: float,
    baseline_flops: int = None,
    baseline_params: int = None,
    latency_ms: float = None,
    baseline_latency: float = None
) -> Dict[str, Any]:
    """
    Normalize metrics relative to baseline

    Args:
        flops: Model FLOPs
        params: Model parameters
        accuracy: Model accuracy (%)
        baseline_acc: Baseline accuracy (%)
        baseline_flops: Baseline FLOPs (optional)
        baseline_params: Baseline params (optional)
        latency_ms: Latency in milliseconds (optional)
        baseline_latency: Baseline latency (optional)

    Returns:
        Dictionary of normalized metrics
    """
    metrics = {
        'flops': flops,
        'flops_g': flops / 1e9,
        'params': params,
        'params_m': params / 1e6,
        'accuracy': accuracy,
        'accuracy_drop': baseline_acc - accuracy,
    }

    if baseline_flops is not None:
        metrics['flops_reduction_pct'] = (1 - flops / baseline_flops) * 100
        metrics['baseline_flops'] = baseline_flops

    if baseline_params is not None:
        metrics['params_reduction_pct'] = (1 - params / baseline_params) * 100
        metrics['baseline_params'] = baseline_params

    if latency_ms is not None:
        metrics['latency_ms'] = latency_ms
        metrics['throughput_fps'] = 1000.0 / latency_ms

        if baseline_latency is not None:
            metrics['latency_reduction_pct'] = (1 - latency_ms / baseline_latency) * 100
            metrics['speedup'] = baseline_latency / latency_ms
            metrics['baseline_latency_ms'] = baseline_latency

    return metrics


def print_metrics_table(metrics: Dict[str, Any], title: str = "Model Metrics"):
    """
    Pretty print metrics in a table format

    Args:
        metrics: Dictionary of metrics
        title: Table title
    """
    print(f'\n{"="*60}')
    print(f'{title:^60}')
    print(f'{"="*60}')

    for key, value in metrics.items():
        if isinstance(value, float):
            if 'pct' in key or 'accuracy' in key or 'drop' in key:
                print(f'{key:<30} {value:>10.2f} %')
            else:
                print(f'{key:<30} {value:>10.4f}')
        elif isinstance(value, int):
            if value > 1e6:
                print(f'{key:<30} {value/1e6:>10.2f} M')
            elif value > 1e3:
                print(f'{key:<30} {value/1e3:>10.2f} K')
            else:
                print(f'{key:<30} {value:>10}')
        else:
            print(f'{key:<30} {str(value):>10}')

    print(f'{"="*60}\n')


def compare_metrics(baseline_metrics: Dict[str, Any], pruned_metrics: Dict[str, Any]):
    """
    Compare baseline and pruned model metrics

    Args:
        baseline_metrics: Baseline model metrics
        pruned_metrics: Pruned model metrics
    """
    print(f'\n{"="*70}')
    print(f'{"BASELINE vs PRUNED COMPARISON":^70}')
    print(f'{"="*70}')
    print(f'{"Metric":<25} {"Baseline":>15} {"Pruned":>15} {"Change":>12}')
    print(f'{"-"*70}')

    keys_to_compare = ['flops_g', 'params_m', 'accuracy', 'latency_ms', 'throughput_fps']

    for key in keys_to_compare:
        if key in baseline_metrics and key in pruned_metrics:
            baseline_val = baseline_metrics[key]
            pruned_val = pruned_metrics[key]

            if key in ['accuracy', 'throughput_fps']:
                # Higher is better
                change_pct = ((pruned_val / baseline_val) - 1) * 100
                print(f'{key:<25} {baseline_val:>15.2f} {pruned_val:>15.2f} {change_pct:>11.1f}%')
            else:
                # Lower is better
                change_pct = (1 - pruned_val / baseline_val) * 100
                print(f'{key:<25} {baseline_val:>15.2f} {pruned_val:>15.2f} {change_pct:>11.1f}%')

    print(f'{"="*70}\n')
