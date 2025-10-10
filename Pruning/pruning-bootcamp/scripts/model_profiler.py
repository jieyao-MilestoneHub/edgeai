"""
Profiling Script - Day 13-14
Comprehensive profiling: FLOPs vs Latency verification
Demonstrates that FLOPs ≠ Latency
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.metrics import compute_flops, compute_params
from tools.latency import measure_latency, profile_layer_latency


def profile_model(model, input_size, device, num_warmup=50, num_runs=200):
    """Comprehensive model profiling"""
    model.eval()

    # Compute FLOPs and params
    flops = compute_flops(model, input_size=input_size)
    params = compute_params(model)

    # Measure latency
    latency_ms = measure_latency(model, input_size=input_size, device=device,
                                 num_warmup=num_warmup, num_runs=num_runs)

    # Theoretical vs actual performance
    theoretical_tflops = flops / (latency_ms * 1e-3) / 1e12

    results = {
        'flops': flops,
        'flops_g': flops / 1e9,
        'params': params,
        'params_m': params / 1e6,
        'latency_ms': latency_ms,
        'throughput_fps': 1000.0 / latency_ms,
        'theoretical_tflops': theoretical_tflops
    }

    return results


def compare_models(baseline_path, pruned_path, config_path, device):
    """Compare baseline and pruned models"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    input_size = config['dataset']['input_size']
    dataset_name = config['dataset']['name']

    if 'cifar' in dataset_name.lower():
        num_classes = 10
    elif 'imagenet' in dataset_name.lower():
        num_classes = 1000
    else:
        num_classes = 1000

    example_input = (1, 3, input_size, input_size)

    # Load baseline model
    from torchvision.models import resnet18
    baseline_model = resnet18(num_classes=num_classes)
    checkpoint = torch.load(baseline_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        baseline_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        baseline_model.load_state_dict(checkpoint)
    baseline_model = baseline_model.to(device)

    # Load pruned model
    pruned_model = resnet18(num_classes=num_classes)
    checkpoint = torch.load(pruned_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        pruned_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        pruned_model.load_state_dict(checkpoint)
    pruned_model = pruned_model.to(device)

    print('=== Profiling Baseline Model ===')
    baseline_results = profile_model(baseline_model, example_input, device)

    print('\n=== Profiling Pruned Model ===')
    pruned_results = profile_model(pruned_model, example_input, device)

    # Compute reductions
    flops_reduction = (1 - pruned_results['flops'] / baseline_results['flops']) * 100
    params_reduction = (1 - pruned_results['params'] / baseline_results['params']) * 100
    latency_reduction = (1 - pruned_results['latency_ms'] / baseline_results['latency_ms']) * 100
    speedup = baseline_results['latency_ms'] / pruned_results['latency_ms']

    # Print comparison
    print('\n' + '='*60)
    print('BASELINE vs PRUNED MODEL COMPARISON')
    print('='*60)

    print(f'\n{"Metric":<20} {"Baseline":>15} {"Pruned":>15} {"Change":>12}')
    print('-'*60)
    print(f'{"FLOPs (G)":<20} {baseline_results["flops_g"]:>15.2f} {pruned_results["flops_g"]:>15.2f} {flops_reduction:>11.1f}%')
    print(f'{"Params (M)":<20} {baseline_results["params_m"]:>15.2f} {pruned_results["params_m"]:>15.2f} {params_reduction:>11.1f}%')
    print(f'{"Latency (ms)":<20} {baseline_results["latency_ms"]:>15.2f} {pruned_results["latency_ms"]:>15.2f} {latency_reduction:>11.1f}%')
    print(f'{"Throughput (FPS)":<20} {baseline_results["throughput_fps"]:>15.2f} {pruned_results["throughput_fps"]:>15.2f} {(pruned_results["throughput_fps"]/baseline_results["throughput_fps"]-1)*100:>11.1f}%')
    print(f'{"Speedup":<20} {"":<15} {speedup:>15.2f}x {"":<12}')

    # Highlight FLOPs vs Latency discrepancy
    print('\n' + '='*60)
    print('FLOPs vs LATENCY ANALYSIS')
    print('='*60)
    print(f'FLOPs Reduction:   {flops_reduction:6.1f}%')
    print(f'Latency Reduction: {latency_reduction:6.1f}%')
    print(f'Discrepancy:       {abs(flops_reduction - latency_reduction):6.1f}% (FLOPs ≠ Latency!)')

    if abs(flops_reduction - latency_reduction) > 10:
        print('\n⚠  Significant discrepancy between FLOPs and Latency reduction!')
        print('   This demonstrates that FLOPs is not always a reliable proxy for latency.')
        print('   Factors: memory bandwidth, cache efficiency, kernel launch overhead, etc.')

    return {
        'baseline': baseline_results,
        'pruned': pruned_results,
        'comparison': {
            'flops_reduction_pct': flops_reduction,
            'params_reduction_pct': params_reduction,
            'latency_reduction_pct': latency_reduction,
            'speedup': speedup,
            'flops_latency_discrepancy': abs(flops_reduction - latency_reduction)
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Model Profiling')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to single model checkpoint to profile')
    parser.add_argument('--baseline', type=str, help='Path to baseline model for comparison')
    parser.add_argument('--pruned', type=str, help='Path to pruned model for comparison')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-warmup', type=int, default=50, help='Number of warmup runs')
    parser.add_argument('--num-runs', type=int, default=200, help='Number of benchmark runs')
    parser.add_argument('--layer-profile', action='store_true', help='Profile individual layers')
    parser.add_argument('--save-results', type=str, help='Path to save profiling results')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device(args.device)
    print(f'Using device: {device}')

    if args.baseline and args.pruned:
        # Compare two models
        results = compare_models(args.baseline, args.pruned, args.config, device)

        if args.save_results:
            import yaml
            save_path = Path(args.save_results)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.dump(results, f)
            print(f'\n✓ Results saved to {save_path}')

    elif args.checkpoint:
        # Profile single model
        input_size = config['dataset']['input_size']
        dataset_name = config['dataset']['name']

        if 'cifar' in dataset_name.lower():
            num_classes = 10
        elif 'imagenet' in dataset_name.lower():
            num_classes = 1000
        else:
            num_classes = 1000

        from torchvision.models import resnet18
        model = resnet18(num_classes=num_classes)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)

        print('=== Profiling Model ===')
        example_input = (1, 3, input_size, input_size)
        results = profile_model(model, example_input, device, args.num_warmup, args.num_runs)

        print(f'\nFLOPs: {results["flops_g"]:.2f} G')
        print(f'Params: {results["params_m"]:.2f} M')
        print(f'Latency: {results["latency_ms"]:.2f} ms')
        print(f'Throughput: {results["throughput_fps"]:.2f} FPS')

        if args.layer_profile:
            print('\n=== Layer-wise Profiling ===')
            layer_latencies = profile_layer_latency(model, example_input, device)
            for name, lat in sorted(layer_latencies.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f'{name:<40} {lat:.3f} ms')

        if args.save_results:
            import yaml
            save_path = Path(args.save_results)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.dump(results, f)
            print(f'\n✓ Results saved to {save_path}')

    else:
        print('Error: Must provide either --checkpoint or both --baseline and --pruned')
        return


if __name__ == '__main__':
    main()
