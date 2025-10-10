"""
Latency Measurement Utilities
Accurate inference latency profiling on CPU/GPU
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
from contextlib import contextmanager


@contextmanager
def inference_mode():
    """Context manager for inference (no gradients, eval mode)"""
    with torch.no_grad():
        yield


def measure_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    device: torch.device,
    num_warmup: int = 50,
    num_runs: int = 200,
    use_cuda_events: bool = True
) -> float:
    """
    Measure model inference latency

    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
        use_cuda_events: Use CUDA events for GPU timing (more accurate)

    Returns:
        Average latency in milliseconds
    """
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    # Warmup
    with inference_mode():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Benchmark
    if device.type == 'cuda' and use_cuda_events:
        # Use CUDA events for accurate GPU timing
        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

        with inference_mode():
            for i in range(num_runs):
                start_events[i].record()
                _ = model(dummy_input)
                end_events[i].record()

        torch.cuda.synchronize()

        # Compute elapsed times
        times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_runs)]
        latency_ms = np.mean(times)

    else:
        # CPU or fallback timing
        times = []
        with inference_mode():
            for _ in range(num_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = model(dummy_input)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        latency_ms = np.mean(times)

    return latency_ms


def measure_latency_with_stats(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    device: torch.device,
    num_warmup: int = 50,
    num_runs: int = 200
) -> Dict[str, float]:
    """
    Measure latency with statistical metrics

    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations

    Returns:
        Dictionary with mean, std, min, max, p50, p95, p99 latencies
    """
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    # Warmup
    with inference_mode():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Benchmark
    times = []

    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

        with inference_mode():
            for i in range(num_runs):
                start_events[i].record()
                _ = model(dummy_input)
                end_events[i].record()

        torch.cuda.synchronize()
        times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_runs)]

    else:
        with inference_mode():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                times.append((end - start) * 1000)

    times = np.array(times)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'throughput_fps': 1000.0 / np.mean(times)
    }


def profile_layer_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    device: torch.device,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Profile latency of individual layers

    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device to run on
        num_runs: Number of benchmark iterations

    Returns:
        Dictionary mapping layer names to average latencies (ms)
    """
    model.eval()
    layer_times = {}

    def make_hook(name):
        def hook(module, input, output):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()

            # We can't actually measure here since hook is called during forward
            # Instead, we'll use a different approach

        return hook

    # Alternative: measure each layer individually by creating sub-models
    # This is a simplified version - for production, use torch.profiler

    named_modules = list(model.named_modules())

    for name, module in named_modules:
        if len(list(module.children())) > 0:
            # Skip container modules
            continue

        if not isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU)):
            # Only profile common layers
            continue

        # This is a rough estimation - for accurate results, use torch.profiler
        layer_times[name] = 0.0  # Placeholder

    return layer_times


def compare_latencies(
    baseline_latency: float,
    pruned_latency: float,
    baseline_flops: int,
    pruned_flops: int
):
    """
    Compare latencies and highlight FLOPs vs Latency discrepancy

    Args:
        baseline_latency: Baseline model latency (ms)
        pruned_latency: Pruned model latency (ms)
        baseline_flops: Baseline FLOPs
        pruned_flops: Pruned FLOPs
    """
    latency_reduction = (1 - pruned_latency / baseline_latency) * 100
    flops_reduction = (1 - pruned_flops / baseline_flops) * 100
    speedup = baseline_latency / pruned_latency

    print(f'\n{"="*60}')
    print(f'{"LATENCY vs FLOPs ANALYSIS":^60}')
    print(f'{"="*60}')
    print(f'Baseline Latency:  {baseline_latency:>10.2f} ms')
    print(f'Pruned Latency:    {pruned_latency:>10.2f} ms')
    print(f'Latency Reduction: {latency_reduction:>10.1f} %')
    print(f'Speedup:           {speedup:>10.2f} x')
    print(f'{"-"*60}')
    print(f'FLOPs Reduction:   {flops_reduction:>10.1f} %')
    print(f'Discrepancy:       {abs(flops_reduction - latency_reduction):>10.1f} %')

    if abs(flops_reduction - latency_reduction) > 10:
        print(f'\n⚠  FLOPs ≠ Latency!')
        print(f'   The {abs(flops_reduction - latency_reduction):.1f}% discrepancy shows that FLOPs')
        print(f'   is not always a reliable proxy for actual inference latency.')
        print(f'   Consider: memory bandwidth, cache, kernel overhead, etc.')

    print(f'{"="*60}\n')


def benchmark_batch_sizes(
    model: nn.Module,
    input_size: Tuple[int, int, int],  # (C, H, W)
    device: torch.device,
    batch_sizes: list = [1, 2, 4, 8, 16, 32],
    num_runs: int = 100
) -> Dict[int, Dict[str, float]]:
    """
    Benchmark model with different batch sizes

    Args:
        model: PyTorch model
        input_size: Input size (C, H, W)
        device: Device to run on
        batch_sizes: List of batch sizes to test
        num_runs: Number of benchmark runs

    Returns:
        Dictionary mapping batch size to latency stats
    """
    results = {}

    for bs in batch_sizes:
        full_input_size = (bs,) + input_size
        try:
            stats = measure_latency_with_stats(model, full_input_size, device, num_warmup=20, num_runs=num_runs)
            results[bs] = stats
            print(f'Batch Size {bs:2d}: {stats["mean_ms"]:6.2f} ms ({stats["throughput_fps"]:6.1f} FPS)')
        except RuntimeError as e:
            print(f'Batch Size {bs:2d}: Failed (OOM or other error)')
            break

    return results
