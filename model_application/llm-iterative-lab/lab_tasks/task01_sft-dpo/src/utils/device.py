"""Device utilities for GPU/CPU management."""

import torch
from typing import Dict, Any


def get_device_info() -> Dict[str, Any]:
    """Get device information.

    Returns:
        Dictionary containing device info
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["cuda_version"] = torch.version.cuda

    return info


def setup_device() -> torch.device:
    """Setup and return the appropriate device.

    Returns:
        torch.device instance

    Raises:
        RuntimeError: If CUDA is not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This project requires GPU for QLoRA training."
        )

    device = torch.device("cuda")

    # Print device info
    info = get_device_info()
    print(f"\n🖥️  Device: {info['device_name']}")
    print(f"   Total VRAM: {info['total_memory_gb']:.2f} GB")
    print(f"   CUDA Version: {info['cuda_version']}\n")

    return device


def print_memory_usage(prefix: str = "") -> None:
    """Print current GPU memory usage.

    Args:
        prefix: Optional prefix for the output
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"{prefix}GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
