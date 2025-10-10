"""
ONNX Export Script - Day 13-14
Exports PyTorch models to ONNX/TorchScript for deployment validation
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def export_to_onnx(model, example_inputs, output_path, opset_version=13):
    """Export model to ONNX format"""
    model.eval()

    torch.onnx.export(
        model,
        example_inputs,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f'✓ Exported to ONNX: {output_path}')


def export_to_torchscript(model, example_inputs, output_path):
    """Export model to TorchScript format"""
    model.eval()

    # Trace the model
    traced_model = torch.jit.trace(model, example_inputs)

    # Save
    traced_model.save(str(output_path))
    print(f'✓ Exported to TorchScript: {output_path}')


def verify_onnx_export(onnx_path, pytorch_model, example_inputs, device):
    """Verify ONNX export by comparing outputs"""
    import onnxruntime as ort
    import numpy as np

    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(example_inputs).cpu().numpy()

    # Get ONNX output
    ort_session = ort.InferenceSession(str(onnx_path))
    onnx_inputs = {ort_session.get_inputs()[0].name: example_inputs.cpu().numpy()}
    onnx_output = ort_session.run(None, onnx_inputs)[0]

    # Compare
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    print(f'Max difference between PyTorch and ONNX: {max_diff:.6f}')

    if max_diff < 1e-5:
        print('✓ ONNX export verified successfully')
        return True
    else:
        print('⚠ Warning: ONNX output differs from PyTorch')
        return False


def verify_torchscript_export(ts_path, pytorch_model, example_inputs, device):
    """Verify TorchScript export by comparing outputs"""
    import numpy as np

    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(example_inputs).cpu().numpy()

    # Load and run TorchScript model
    ts_model = torch.jit.load(str(ts_path))
    ts_model.eval()
    with torch.no_grad():
        ts_output = ts_model(example_inputs).cpu().numpy()

    # Compare
    max_diff = np.max(np.abs(pytorch_output - ts_output))
    print(f'Max difference between PyTorch and TorchScript: {max_diff:.6f}')

    if max_diff < 1e-7:
        print('✓ TorchScript export verified successfully')
        return True
    else:
        print('⚠ Warning: TorchScript output differs from PyTorch')
        return False


def main():
    parser = argparse.ArgumentParser(description='Export Model to ONNX/TorchScript')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--format', type=str, default='both',
                       choices=['onnx', 'torchscript', 'both'],
                       help='Export format')
    parser.add_argument('--opset-version', type=int, default=13, help='ONNX opset version')
    parser.add_argument('--verify', action='store_true', help='Verify exported models')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: same as checkpoint)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device(args.device)
    print(f'Using device: {device}')

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    from torchvision.models import resnet18

    # Determine number of classes from config
    dataset_name = config['dataset']['name']
    if 'cifar' in dataset_name.lower():
        num_classes = 10
    elif 'imagenet' in dataset_name.lower():
        num_classes = 1000
    else:
        num_classes = 1000

    model = resnet18(num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f'\n✓ Loaded checkpoint from {args.checkpoint}')

    # Create example inputs
    input_size = config['dataset']['input_size']
    example_inputs = torch.randn(1, 3, input_size, input_size).to(device)

    # Export
    print(f'\n=== Exporting Model ===')

    if args.format in ['onnx', 'both']:
        onnx_path = output_dir / 'model.onnx'
        export_to_onnx(model, example_inputs, onnx_path, args.opset_version)

        if args.verify:
            verify_onnx_export(onnx_path, model, example_inputs, device)

    if args.format in ['torchscript', 'both']:
        ts_path = output_dir / 'model.pt'
        export_to_torchscript(model, example_inputs, ts_path)

        if args.verify:
            verify_torchscript_export(ts_path, model, example_inputs, device)

    print(f'\n✓ Export complete. Files saved to {output_dir}')


if __name__ == '__main__':
    main()
