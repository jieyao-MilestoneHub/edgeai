#!/usr/bin/env python3
"""環境檢查腳本"""

import sys

def check_python_version():
    """檢查 Python 版本"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("   ❌ Python 3.10+ required")
        return False
    print("   ✅ Python version OK")
    return True

def check_pytorch():
    """檢查 PyTorch"""
    print("\n🔍 Checking PyTorch...")
    try:
        import torch
        print(f"   PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("   ⚠️  CUDA not available (CPU only)")
        return True
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False

def check_transformers():
    """檢查 Transformers"""
    print("\n🔍 Checking Transformers...")
    try:
        import transformers
        print(f"   Transformers {transformers.__version__}")
        print("   ✅ Transformers OK")
        return True
    except ImportError:
        print("   ❌ Transformers not installed")
        return False

def check_datasets():
    """檢查 Datasets"""
    print("\n🔍 Checking Datasets...")
    try:
        import datasets
        print(f"   Datasets {datasets.__version__}")
        print("   ✅ Datasets OK")
        return True
    except ImportError:
        print("   ❌ Datasets not installed")
        return False

def main():
    print("=" * 60)
    print("🧪 LLM Tuning Lab - Environment Check")
    print("=" * 60)

    checks = [
        check_python_version(),
        check_pytorch(),
        check_transformers(),
        check_datasets(),
    ]

    print("\n" + "=" * 60)
    if all(checks):
        print("✅ All checks passed! You're ready to start.")
        print("=" * 60)
        print("\n📚 Start with: cd lab_tasks/task01_lora_basic\n")
        sys.exit(0)
    else:
        print("❌ Some checks failed. Please install missing packages.")
        print("=" * 60)
        print("\n💡 Run: pip install -r requirements.txt\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
