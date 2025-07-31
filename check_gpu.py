"""
GPU検出とCUDA環境の確認
"""

import torch
import sys

print("=== GPU/CUDA環境チェック ===\n")

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(
            f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
        )
else:
    print("\nGPUが検出されません。以下を確認してください：")
    print("1. NVIDIAドライバが最新か")
    print("2. CUDA対応のPyTorchがインストールされているか")
    print("3. 環境変数CUDA_VISIBLEが設定されていないか")

print("\n=== 推奨インストールコマンド ===")
print("GPU版PyTorchのインストール (CUDA 11.8):")
print(
    "uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
)
print("\nGPU版PyTorchのインストール (CUDA 12.1):")
print(
    "uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
)
