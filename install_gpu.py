"""
GPU版PyTorchのインストールスクリプト
"""

import subprocess
import sys

print("=== GPU版PyTorchインストール ===\n")

# CUDA 12.1版をインストール（RTX 4070 Ti Super対応）
cmd = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "torch",
    "torchvision",
    "torchaudio",
    "--index-url",
    "https://download.pytorch.org/whl/cu121",
    "--force-reinstall",
]

print("実行コマンド:")
print(" ".join(cmd))
print("\nインストール中...")

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("\n✓ インストール成功")
    print("\n次のステップ:")
    print("1. uv run python check_gpu.py  # GPU検出確認")
    print("2. uv run python setup_hubert.py  # HuBERTエンコード（GPU版）")
else:
    print("\n✗ インストール失敗")
    print(result.stderr)
