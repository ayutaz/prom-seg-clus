# GPU版PyTorchセットアップガイド

## 問題
デフォルトでインストールされるPyTorchはCPU版のため、GPUが検出されません。

## 解決方法

### 1. GPU版PyTorchのインストール

RTX 4070 Ti SuperはCUDA 12.1に対応しているため、以下のコマンドでGPU版をインストールします：

```bash
# 方法1: install_gpu.pyスクリプトを使用
uv run python install_gpu.py

# 方法2: 手動インストール
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### 2. GPU検出の確認

```bash
uv run python check_gpu.py
```

成功すると以下のような出力が表示されます：
```
CUDA available: True
GPU 0:
  Name: NVIDIA GeForce RTX 4070 Ti SUPER
  Memory: 16.0 GB
```

### 3. 再度HuBERTエンコードを実行

```bash
uv run python setup_hubert.py
```

GPU版では以下のように表示されます：
```
GPU検出: NVIDIA GeForce RTX 4070 Ti SUPER
GPUメモリ: 16.0 GB
使用デバイス: cuda
バッチサイズ: 4
```

## トラブルシューティング

### CUDAバージョンの確認
```bash
nvidia-smi
```

### 環境変数の確認
CUDA_VISIBLE_DEVICESが設定されていないか確認：
```bash
echo $env:CUDA_VISIBLE_DEVICES
```

### PyTorchの再インストール
```bash
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```