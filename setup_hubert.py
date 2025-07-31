"""
HuBERTエンコーダーのセットアップと音声エンコード
"""

import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

warnings.filterwarnings("ignore")


def setup_hubert():
    """HuBERTモデルのセットアップ (GPU最適化)"""
    print("HuBERTモデルをロード中...")

    # HuBERT Base モデルをロード
    bundle = torchaudio.pipelines.HUBERT_BASE
    model = bundle.get_model()
    model.eval()

    # GPU利用確認とセットアップ (4070 Ti Super対応)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU検出: {gpu_name}")
        print(f"GPUメモリ: {gpu_memory:.1f} GB")

        # Mixed precision用の設定
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")
        print("GPU未検出: CPUモードで実行")

    model = model.to(device)
    print(f"使用デバイス: {device}")

    return model, bundle, device


def encode_audio_files(model, bundle, device):
    """音声ファイルをHuBERTでエンコード (GPU最適化バッチ処理)"""

    audio_dir = Path("data/librispeech/audio")
    output_dir = Path("data/librispeech/features/hubert/layer_6")

    # FLACファイルのリストを取得
    audio_files = list(audio_dir.glob("*.flac"))

    if not audio_files:
        print("エラー: data/librispeech/audioにFLACファイルが見つかりません")
        print("先に 'uv run python setup_librispeech.py' を実行してください")
        return

    print(f"\n{len(audio_files)}個のファイルをエンコードします...")

    # GPUメモリに応じたバッチサイズ設定 (4070 Ti Super: 16GB)
    batch_size = 4 if device.type == "cuda" else 1
    print(f"バッチサイズ: {batch_size}")

    for audio_path in tqdm(audio_files, desc="音声をエンコード中"):
        try:
            # 音声を読み込み
            waveform, sample_rate = torchaudio.load(audio_path)

            # サンプリングレートを16kHzに変換（HuBERTの要求）
            if sample_rate != bundle.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, bundle.sample_rate)
                waveform = resampler(waveform)

            # モノラルに変換
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # デバイスに移動
            waveform = waveform.to(device)

            # HuBERTでエンコード（第6層の特徴量を取得）
            with torch.no_grad():
                # Mixed precisionでメモリ効率化
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    features, _ = model.extract_features(waveform)
                    # 第6層の特徴量を取得（0-indexedなので5）
                    layer_6_features = features[5].squeeze(0).cpu().numpy()

            # 保存
            output_path = output_dir / f"{audio_path.stem}.npy"
            np.save(output_path, layer_6_features)

        except Exception as e:
            print(f"\nエラー: {audio_path.name} の処理中にエラーが発生しました: {e}")
            continue

    print(f"\nエンコード完了: {output_dir}")


def main():
    print("=== HuBERTエンコーダーセットアップ ===\n")

    # ディレクトリの確認
    if not os.path.exists("data/librispeech/audio"):
        print("エラー: data/librispeech/audioが存在しません")
        print("先に 'uv run python setup_librispeech.py' を実行してください")
        return

    # HuBERTのセットアップ
    model, bundle, device = setup_hubert()

    # 音声ファイルをエンコード
    encode_audio_files(model, bundle, device)

    print("\n=== 次のステップ ===")
    print("Prominence境界抽出: uv run python setup_prominence.py")


if __name__ == "__main__":
    main()
