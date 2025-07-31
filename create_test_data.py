"""
MFCCを使った簡易テストデータ作成
実際に動作する最小限のテストデータを生成
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def create_test_data():
    """テスト用の音声データとMFCC特徴量を生成"""

    # テスト用ディレクトリ作成
    dirs = [
        "test_data/audio",
        "test_data/features/mfcc",
        "test_data/boundaries",
        "test_data/output",
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    # 複数のテスト音声を生成
    sr = 16000

    for i in range(3):  # 3つのサンプル
        # 1. テスト音声を生成（3秒）
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))

        # 異なる周波数のセグメントを作成（疑似的な「単語」）
        audio = np.zeros_like(t)

        # パターンを変える
        if i == 0:
            # パターン1: A-B-A (440Hz-880Hz-440Hz)
            audio[0:sr] = 0.3 * np.sin(2 * np.pi * 440 * t[0:sr])
            audio[sr : 2 * sr] = 0.3 * np.sin(2 * np.pi * 880 * t[sr : 2 * sr])
            audio[2 * sr :] = 0.3 * np.sin(2 * np.pi * 440 * t[2 * sr :])
        elif i == 1:
            # パターン2: B-A-B (880Hz-440Hz-880Hz)
            audio[0:sr] = 0.3 * np.sin(2 * np.pi * 880 * t[0:sr])
            audio[sr : 2 * sr] = 0.3 * np.sin(2 * np.pi * 440 * t[sr : 2 * sr])
            audio[2 * sr :] = 0.3 * np.sin(2 * np.pi * 880 * t[2 * sr :])
        else:
            # パターン3: A-A-B (440Hz-440Hz-880Hz)
            audio[0:sr] = 0.3 * np.sin(2 * np.pi * 440 * t[0:sr])
            audio[sr : 2 * sr] = 0.3 * np.sin(2 * np.pi * 440 * t[sr : 2 * sr])
            audio[2 * sr :] = 0.3 * np.sin(2 * np.pi * 880 * t[2 * sr :])

        # ノイズを少し追加（よりリアルに）
        audio += 0.01 * np.random.randn(len(audio))

        # 音声ファイル保存
        audio_path = f"test_data/audio/test_{i:03d}.wav"
        sf.write(audio_path, audio, sr)
        print(f"テスト音声を作成: {audio_path}")

        # 2. MFCC特徴量を抽出
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=13, hop_length=160, n_fft=400
        )  # 10msホップ
        mfcc = mfcc.T  # (time, features)の形に転置

        # 特徴量保存
        feature_path = f"test_data/features/mfcc/test_{i:03d}.npy"
        np.save(feature_path, mfcc)
        print(f"MFCC特徴量を保存: {feature_path} (shape: {mfcc.shape})")

        # 3. 単語境界を作成（1秒ごと）
        boundaries = [1.0, 2.0, 3.0]
        boundary_path = f"test_data/boundaries/test_{i:03d}.list"
        with open(boundary_path, "w") as f:
            for b in boundaries:
                f.write(f"{b}\n")
        print(f"境界ファイルを作成: {boundary_path}")

    print(f"\n合計{3}個のテストファイルを作成しました")


def main():
    print("=== テストデータ作成 ===\n")

    # テストデータ作成
    create_test_data()

    print("\n=== 次のステップ ===")
    print("以下のコマンドでクラスタリングを実行:")
    print(
        "uv run python prom_seg_clus.py mfcc -1 test_data/audio test_data/features test_data/boundaries test_data/output 5 --extension .wav"
    )


if __name__ == "__main__":
    main()
