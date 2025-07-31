"""
MFCCを使った簡易テスト
外部ツールなしで、最小限のデータで動作確認
"""

from pathlib import Path

import librosa
import numpy as np


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

    # 1. テスト音声を生成（3秒のサイン波）
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    # 異なる周波数のセグメントを作成（疑似的な「単語」）
    audio = np.zeros_like(t)
    audio[0:sr] = 0.3 * np.sin(2 * np.pi * 440 * t[0:sr])  # "単語1"
    audio[sr : 2 * sr] = 0.3 * np.sin(2 * np.pi * 880 * t[sr : 2 * sr])  # "単語2"
    audio[2 * sr :] = 0.3 * np.sin(2 * np.pi * 440 * t[2 * sr :])  # "単語1"の繰り返し

    # 音声ファイル保存
    audio_path = "test_data/audio/test_001.wav"
    librosa.output.write_wav(audio_path, audio.astype(np.float32), sr)
    print(f"テスト音声を作成: {audio_path}")

    # 2. MFCC特徴量を抽出
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=160)  # 10msホップ
    mfcc = mfcc.T  # (time, features)の形に転置

    # 特徴量保存
    feature_path = "test_data/features/mfcc/test_001.npy"
    np.save(feature_path, mfcc)
    print(f"MFCC特徴量を保存: {feature_path} (shape: {mfcc.shape})")

    # 3. 単語境界を作成（1秒ごと）
    boundaries = [1.0, 2.0, 3.0]
    boundary_path = "test_data/boundaries/test_001.list"
    with open(boundary_path, "w") as f:
        for b in boundaries:
            f.write(f"{b}\n")
    print(f"境界ファイルを作成: {boundary_path}")

    return audio_path, feature_path, boundary_path


def run_test():
    """テスト実行"""
    print("=== 簡易テストデータ作成 ===\n")

    # librosaがインストールされているか確認
    try:
        import importlib.util

        if importlib.util.find_spec("librosa") is None:
            raise ImportError
    except ImportError:
        print("エラー: librosaが必要です。以下でインストールしてください:")
        print("uv add librosa")
        return

    # テストデータ作成
    audio_path, feature_path, boundary_path = create_test_data()

    print("\n=== 実行コマンド ===")
    print(
        "uv run python prom_seg_clus.py mfcc -1 test_data/audio test_data/features test_data/boundaries test_data/output 3 --extension .wav"
    )
    print("\n期待される結果:")
    print("- test_data/output/test_001.list が作成される")
    print("- 3つのセグメントがそれぞれクラスタに割り当てられる")
    print("- 1番目と3番目のセグメントは同じクラスタになる可能性が高い")


if __name__ == "__main__":
    run_test()
