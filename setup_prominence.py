"""
Prominence-based境界抽出のセットアップ
簡易版の実装（本来はhttps://github.com/s-malan/prom-word-segを使用）
"""

import warnings
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

warnings.filterwarnings("ignore")


def extract_prominence_boundaries(audio_path, sr=16000):
    """
    音声からProminence-basedの境界を抽出
    簡易実装：エネルギーベースの谷検出
    """

    # 音声を読み込み
    y, _ = librosa.load(audio_path, sr=sr)

    # 短時間エネルギーを計算
    hop_length = int(0.01 * sr)  # 10ms
    frame_length = int(0.025 * sr)  # 25ms

    energy = np.array(
        [np.sum(y[i : i + frame_length] ** 2) for i in range(0, len(y) - frame_length, hop_length)]
    )

    # エネルギーを平滑化
    from scipy.ndimage import gaussian_filter1d

    smoothed_energy = gaussian_filter1d(energy, sigma=5)

    # エネルギーの谷（極小値）を見つける
    inverted_energy = -smoothed_energy
    peaks, properties = find_peaks(
        inverted_energy,
        distance=int(0.1 * sr / hop_length),  # 最小100ms間隔
        prominence=np.std(inverted_energy) * 0.3,
    )  # ある程度の深さの谷のみ

    # フレーム番号を時間（秒）に変換
    boundaries = peaks * hop_length / sr

    # 音声の最後に境界を追加
    boundaries = np.append(boundaries, len(y) / sr)

    return boundaries.tolist()


def process_all_audio_files():
    """すべての音声ファイルの境界を抽出"""

    audio_dir = Path("data/librispeech/audio")
    boundary_dir = Path("data/librispeech/boundaries")

    audio_files = list(audio_dir.glob("*.flac"))

    if not audio_files:
        print("エラー: data/librispeech/audioにFLACファイルが見つかりません")
        print("先に以下を実行してください:")
        print("1. uv run python setup_librispeech.py")
        return

    print(f"\n{len(audio_files)}個のファイルから境界を抽出します...")

    for audio_path in tqdm(audio_files, desc="境界抽出中"):
        try:
            # 境界を抽出
            boundaries = extract_prominence_boundaries(audio_path)

            # 境界が少なすぎる場合は固定間隔で分割
            if len(boundaries) < 3:
                duration = librosa.get_duration(path=audio_path)
                # 0.5秒間隔で分割
                boundaries = np.arange(0.5, duration, 0.5).tolist()
                boundaries.append(duration)

            # 保存
            boundary_path = boundary_dir / f"{audio_path.stem}.list"
            with open(boundary_path, "w") as f:
                for boundary in boundaries:
                    f.write(f"{boundary:.3f}\n")

        except Exception as e:
            print(f"\nエラー: {audio_path.name} の処理中にエラーが発生しました: {e}")
            continue

    print(f"\n境界抽出完了: {boundary_dir}")


def main():
    print("=== Prominence境界抽出セットアップ ===\n")
    print("注意: これは簡易実装です。")
    print("本格的な実装は https://github.com/s-malan/prom-word-seg を使用してください。\n")

    # ディレクトリの確認
    if not os.path.exists("data/librispeech/audio"):
        print("エラー: data/librispeech/audioが存在しません")
        print("先に 'uv run python setup_librispeech.py' を実行してください")
        return

    # 境界抽出
    process_all_audio_files()

    print("\n=== 次のステップ ===")
    print("完全なパイプライン実行: uv run python run_librispeech_pipeline.py")


if __name__ == "__main__":
    import os

    main()
