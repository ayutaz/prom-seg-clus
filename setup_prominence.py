"""
Prominence-based境界抽出のセットアップ
https://github.com/s-malan/prom-word-seg の実装を基にした正確な実装
"""

import os
import warnings
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")


class ProminenceSegmentor:
    """
    Prominence-based word segmentation following s-malan/prom-word-seg implementation
    """

    def __init__(self, frames_per_ms=20, window_size=5, prominence_threshold=0.5):
        self.frames_per_ms = frames_per_ms
        self.window_size = window_size
        self.prominence_threshold = prominence_threshold

    def extract_features(self, audio_path, sr=16000, feature_type="mfcc"):
        """Extract frame-level features from audio"""
        y, _ = librosa.load(audio_path, sr=sr)

        if feature_type == "mfcc":
            features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(0.01 * sr))
            return features.T  # Shape: [frames, features]
        elif feature_type == "melspec":
            features = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=int(0.01 * sr))
            return librosa.power_to_db(features).T
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

    def get_distance(self, embeddings, distance_type="euclidean"):
        """Calculate normalized distances between consecutive embeddings"""
        distances = []

        for i in range(len(embeddings) - 1):
            if distance_type == "euclidean":
                dist = np.linalg.norm(embeddings[i + 1] - embeddings[i])
            elif distance_type == "cosine":
                # Cosine distance = 1 - cosine similarity
                dot_product = np.dot(embeddings[i], embeddings[i + 1])
                norm_product = np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                if norm_product == 0:
                    dist = 1.0
                else:
                    dist = 1 - (dot_product / norm_product)
            else:
                raise ValueError(f"Unsupported distance type: {distance_type}")

            distances.append(dist)

        # Normalize distances using StandardScaler
        distances = np.array(distances).reshape(-1, 1)
        distances = StandardScaler().fit_transform(distances).flatten()

        return distances

    def moving_average(self, distances, window_size=None):
        """Apply moving average filter to distance sequence"""
        if window_size is None:
            window_size = self.window_size

        if len(distances) < window_size:
            return distances

        kernel = np.ones(window_size) / window_size
        return np.convolve(distances, kernel, mode="valid")

    def peak_detection(self, smoothed_distances, prominence_threshold=None):
        """Identify peaks in smoothed distance signals"""
        if prominence_threshold is None:
            prominence_threshold = self.prominence_threshold

        peaks, properties = find_peaks(smoothed_distances, prominence=prominence_threshold)

        return peaks, properties.get("prominences", [])

    def frames_to_seconds(self, frame_indices, sr=16000):
        """Convert frame indices to time in seconds"""
        hop_length = int(0.01 * sr)  # 10ms hop
        return frame_indices * hop_length / sr

    def segment_audio(self, audio_path, sr=16000, feature_type="mfcc", distance_type="euclidean"):
        """
        Complete segmentation pipeline
        """
        # 1. Extract features
        features = self.extract_features(audio_path, sr, feature_type)

        # 2. Calculate distances
        distances = self.get_distance(features, distance_type)

        # 3. Apply smoothing
        smoothed_distances = self.moving_average(distances)

        # 4. Detect peaks
        peaks, prominences = self.peak_detection(smoothed_distances)

        # 5. Convert to time boundaries
        # Adjust peak indices for smoothing offset
        offset = (self.window_size - 1) // 2
        adjusted_peaks = peaks + offset

        boundaries = self.frames_to_seconds(adjusted_peaks, sr)

        # Add end boundary
        duration = librosa.get_duration(path=audio_path)
        boundaries = np.append(boundaries, duration)

        return boundaries.tolist()


def extract_prominence_boundaries(audio_path, sr=16000, **kwargs):
    """
    音声からProminence-basedの境界を抽出
    s-malan/prom-word-seg の正確な実装
    """
    segmentor = ProminenceSegmentor(
        window_size=kwargs.get("window_size", 5),
        prominence_threshold=kwargs.get("prominence_threshold", 0.5),
    )

    return segmentor.segment_audio(
        audio_path,
        sr=sr,
        feature_type=kwargs.get("feature_type", "mfcc"),
        distance_type=kwargs.get("distance_type", "euclidean"),
    )


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
    print("実装: s-malan/prom-word-seg based prominence detection")

    for audio_path in tqdm(audio_files, desc="境界抽出中"):
        try:
            # 境界を抽出（正確な実装）
            boundaries = extract_prominence_boundaries(
                audio_path,
                feature_type="mfcc",
                distance_type="euclidean",
                window_size=5,
                prominence_threshold=0.5,
            )

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
    print("実装: https://github.com/s-malan/prom-word-seg based")
    print("アルゴリズム: Frame-level feature distance + prominence-based peak detection\n")

    # ディレクトリの確認
    if not os.path.exists("data/librispeech/audio"):
        print("エラー: data/librispeech/audioが存在しません")
        print("先に 'uv run python setup_librispeech.py' を実行してください")
        return

    # 境界抽出
    process_all_audio_files()

    print("\n=== アルゴリズム詳細 ===")
    print("1. MFCC特徴量抽出 (13次元)")
    print("2. フレーム間Euclidean距離計算")
    print("3. StandardScaler正規化")
    print("4. Moving average平滑化 (window=5)")
    print("5. Prominence-based peak detection (threshold=0.5)")
    print("6. 時間境界変換")

    print("\n=== 次のステップ ===")
    print("完全なパイプライン実行: uv run python run_librispeech_pipeline.py")


if __name__ == "__main__":
    main()
