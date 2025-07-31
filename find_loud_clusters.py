"""
振幅が大きいクラスタを探す
"""

import soundfile as sf
import numpy as np
from pathlib import Path
from collections import defaultdict


def find_loud_clusters():
    """振幅が大きいセグメントを持つクラスタを探す"""
    
    audio_dir = Path("data/librispeech/audio")
    output_dir = Path("data/librispeech/output")
    boundary_dir = Path("data/librispeech/boundaries")
    
    # クラスタごとの最大振幅を記録
    cluster_amplitudes = defaultdict(list)
    
    # 最初の数ファイルをチェック
    list_files = list(output_dir.glob("*.list"))[:5]
    
    for list_file in list_files:
        audio_name = list_file.stem
        audio_path = audio_dir / f"{audio_name}.flac"
        boundary_file = boundary_dir / f"{audio_name}.list"
        
        if not audio_path.exists() or not boundary_file.exists():
            continue
        
        print(f"\n処理中: {audio_name}")
        
        # 音声ファイルを読み込み
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # 境界情報を読み込み
        boundaries = [0.0]
        with open(boundary_file, 'r') as f:
            for line in f:
                boundaries.append(float(line.strip()))
        
        # クラスタ情報を読み込み
        with open(list_file, 'r') as f:
            segment_idx = 0
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and segment_idx < len(boundaries) - 1:
                    cluster_id = int(parts[1])
                    
                    # セグメントを抽出
                    start_time = boundaries[segment_idx]
                    end_time = boundaries[segment_idx + 1]
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    
                    if end_sample <= len(audio):
                        segment = audio[start_sample:end_sample]
                        max_amp = np.max(np.abs(segment))
                        avg_amp = np.mean(np.abs(segment))
                        
                        cluster_amplitudes[cluster_id].append({
                            'max_amp': max_amp,
                            'avg_amp': avg_amp,
                            'file': audio_name,
                            'segment_idx': segment_idx,
                            'duration': end_time - start_time
                        })
                    
                    segment_idx += 1
    
    # 各クラスタの平均振幅を計算
    cluster_avg_amps = {}
    for cluster_id, segments in cluster_amplitudes.items():
        avg_max_amp = np.mean([s['max_amp'] for s in segments])
        cluster_avg_amps[cluster_id] = (avg_max_amp, len(segments))
    
    # 振幅が大きい順にソート
    sorted_clusters = sorted(cluster_avg_amps.items(), key=lambda x: x[1][0], reverse=True)
    
    print("\n\n=== 振幅が大きいクラスタ TOP 20 ===")
    for i, (cluster_id, (avg_amp, count)) in enumerate(sorted_clusters[:20]):
        print(f"\n{i+1}. クラスタ {cluster_id}:")
        print(f"   平均最大振幅: {avg_amp:.4f}")
        print(f"   出現回数: {count}")
        
        # 具体例を表示
        examples = cluster_amplitudes[cluster_id][:2]
        for ex in examples:
            print(f"   例: {ex['file']} #{ex['segment_idx']}, 振幅={ex['max_amp']:.4f}, 長さ={ex['duration']:.2f}s")
    
    # 推奨クラスタを出力
    print("\n\n=== 推奨クラスタ（音声がはっきりしている） ===")
    recommended = []
    for cluster_id, (avg_amp, count) in sorted_clusters[:10]:
        if avg_amp > 0.1 and count >= 2:  # 振幅が0.1以上、2回以上出現
            recommended.append(cluster_id)
    
    print("以下のクラスタを試してみてください:")
    for cluster_id in recommended[:5]:
        print(f"  uv run python export_cluster_audio_fixed.py {cluster_id}")


def main():
    find_loud_clusters()


if __name__ == "__main__":
    main()
