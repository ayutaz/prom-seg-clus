"""
より多くのクラスタを評価し、統計を取る
"""

import random
from pathlib import Path
import soundfile as sf
import numpy as np
from collections import defaultdict, Counter


def analyze_cluster_statistics():
    """クラスタの統計情報を分析"""
    
    audio_dir = Path("data/librispeech/audio")
    output_dir = Path("data/librispeech/output")
    boundary_dir = Path("data/librispeech/boundaries")
    
    # すべてのクラスタ情報を収集
    cluster_stats = defaultdict(lambda: {
        'segments': [],
        'durations': [],
        'amplitudes': [],
        'files': set()
    })
    
    print("データを分析中...")
    
    for list_file in list(output_dir.glob("*.list"))[:20]:  # 最初の20ファイル
        audio_name = list_file.stem
        audio_path = audio_dir / f"{audio_name}.flac"
        boundary_file = boundary_dir / f"{audio_name}.list"
        
        if not audio_path.exists() or not boundary_file.exists():
            continue
            
        # 音声を読み込み
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
                    
                    start_time = boundaries[segment_idx]
                    end_time = boundaries[segment_idx + 1]
                    duration = end_time - start_time
                    
                    # 振幅を計算
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    if end_sample <= len(audio):
                        segment = audio[start_sample:end_sample]
                        max_amp = np.max(np.abs(segment))
                        
                        cluster_stats[cluster_id]['segments'].append({
                            'file': audio_name,
                            'start': start_time,
                            'end': end_time,
                            'idx': segment_idx
                        })
                        cluster_stats[cluster_id]['durations'].append(duration)
                        cluster_stats[cluster_id]['amplitudes'].append(max_amp)
                        cluster_stats[cluster_id]['files'].add(audio_name)
                    
                    segment_idx += 1
    
    # 統計を計算
    print("\n=== クラスタ統計 ===")
    
    # 長さの一貫性が高いクラスタを探す
    consistent_clusters = []
    
    for cluster_id, stats in cluster_stats.items():
        if len(stats['durations']) >= 3:  # 3回以上出現
            durations = stats['durations']
            mean_duration = np.mean(durations)
            std_duration = np.std(durations)
            cv = std_duration / mean_duration if mean_duration > 0 else 1.0
            
            # 変動係数が小さい（一貫性が高い）かつ振幅が大きい
            mean_amp = np.mean(stats['amplitudes'])
            if cv < 0.3 and mean_amp > 0.1:
                consistent_clusters.append({
                    'id': cluster_id,
                    'cv': cv,
                    'mean_duration': mean_duration,
                    'mean_amp': mean_amp,
                    'count': len(durations),
                    'n_files': len(stats['files'])
                })
    
    # 一貫性順にソート
    consistent_clusters.sort(key=lambda x: x['cv'])
    
    print("\n=== 長さの一貫性が高いクラスタ TOP 10 ===")
    print("変動係数(CV)が小さいほど一貫性が高い")
    print("")
    
    export_dir = Path("consistent_clusters")
    export_dir.mkdir(exist_ok=True)
    
    for i, cluster in enumerate(consistent_clusters[:10]):
        cluster_id = cluster['id']
        print(f"\n{i+1}. クラスタ {cluster_id}:")
        print(f"   変動係数: {cluster['cv']:.3f}")
        print(f"   平均長さ: {cluster['mean_duration']:.2f}秒 ± {cluster['cv']*cluster['mean_duration']:.2f}秒")
        print(f"   平均振幅: {cluster['mean_amp']:.3f}")
        print(f"   出現回数: {cluster['count']}回 ({cluster['n_files']}ファイル)")
        
        # このクラスタの音声を出力
        if i < 5:  # 上位5つを出力
            cluster_dir = export_dir / f"cluster_{cluster_id}_cv{cluster['cv']:.3f}"
            cluster_dir.mkdir(exist_ok=True)
            
            # 最大3つの例を出力
            segments = cluster_stats[cluster_id]['segments'][:3]
            for j, seg in enumerate(segments):
                audio_path = audio_dir / f"{seg['file']}.flac"
                audio, sr = sf.read(audio_path)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                start_sample = int(seg['start'] * sr)
                end_sample = int(seg['end'] * sr)
                segment_audio = audio[start_sample:end_sample]
                
                filename = cluster_dir / f"example_{j+1}_{seg['file']}.wav"
                sf.write(filename, segment_audio, sr)
    
    print(f"\n\n音声ファイルは {export_dir} に保存されました")
    print("これらは長さが一貫しているクラスタなので、")
    print("同じ単語である可能性が高いです。")


def main():
    print("=== 一貫性の高いクラスタの探索 ===")
    analyze_cluster_statistics()


if __name__ == "__main__":
    main()
