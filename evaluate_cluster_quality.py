"""
クラスタの品質を評価するためのツール
複数のクラスタをランダムにサンプリングして出力
"""

import random
from pathlib import Path
import soundfile as sf
import numpy as np
from collections import defaultdict


def export_random_clusters(n_clusters=10, examples_per_cluster=3):
    """ランダムに選んだクラスタの音声を出力"""
    
    audio_dir = Path("data/librispeech/audio")
    output_dir = Path("data/librispeech/output")
    boundary_dir = Path("data/librispeech/boundaries")
    
    # すべてのクラスタを収集
    all_clusters = defaultdict(list)
    
    for list_file in output_dir.glob("*.list"):
        audio_name = list_file.stem
        audio_path = audio_dir / f"{audio_name}.flac"
        boundary_file = boundary_dir / f"{audio_name}.list"
        
        if not audio_path.exists() or not boundary_file.exists():
            continue
            
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
                    
                    all_clusters[cluster_id].append({
                        'audio_path': audio_path,
                        'start_time': boundaries[segment_idx],
                        'end_time': boundaries[segment_idx + 1],
                        'segment_idx': segment_idx,
                        'file': audio_name
                    })
                    
                    segment_idx += 1
    
    # 複数回出現するクラスタのみを対象
    multi_occurrence_clusters = {cid: segs for cid, segs in all_clusters.items() if len(segs) >= 2}
    
    print(f"\n複数回出現するクラスタ数: {len(multi_occurrence_clusters)}")
    print(f"全クラスタ数: {len(all_clusters)}")
    
    # ランダムにクラスタを選択
    selected_clusters = random.sample(list(multi_occurrence_clusters.keys()), 
                                    min(n_clusters, len(multi_occurrence_clusters)))
    
    # 振幅でフィルタリング
    good_clusters = []
    
    for cluster_id in selected_clusters:
        segments = multi_occurrence_clusters[cluster_id][:examples_per_cluster]
        
        # 最初のセグメントの振幅をチェック
        seg = segments[0]
        audio, sr = sf.read(seg['audio_path'])
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        start_sample = int(seg['start_time'] * sr)
        end_sample = int(seg['end_time'] * sr)
        if end_sample <= len(audio):
            segment_audio = audio[start_sample:end_sample]
            max_amp = np.max(np.abs(segment_audio))
            
            if max_amp > 0.1:  # 振幅が0.1以上
                good_clusters.append((cluster_id, max_amp))
    
    # 振幅順にソート
    good_clusters.sort(key=lambda x: x[1], reverse=True)
    
    print("\n=== 評価用クラスタ（音声がはっきりしているもの） ===")
    
    export_base_dir = Path("cluster_evaluation")
    export_base_dir.mkdir(exist_ok=True)
    
    for i, (cluster_id, amp) in enumerate(good_clusters[:n_clusters]):
        print(f"\nクラスタ {cluster_id} (出現回数: {len(all_clusters[cluster_id])})")
        
        # このクラスタの音声を出力
        cluster_dir = export_base_dir / f"cluster_{cluster_id}"
        cluster_dir.mkdir(exist_ok=True)
        
        segments = multi_occurrence_clusters[cluster_id][:examples_per_cluster]
        
        for j, seg in enumerate(segments):
            # 音声を読み込み
            audio, sr = sf.read(seg['audio_path'])
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # セグメント抽出
            start_sample = int(seg['start_time'] * sr)
            end_sample = int(seg['end_time'] * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # 保存
            filename = cluster_dir / f"example_{j+1}_{seg['file']}.wav"
            sf.write(filename, segment_audio, sr)
            
            duration = seg['end_time'] - seg['start_time']
            print(f"  例{j+1}: {seg['file']} [{seg['start_time']:.2f}s-{seg['end_time']:.2f}s] ({duration:.2f}秒)")
    
    print(f"\n\n=== 結果 ===")
    print(f"音声ファイルは {export_base_dir} に保存されました")
    print("\n各クラスタフォルダ内の音声ファイルを聞き比べて、")
    print("同じクラスタの音声が本当に似ているか確認してください。")
    
    # 簡単な統計
    print("\n=== クラスタ品質の評価方法 ===")
    print("1. 同じクラスタ内の音声が同じ単語に聞こえるか？")
    print("2. 異なるクラスタの音声は異なる単語に聞こえるか？")
    print("3. 音声の長さは一貫しているか？")
    print("\n良いクラスタの例：")
    print("- 3つの音声すべてが同じ単語（例：'the'）")
    print("- 似た長さ（例：0.2-0.3秒）")
    print("\n悪いクラスタの例：")
    print("- 異なる単語が混在")
    print("- 長さがバラバラ（例：0.1秒と0.8秒）")


def main():
    print("=== クラスタ品質評価ツール ===")
    
    # シード設定で再現性を確保
    random.seed(42)
    
    # 10個のクラスタをランダムに選んで、各3つの例を出力
    export_random_clusters(n_clusters=10, examples_per_cluster=3)


if __name__ == "__main__":
    main()
