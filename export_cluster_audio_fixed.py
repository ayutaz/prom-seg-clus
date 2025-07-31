"""
特定クラスタの音声をWAVファイルとして出力（修正版）
"""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from collections import defaultdict


def load_boundaries(boundary_file):
    """境界ファイルから時間情報を読み込み"""
    boundaries = [0.0]  # 最初は0秒から
    with open(boundary_file, 'r') as f:
        for line in f:
            time = float(line.strip())
            boundaries.append(time)
    return boundaries


def export_cluster_segments(cluster_id, max_examples=5):
    """指定クラスタの音声セグメントをWAVファイルとして出力"""
    
    audio_dir = Path("data/librispeech/audio")
    output_dir = Path("data/librispeech/output")
    boundary_dir = Path("data/librispeech/boundaries")
    
    # セグメント情報を読み込み
    segment_data = defaultdict(list)
    
    for list_file in output_dir.glob("*.list"):
        audio_name = list_file.stem
        audio_path = audio_dir / f"{audio_name}.flac"
        boundary_file = boundary_dir / f"{audio_name}.list"
        
        if not audio_path.exists() or not boundary_file.exists():
            continue
        
        # 境界情報を読み込み
        boundaries = load_boundaries(boundary_file)
        
        # クラスタ情報を読み込み
        with open(list_file, 'r') as f:
            segment_idx = 0
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # 注意: outputファイルの最初の列は終了時刻ではなく、
                    # セグメントの終了時刻を示している
                    end_time_in_file = float(parts[0])
                    cid = int(parts[1])
                    
                    # 境界ファイルから実際の開始・終了時刻を取得
                    if segment_idx < len(boundaries) - 1:
                        start_time = boundaries[segment_idx]
                        end_time = boundaries[segment_idx + 1]
                        
                        if cid == cluster_id:
                            segment_data[cid].append({
                                'audio_path': audio_path,
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': end_time - start_time,
                                'segment_idx': segment_idx
                            })
                    
                    segment_idx += 1
    
    if cluster_id not in segment_data:
        print(f"クラスタ {cluster_id} が見つかりません")
        return
    
    # 出力ディレクトリ作成
    export_dir = Path(f"exported_audio_fixed/cluster_{cluster_id}")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    examples = segment_data[cluster_id][:max_examples]
    print(f"\nクラスタ {cluster_id} の音声セグメントを出力中...")
    print(f"総数: {len(segment_data[cluster_id])}個中{len(examples)}個を出力")
    
    for i, seg in enumerate(examples):
        # 音声データ読み込み
        audio, sr = sf.read(seg['audio_path'])
        
        # モノラルに変換
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # セグメント抽出
        start_sample = int(seg['start_time'] * sr)
        end_sample = int(seg['end_time'] * sr)
        segment_audio = audio[start_sample:end_sample]
        
        # WAVファイルとして保存
        filename = export_dir / f"example_{i+1:02d}_{seg['start_time']:.2f}s-{seg['end_time']:.2f}s.wav"
        sf.write(filename, segment_audio, sr)
        
        print(f"  {i+1}. {filename.name} (長さ: {seg['duration']:.2f}秒)")
        print(f"     ファイル: {seg['audio_path'].name}, セグメント#{seg['segment_idx']}")
        print(f"     最大振幅: {np.max(np.abs(segment_audio)):.4f}")
    
    print(f"\n完了！")
    print(f"音声ファイルは {export_dir} に保存されました")
    print(f"\nWindowsで再生するには:")
    print(f"1. エクスプローラーで {export_dir} を開く")
    print(f"2. WAVファイルをダブルクリックして再生")
    
    return export_dir


def main():
    if len(sys.argv) < 2:
        # デフォルトで最頻出クラスタ871を出力
        cluster_id = 871
        print(f"デフォルト: クラスタ {cluster_id} を出力します")
    else:
        try:
            cluster_id = int(sys.argv[1])
        except ValueError:
            print("エラー: クラスタIDは整数で指定してください")
            print("使用法: python export_cluster_audio_fixed.py [クラスタID]")
            return
    
    export_cluster_segments(cluster_id)


if __name__ == "__main__":
    main()
