"""
音声セグメント再生ツール
同じクラスタIDのセグメントを聴き比べる
"""

import os
from pathlib import Path
import numpy as np
import soundfile as sf
import sounddevice as sd
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import librosa
import librosa.display


class SegmentPlayer:
    """音声セグメントの再生と分析"""
    
    def __init__(self, audio_dir="data/librispeech/audio", output_dir="data/librispeech/output"):
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.segment_data = self._load_all_segments()
        
    def _load_all_segments(self):
        """すべてのセグメント情報を読み込む"""
        segment_data = defaultdict(list)
        
        for list_file in self.output_dir.glob("*.list"):
            audio_name = list_file.stem
            audio_path = self.audio_dir / f"{audio_name}.flac"
            
            if not audio_path.exists():
                continue
                
            with open(list_file, 'r') as f:
                prev_time = 0
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        end_time = float(parts[0])
                        cluster_id = int(parts[1])
                        
                        segment_data[cluster_id].append({
                            'audio_path': audio_path,
                            'start_time': prev_time,
                            'end_time': end_time,
                            'duration': end_time - prev_time
                        })
                        
                        prev_time = end_time
        
        return segment_data
    
    def get_cluster_examples(self, cluster_id, max_examples=5):
        """特定クラスタの例を取得"""
        if cluster_id not in self.segment_data:
            print(f"クラスタ {cluster_id} が見つかりません")
            return []
        
        examples = self.segment_data[cluster_id][:max_examples]
        print(f"\nクラスタ {cluster_id} の例 ({len(self.segment_data[cluster_id])}個中{len(examples)}個):")
        
        for i, seg in enumerate(examples):
            print(f"{i+1}. {seg['audio_path'].name} [{seg['start_time']:.2f}s - {seg['end_time']:.2f}s] (長さ: {seg['duration']:.2f}s)")
        
        return examples
    
    def load_segment(self, segment_info, sr=16000):
        """セグメントの音声データを読み込む"""
        audio, orig_sr = sf.read(segment_info['audio_path'])
        
        # モノラルに変換
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # リサンプリング
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        
        # セグメント抽出
        start_sample = int(segment_info['start_time'] * sr)
        end_sample = int(segment_info['end_time'] * sr)
        
        return audio[start_sample:end_sample], sr
    
    def play_segment(self, segment_info):
        """セグメントを再生"""
        audio, sr = self.load_segment(segment_info)
        print(f"再生中: {segment_info['start_time']:.2f}s - {segment_info['end_time']:.2f}s")
        sd.play(audio, sr)
        sd.wait()
    
    def visualize_segments(self, cluster_id, max_examples=5, save_path=None):
        """同じクラスタのセグメントを可視化"""
        examples = self.get_cluster_examples(cluster_id, max_examples)
        
        if not examples:
            return
        
        fig, axes = plt.subplots(len(examples), 2, figsize=(12, 3*len(examples)))
        if len(examples) == 1:
            axes = axes.reshape(1, -1)
        
        for i, seg in enumerate(examples):
            audio, sr = self.load_segment(seg)
            
            # 波形
            ax1 = axes[i, 0]
            time = np.linspace(0, len(audio)/sr, len(audio))
            ax1.plot(time, audio)
            ax1.set_title(f"{seg['audio_path'].name} [{seg['start_time']:.2f}s - {seg['end_time']:.2f}s]")
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            
            # スペクトログラム
            ax2 = axes[i, 1]
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
            ax2.set_title('Spectrogram')
            plt.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        plt.suptitle(f'クラスタ {cluster_id} のセグメント例', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可視化を保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_clusters(self, cluster_ids, examples_per_cluster=3):
        """複数のクラスタを比較"""
        fig, axes = plt.subplots(len(cluster_ids), examples_per_cluster, 
                                figsize=(4*examples_per_cluster, 3*len(cluster_ids)))
        
        for i, cluster_id in enumerate(cluster_ids):
            examples = self.get_cluster_examples(cluster_id, examples_per_cluster)
            
            for j, seg in enumerate(examples):
                audio, sr = self.load_segment(seg)
                
                ax = axes[i, j] if len(cluster_ids) > 1 else axes[j]
                
                # スペクトログラム
                D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
                librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
                
                if j == 0:
                    ax.set_ylabel(f'Cluster {cluster_id}')
                ax.set_title(f"{seg['duration']:.2f}s")
        
        plt.suptitle('クラスタ間の比較', fontsize=16)
        plt.tight_layout()
        plt.savefig('cluster_comparison.png', dpi=300, bbox_inches='tight')
        print("クラスタ比較を保存: cluster_comparison.png")
        plt.close()
    
    def export_segments(self, cluster_id, output_dir="exported_segments", max_examples=10):
        """特定クラスタのセグメントをWAVファイルとして出力"""
        output_path = Path(output_dir) / f"cluster_{cluster_id}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        examples = self.get_cluster_examples(cluster_id, max_examples)
        
        for i, seg in enumerate(examples):
            audio, sr = self.load_segment(seg)
            
            filename = f"cluster_{cluster_id}_example_{i+1}.wav"
            sf.write(output_path / filename, audio, sr)
            
        print(f"\n{len(examples)}個のセグメントを {output_path} に出力しました")


def interactive_session():
    """対話的なセグメント再生セッション"""
    
    print("=== 音声セグメント再生ツール ===\n")
    
    # sounddeviceのインストール確認
    try:
        import sounddevice as sd
    except ImportError:
        print("エラー: sounddeviceが必要です。以下でインストールしてください:")
        print("uv pip install sounddevice")
        return
    
    player = SegmentPlayer()
    
    print("利用可能なクラスタ数:", len(player.segment_data))
    
    # 最頻出クラスタを表示
    cluster_counts = {cid: len(segs) for cid, segs in player.segment_data.items()}
    top_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\n最頻出クラスタ TOP 10:")
    for cid, count in top_clusters:
        print(f"  クラスタ {cid}: {count}個")
    
    while True:
        print("\n" + "="*50)
        print("コマンド:")
        print("  play <cluster_id>    - クラスタの例を再生")
        print("  vis <cluster_id>     - クラスタを可視化")
        print("  compare <id1> <id2>  - 2つのクラスタを比較")
        print("  export <cluster_id>  - セグメントをWAVファイルで出力")
        print("  quit                 - 終了")
        
        command = input("\n> ").strip().split()
        
        if not command:
            continue
        
        if command[0] == "quit":
            break
        
        elif command[0] == "play" and len(command) > 1:
            try:
                cluster_id = int(command[1])
                examples = player.get_cluster_examples(cluster_id, 3)
                
                for i, seg in enumerate(examples):
                    print(f"\n例 {i+1}を再生...")
                    player.play_segment(seg)
                    
                    if i < len(examples) - 1:
                        input("Enterで次の例を再生...")
                        
            except ValueError:
                print("エラー: クラスタIDは整数で指定してください")
            except Exception as e:
                print(f"エラー: {e}")
        
        elif command[0] == "vis" and len(command) > 1:
            try:
                cluster_id = int(command[1])
                player.visualize_segments(cluster_id, save_path=f"cluster_{cluster_id}_vis.png")
            except ValueError:
                print("エラー: クラスタIDは整数で指定してください")
        
        elif command[0] == "compare" and len(command) > 2:
            try:
                cluster_ids = [int(command[1]), int(command[2])]
                player.compare_clusters(cluster_ids)
            except ValueError:
                print("エラー: クラスタIDは整数で指定してください")
        
        elif command[0] == "export" and len(command) > 1:
            try:
                cluster_id = int(command[1])
                player.export_segments(cluster_id)
            except ValueError:
                print("エラー: クラスタIDは整数で指定してください")
        
        else:
            print("不明なコマンドです")


def main():
    # 非対話モードでの実行例
    player = SegmentPlayer()
    
    # 最頻出クラスタの可視化
    cluster_counts = {cid: len(segs) for cid, segs in player.segment_data.items()}
    top_cluster = max(cluster_counts.items(), key=lambda x: x[1])[0]
    
    print(f"最頻出クラスタ {top_cluster} を可視化中...")
    player.visualize_segments(top_cluster, save_path=f"top_cluster_{top_cluster}.png")
    
    # 対話モード
    print("\n対話モードを開始するには 'python play_segments.py --interactive' を実行してください")


if __name__ == "__main__":
    import sys
    if "--interactive" in sys.argv:
        interactive_session()
    else:
        main()