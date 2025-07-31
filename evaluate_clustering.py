"""
クラスタリング評価実験
LibriSpeechの正解ラベルとの比較とZeroSpeech Challenge評価指標
"""

import os
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class ClusteringEvaluator:
    """クラスタリング結果の評価"""
    
    def __init__(self, output_dir="data/librispeech/output", 
                 alignment_dir="data/librispeech/alignments"):
        self.output_dir = Path(output_dir)
        self.alignment_dir = Path(alignment_dir)
        self.results = self._load_clustering_results()
        
    def _load_clustering_results(self):
        """クラスタリング結果を読み込む"""
        results = {}
        
        for list_file in self.output_dir.glob("*.list"):
            segments = []
            with open(list_file, 'r') as f:
                prev_time = 0
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        end_time = float(parts[0])
                        cluster_id = int(parts[1])
                        segments.append({
                            'start': prev_time,
                            'end': end_time,
                            'cluster': cluster_id
                        })
                        prev_time = end_time
            
            results[list_file.stem] = segments
        
        return results
    
    def compute_boundary_metrics(self, tolerance=0.02):
        """境界検出の精度を評価（tolerance: 許容誤差秒）"""
        
        # この関数は実際のアライメントデータが必要
        # LibriSpeechの場合、別途ダウンロードが必要
        
        print("\n=== 境界検出評価 ===")
        print(f"許容誤差: {tolerance*1000:.0f}ms")
        
        # プレースホルダー（実際の評価には正解データが必要）
        print("注意: 境界評価には正解アライメントデータが必要です")
        print("https://zenodo.org/records/2619474 からダウンロードしてください")
        
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    def compute_clustering_metrics(self):
        """クラスタリング品質の評価"""
        
        print("\n=== クラスタリング評価 ===")
        
        # クラスタ統計
        all_clusters = []
        for segments in self.results.values():
            all_clusters.extend([s['cluster'] for s in segments])
        
        cluster_counts = Counter(all_clusters)
        
        metrics = {
            'n_clusters': len(cluster_counts),
            'n_segments': len(all_clusters),
            'avg_cluster_size': len(all_clusters) / len(cluster_counts),
            'singleton_ratio': sum(1 for c in cluster_counts.values() if c == 1) / len(cluster_counts)
        }
        
        print(f"クラスタ数: {metrics['n_clusters']}")
        print(f"総セグメント数: {metrics['n_segments']}")
        print(f"平均クラスタサイズ: {metrics['avg_cluster_size']:.2f}")
        print(f"シングルトン率: {metrics['singleton_ratio']:.2%}")
        
        return metrics
    
    def compute_token_type_metrics(self):
        """トークン・タイプ比率の計算"""
        
        print("\n=== トークン・タイプ分析 ===")
        
        type_token_ratios = []
        
        for file_id, segments in self.results.items():
            tokens = [s['cluster'] for s in segments]
            types = set(tokens)
            
            if len(tokens) > 0:
                ttr = len(types) / len(tokens)
                type_token_ratios.append(ttr)
        
        avg_ttr = np.mean(type_token_ratios)
        
        print(f"平均タイプ・トークン比: {avg_ttr:.3f}")
        print(f"標準偏差: {np.std(type_token_ratios):.3f}")
        
        return {
            'avg_ttr': avg_ttr,
            'std_ttr': np.std(type_token_ratios),
            'ttrs': type_token_ratios
        }
    
    def analyze_cluster_consistency(self, player=None):
        """クラスタの一貫性分析（音響的類似性）"""
        
        print("\n=== クラスタ一貫性分析 ===")
        
        # 各クラスタのセグメント長の分散を計算
        cluster_durations = defaultdict(list)
        
        for segments in self.results.values():
            for seg in segments:
                duration = seg['end'] - seg['start']
                cluster_durations[seg['cluster']].append(duration)
        
        # 持続時間の一貫性
        consistency_scores = {}
        for cluster_id, durations in cluster_durations.items():
            if len(durations) > 1:
                # 変動係数（CV）= 標準偏差 / 平均
                cv = np.std(durations) / np.mean(durations)
                consistency_scores[cluster_id] = 1 - cv  # CVが小さいほど一貫性が高い
        
        avg_consistency = np.mean(list(consistency_scores.values()))
        
        print(f"平均一貫性スコア: {avg_consistency:.3f}")
        print(f"最も一貫性の高いクラスタ TOP 10:")
        
        sorted_clusters = sorted(consistency_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        
        for cluster_id, score in sorted_clusters:
            n_segments = len(cluster_durations[cluster_id])
            avg_duration = np.mean(cluster_durations[cluster_id]) * 1000  # ms
            print(f"  クラスタ {cluster_id}: スコア={score:.3f}, "
                  f"セグメント数={n_segments}, 平均長={avg_duration:.0f}ms")
        
        return consistency_scores
    
    def plot_evaluation_results(self, metrics, save_path="evaluation_results.png"):
        """評価結果の可視化"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. クラスタサイズ分布
        ax1 = axes[0, 0]
        all_clusters = []
        for segments in self.results.values():
            all_clusters.extend([s['cluster'] for s in segments])
        
        cluster_counts = Counter(all_clusters)
        sizes = list(cluster_counts.values())
        
        ax1.hist(sizes, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('クラスタサイズ')
        ax1.set_ylabel('頻度')
        ax1.set_title('クラスタサイズ分布')
        ax1.set_yscale('log')
        
        # 2. セグメント長分布
        ax2 = axes[0, 1]
        all_durations = []
        for segments in self.results.values():
            for seg in segments:
                duration = (seg['end'] - seg['start']) * 1000  # ms
                all_durations.append(duration)
        
        ax2.hist(all_durations, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('セグメント長 (ms)')
        ax2.set_ylabel('頻度')
        ax2.set_title('セグメント長分布')
        
        # 3. ファイルごとのセグメント数
        ax3 = axes[1, 0]
        file_segment_counts = [len(segs) for segs in self.results.values()]
        
        ax3.hist(file_segment_counts, bins=30, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('セグメント数/ファイル')
        ax3.set_ylabel('頻度')
        ax3.set_title('ファイルごとのセグメント数分布')
        
        # 4. タイプ・トークン比分布
        ax4 = axes[1, 1]
        if 'ttrs' in metrics:
            ax4.hist(metrics['ttrs'], bins=30, edgecolor='black', alpha=0.7)
            ax4.set_xlabel('タイプ・トークン比')
            ax4.set_ylabel('頻度')
            ax4.set_title('タイプ・トークン比分布')
            ax4.axvline(x=metrics['avg_ttr'], color='r', linestyle='--', 
                       label=f"平均: {metrics['avg_ttr']:.3f}")
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n評価結果を保存: {save_path}")
        plt.close()
    
    def generate_report(self, output_file="evaluation_report.txt"):
        """評価レポートの生成"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== クラスタリング評価レポート ===\n\n")
            
            # 基本統計
            clustering_metrics = self.compute_clustering_metrics()
            f.write("## クラスタリング統計\n")
            for key, value in clustering_metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # トークン・タイプ分析
            ttr_metrics = self.compute_token_type_metrics()
            f.write("## トークン・タイプ分析\n")
            f.write(f"平均TTR: {ttr_metrics['avg_ttr']:.3f}\n")
            f.write(f"標準偏差: {ttr_metrics['std_ttr']:.3f}\n")
            f.write("\n")
            
            # 一貫性分析
            consistency_scores = self.analyze_cluster_consistency()
            f.write("## クラスタ一貫性\n")
            f.write(f"平均一貫性スコア: {np.mean(list(consistency_scores.values())):.3f}\n")
            
        print(f"\n評価レポートを保存: {output_file}")


def compute_zerospeech_metrics(evaluator):
    """ZeroSpeech Challenge形式の評価指標"""
    
    print("\n=== ZeroSpeech評価指標 ===")
    
    # ABX識別スコア（音素識別能力）
    # 実際の計算には音響特徴量が必要
    print("ABX識別スコア: 音響特徴量が必要です")
    
    # NED (Normalized Edit Distance)
    # 実際の計算には正解転写が必要
    print("NED (正規化編集距離): 正解転写が必要です")
    
    # Coverage（語彙カバレッジ）
    all_clusters = []
    for segments in evaluator.results.values():
        all_clusters.extend([s['cluster'] for s in segments])
    
    cluster_counts = Counter(all_clusters)
    
    # 頻度順に並べ替え
    sorted_counts = sorted(cluster_counts.values(), reverse=True)
    total = sum(sorted_counts)
    
    # 上位N%のクラスタがカバーするトークンの割合
    coverage_ratios = []
    for top_percent in [10, 20, 50]:
        n_clusters = int(len(sorted_counts) * top_percent / 100)
        covered = sum(sorted_counts[:n_clusters])
        ratio = covered / total
        coverage_ratios.append((top_percent, ratio))
        print(f"上位{top_percent}%のクラスタがカバーするトークン: {ratio:.2%}")
    
    return {
        'coverage_ratios': coverage_ratios,
        'vocabulary_size': len(cluster_counts)
    }


def main():
    print("=== クラスタリング評価実験 ===\n")
    
    # 評価器の初期化
    evaluator = ClusteringEvaluator()
    
    # 各種評価の実行
    clustering_metrics = evaluator.compute_clustering_metrics()
    ttr_metrics = evaluator.compute_token_type_metrics()
    consistency_scores = evaluator.analyze_cluster_consistency()
    
    # ZeroSpeech形式の評価
    zerospeech_metrics = compute_zerospeech_metrics(evaluator)
    
    # 結果の可視化
    all_metrics = {**clustering_metrics, **ttr_metrics}
    evaluator.plot_evaluation_results(all_metrics)
    
    # レポート生成
    evaluator.generate_report()
    
    print("\n評価完了！")
    print("生成されたファイル:")
    print("  - evaluation_results.png: 評価結果の可視化")
    print("  - evaluation_report.txt: 詳細レポート")


if __name__ == "__main__":
    main()