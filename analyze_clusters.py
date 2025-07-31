"""
クラスタ分析と可視化
"""

import os
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def load_cluster_data(output_dir="data/librispeech/output"):
    """すべての出力ファイルからクラスタデータを読み込む"""
    
    all_clusters = []
    all_durations = []
    file_clusters = defaultdict(list)
    
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"エラー: {output_dir} が見つかりません")
        return None, None, None
    
    # すべての.listファイルを処理
    list_files = list(output_path.glob("*.list"))
    print(f"{len(list_files)}個のファイルを分析中...")
    
    for list_file in tqdm(list_files, desc="ファイル読み込み"):
        with open(list_file, 'r') as f:
            prev_time = 0
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    time = float(parts[0])
                    cluster_id = int(parts[1])
                    
                    all_clusters.append(cluster_id)
                    duration = time - prev_time
                    all_durations.append(duration)
                    file_clusters[list_file.stem].append(cluster_id)
                    
                    prev_time = time
    
    return all_clusters, all_durations, file_clusters


def analyze_cluster_distribution(all_clusters):
    """クラスタ分布の分析"""
    
    print("\n=== クラスタ分布分析 ===")
    
    # 頻度カウント
    cluster_counts = Counter(all_clusters)
    
    print(f"総セグメント数: {len(all_clusters)}")
    print(f"ユニーククラスタ数: {len(cluster_counts)}")
    print(f"平均出現回数: {len(all_clusters) / len(cluster_counts):.2f}")
    
    # 最頻出クラスタ
    print("\n最頻出クラスタ TOP 20:")
    for cluster_id, count in cluster_counts.most_common(20):
        print(f"  クラスタ {cluster_id:4d}: {count:4d}回 ({count/len(all_clusters)*100:5.2f}%)")
    
    # 希少クラスタ
    rare_clusters = [c for c, count in cluster_counts.items() if count == 1]
    print(f"\n1回しか出現しないクラスタ: {len(rare_clusters)}個 ({len(rare_clusters)/len(cluster_counts)*100:.1f}%)")
    
    return cluster_counts


def plot_cluster_distribution(cluster_counts, save_path="cluster_analysis.png"):
    """クラスタ分布の可視化"""
    
    plt.figure(figsize=(15, 10))
    
    # 1. 頻度分布のヒストグラム
    plt.subplot(2, 2, 1)
    frequencies = list(cluster_counts.values())
    plt.hist(frequencies, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('出現回数')
    plt.ylabel('クラスタ数')
    plt.title('クラスタ出現頻度分布')
    plt.yscale('log')
    
    # 2. 上位クラスタの棒グラフ
    plt.subplot(2, 2, 2)
    top_clusters = cluster_counts.most_common(30)
    cluster_ids = [str(c[0]) for c in top_clusters]
    counts = [c[1] for c in top_clusters]
    
    plt.bar(range(len(cluster_ids)), counts, color='skyblue', edgecolor='navy')
    plt.xlabel('クラスタID')
    plt.ylabel('出現回数')
    plt.title('上位30クラスタの出現回数')
    plt.xticks(range(len(cluster_ids)), cluster_ids, rotation=90, fontsize=8)
    
    # 3. 累積分布
    plt.subplot(2, 2, 3)
    sorted_counts = sorted(cluster_counts.values(), reverse=True)
    cumsum = np.cumsum(sorted_counts)
    total = sum(sorted_counts)
    
    plt.plot(range(len(cumsum)), cumsum / total * 100, linewidth=2)
    plt.xlabel('クラスタ数（頻度順）')
    plt.ylabel('累積カバー率 (%)')
    plt.title('クラスタの累積カバー率')
    plt.grid(True, alpha=0.3)
    
    # カバー率のマーカー
    coverage_80 = next(i for i, c in enumerate(cumsum) if c/total >= 0.8)
    coverage_90 = next(i for i, c in enumerate(cumsum) if c/total >= 0.9)
    plt.axhline(y=80, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=90, color='r', linestyle='--', alpha=0.5)
    plt.text(len(cumsum)*0.7, 82, f'80%カバー: {coverage_80}クラスタ', fontsize=10)
    plt.text(len(cumsum)*0.7, 92, f'90%カバー: {coverage_90}クラスタ', fontsize=10)
    
    # 4. Zipfの法則プロット
    plt.subplot(2, 2, 4)
    ranks = range(1, len(sorted_counts) + 1)
    plt.loglog(ranks, sorted_counts, 'b-', linewidth=2)
    plt.xlabel('ランク（log）')
    plt.ylabel('頻度（log）')
    plt.title('Zipfの法則プロット')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n可視化を保存: {save_path}")
    plt.close()


def analyze_segment_durations(all_durations):
    """セグメント長の分析"""
    
    print("\n=== セグメント長分析 ===")
    
    durations_ms = [d * 1000 for d in all_durations]  # ミリ秒に変換
    
    print(f"平均セグメント長: {np.mean(durations_ms):.1f} ms")
    print(f"中央値: {np.median(durations_ms):.1f} ms")
    print(f"標準偏差: {np.std(durations_ms):.1f} ms")
    print(f"最短: {np.min(durations_ms):.1f} ms")
    print(f"最長: {np.max(durations_ms):.1f} ms")
    
    # セグメント長の分布を可視化
    plt.figure(figsize=(10, 6))
    plt.hist(durations_ms, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('セグメント長 (ms)')
    plt.ylabel('頻度')
    plt.title('セグメント長の分布')
    plt.axvline(x=np.mean(durations_ms), color='r', linestyle='--', label=f'平均: {np.mean(durations_ms):.1f}ms')
    plt.axvline(x=np.median(durations_ms), color='g', linestyle='--', label=f'中央値: {np.median(durations_ms):.1f}ms')
    plt.legend()
    plt.savefig('segment_durations.png', dpi=300, bbox_inches='tight')
    print("セグメント長分布を保存: segment_durations.png")
    plt.close()


def find_cluster_examples(file_clusters, target_cluster, n_examples=5):
    """特定クラスタの出現例を探す"""
    
    examples = []
    for filename, clusters in file_clusters.items():
        indices = [i for i, c in enumerate(clusters) if c == target_cluster]
        for idx in indices[:n_examples]:
            examples.append((filename, idx))
            if len(examples) >= n_examples:
                break
        if len(examples) >= n_examples:
            break
    
    return examples


def main():
    print("=== クラスタ分析と可視化 ===\n")
    
    # データ読み込み
    all_clusters, all_durations, file_clusters = load_cluster_data()
    
    if all_clusters is None:
        return
    
    # クラスタ分布の分析
    cluster_counts = analyze_cluster_distribution(all_clusters)
    
    # 可視化
    plot_cluster_distribution(cluster_counts)
    
    # セグメント長の分析
    analyze_segment_durations(all_durations)
    
    # 最頻出クラスタの例を表示
    print("\n=== 最頻出クラスタの出現例 ===")
    top_5_clusters = cluster_counts.most_common(5)
    for cluster_id, count in top_5_clusters:
        print(f"\nクラスタ {cluster_id} (出現回数: {count}):")
        examples = find_cluster_examples(file_clusters, cluster_id, 3)
        for filename, idx in examples:
            print(f"  - {filename} のセグメント #{idx}")
    
    print("\n分析完了！")


if __name__ == "__main__":
    main()