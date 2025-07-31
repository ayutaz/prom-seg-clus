"""
LibriSpeech dev-cleanの完全な実行パイプライン
"""

import os
import subprocess
from pathlib import Path


def check_requirements():
    """必要なデータが揃っているか確認"""

    required_dirs = [
        "data/librispeech/audio",
        "data/librispeech/features/hubert/layer_6",
        "data/librispeech/boundaries",
    ]

    missing = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing.append(dir_path)
        else:
            # ファイルが存在するかチェック
            files = list(Path(dir_path).glob("*"))
            if len(files) == 0:
                missing.append(f"{dir_path} (空)")

    return missing


def run_clustering():
    """prom_seg_clusを実行"""

    import sys

    # uvで実行するように修正
    cmd = [
        "uv",
        "run",
        "python",
        "prom_seg_clus.py",
        "hubert",  # モデル名
        "6",  # レイヤー番号
        "data/librispeech/audio",
        "data/librispeech/features",
        "data/librispeech/boundaries",
        "data/librispeech/output",
        "5000",  # K_max（クラスタ数）
        "--extension",
        ".flac",
        "--sample_size",
        "-1",  # すべてのデータを使用
    ]

    print("実行コマンド:")
    print(" ".join(cmd))
    print()

    # 実行
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("クラスタリング完了!")
        print(result.stdout)
    else:
        print("エラーが発生しました:")
        print(result.stderr)


def analyze_results():
    """結果の簡単な分析"""

    output_dir = Path("data/librispeech/output")

    if not output_dir.exists():
        print("\n出力ディレクトリが見つかりません")
        return

    output_files = list(output_dir.glob("*.list"))

    if not output_files:
        print("\n出力ファイルが見つかりません")
        return

    print("\n=== 結果分析 ===")
    print(f"処理されたファイル数: {len(output_files)}")

    # 最初のファイルの内容を表示
    if output_files:
        print(f"\nサンプル出力 ({output_files[0].name}):")
        with open(output_files[0]) as f:
            lines = f.readlines()[:10]  # 最初の10行
            for line in lines:
                print(f"  {line.strip()}")

        if len(lines) == 10:
            print("  ...")

    # クラスタ統計
    all_clusters = []
    for output_file in output_files:
        with open(output_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    cluster_id = int(parts[1])
                    all_clusters.append(cluster_id)

    if all_clusters:
        unique_clusters = len(set(all_clusters))
        print(f"\n使用されたクラスタ数: {unique_clusters}")
        print(f"総セグメント数: {len(all_clusters)}")
        print(f"平均セグメント/ファイル: {len(all_clusters) / len(output_files):.1f}")


def main():
    print("=== LibriSpeech dev-clean 実行パイプライン ===\n")

    # 要件チェック
    missing = check_requirements()

    if missing:
        print("エラー: 以下のディレクトリ/ファイルが不足しています:")
        for item in missing:
            print(f"  - {item}")

        print("\n以下の順番で実行してください:")
        print("1. uv sync  # 依存関係のインストール")
        print("2. uv run python setup_librispeech.py  # データダウンロード")
        print("3. uv run python setup_hubert.py  # HuBERTエンコード")
        print("4. uv run python setup_prominence.py  # 境界抽出")
        print("5. uv run python run_librispeech_pipeline.py  # このスクリプト")
        return

    print("すべての要件が揃っています\n")

    # クラスタリング実行
    print("=== クラスタリング実行 ===\n")
    run_clustering()

    # 結果分析
    analyze_results()

    print("\n=== 完了 ===")
    print("結果は data/librispeech/output/ に保存されました")


if __name__ == "__main__":
    main()
