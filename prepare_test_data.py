"""
LibriSpeechデータセット準備スクリプト
このスクリプトは、LibriSpeech dev-cleanデータセットをダウンロードし、
prom-seg-clusプロジェクトで使用できる形式に準備します。
"""

import os
import urllib.request
import tarfile
from pathlib import Path
import subprocess

def download_librispeech():
    """LibriSpeech dev-cleanをダウンロード"""
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    output_path = "dev-clean.tar.gz"
    
    if not os.path.exists(output_path):
        print("LibriSpeech dev-cleanをダウンロード中...")
        urllib.request.urlretrieve(url, output_path)
        print("ダウンロード完了")
    
    # 解凍
    if not os.path.exists("LibriSpeech"):
        print("解凍中...")
        with tarfile.open(output_path, "r:gz") as tar:
            tar.extractall()
        print("解凍完了")
    
    return "LibriSpeech/dev-clean"

def prepare_directory_structure():
    """必要なディレクトリ構造を作成"""
    dirs = [
        "data/audio",
        "data/features/hubert/layer_6",
        "data/boundaries",
        "data/output"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("ディレクトリ構造を作成しました")
    return dirs

def convert_to_wav(librispeech_dir, output_dir):
    """FLACファイルをWAVに変換（必要に応じて）"""
    import glob
    
    flac_files = glob.glob(f"{librispeech_dir}/**/*.flac", recursive=True)
    print(f"{len(flac_files)}個のFLACファイルが見つかりました")
    
    # 最初の10ファイルだけ処理（テスト用）
    for i, flac_file in enumerate(flac_files[:10]):
        wav_file = os.path.join(output_dir, f"{i:04d}.wav")
        # Note: ffmpegが必要
        # subprocess.run(["ffmpeg", "-i", flac_file, wav_file])
        print(f"変換が必要: {flac_file} -> {wav_file}")
    
    return len(flac_files[:10])

def main():
    print("=== LibriSpeechテストデータ準備 ===\n")
    
    # 1. ディレクトリ構造を作成
    dirs = prepare_directory_structure()
    
    # 2. LibriSpeechをダウンロード
    librispeech_dir = download_librispeech()
    
    # 3. 音声ファイルの準備
    num_files = convert_to_wav(librispeech_dir, "data/audio")
    
    print("\n=== 次のステップ ===")
    print("1. HuBERTで音声をエンコード:")
    print("   https://github.com/bshall/hubert を参照")
    print("\n2. Prominence-based境界を抽出:")
    print("   https://github.com/s-malan/prom-word-seg を参照")
    print("\n3. prom-seg-clusを実行:")
    print("   uv run python prom_seg_clus.py hubert 6 data/audio data/features data/boundaries data/output 5000")

if __name__ == "__main__":
    main()