"""
LibriSpeech dev-cleanデータセットのセットアップスクリプト
音声ファイルのダウンロード、解凍、整理を行う
"""

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm


def download_file(url, filename):
    """URLからファイルをダウンロード（プログレスバー付き）"""

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)


def setup_librispeech():
    """LibriSpeech dev-cleanをセットアップ"""

    # ディレクトリ構造作成
    dirs = [
        "data/librispeech/audio",
        "data/librispeech/features/hubert/layer_6",
        "data/librispeech/boundaries",
        "data/librispeech/output",
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # LibriSpeech dev-cleanをダウンロード
    librispeech_url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    tar_path = "dev-clean.tar.gz"

    if not os.path.exists(tar_path):
        print("LibriSpeech dev-cleanをダウンロード中...")
        download_file(librispeech_url, tar_path)
    else:
        print("dev-clean.tar.gz は既に存在します")

    # 解凍
    if not os.path.exists("LibriSpeech"):
        print("\n解凍中...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall()
        print("解凍完了")

    # FLACファイルを整理してコピー
    print("\n音声ファイルを整理中...")
    source_dir = Path("LibriSpeech/dev-clean")
    target_dir = Path("data/librispeech/audio")

    flac_files = list(source_dir.glob("**/*.flac"))
    print(f"{len(flac_files)}個のFLACファイルが見つかりました")

    # サンプルとして最初の50ファイルのみ処理（全部だと時間がかかるため）
    for _i, flac_path in enumerate(tqdm(flac_files[:50], desc="音声ファイルをコピー")):
        # スピーカーID-チャプターID-発話ID の形式でファイル名を作成
        parts = flac_path.parts
        speaker_id = parts[-3]
        chapter_id = parts[-2]
        utterance_id = flac_path.stem

        new_filename = f"{speaker_id}-{chapter_id}-{utterance_id}.flac"
        target_path = target_dir / new_filename

        shutil.copy2(flac_path, target_path)

    print(f"\n{min(50, len(flac_files))}個のファイルをdata/librispeech/audioにコピーしました")

    # アライメント情報のダウンロード
    # alignment_url = "https://zenodo.org/records/2619474/files/librispeech_alignments.zip"
    # 注：実際のアライメントデータは大きいため、ここでは省略

    print("\n=== セットアップ完了 ===")
    print("次のステップ:")
    print("1. HuBERTエンコーダーのセットアップ: uv run python setup_hubert.py")
    print("2. Prominence境界抽出のセットアップ: uv run python setup_prominence.py")
    print("3. 完全なパイプライン実行: uv run python run_librispeech_pipeline.py")


if __name__ == "__main__":
    setup_librispeech()
