# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは、音声セグメントのプロミネンスベース単語分割とクラスタリングによる語彙学習のためのコードです。FAISSを使用したK-meansクラスタリングにより、音声特徴から抽出された単語セグメントの埋め込みをクラスタリングします。

## セットアップ

### 依存関係のインストール
```bash
uv sync
```

### Python環境の有効化
```bash
# Windows
.venv\Scripts\activate
# Unix/macOS
source .venv/bin/activate
```

## 開発環境

### コード品質管理
このプロジェクトはruffを使用してコード品質を管理しています。

```bash
# フォーマットとlintチェック
make check

# 個別実行
uv run ruff format .  # コードフォーマット
uv run ruff check . --fix  # lintチェックと自動修正

# pre-commitフックのインストール（初回のみ）
uv run pre-commit install
```

## 主要コマンド

### 特徴操作とクラスタリング実行
```bash
uv run python prom_seg_clus.py <model_name> <layer> <path/to/audio> <path/to/features> <path/to/boundaries> <path/to/output> <k_max> [--extension] [--sample_size] [--speaker]
```

引数:
- `model_name`: 使用するモデル名（例: hubert, mfcc, melspec）
- `layer`: モデルレイヤー番号（レイヤーがない場合は-1）
- `k_max`: K-meansクラスタ数
- `--extension`: オーディオファイル形式（デフォルト: .flac）
- `--sample_size`: サンプル数（デフォルト: -1で全データ）
- `--speaker`: 話者別クラスタリング用のJSONファイル

## アーキテクチャ

### ディレクトリ構造
- `prom_seg_clus.py`: メインスクリプト。音声特徴の前処理、クラスタリング、セグメント保存を行う
- `utils/data_process.py`: Features クラスによる音声特徴のサンプリング、正規化、PCA適用
- `wordseg/cluster.py`: CLUSTER クラスによるFAISS K-meansクラスタリング実装
- `wordseg/subsample.py`: セグメント埋め込みのダウンサンプリング

### 主要処理フロー
1. **データ準備**: VADで発話を抽出し、HuBERTなどで音声エンコード
2. **境界抽出**: プロミネンスベースの単語境界を事前に抽出
3. **特徴処理**: PCA（250次元）で次元削減、L2正規化
4. **クラスタリング**: FAISS K-meansでセグメント埋め込みをクラスタリング
5. **保存**: 各発話のセグメント境界とクラスタIDを.listファイルに保存

### モデル固有設定
- MFCC/Melspec: フレーム長10ms、PCAなし
- その他（HuBERT等）: フレーム長20ms、PCA適用

### 依存関係管理
このプロジェクトはuvを使用して依存関係を管理しています。`pyproject.toml`で定義された依存関係:
- numpy>=1.24.0
- scikit-learn>=1.3.0 (PCA, StandardScaler)
- faiss-cpu>=1.7.0
- torch>=2.0.0
- tqdm>=4.65.0