# LibriSpeech dev-clean 実行ガイド

このガイドでは、LibriSpeech dev-cleanデータセットを使用してprom-seg-clusを実行する手順を説明します。

## 必要な環境

- Python 3.10以上
- 約2GB以上の空きディスク容量
- （オプション）CUDA対応GPU（HuBERTエンコードの高速化）

## 実行手順

### 1. 依存関係のインストール

```bash
uv sync
```

### 2. LibriSpeech dev-cleanのダウンロードと準備

```bash
uv run python setup_librispeech.py
```

このスクリプトは以下を実行します：
- LibriSpeech dev-clean（約337MB）のダウンロード
- 音声ファイルの解凍と整理
- サンプルとして最初の50ファイルを処理用にコピー

### 3. HuBERTエンコード

```bash
uv run python setup_hubert.py
```

このスクリプトは以下を実行します：
- HuBERT Baseモデルのダウンロード（初回のみ）
- 音声ファイルをHuBERTでエンコード（第6層の特徴量を抽出）
- 特徴量を.npy形式で保存

### 4. Prominence境界抽出

```bash
uv run python setup_prominence.py
```

このスクリプトは以下を実行します：
- エネルギーベースの簡易的な境界検出
- 各音声ファイルの単語境界を.list形式で保存

注意：これは簡易実装です。本格的な実装は[prom-word-seg](https://github.com/s-malan/prom-word-seg)を使用してください。

### 5. クラスタリング実行

```bash
uv run python run_librispeech_pipeline.py
```

このスクリプトは以下を実行します：
- すべての前処理が完了しているかチェック
- prom_seg_clus.pyを適切なパラメータで実行
- 結果の簡単な統計を表示

## 期待される結果

- `data/librispeech/output/`に各音声ファイルのセグメントとクラスタIDが保存されます
- 各.listファイルには以下の形式でデータが含まれます：
  ```
  時間（秒） クラスタID
  0.450 234
  0.870 1567
  1.230 234
  ...
  ```

## トラブルシューティング

### メモリ不足エラー
- `--sample_size`パラメータで処理するファイル数を減らしてください
- より小さいK_max（クラスタ数）を使用してください

### HuBERTのダウンロードが遅い
- 初回実行時はモデルのダウンロードに時間がかかります
- 一度ダウンロードされればキャッシュされます

### GPU最適化
- RTX 4070 Ti Super (16GB VRAM)に最適化されています
- Mixed precisionとcuDNN最適化が自動的に有効になります
- バッチサイズはGPUメモリに応じて自動調整されます

### 境界抽出アルゴリズム
- s-malan/prom-word-seg の正確な実装を使用
- フレーム間特徴量距離とprominence-based peak detectionを実装
- パラメータ調整可能（window_size、prominence_threshold等）

## カスタマイズ

### より多くのファイルを処理
`setup_librispeech.py`の50ファイル制限を変更：
```python
# 50を任意の数に変更（または制限を削除）
for i, flac_path in enumerate(tqdm(flac_files[:50], desc="音声ファイルをコピー")):
```

### クラスタ数の変更
`run_librispeech_pipeline.py`のK_maxパラメータを変更：
```python
"5000",  # この値を変更
```