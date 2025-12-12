# Defect Classification (良品/不良品分類)

本プロジェクトは、製造ラインにおける製品の良品・不良品を自動識別するためのAIモデル開発・検証リポジトリです。
転移学習を用いた **ResNet50** および **EfficientNet-B4** の性能比較を行い、異常検知（分類）タスクにおける有効性を確認しています。

主な実験は Google Colab 上で実施されており、主要なノートブックとして `notebooks/resnet_efficientnet.ipynb` があります。

## 📊 モデル性能比較結果

`notebooks/resnet_efficientnet.ipynb` における実験結果は以下の通りです。
EfficientNet-B4 が非常に高い精度（ROC-AUC 1.0）を達成していますが、推論速度（レイテンシ）は ResNet50 が優れています。

| Model | ROC-AUC | F1 Score | Latency (ms/image) | 備考 |
| :--- | :---: | :---: | :---: | :--- |
| **ResNet50** | 0.9757 | 0.9275 | **15.64** | 高速な推論が可能 |
| **EfficientNet-B4** | **1.0** | **0.9859** | 26.50 | 非常に高い分類精度 |

## 📁 ディレクトリ構成

- **`notebooks/`**
  - **`resnet_efficientnet.ipynb`**: 本プロジェクトの主要な実験ノートブック。ResNet50とEfficientNet-B4の学習、評価、比較を行っています。
  
- **`dataset/`**
  - 学習・評価用データセット（Phase 1/2 データセット等）

- **`archive/`**
  - `src/`: 過去の実験スクリプト（EfficientADの実験など）
    - `train_efficientad.py`
    - `evaluate_efficientad.py`
    - `visualize_inference.py`
  - `config.yaml`: 設定ファイル
  - `requirements.txt`: 依存ライブラリ
  - `results/`: 実験ログ・チェックポイント

## 🚀 実験の概要と実行

### `notebooks/resnet_efficientnet.ipynb` の内容
このノートブックでは以下の手順で実験を行っています。

1. **環境設定**: PyTorch, WandB, 必要なライブラリのインストール
2. **データセット準備**: `ClassificationDataset` クラスによる独自データセットの読み込み（Data Augmentation含む）
3. **モデル構築**:
   - **ResNet50**: ImageNet事前学習済みモデルをベースに、最終層を2クラス分類用に変更
   - **EfficientNet-B4**: ImageNet事前学習済みモデルをベースに、Classifierを変更
4. **学習**: F1 Scoreを最大化する閾値探索を行いながら学習を実施
5. **評価**: ROC-AUC, F1 Score, 混同行列による性能評価

### 実行方法
Google Colabにノートブックをアップロードし、GPUランタイムで実行してください。
データセットは適切なパス（例: `/content/phase1_dataset.zip`）に配置する必要があります。

## 🧠 相対比較と考察
- **精度面**: EfficientNet-B4 は非常に強力な特徴抽出能力を持ち、今回のタスクにおいて完璧に近い分類性能を示しました。
- **速度面**: ResNet50 は軽量で高速であり、リアルタイム性が求められる環境では依然として有力な選択肢となります。

## 📝 過去の実験 (EfficientAD)
`archive/src/` には、教師なし異常検知モデルである **EfficientAD** を用いた実験コードも含まれています。
こちらは良品画像のみで学習を行い、異常箇所をセグメンテーション（可視化）するタスクに特化しています。
