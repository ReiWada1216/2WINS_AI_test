# EfficientAD Anomaly Detection Project

このプロジェクトは、Google Colab上で **EfficientAD** を用いた異常検知モデルの実装・実験を行ったものです。
技術面接用のポートフォリオとして、リポジトリ構成を整理しています。

## 📁 ディレクトリ構成

プロジェクトは以下の構成で管理されています。

- **`notebooks/`**
  - モデルの学習および推論を行うための主要な実装が含まれています。
  - Google Colabでの実行を想定しています。
  - 主なファイル:
    - `colab_efficientad.ipynb`: EfficientADの学習・推論用ノートブック
    - `colab_efficientad_training.ipynb`: トレーニング実験用
    - `colab_phase1.ipynb`, `colab_phase2.ipynb`: 各フェーズの実装

- **`dataset/`**
  - モデルの学習・評価に使用するデータセットが格納されています。
  - `phase1_dataset`, `phase2_dataset` などが含まれます。

- **`weights/`**
  - 学習済みのモデル重みファイルが保存されています。
  - `efficientAD_rocauc_0.983.ckpt`: 高精度を達成したEfficientADのチェックポイント
  - その他、比較実験用のバックボーンモデルなど

- **`archive/`**
  - 開発過程で使用したローカルスクリプトや、現在は使用していない試行錯誤のファイル群です。
  - 技術的な詳細を確認したい場合の参照用として残しています。

## 🚀 実行方法

本プロジェクトの実装は **Google Colab** 上で動作することを前提としています。
`notebooks/` ディレクトリ内の `.ipynb` ファイルをGoogle Colabにアップロードし、ランタイムをGPUに設定して実行してください。

## 🧠 モデルと成果

- **Model**: EfficientAD
- **Task**: 異常検知 (Anomaly Detection)
- **Performance**:
  - ROC AUC: 0.983 (Best checkpoint)
  - 非常に高い精度で異常箇所を特定可能です。

## 📝 補足

- コード内のコメントや説明は、実験の経緯を含めて記述されています。
- ローカル環境での再現よりも、Colab環境での再現性を重視しています。
