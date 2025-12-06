# 不良品識別AIモデル (Defect Classification Model)

本リポジトリは、製造現場の外観検査をAIで自動化するため、製品画像から良品/不良品を予測する二値分類モデルを開発するものである。

**【課題の焦点】**
熟練検査員の経験に依存する不安定な目視検査を代替し、**わずかな見落としによるトラブルを防ぐ**ことを目的としている。

## 評価指標 (KPI)

クライアントの要件「わずかな見落としが後のトラブルにつながる」を踏まえ、以下の指標を最重要KPIに設定する。

* **最重要 KPI:** **再現率 (Recall)**
    * 不良品をどれだけ正確に不良品と判定できたかを示す指標。見逃し（偽陰性/False Negative）の最小化を目的とします。
* **サブ KPI:** Precision, F1-Score

## ⚙️ 開発環境と再現手順

### 必要なもの
- Python 3.x
- Git
- データセット (dataset.zip)

### 環境構築
1. リポジトリをクローンし、ディレクトリへ移動します。
```bash
git clone https://github.com/ReiWada1216/2WINS_AI_test
cd 2WINS_AI_test
```

2. Python仮想環境を構築し、依存関係をインストール。
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. 依存関係をインストール。
```bash
pip install -r requirements.txt
```

4. データセットを解凍。
```bash
unzip dataset.zip
```

5. 学習の実行。
```bash
python src/main.py
```

## 📂 プロジェクト構造
- `src/`: データローダー、モデル、学習ロジックなどの共通コード
- `notebooks/`: 実験の流れと評価を実行するノートブック
- `weights/`: 学習済み重みの保存場所 (Git管理外)
- `results/`: 混同行列、Grad-CAM画像などの可視化結果

## 🔗 成果物と外部リンク

| 提出物 | ファイル/リンク | 備考 |
| :--- | :--- | :--- |
| **報告レポート** | `[Google Drive 共有リンク貼る]` | KPI、試した技術、考察をまとめた最終報告書です。 |
| **学習済み重み** | `[Google Drive 共有リンク貼る]` | 容量制限のため、重みファイルは別途アップロードしています。 |
| **可視化結果** | `results/figures/` 以下に格納 | 混同行列、学習曲線、Grad-CAMの例などを確認できます。 |
