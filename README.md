
project_root/
├── data/
│   ├── raw/            <- dataset.zip と、そこから解凍された元の good/bad フォルダ
│   └── processed/      <- 学習用にリサイズした画像や、分割済みリスト(train.csvなど)
├── models/             <- 学習済みモデルの重み (.pth) ※Gitにはあげない
├── notebooks/          <- 試行錯誤とレポート用
│   ├── 01_eda_and_preprocess.ipynb  <- データ確認、リサイズ、分割
│   └── 02_train_and_report.ipynb    <- 学習実行、評価、可視化、考察
├── src/                <- エンジニアリング用（共通部品）
│   ├── __init__.py
│   ├── config.py       <- ★重要：パスや定数（画像サイズ、学習率）を一箇所で管理
│   ├── dataset.py      <- 画像の読み込み、Transform(データ拡張)の定義
│   ├── model.py        <- モデル構造(ResNetなど)の定義
│   ├── trainer.py      <- 学習ループの関数化（train_one_epochなど）
│   └── utils.py        <- 乱数固定、可視化関数(Grad-CAMなど)
├── results/            <- 提出用の成果物
│   ├── figures/        <- 混同行列、Lossカーブ、ヒートマップ画像
│   └── report.pdf      <- 最終レポート
├── .gitignore          <- data/ や models/ を除外設定
├── README.md           <- プロジェクトの「表紙」。実行手順を記載
└── requirements.txt    <- 必要なライブラリ一覧