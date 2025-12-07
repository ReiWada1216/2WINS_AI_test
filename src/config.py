import os
from pathlib import Path

# ==========================================
# プロジェクト共通の定数設定
# ==========================================

# 乱数シード（再現性確保のため）
SEED: int = 42

# 画像サイズ設定
ORIGINAL_SIZE: int = 1024  # 元画像のサイズ
PATCH_SIZE: int = 224       # 学習・推論に使用するパッチサイズ

# 学習設定
BATCH_SIZE: int = 32
NUM_WORKERS: int = 2     # DataLoaderのワーカー数
EPOCHS: int = 100
LEARNING_RATE: float = 1e-4

# クラスラベル定義
CLASS_LABELS: dict[int, str] = {
    0: "Good",
    1: "Bad"
}
NUM_CLASSES: int = len(CLASS_LABELS)

# 判定閾値 (Evaluation)
THRESHOLD: float = 0.8

# クラスの不均衡対策 (Good: 1000, Bad: 350)
# Badクラスの重みを高く設定する
# 計算例: N_total / (N_class * n_classes) or Simple inverse ratio
# ここでは簡易的に Good=1.0 としたときの相対比率を設定
# 1000 / 350 ≈ 2.857
CLASS_WEIGHTS: list[float] = [1.0, 2.86]

# ==========================================
# パス設定
# ==========================================

# プロジェクトルートディレクトリ（このファイルの2つ上の階層をルートとする）
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()

# データディレクトリ
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# モデル重み保存ディレクトリ
WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"

# 出力ディレクトリ（ログや結果）
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# ディレクトリが存在しない場合は作成
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, WEIGHTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)