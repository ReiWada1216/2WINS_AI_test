import os
from pathlib import Path

# ==========================================
# 共通定数 (Common)
# ==========================================

# 乱数シード
SEED: int = 42

# パス設定
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# ディレクトリ作成
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, WEIGHTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 画像設定
ORIGINAL_SIZE: int = 1024
PATCH_SIZE: int = 256
STRIDE: int = 128
CHANNELS: int = 1
BACKGROUND_THRESHOLD: int = 40

# データローダー設定
NUM_WORKERS: int = 2

# クラスラベル
CLASS_LABELS: dict[int, str] = {
    0: "Good",
    1: "Bad"
}
NUM_CLASSES: int = len(CLASS_LABELS)
CLASS_WEIGHTS: list[float] = [1.0, 2.86] # Phase 1用

# ==========================================
# Phase 1: Classification (CNN) 設定
# ==========================================
PHASE1_CONFIG = {
    "phase_name": "Phase 1 (Classification)",
    "model_type": "simple_cnn",
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 1e-3,
    "img_size": PATCH_SIZE,
    "threshold": 0.5, # 判定確率の閾値
}

# ==========================================
# Phase 2: Anomaly Detection (CAE) 設定
# ==========================================
PHASE2_CONFIG = {
    "phase_name": "Phase 2 (AE)",
    "model_type": "cae",
    "batch_size": 32,
    "epochs": 30, # デフォルト30だがCAEは長めが良い場合も
    "learning_rate": 1e-3,
    "img_size": PATCH_SIZE,
    "latent_dim": 128, # ボトルネック等の次元数 (参考値)
    "threshold": 0.05, # MSE閾値 (初期値)
}