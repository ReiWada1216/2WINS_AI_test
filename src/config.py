
# Configuration for Binary Classification Project

# 画像サイズ
IMAGE_SIZE = 256  # 画像をこのサイズにリサイズ

# 学習設定
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42
EPOCHS = 10
LEARNING_RATE = 1e-4

# データパス
DATA_PROCESSED_DIR = "data/processed"
DATA_RAW_DIR = "data/raw"

# クラス重み
N_GOOD = 1000
N_BAD = 350
CLASS_WEIGHTS_TENSOR = torch.tensor([1.0, N_GOOD / N_BAD], dtype=torch.float32)